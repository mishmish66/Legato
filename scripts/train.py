import argparse
import io
import os
from pathlib import Path

import hydra
import numpy as np
import requests
import torch
from einops import einsum, pack, rearrange, repeat
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

import wandb
from legato.loss import (
    CondensationLoss,
    ConsistencyLoss,
    CoverageLoss,
    SmoothnessLoss,
    TransitionLoss,
)
from legato.nets import DoublePerceptron, Freqceptron, Perceptron, TransitionModel
from legato.sampler import PBallSampler


def train(
    state_encoder,
    action_encoder,
    transition_model,
    state_decoder,
    action_decoder,
    states,
    actions,
    states_test,
    actions_test,
    np_rng,
    config,
    slow_bar=False,
):
    """Train the networks."""

    mininterval = 30 if slow_bar else 0.1

    perturbation_generator = PBallSampler(2, 1, 1.0, device="cuda")
    latent_state_sampler = PBallSampler(4, 1, config.state_space_size, device="cuda")
    latent_action_sampler = PBallSampler(2, 1, config.action_space_size, device="cuda")
    latent_action_state_sampler = lambda n: (
        latent_action_sampler(n),
        latent_state_sampler(n),
    )

    action_mse = nn.MSELoss()
    state_mse = nn.MSELoss()

    transition_loss_func = torch.compile(TransitionLoss())

    smoothness_loss_func = SmoothnessLoss()

    state_coverage_loss_params = config.loss_params.state_coverage_loss_params
    state_coverage_loss_func = torch.compile(
        CoverageLoss(
            latent_state_sampler,
            latent_samples=state_coverage_loss_params.latent_samples,
            selection_tail_size=state_coverage_loss_params.selection_tail_size,
            far_sample_count=state_coverage_loss_params.far_sample_count,
            pushing_sample_size=state_coverage_loss_params.pushing_sample_size,
        )  # , disable=True
    )
    action_coverage_loss_params = config.loss_params.action_coverage_loss_params
    action_coverage_loss_func = torch.compile(
        CoverageLoss(
            latent_action_sampler,
            latent_samples=action_coverage_loss_params.latent_samples,
            selection_tail_size=action_coverage_loss_params.selection_tail_size,
            far_sample_count=action_coverage_loss_params.far_sample_count,
            pushing_sample_size=action_coverage_loss_params.pushing_sample_size,
        )  # , disable=True
    )

    consistency_loss_func = torch.compile(
        ConsistencyLoss(
            latent_state_sampler=latent_state_sampler,
            latent_action_sampler=latent_action_sampler,
            state_decoder=state_decoder,
            action_decoder=action_decoder,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
        )  # , disable=True
    )

    condensation_loss_func = CondensationLoss(
        state_space_size=config.state_space_size,
        action_space_size=config.action_space_size,
        loss_function=nn.L1Loss(),
    )

    epoch_state_actions = config.batching_params.epoch_state_actions
    epoch_trajectories = config.batching_params.epoch_trajectories

    encoder_batch_size = config.batching_params.encoder_batch_size
    transition_batch_size = config.batching_params.transition_batch_size

    test_epoch_steps = config.batching_params.test_epoch_steps

    encoder_grad_skips = config.batching_params.encoder_grad_skips

    encoder_epochs = config.batching_params.encoder_epochs
    transition_epochs = config.batching_params.transition_epochs

    train_epochs = config.batching_params.train_epochs

    transition_warmup_epochs = config.batching_params.transition_warmup_epochs
    encoder_warmup_epochs = config.batching_params.encoder_warmup_epochs
    transition_finetune_epochs = config.batching_params.transition_finetune_epochs

    encoder_optimizer = torch.optim.AdamW(
        [
            param
            for net in [
                state_encoder,
                action_encoder,
                state_decoder,
                action_decoder,
            ]
            for param in net.parameters()
        ],
        lr=config.optimizer_params.encoder_lr,
    )
    encoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        encoder_optimizer,
        gamma=1.0 ** (1 / train_epochs),
    )

    transition_optimizer = torch.optim.AdamW(
        transition_model.parameters(),
        lr=config.optimizer_params.transition_lr,
    )
    transition_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        transition_optimizer,
        gamma=1.0 ** (1 / train_epochs),
    )

    def get_state_action_batch(np_rng, batch_size, states=states, actions=actions):
        """Get a batch of states and actions."""

        np_encoder_batch_raveled_inds = np_rng.permutation(np.prod(states.shape[:-1]))[
            :batch_size
        ]
        encoder_batch_raveled_inds = torch.tensor(
            np_encoder_batch_raveled_inds, device=states.device
        )
        encoder_batch_inds = torch.unravel_index(
            encoder_batch_raveled_inds, states.shape[:-1]
        )

        state_batch = states[encoder_batch_inds].to("cuda")
        action_batch = actions[encoder_batch_inds].to("cuda")

        return state_batch, action_batch

    def get_traj_batch(np_rng, batch_size, states=states, actions=actions):
        """Get a batch of trajectories."""

        perm_num = states.shape[0]
        if batch_size > states.shape[0]:
            repeat_count = batch_size // states.shape[0] + 1
            perm_num = perm_num * repeat_count
        np_transition_traj_batch_inds = (
            np_rng.permutation(perm_num)[:batch_size] % states.shape[0]
        )
        transition_traj_batch_inds = torch.tensor(
            np_transition_traj_batch_inds, device=states.device
        )
        transition_start_inds = torch.randint(
            0,
            int(states.shape[-2] // 1.1),
            (batch_size,),
            device=states.device,
        )

        state_traj_batch = states[transition_traj_batch_inds].to("cuda")
        action_traj_batch = actions[transition_traj_batch_inds].to("cuda")

        return state_traj_batch, action_traj_batch, transition_start_inds

    def perturb_actions(latent_traj_actions):

        # Generate random perturbations
        perturbations = perturbation_generator(len(latent_traj_actions))
        perturbations = torch.randn_like(latent_traj_actions)
        # Normalize the perturbations
        perturbations = perturbations / torch.norm(
            perturbations, p=1, dim=-1, keepdim=True
        )
        # Scale the perturbations between 0 and 1
        perturbations = perturbations * torch.rand(
            (*perturbations.shape[:-1], 1), device="cuda"
        )
        # Get perturbed actions
        latent_traj_actions_perturbed = latent_traj_actions + perturbations
        # Get the norms of the perturbed actions
        perturbed_action_norms = torch.norm(
            latent_traj_actions_perturbed, p=1, dim=-1, keepdim=True
        )
        # Scale the perturbed actions to the action space size
        perturbed_actions_scaled = (
            latent_traj_actions_perturbed
            / perturbed_action_norms
            * config.action_space_size
        )
        # Now use the scaled perturbed actions if the norms are too large
        latent_traj_actions_perturbed = torch.where(
            perturbed_action_norms > config.action_space_size,
            perturbed_actions_scaled,
            latent_traj_actions_perturbed,
        )

        return latent_traj_actions_perturbed

    def encoder_step(
        step,
        flat_states,
        flat_actions,
        traj_start_inds,
        traj_states,
        traj_actions,
    ):
        if step % encoder_grad_skips == 0:
            encoder_optimizer.zero_grad()
            transition_optimizer.zero_grad()

        flat_encoded_states = state_encoder(flat_states)
        flat_encoded_actions = action_encoder((flat_actions, flat_encoded_states))

        flat_reconstructed_states = state_decoder(flat_encoded_states)
        flat_reconstructed_actions = action_decoder(
            (flat_encoded_actions, flat_encoded_states)
        )

        state_reconstruction_loss = state_mse(flat_reconstructed_states, flat_states)
        action_reconstruction_loss = action_mse(
            flat_reconstructed_actions, flat_actions
        )

        state_coverage_loss = state_coverage_loss_func(flat_encoded_states)
        action_coverage_loss = action_coverage_loss_func(flat_encoded_actions)
        condensation_loss = condensation_loss_func(
            flat_encoded_states, flat_encoded_actions
        )

        traj_start_states = traj_states[torch.arange(len(traj_states)), traj_start_inds]
        traj_latent_start_states = state_encoder(traj_start_states)
        traj_latent_states = state_encoder(traj_states)
        traj_latent_actions = action_encoder((traj_actions, traj_latent_states))

        traj_latent_fut_states_prime, mask = transition_model(
            traj_latent_start_states,
            traj_latent_actions,
            start_indices=traj_start_inds.cuda(),
            return_mask=True,
        )

        transition_loss = transition_loss_func(
            traj_latent_fut_states_prime, traj_latent_states, mask
        )

        all_perturbed_latent_traj_actions = perturb_actions(traj_latent_actions)
        perturb_inds = np_rng.integers(traj_start_inds.cpu(), traj_states.shape[-2])
        perturbed_latent_traj_actions = traj_latent_actions.clone()
        perturbed_latent_traj_actions[
            torch.arange(len(perturbed_latent_traj_actions)), perturb_inds
        ] = all_perturbed_latent_traj_actions[
            torch.arange(len(all_perturbed_latent_traj_actions)), perturb_inds
        ]

        traj_latent_fut_states_prime_perturbed = transition_model(
            traj_latent_start_states,
            perturbed_latent_traj_actions,
            start_indices=traj_start_inds.cuda(),
        )

        smoothness_loss = smoothness_loss_func(
            traj_latent_actions,
            traj_latent_fut_states_prime,
            perturbed_latent_traj_actions,
            traj_latent_fut_states_prime_perturbed,
            mask,
        )

        consistency_loss = consistency_loss_func()

        loss_weights = config.loss_params.loss_weights
        encoder_loss = (
            state_reconstruction_loss * loss_weights.state_reconstruction
            + action_reconstruction_loss * loss_weights.action_reconstruction
            + condensation_loss * loss_weights.condensation
            + smoothness_loss * loss_weights.smoothness
            + transition_loss * loss_weights.transition
            + state_coverage_loss * loss_weights.state_coverage
            + action_coverage_loss * loss_weights.action_coverage
            + consistency_loss * loss_weights.consistency
        )

        encoder_loss.backward()

        if step % encoder_grad_skips == encoder_grad_skips - 1:
            encoder_optimizer.step()

        wandb.log(
            {
                "state_reconstruction_loss": state_reconstruction_loss.item(),
                "action_reconstruction_loss": action_reconstruction_loss.item(),
                "state_coverage_loss": state_coverage_loss.item(),
                "action_coverage_loss": action_coverage_loss.item(),
                "condensation_loss": condensation_loss.item(),
                "transition_loss": transition_loss.item(),
                "smoothness_loss": smoothness_loss.item(),
                "consistency_loss": consistency_loss.item(),
                "encoder_loss": encoder_loss.item(),
                "encoder_lr": encoder_lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

    def transition_step(step, traj_states, traj_actions, traj_start_inds):
        transition_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        traj_start_states = traj_states[torch.arange(len(traj_states)), traj_start_inds]
        traj_latent_start_states = state_encoder(traj_start_states)
        traj_latent_states = state_encoder(traj_states)
        traj_latent_actions = action_encoder((traj_actions, traj_latent_states))

        traj_latent_fut_states_prime, mask = transition_model(
            traj_latent_start_states,
            traj_latent_actions,
            start_indices=traj_start_inds.cuda(),
            return_mask=True,
        )

        transition_loss = transition_loss_func(
            traj_latent_fut_states_prime, traj_latent_states, mask
        )

        transition_loss.backward()
        transition_optimizer.step()

        wandb.log(
            {
                "transition_loss": transition_loss.item(),
                "transition_lr": transition_lr_scheduler.get_last_lr()[0],
            },
            step=step,
        )

    def encoder_epoch(step, epoch):
        epoch_steps = epoch_state_actions // encoder_batch_size
        _epoch_state_actions = int(encoder_batch_size * epoch_steps)
        _epoch_trajectories = int(transition_batch_size * epoch_steps)

        flat_states, flat_actions = get_state_action_batch(np_rng, _epoch_state_actions)
        batchified_states = rearrange(
            flat_states, "(b n) e -> b n e", n=encoder_batch_size
        )
        batchified_actions = rearrange(
            flat_actions, "(b n) e -> b n e", n=encoder_batch_size
        )

        traj_states, traj_actions, start_inds = get_traj_batch(
            np_rng, _epoch_trajectories
        )

        batchified_traj_states = rearrange(
            traj_states, "(b n) t e -> b n t e", n=transition_batch_size
        )
        batchified_traj_actions = rearrange(
            traj_actions, "(b n) t e -> b n t e", n=transition_batch_size
        )
        batchified_start_inds = rearrange(
            start_inds, "(b n) -> b n", n=transition_batch_size
        )

        for i in tqdm(
            range(epoch_steps), desc=f"Epoch {epoch}, Encoder", mininterval=mininterval
        ):
            encoder_step(
                step + i,
                batchified_states[i],
                batchified_actions[i],
                batchified_start_inds[i],
                batchified_traj_states[i],
                batchified_traj_actions[i],
            )

        return step + epoch_steps

    def transition_epoch(step, epoch):
        epoch_steps = epoch_trajectories // transition_batch_size
        _epoch_trajectories = int(transition_batch_size * epoch_steps)

        traj_states, traj_actions, start_inds = get_traj_batch(
            np_rng, _epoch_trajectories
        )

        batchified_traj_states = rearrange(
            traj_states, "(b n) t e -> b n t e", n=transition_batch_size
        )
        batchified_traj_actions = rearrange(
            traj_actions, "(b n) t e -> b n t e", n=transition_batch_size
        )
        batchified_start_inds = rearrange(
            start_inds, "(b n) -> b n", n=transition_batch_size
        )

        for i in tqdm(
            range(epoch_steps),
            desc=f"Epoch {epoch}, Transition",
            mininterval=mininterval,
        ):
            transition_step(
                step + i,
                batchified_traj_states[i],
                batchified_traj_actions[i],
                batchified_start_inds[i],
            )

        return step + epoch_steps

    def test_epoch(step):
        with torch.no_grad():
            flat_states, flat_actions = get_state_action_batch(
                np_rng,
                test_epoch_steps * encoder_batch_size,
                states=states_test,
                actions=actions_test,
            )
            traj_states, traj_actions, start_inds = get_traj_batch(
                np_rng,
                test_epoch_steps * transition_batch_size,
                states=states_test,
                actions=actions_test,
            )
            batchified_states = rearrange(
                flat_states, "(b n) e -> b n e", n=encoder_batch_size
            )
            batchified_actions = rearrange(
                flat_actions, "(b n) e -> b n e", n=encoder_batch_size
            )
            batchified_traj_states = rearrange(
                traj_states, "(b n) t e -> b n t e", n=transition_batch_size
            )
            batchified_traj_actions = rearrange(
                traj_actions, "(b n) t e -> b n t e", n=transition_batch_size
            )
            batchified_start_inds = rearrange(
                start_inds, "(b n) -> b n", n=transition_batch_size
            )

            state_reconstruction_losses = []
            action_reconstruction_losses = []
            state_coverage_losses = []
            action_coverage_losses = []
            condensation_losses = []
            transition_losses = []
            smoothness_losses = []

            for i in tqdm(
                range(test_epoch_steps), desc=f"Testing", mininterval=mininterval
            ):
                flat_batch_states = batchified_states[i]
                flat_batch_actions = batchified_actions[i]

                batch_traj_states = batchified_traj_states[i]
                batch_traj_actions = batchified_traj_actions[i]

                batch_traj_start_inds = batchified_start_inds[i]

                flat_encoded_states = state_encoder(flat_batch_states)
                flat_encoded_actions = action_encoder(
                    (flat_batch_actions, flat_encoded_states)
                )

                flat_reconstructed_states = state_decoder(flat_encoded_states)
                flat_reconstructed_actions = action_decoder(
                    (flat_encoded_actions, flat_encoded_states)
                )

                state_reconstruction_loss = state_mse(
                    flat_reconstructed_states, flat_batch_states
                )
                action_reconstruction_loss = action_mse(
                    flat_reconstructed_actions, flat_batch_actions
                )

                state_coverage_loss = state_coverage_loss_func(flat_encoded_states)
                action_coverage_loss = action_coverage_loss_func(flat_encoded_actions)
                condensation_loss = condensation_loss_func(
                    flat_encoded_states, flat_encoded_actions
                )

                traj_start_states = batch_traj_states[
                    torch.arange(len(batch_traj_states)), batch_traj_start_inds
                ]
                traj_latent_start_states = state_encoder(traj_start_states)
                traj_latent_states = state_encoder(batch_traj_states)
                traj_latent_actions = action_encoder(
                    (batch_traj_actions, traj_latent_states)
                )

                traj_latent_fut_states_prime, mask = transition_model(
                    traj_latent_start_states,
                    traj_latent_actions,
                    start_indices=batch_traj_start_inds.cuda(),
                    return_mask=True,
                )

                transition_loss = transition_loss_func(
                    traj_latent_fut_states_prime, traj_latent_states, mask
                )

                all_perturbed_latent_traj_actions = perturb_actions(traj_latent_actions)
                perturb_inds = np_rng.integers(
                    batch_traj_start_inds.cpu(), traj_states.shape[-2]
                )
                perturbed_latent_traj_actions = traj_latent_actions.clone()
                perturbed_latent_traj_actions[
                    torch.arange(len(perturbed_latent_traj_actions)), perturb_inds
                ] = all_perturbed_latent_traj_actions[
                    torch.arange(len(all_perturbed_latent_traj_actions)), perturb_inds
                ]

                traj_latent_fut_states_prime_perturbed = transition_model(
                    traj_latent_start_states,
                    perturbed_latent_traj_actions,
                    start_indices=batch_traj_start_inds.cuda(),
                )

                smoothness_loss = smoothness_loss_func(
                    traj_latent_actions,
                    traj_latent_fut_states_prime,
                    perturbed_latent_traj_actions,
                    traj_latent_fut_states_prime_perturbed,
                    mask,
                )

                state_reconstruction_losses.append(state_reconstruction_loss.item())
                action_reconstruction_losses.append(action_reconstruction_loss.item())
                state_coverage_losses.append(state_coverage_loss.item())
                action_coverage_losses.append(action_coverage_loss.item())
                condensation_losses.append(condensation_loss.item())
                transition_losses.append(transition_loss.item())
                smoothness_losses.append(smoothness_loss.item())

            wandb.log(
                {
                    "test_state_reconstruction_loss": np.mean(
                        state_reconstruction_losses
                    ),
                    "test_action_reconstruction_loss": np.mean(
                        action_reconstruction_losses
                    ),
                    "test_state_coverage_loss": np.mean(state_coverage_losses),
                    "test_action_coverage_loss": np.mean(action_coverage_losses),
                    "test_condensation_loss": np.mean(condensation_losses),
                    "test_transition_loss": np.mean(transition_losses),
                    "test_smoothness_loss": np.mean(smoothness_losses),
                },
                step=step,
            )

    step = 0

    # Start with the transition warmup
    print("Transition warmup")
    for epoch in range(transition_warmup_epochs):
        step = transition_epoch(step, epoch)

    # Start with the encoder warmup
    print("Encoder warmup")
    for epoch in range(encoder_warmup_epochs):
        step = encoder_epoch(step, epoch)

    # Now do the main training loop
    epoch = 0
    print("Main training loop")
    while epoch < train_epochs:

        if epoch % (transition_epochs + encoder_epochs) < transition_epochs:
            step = transition_epoch(step, epoch)
        else:
            step = encoder_epoch(step, epoch)

        test_epoch(step)

        # Step the learning rate
        encoder_lr_scheduler.step()
        transition_lr_scheduler.step()
        epoch += 1

    # Now do the transition finetuning
    print("Transition finetuning")
    for _ in range(transition_finetune_epochs):
        step = transition_epoch(step, epoch)
        epoch += 1


@hydra.main(config_path="../config", config_name="config")
def main(cfg):

    # Add an argument for dataset location (default: data.npz)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--wandb_api_key", type=str, default=None)
    # parser.add_argument("--data", type=str, default="data.npz")
    # parser.add_argument("--url", type=str, default=None)

    # args = parser.parse_args()

    # Not sure why but torch told me to do this
    torch.set_float32_matmul_precision("high")
    # Set random seed
    np_rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Initialize wandb

    if cfg.system_params.wandb_api_key is not None:
        wandb.login(key=cfg.system_params.wandb_api_key)

    config_dict = OmegaConf.to_container(cfg)
    wandb.init(
        project="legato",
        config=config_dict,
        mode="disabled" if cfg.system_params.wandb_api_key is None else "online",
    )

    # Load data stuff

    if cfg.system_params.data_url is not None:
        # make target destination for streaming (in memory)
        dest = io.BytesIO()
        chunk_size = 1024

        resp = requests.get(cfg.system_params.data_url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with tqdm(
            desc="Downloading data",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = dest.write(data)
                bar.update(size)
        # Download the data from the url
        print(f"Downloading data from {cfg.system_params.data_url}...")
        # Load data from the response
        print("Loading data...")
        dest.seek(0)
        data = np.load(dest)
        print("Data loaded.")
        del dest
        del resp

    else:
        # Load data from data.npz
        data_file_path = Path(cfg.system_params.data_file_path)
        data_file_path = hydra.utils.to_absolute_path(data_file_path)
        data = np.load(data_file_path)

    observations = torch.tensor(
        data["observations"],
        dtype=torch.float32,
        device="cuda",
    )
    actions = torch.tensor(
        data["actions"],
        dtype=torch.float32,
        device="cuda",
    )

    train_proportion = 0.8
    n_train = int(train_proportion * len(observations))
    indices = np_rng.permutation(np.arange(len(observations)))

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    observations_train = observations[train_indices]
    actions_train = actions[train_indices]

    observations_test = observations[test_indices]
    actions_test = actions[test_indices]

    # Define networks

    state_dim = observations_train.shape[-1]
    action_dim = actions_train.shape[-1]

    transition_model = TransitionModel(
        2,
        4,
        cfg.net_params.transition_model_params.latent_dim,
        cfg.net_params.transition_model_params.n_heads,
        n_layers=cfg.net_params.transition_model_params.n_layers,
        pe_wavelength_range=cfg.net_params.transition_model_params.pe_wavelength_range,
    ).cuda()

    freqs = torch.logspace(
        start=np.log(0.1),
        end=np.log(100),
        steps=32,
        base=torch.e,
        device="cuda",
    )
    # state_encoder = Perceptron(
    #     state_dim, cfg.net_params.state_encoder_params.layer_sizes, state_dim
    # ).cuda()
    state_encoder = Freqceptron(
        state_dim,
        cfg.net_params.state_encoder_params.layer_sizes,
        state_dim,
        freqs=freqs,
    ).cuda()
    action_encoder = DoublePerceptron(
        action_dim,
        state_dim,
        cfg.net_params.action_encoder_params.layer_sizes,
        action_dim,
    ).cuda()
    # state_decoder = Perceptron(
    #     state_dim, cfg.net_params.state_decoder_params.layer_sizes, state_dim
    # ).cuda()
    state_decoder = Freqceptron(
        state_dim,
        cfg.net_params.state_decoder_params.layer_sizes,
        state_dim,
        freqs=freqs,
    ).cuda()
    action_decoder = DoublePerceptron(
        action_dim,
        state_dim,
        cfg.net_params.action_decoder_params.layer_sizes,
        action_dim,
    ).cuda()

    # Check if we are in a Vast.ai instance
    in_vast = os.getenv("CONTAINER_API_KEY") is not None

    train(
        state_encoder,
        action_encoder,
        torch.compile(transition_model),
        state_decoder,
        action_decoder,
        observations_train,
        actions_train,
        observations_test,
        actions_test,
        np_rng,
        cfg,
        in_vast,
    )

    trained_param_dir = Path(__file__).parent.parent / "trained_net_params"

    # Save the models
    torch.save(state_encoder, trained_param_dir / "state_encoder.pt")
    torch.save(action_encoder, trained_param_dir / "action_encoder.pt")
    torch.save(transition_model, trained_param_dir / "transition_model.pt")
    torch.save(state_decoder, trained_param_dir / "state_decoder.pt")
    torch.save(action_decoder, trained_param_dir / "action_decoder.pt")

    # Save the train test split
    np.savez(
        trained_param_dir / "indices",
        train_indices=train_indices,
        test_indices=test_indices,
    )

    # Upload to wandb
    wandb.save(trained_param_dir / "state_encoder.pt")
    wandb.save(trained_param_dir / "action_encoder.pt")
    wandb.save(trained_param_dir / "transition_model.pt")
    wandb.save(trained_param_dir / "state_decoder.pt")
    wandb.save(trained_param_dir / "action_decoder.pt")
    wandb.save(trained_param_dir / "indices.npz")

    # Block until all files are uploaded
    wandb.finish()

    # Check if this is a Vast.ai instance
    # Check if CONTAINER_API_KEY is set
    if in_vast:
        # Call "vastai destroy instance $CONTAINER_ID" to stop the instance
        print("Killing instance...")
        command = (
            f"vastai destroy instance {os.getenv('CONTAINER_ID')}"  # Kill the instance
            + f" --api-key {os.getenv('CONTAINER_API_KEY')}"  # Add the api key
        )
        print("Running command:    " + command)
        os.system(command)
        print("Instance stopped?")


if __name__ == "__main__":
    main()
