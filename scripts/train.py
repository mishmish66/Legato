import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.autograd.profiler as profiler
from einops import einsum, pack, rearrange, repeat
from torch import nn
from tqdm import tqdm

import wandb
from legato.loss import (
    CoverageLoss,
    SmoothnessLoss,
    TransitionLoss,
    CondensationLoss,
    ConsistencyLoss,
)
from legato.nets import DoublePerceptron, Perceptron, TransitionModel
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
    state_space_size=2.0,
    action_space_size=1.0,
):
    """Train the networks."""

    perturbation_generator = PBallSampler(2, 1, 1.0, device="cuda")
    latent_state_sampler = PBallSampler(4, 1, state_space_size, device="cuda")
    latent_action_sampler = PBallSampler(2, 1, action_space_size, device="cuda")
    latent_action_state_sampler = lambda n: (
        latent_action_sampler(n),
        latent_state_sampler(n),
    )

    action_mse = nn.MSELoss()
    state_mse = nn.MSELoss()

    transition_loss_func = torch.compile(TransitionLoss())

    smoothness_loss_func = SmoothnessLoss()

    state_coverage_loss_func = torch.compile(
        CoverageLoss(
            latent_state_sampler,
            latent_samples=4096,
            selection_tail_size=4,
            far_sample_count=64,
            pushing_sample_size=16,
        )  # , disable=True
    )
    action_coverage_loss_func = torch.compile(
        CoverageLoss(
            latent_action_sampler,
            latent_samples=1024,
            selection_tail_size=4,
            far_sample_count=16,
            pushing_sample_size=64,
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
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        loss_function=nn.L1Loss(),
    )

    epoch_state_actions = int(5e5)
    epoch_trajectories = int(2.5e4)

    encoder_batch_size = 4096
    transition_batch_size = 128

    test_epoch_steps = 8

    encoder_grad_skips = 1

    encoder_epochs = 1
    transition_epochs = 1

    train_epochs = 256

    transition_warmup_epochs = 1
    encoder_warmup_epochs = 2
    transition_finetune_epochs = 8

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
        lr=5e-5,
    )
    encoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        encoder_optimizer,
        gamma=1.0 ** (1 / train_epochs),
    )

    transition_optimizer = torch.optim.AdamW(
        transition_model.parameters(),
        lr=5e-5,
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
            latent_traj_actions_perturbed / perturbed_action_norms * action_space_size
        )
        # Now use the scaled perturbed actions if the norms are too large
        latent_traj_actions_perturbed = torch.where(
            perturbed_action_norms > action_space_size,
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
        flat_encoded_actions = action_encoder((flat_actions, flat_states))

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
        traj_latent_actions = action_encoder((traj_actions, traj_states))

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

        encoder_loss = (
            state_reconstruction_loss
            + action_reconstruction_loss
            + condensation_loss * 10.0
            + smoothness_loss * 0.01
            + transition_loss * 0.01
            + state_coverage_loss * 0.1
            + action_coverage_loss * 0.1
            + consistency_loss * 1.0
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
        traj_latent_actions = action_encoder((traj_actions, traj_states))

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

        for i in tqdm(range(epoch_steps), desc=f"Epoch {epoch}, Encoder"):
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

        for i in tqdm(range(epoch_steps), desc=f"Epoch {epoch}, Transition"):
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

            for i in tqdm(range(test_epoch_steps), desc=f"Testing"):
                flat_batch_states = batchified_states[i]
                flat_batch_actions = batchified_actions[i]

                batch_traj_states = batchified_traj_states[i]
                batch_traj_actions = batchified_traj_actions[i]

                batch_traj_start_inds = batchified_start_inds[i]

                flat_encoded_states = state_encoder(flat_batch_states)
                flat_encoded_actions = action_encoder(
                    (flat_batch_actions, flat_batch_states)
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
                    (batch_traj_actions, batch_traj_states)
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


if __name__ == "__main__":

    # Not sure why but torch told me to do this
    torch.set_float32_matmul_precision("high")
    # Set random seed
    np_rng = np.random.default_rng(0)
    torch.manual_seed(0)

    wandb.init(project="legato")

    # Load data from data.npz
    data = np.load("data.npz")

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
        2, 4, 128, 4, n_layers=3, pe_wavelength_range=[1, 2048]
    ).cuda()

    state_encoder = Perceptron(state_dim, [512, 1024, 512], state_dim).cuda()
    action_encoder = DoublePerceptron(
        action_dim, state_dim, [512, 1024, 512], action_dim
    ).cuda()
    state_decoder = Perceptron(state_dim, [512, 1024, 512], state_dim).cuda()
    action_decoder = DoublePerceptron(
        action_dim, state_dim, [512, 1024, 512], action_dim
    ).cuda()

    with profiler.profile(
        enabled=False,
        with_stack=True,
        use_cuda=True,
        profile_memory=True,
    ) as prof:
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
        )

    # prof.export_chrome_trace("profile_results.json")
    # print(
    #     prof.key_averages(group_by_stack_n=5).table(
    #         sort_by="self_cpu_time_total", row_limit=5
    #     )
    # )

    # Save the models
    torch.save(state_encoder, "trained_net_params/state_encoder.pt")
    torch.save(action_encoder, "trained_net_params/action_encoder.pt")
    torch.save(transition_model, "trained_net_params/transition_model.pt")
    torch.save(state_decoder, "trained_net_params/state_decoder.pt")
    torch.save(action_decoder, "trained_net_params/action_decoder.pt")

    # Save the train test split
    np.savez(
        "trained_net_params/indices",
        train_indices=train_indices,
        test_indices=test_indices,
    )
    
    # Upload to wandb
    wandb.save("trained_net_params/state_encoder.pt")
    wandb.save("trained_net_params/action_encoder.pt")
    wandb.save("trained_net_params/transition_model.pt")
    wandb.save("trained_net_params/state_decoder.pt")
    wandb.save("trained_net_params/action_decoder.pt")
    wandb.save("trained_net_params/indices.npz")
