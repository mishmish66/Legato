import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.autograd.profiler as profiler
import wandb
from einops import einsum, pack, rearrange, repeat
from torch import nn
from tqdm import tqdm

from loss import CoverageLoss, SmoothnessLoss, TransitionLoss
from nets import Perceptron, TransitionModel


def train(
    state_encoder,
    action_encoder,
    transition_model,
    state_decoder,
    action_decoder,
    states,
    actions,
    np_rng,
):
    """Train the networks."""

    action_mse = nn.MSELoss()
    state_mse = nn.MSELoss()
    transition_loss_func = torch.compile(TransitionLoss())

    smoothness_loss_func = SmoothnessLoss()

    coverage_loss_func = torch.compile(
        CoverageLoss(
            state_space_size=1.5,
            action_space_size=1.75,
            latent_samples=16_384,
            # latent_state_samples=1024,
            # latent_action_samples=1024,
        )
    )

    epoch_state_actions = int(1e6)
    epoch_trajectories = int(1e4)

    encoder_batch_size = 4096
    transition_batch_size = 256

    encoder_epochs = 1
    transition_epochs = 4

    train_epochs = 256

    transition_warmup_epochs = 1
    encoder_warmup_epochs = 1
    transition_finetune_epochs = 16

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
        lr=1e-3,
    )
    encoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        encoder_optimizer,
        gamma=0.01 ** (1 / train_epochs),
    )

    transition_optimizer = torch.optim.AdamW(
        transition_model.parameters(),
        lr=1e-3,
    )
    transition_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        transition_optimizer,
        gamma=(0.5) ** (1 / train_epochs),
    )

    def get_state_action_batch(batch_size):
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

    def get_traj_batch(np_rng, batch_size):
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

    def perturb_actions(latent_traj_actions, mask):

        # Generate random perturbations
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
            * coverage_loss_func.action_space_size
        )
        # Now use the scaled perturbed actions if the norms are too large
        latent_traj_actions_perturbed = torch.where(
            perturbed_action_norms > coverage_loss_func.action_space_size,
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
        encoder_optimizer.zero_grad()
        transition_optimizer.zero_grad()

        flat_encoded_states = state_encoder(flat_states)
        flat_encoded_actions = action_encoder(
            torch.cat([flat_actions, flat_states], dim=-1)
        )

        flat_reconstructed_states = state_decoder(flat_encoded_states)
        flat_reconstructed_actions = action_decoder(
            torch.cat([flat_encoded_actions, flat_encoded_states], dim=-1)
        )

        state_reconstruction_loss = state_mse(flat_reconstructed_states, flat_states)
        action_reconstruction_loss = action_mse(
            flat_reconstructed_actions, flat_actions
        )

        coverage_loss = coverage_loss_func(flat_encoded_states, flat_encoded_actions)

        traj_start_states = traj_states[torch.arange(len(traj_states)), traj_start_inds]
        traj_latent_start_states = state_encoder(traj_start_states)
        traj_latent_states = state_encoder(traj_states)
        traj_latent_actions = action_encoder(
            torch.cat([traj_actions, traj_states], dim=-1)
        )

        traj_latent_fut_states_prime, mask = transition_model(
            traj_latent_start_states,
            traj_latent_actions,
            start_indices=traj_start_inds.cuda(),
            return_mask=True,
        )

        perturbed_latent_traj_actions = perturb_actions(traj_latent_actions, mask)

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

        encoder_loss = (
            state_reconstruction_loss
            + action_reconstruction_loss
            + smoothness_loss
            + coverage_loss * 0.01
        )

        encoder_loss.backward()
        encoder_optimizer.step()

        wandb.log(
            {
                "state_reconstruction_loss": state_reconstruction_loss.item(),
                "action_reconstruction_loss": action_reconstruction_loss.item(),
                "coverage_loss": coverage_loss.item(),
                "smoothness_loss": smoothness_loss.item(),
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
        traj_latent_actions = action_encoder(
            torch.cat([traj_actions, traj_states], dim=-1)
        )

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

        flat_states, flat_actions = get_state_action_batch(_epoch_state_actions)
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

        # Step the learning rate
        encoder_lr_scheduler.step()
        transition_lr_scheduler.step()
        epoch += 1

    # Now do the transition finetuning
    print("Transition finetuning")
    for epoch in range(transition_finetune_epochs):
        step = transition_epoch(step, epoch - transition_finetune_epochs)
        epoch += 1

    # for i in tqdm(range(), disable=False):

    #     # Yoink a batch of data
    #     np_encoder_batch_raveled_inds = np_rng.permutation(np.prod(states.shape[0]))[
    #         :encoder_batch_size
    #     ]
    #     encoder_batch_raveled_inds = torch.tensor(
    #         np_encoder_batch_raveled_inds, device=states.device
    #     )
    #     encoder_batch_inds = torch.unravel_index(
    #         encoder_batch_raveled_inds, states.shape[:-1]
    #     )

    #     np_transition_traj_batch_inds = np_rng.permutation(states.shape[0])[
    #         :transition_batch_size
    #     ]
    #     transition_traj_batch_inds = torch.tensor(
    #         np_transition_traj_batch_inds, device=states.device
    #     )[:transition_batch_size]
    #     transition_start_inds = torch.randint(
    #         0,
    #         int(states.shape[-2] // 1.1),
    #         (transition_batch_size,),
    #         device=states.device,
    #     )

    #     state_batch = states[encoder_batch_inds].to("cuda")
    #     action_batch = actions[encoder_batch_inds].to("cuda")

    #     starting_states = states[transition_traj_batch_inds, transition_start_inds].to(
    #         "cuda"
    #     )
    #     state_traj_batch = states[transition_traj_batch_inds].to("cuda")
    #     action_traj_batch = actions[transition_traj_batch_inds].to("cuda")

    #     # Now do a forward pass

    #     if i % encoder_step_every == 0 and i < steps:
    #         perceptron_optimizer.zero_grad()
    #         transformer_optimizer.zero_grad()

    #         latent_states = state_encoder(state_batch)
    #         latent_actions = action_encoder(
    #             torch.cat([action_batch, state_batch], dim=-1)
    #         )

    #         reconstructed_states = state_decoder(latent_states)
    #         reconstructed_actions = action_decoder(
    #             torch.cat([latent_actions, latent_states], dim=-1)
    #         )

    #         state_reconstruction_loss = state_mse(reconstructed_states, state_batch)
    #         action_reconstruction_loss = action_mse(reconstructed_actions, action_batch)

    #         latent_start_states = state_encoder(starting_states)
    #         latent_traj_actions = action_encoder(
    #             torch.cat([action_traj_batch, state_traj_batch], dim=-1)
    #         )
    #         latent_fut_states_prime, mask = transition_model(
    #             latent_start_states,
    #             latent_traj_actions,
    #             start_indices=transition_start_inds.cuda(),
    #             return_mask=True,
    #         )
    #         latent_fut_states_gt = state_encoder(state_traj_batch)

    #         perturbations = torch.randn_like(latent_traj_actions)
    #         perturbations = perturbations / torch.norm(
    #             perturbations, p=1, dim=-1, keepdim=True
    #         )
    #         perturbations = perturbations * torch.rand(
    #             (*perturbations.shape[:-1], 1), device="cuda"
    #         )

    #         latent_traj_actions_perturbed = latent_traj_actions + perturbations
    #         # Normalize the perturbations if they are too large
    #         perturbed_action_norms = torch.norm(
    #             latent_traj_actions_perturbed, p=1, dim=-1, keepdim=True
    #         )
    #         latent_traj_actions_perturbed = torch.where(
    #             perturbed_action_norms > coverage_loss_func.action_space_size,
    #             latent_traj_actions_perturbed
    #             * coverage_loss_func.action_space_size
    #             / perturbed_action_norms,
    #             latent_traj_actions_perturbed,
    #         )
    #         latent_fut_states_prime_perturbed = transition_model(
    #             latent_start_states,
    #             latent_traj_actions_perturbed,
    #             start_indices=transition_start_inds.cuda(),
    #         )

    #         transition_loss = transition_loss_func(
    #             latent_fut_states_prime, latent_fut_states_gt, mask
    #         )
    #         smoothness_loss = smoothness_loss_func(
    #             latent_traj_actions,
    #             latent_fut_states_prime,
    #             latent_traj_actions_perturbed,
    #             latent_fut_states_prime_perturbed,
    #             mask,
    #         )
    #         coverage_loss = coverage_loss_func(latent_states, latent_actions)

    #         perceptron_loss = (
    #             state_reconstruction_loss
    #             + action_reconstruction_loss
    #             + smoothness_loss
    #             + coverage_loss * 0.01
    #         )

    #         perceptron_loss.backward()
    #         perceptron_optimizer.step()
    #         perceptron_lr_scheduler.step()

    #     # Now do a forward pass for the transformer

    #     if True:
    #         transformer_optimizer.zero_grad()
    #         perceptron_optimizer.zero_grad()

    #         latent_start_states = state_encoder(starting_states)
    #         latent_traj_actions = action_encoder(
    #             torch.cat([action_traj_batch, state_traj_batch], dim=-1)
    #         )
    #         latent_fut_states_prime = transition_model(
    #             latent_start_states,
    #             latent_traj_actions,
    #             start_indices=transition_start_inds.cuda(),
    #         )
    #         latent_fut_states_gt = state_encoder(state_traj_batch)

    #         transition_loss = transition_loss_func(
    #             latent_fut_states_prime, latent_fut_states_gt, mask
    #         )

    #         transition_loss.backward()
    #         transformer_optimizer.step()

    #         if i < steps:
    #             transformer_lr_scheduler.step()

    #     wandb.log(
    #         {
    #             "perceptron_loss": perceptron_loss.item(),
    #             "transition_loss": transition_loss.item(),
    #             "perceptron_lr": perceptron_lr_scheduler.get_last_lr()[0],
    #             "transformer_lr": transformer_lr_scheduler.get_last_lr()[0],
    #             "smoothness_loss": smoothness_loss.item(),
    #             "coverage_loss": coverage_loss.item(),
    #             "state_reconstruction_loss": state_reconstruction_loss.item(),
    #             "action_reconstruction_loss": action_reconstruction_loss.item(),
    #         }
    #     )

    #     # if i % 16 == 15:
    #     #     print(
    #     #         f"Iteration {i}:\n"
    #     #         + f"    perceptron loss:            {perceptron_loss.item()}\n"
    #     #         + f"    transformer loss:           {transition_loss.item()}\n"
    #     #         + f"    perceptron lr:              {perceptron_lr_scheduler.get_last_lr()}\n"
    #     #         + f"    transformer lr:             {transformer_lr_scheduler.get_last_lr()}\n"
    #     #         + f"    smoothness loss:            {smoothness_loss.item()}\n"
    #     #         + f"    coverage loss:              {coverage_loss.item()}\n"
    #     #         + f"    state reconstruction loss:  {state_reconstruction_loss.item()}\n"
    #     #         + f"    action reconstruction loss: {action_reconstruction_loss.item()}"
    #     #     )


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
        2, 4, 32, 4, n_layers=3, pe_wavelength_range=[1, 2048]
    ).cuda()

    state_encoder = Perceptron(state_dim, [32, 64, 32], state_dim).cuda()
    action_encoder = Perceptron(action_dim + state_dim, [32, 32, 32], action_dim).cuda()
    state_decoder = Perceptron(state_dim, [32, 64, 32], state_dim).cuda()
    action_decoder = Perceptron(action_dim + state_dim, [32, 32, 32], action_dim).cuda()

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
