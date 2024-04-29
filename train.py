import gymnasium as gym
import numpy as np
import torch
from einops import einsum, pack, rearrange, repeat
from torch import nn
from tqdm import tqdm

from loss import CoverageLoss, SmoothnessLoss, TransitionLoss
from nets import Perceptron, TransitionModel

# Define loss functions


def train(
    state_encoder,
    action_encoder,
    transition_model,
    state_decoder,
    action_decoder,
    states,
    actions,
):

    # Now launch the training loop

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

    perceptron_optimizer = torch.optim.AdamW(
        [
            param
            for net in [
                state_encoder,
                action_encoder,
                state_decoder,
                action_decoder,
                transition_model, # Temporarily adding this back to see
            ]
            for param in net.parameters()
        ],
        lr=1e-2,
    )
    perceptron_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        perceptron_optimizer,
        gamma=0.995,
        last_epoch=-1,
    )

    # transformer_optimizer = torch.optim.AdamW(transition_model.parameters(), lr=1e-2)

    # transformer_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     transformer_optimizer,
    #     factor=0.5,
    #     patience=128,
    #     threshold=1e-4,
    #     threshold_mode="rel",
    #     cooldown=8,
    #     min_lr=1e-6,
    #     eps=1e-8,
    #     verbose=True,
    # )
    encoder_batch_size = 4096
    transition_batch_size = 64

    for i in tqdm(range(2048), disable=True):

        # Yoink a batch of data
        encoder_batch_raveled_inds = torch.randperm(np.prod(states.shape[:-1]))[
            :encoder_batch_size
        ]
        encoder_batch_inds = torch.unravel_index(
            encoder_batch_raveled_inds, states.shape[:-1]
        )

        transition_traj_batch_inds = torch.randperm(states.shape[0], device="cuda")[
            :transition_batch_size
        ]
        transition_start_inds = torch.randint(
            0,
            int(states.shape[-2] // 1.1),
            (transition_batch_size,),
            device="cuda",
        )

        state_batch = states[encoder_batch_inds].to("cuda")
        action_batch = actions[encoder_batch_inds].to("cuda")

        starting_states = states[transition_traj_batch_inds, transition_start_inds].to(
            "cuda"
        )
        state_traj_batch = states[transition_traj_batch_inds].to("cuda")
        action_traj_batch = actions[transition_traj_batch_inds].to("cuda")

        # Now do a forward pass
        perceptron_optimizer.zero_grad()
        # transformer_optimizer.zero_grad()

        latent_states = state_encoder(state_batch)
        latent_actions = action_encoder(torch.cat([action_batch, state_batch], dim=-1))

        reconstructed_states = state_decoder(latent_states)
        reconstructed_actions = action_decoder(
            torch.cat([latent_actions, latent_states], dim=-1)
        )

        state_reconstruction_loss = state_mse(reconstructed_states, state_batch)
        action_reconstruction_loss = action_mse(reconstructed_actions, action_batch)

        latent_start_states = state_encoder(starting_states)
        latent_traj_actions = action_encoder(
            torch.cat([action_traj_batch, state_traj_batch], dim=-1)
        )
        latent_fut_states_prime, mask = transition_model(
            latent_start_states,
            latent_traj_actions,
            start_indices=transition_start_inds,
            return_mask=True,
        )
        latent_fut_states_gt = state_encoder(state_traj_batch)

        perturbations = torch.randn_like(latent_traj_actions)
        perturbations = perturbations / torch.norm(
            perturbations, p=1, dim=-1, keepdim=True
        )
        perturbations = perturbations * torch.rand(
            (*perturbations.shape[:-1], 1), device="cuda"
        )

        latent_traj_actions_perturbed = latent_traj_actions + perturbations
        # Normalize the perturbations if they are too large
        perturbed_action_norms = torch.norm(
            latent_traj_actions_perturbed, p=1, dim=-1, keepdim=True
        )
        latent_traj_actions_perturbed = torch.where(
            perturbed_action_norms > coverage_loss_func.action_space_size,
            latent_traj_actions_perturbed
            * coverage_loss_func.action_space_size
            / perturbed_action_norms,
            latent_traj_actions_perturbed,
        )
        latent_fut_states_prime_perturbed = transition_model(
            latent_start_states,
            latent_traj_actions_perturbed,
            start_indices=transition_start_inds,
        )

        transition_loss = transition_loss_func(
            latent_fut_states_prime, latent_fut_states_gt, mask
        )
        smoothness_loss = smoothness_loss_func(
            latent_traj_actions,
            latent_fut_states_prime,
            latent_traj_actions_perturbed,
            latent_fut_states_prime_perturbed,
            mask,
        )
        coverage_loss = coverage_loss_func(latent_states, latent_actions)

        perceptron_loss = (
            state_reconstruction_loss
            + action_reconstruction_loss
            + transition_loss  # Temporarily adding this back to see
            + smoothness_loss
            + coverage_loss * 0.01
        )

        if True:  # i < 1024: This is a temporary hack to train the transformer first
            perceptron_loss.backward()
            perceptron_optimizer.step()
            perceptron_lr_scheduler.step()

        # Now do a forward pass for the transformer (or a few)

        for _ in range(0):  # (16):

            transition_traj_batch_inds = torch.randperm(states.shape[0], device="cuda")[
                :transition_batch_size
            ]
            transition_start_inds = torch.randint(
                0,
                int(states.shape[-2] // 1.1),
                (transition_batch_size,),
                device="cuda",
            )
            starting_states = states[
                transition_traj_batch_inds, transition_start_inds
            ].to("cuda")
            state_traj_batch = states[transition_traj_batch_inds].to("cuda")
            action_traj_batch = actions[transition_traj_batch_inds].to("cuda")

            transition_traj_batch_inds = torch.randperm(states.shape[0], device="cuda")[
                :transition_batch_size
            ]
            transition_start_inds = torch.randint(
                0,
                int(states.shape[-2] // 1.1),
                (transition_batch_size,),
                device="cuda",
            )
            perceptron_optimizer.zero_grad()
            transformer_optimizer.zero_grad()

            latent_fut_states_gt = state_encoder(state_traj_batch)

            latent_start_states = state_encoder(starting_states)
            latent_traj_actions = action_encoder(
                torch.cat([action_traj_batch, state_traj_batch], dim=-1)
            )

            latent_fut_states_prime, mask = transition_model(
                latent_start_states,
                latent_traj_actions,
                start_indices=transition_start_inds,
                return_mask=True,
            )
            latent_fut_states_gt = state_encoder(state_traj_batch)

            transition_loss = transition_loss_func(
                latent_fut_states_prime, latent_fut_states_gt, mask
            )

            transition_loss.backward()
            transformer_optimizer.step()
            transformer_lr_scheduler.step(transition_loss.item())

        if i % 16 == 0:
            print(
                f"Iteration {i}:\n"
                + f"    perceptron loss:            {perceptron_loss.item()}\n"
                + f"    transformer loss:           {transition_loss.item()}\n"
                + f"    perceptron lr:              {perceptron_lr_scheduler.get_last_lr()}\n"
                # + f"    transformer lr:             {transformer_lr_scheduler.get_last_lr()}\n"
                + f"    smoothness loss:            {smoothness_loss.item()}\n"
                + f"    coverage loss:              {coverage_loss.item()}\n"
                + f"    state reconstruction loss:  {state_reconstruction_loss.item()}\n"
                + f"    action reconstruction loss: {action_reconstruction_loss.item()}"
            )


if __name__ == "__main__":

    # Set random seed
    np_rng = np.random.default_rng(0)

    # Load data from data.npz
    data = np.load("data.npz")

    observations = torch.tensor(
        data["observations"], dtype=torch.float32, device="cuda"
    )
    actions = torch.tensor(data["actions"], dtype=torch.float32, device="cuda")

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

    transition_model_raw = TransitionModel(
        2, 4, 16, 1, pe_wavelength_range=[1, 2048]
    ).cuda()

    state_encoder = Perceptron(state_dim, [32, 64, 32], state_dim).cuda()
    action_encoder = Perceptron(action_dim + state_dim, [32, 32, 32], action_dim).cuda()
    state_decoder = Perceptron(state_dim, [32, 64, 32], state_dim).cuda()
    action_decoder = Perceptron(action_dim + state_dim, [32, 32, 32], action_dim).cuda()

    # Load models, this is temporary
    state_encoder = torch.load("trained_net_params/state_encoder.pt")
    action_encoder = torch.load("trained_net_params/action_encoder.pt")
    # transiton_model_raw = torch.load("trained_net_params/transition_model.pt")
    state_decoder = torch.load("trained_net_params/state_decoder.pt")
    action_decoder = torch.load("trained_net_params/action_decoder.pt")

    train(
        state_encoder,
        action_encoder,
        # torch.compile(transition_model_raw),
        transition_model_raw,
        state_decoder,
        action_decoder,
        observations_train,
        actions_train,
    )

    # Save the models
    # torch.save(state_encoder, "trained_net_params/state_encoder.pt")
    # torch.save(action_encoder, "trained_net_params/action_encoder.pt")
    torch.save(transition_model_raw, "trained_net_params/transition_model.pt")
    # torch.save(state_decoder, "trained_net_params/state_decoder.pt")
    # torch.save(action_decoder, "trained_net_params/action_decoder.pt")

    # Save the train test split
    np.savez(
        "trained_net_params/indices",
        train_indices=train_indices,
        test_indices=test_indices,
    )
