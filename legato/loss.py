import numpy as np
import torch
from einops import einsum, pack, rearrange, repeat
from torch import nn


class TransitionLoss(nn.Module):
    def __init__(self, loss_fn=nn.HuberLoss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, latent_fut_states_prime, latent_fut_states_gt, mask):
        masked_latent_fut_states_prime = torch.where(
            mask[..., None], latent_fut_states_gt, latent_fut_states_prime
        )
        loss = self.loss_fn(masked_latent_fut_states_prime, latent_fut_states_gt)
        return loss


class SmoothnessLoss(nn.Module):
    def __init__(self, norm_p=1, discount=0.998, loss_fn=nn.HuberLoss()):
        super().__init__()
        self.discount = discount
        self.norm_p = norm_p
        self.loss_fn = loss_fn

    def forward(
        self,
        latent_actions,
        latent_next_states,
        latent_actions_perturbed,
        latent_next_states_perturbed,
        mask,
    ):
        action_diffs = latent_actions_perturbed - latent_actions
        action_dists = torch.norm(action_diffs, p=self.norm_p, dim=-1)
        cum_act_dists = torch.cumsum(action_dists, dim=-1, dtype=torch.float32)

        state_diffs = latent_next_states_perturbed - latent_next_states
        state_dists = torch.norm(state_diffs, p=self.norm_p, dim=-1)

        future_indices = torch.cumsum(~mask, dim=-1, dtype=torch.float32)
        future_discounts = self.discount**future_indices
        dist_limits = cum_act_dists / future_discounts
        state_violations = torch.relu(state_dists - dist_limits) / (
            dist_limits + 1e-3
        )  # Add constant to avoid exploding gradients
        loss_input = torch.where(mask, 0, state_violations)
        # Scale the loss for future elements only
        loss = (
            self.loss_fn(
                loss_input, torch.zeros_like(loss_input)
            ).mean()  # The mean huber loss
            * np.prod(mask.shape)  # The mean denominator
            / torch.sum(~mask)  # Number of non-zero elements
        )

        return loss


class CoverageLoss(nn.Module):

    def __init__(
        self,
        latent_sampler,
        norm_p=1,
        latent_samples=2048,
        selection_tail_size=4,
        far_sample_count=64,
        pushing_sample_size=4,
        loss_function=nn.HuberLoss(),
    ):
        super().__init__()

        self.latent_sampler = latent_sampler
        self.norm_p = norm_p
        self.latent_samples = latent_samples
        self.selection_tail_size = selection_tail_size
        self.far_sample_count = far_sample_count
        self.pushing_sample_size = pushing_sample_size
        self.loss_function = loss_function

    def forward(self, latents):
        # penalize for empty space within the state space
        # Sample random points in the latent space
        space_samples = self.latent_sampler(self.latent_samples)

        # Find the sample that is the farthest from any of the latent states
        space_dists = torch.cdist(space_samples, latents, p=self.norm_p)
        tail_dists = torch.topk(
            space_dists, self.selection_tail_size, dim=-1, largest=False
        ).values.mean(-1)
        farthest_dist_inds = torch.topk(
            tail_dists, self.far_sample_count, dim=-1
        ).indices
        far_samples = space_samples[farthest_dist_inds]

        # Now make the states by the latent states closer to the farthest samples
        empty_space_dists = torch.cdist(far_samples, latents, p=self.norm_p)
        close_empty_space_dists = torch.topk(
            empty_space_dists, self.pushing_sample_size, dim=-1, largest=False
        ).values.mean(-1)
        space_coverage_loss = self.loss_function(
            close_empty_space_dists, torch.zeros_like(close_empty_space_dists)
        ).mean()

        return space_coverage_loss


class ConsistencyLoss(nn.Module):

    def __init__(
        self,
        latent_state_sampler,
        latent_action_sampler,
        state_decoder,
        action_decoder,
        state_encoder,
        action_encoder,
        state_samples=2048,
        action_samples=2048,
        loss_function=nn.HuberLoss(),
    ):
        super().__init__()

        self.latent_state_sampler = latent_state_sampler
        self.latent_action_sampler = latent_action_sampler
        self.state_decoder = state_decoder
        self.action_decoder = action_decoder
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.state_samples = state_samples
        self.action_samples = action_samples
        self.loss_function = loss_function

    def forward(self):
        # penalize for empty space within the state space
        # Classify the test samples with a modified knn classifier

        latent_states = self.latent_state_sampler(self.state_samples)
        latent_actions = self.latent_action_sampler(self.action_samples)
        latent_action_states = self.latent_state_sampler(self.action_samples)

        action_states = self.state_decoder(latent_action_states)
        actions = self.action_decoder((latent_actions, latent_action_states))
        recovered_latent_actions = self.action_encoder((actions, action_states))

        states = self.state_decoder(latent_states)
        recovered_latent_states = self.state_encoder(states)

        action_consistency_losses = self.loss_function(
            latent_actions, recovered_latent_actions
        )
        state_consistency_losses = self.loss_function(
            latent_states, recovered_latent_states
        )

        return action_consistency_losses.mean() + state_consistency_losses.mean()


class CondensationLoss(nn.Module):
    def __init__(self, state_space_size, action_space_size, space_ball_p=1):
        super().__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.space_ball_p = space_ball_p

    def forward(self, latent_states, latent_actions):

        latent_states = rearrange(latent_states, "... e -> (...) e")
        latent_actions = rearrange(latent_actions, "... e -> (...) e")

        state_norms = torch.norm(latent_states, p=self.space_ball_p, dim=-1)
        action_norms = torch.norm(latent_actions, p=self.space_ball_p, dim=-1)

        state_violations = torch.relu(state_norms - self.state_space_size)
        action_violations = torch.relu(action_norms - self.action_space_size)

        state_size_violations = state_violations**2
        action_size_violations = action_violations**2

        state_size_loss = state_size_violations.mean()
        action_size_loss = action_size_violations.mean()

        return state_size_loss + action_size_loss
