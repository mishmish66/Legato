import numpy as np
import torch
from einops import einsum, pack, rearrange, repeat
from torch import nn


class TransitionLoss(nn.Module):
    def __init__(self, loss_fn=nn.MSELoss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, latent_fut_states_prime, latent_fut_states_gt, mask):
        masked_latent_fut_states_prime = torch.where(
            mask[..., None], latent_fut_states_gt, latent_fut_states_prime
        )
        loss = self.loss_fn(masked_latent_fut_states_prime, latent_fut_states_gt)
        return loss


class SmoothnessLoss(nn.Module):
    def __init__(self, norm_p=1, discount=0.99):
        super().__init__()
        self.discount = discount
        self.norm_p = norm_p

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

        state_diffs = latent_next_states_perturbed - latent_next_states
        state_dists = torch.norm(state_diffs, p=self.norm_p, dim=-1)

        future_indices = torch.cumsum(~mask, dim=-1, dtype=torch.float32)
        future_discounts = self.discount**future_indices
        dist_limits = action_dists / future_discounts
        state_violations = torch.relu(state_dists - dist_limits)
        losses = state_violations**2
        masked_losses = torch.where(mask, 0, losses)

        return masked_losses.mean()


class CoverageLoss(nn.Module):
    def __init__(
        self,
        state_space_size,
        action_space_size,
        latent_samples=2048,
        space_ball_p=1,
        selection_tail_size=4,
        far_sample_count=64,
        pushing_sample_size=4,
    ):
        super().__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.latent_samples = latent_samples
        self.space_ball_p = space_ball_p
        self.selection_tail_size = selection_tail_size
        self.far_sample_count = far_sample_count
        self.pushing_sample_size = pushing_sample_size

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

        # penalize for empty space within the state space
        # Sample random points in the latent space
        if self.space_ball_p != 1:
            raise NotImplementedError("Only L1 norm is supported :(")

        state_space_samples = (
            torch.rand(
                self.latent_samples,
                latent_states.shape[-1],
                device=latent_states.device,
            )
            * 2
            - 1
        ) * self.state_space_size
        action_space_samples = (
            torch.rand(
                self.latent_samples,
                latent_actions.shape[-1],
                device=latent_actions.device,
            )
            * 2
            - 1
        ) * self.action_space_size

        # Find the state_space that is the farthest from any of the latent_states
        state_space_dists = torch.cdist(state_space_samples, latent_states, p=1)
        state_space_dist_tail = (
            state_space_dists.sort(dim=-1)
            .values[:, : self.selection_tail_size]
            .mean(dim=-1)
        )
        farthest_idxs = state_space_dist_tail.argsort(descending=True)[
            : self.far_sample_count
        ]
        farthest_state_samples = state_space_samples[farthest_idxs]

        action_space_dists = torch.cdist(action_space_samples, latent_actions, p=1)
        action_space_dist_tail = (
            action_space_dists.sort(dim=-1)
            .values[:, : self.selection_tail_size]
            .mean(dim=-1)
        )
        farthest_idxs = action_space_dist_tail.argsort(descending=True)[
            : self.far_sample_count
        ]
        farthest_action_samples = action_space_samples[farthest_idxs]

        # Now make the states by the farthest latent states closer to the farthest samples
        # Maybe in the future make just the few closest ones closer
        empty_state_space_dists = torch.cdist(
            farthest_state_samples, latent_states, p=1
        )
        close_empty_state_space_dists = empty_state_space_dists.sort(dim=-1).values[
            :, : self.pushing_sample_size
        ]
        state_coverage_losses = close_empty_state_space_dists**2

        empty_action_space_dists = torch.cdist(
            farthest_action_samples, latent_actions, p=1
        )
        close_empty_action_space_dists = empty_action_space_dists.sort(dim=-1).values[
            :, : self.pushing_sample_size
        ]
        action_coverage_losses = close_empty_action_space_dists**2

        return (
            state_size_loss
            + action_size_loss
            + state_coverage_losses.mean()
            + action_coverage_losses.mean()
        )
