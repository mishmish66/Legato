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
        decoder,
        latent_sampler,
        k=16,
        loss_function=nn.HuberLoss(),
    ):
        super().__init__()

        self.decoder = decoder
        self.latent_sampler = latent_sampler
        self.k = k
        self.loss_function = loss_function

    def forward(self, gt_vals):
        # penalize for empty space within the state space

        train_latent_space_samples = self.latent_sampler(len(gt_vals))
        test_latent_space_samples = self.latent_sampler(len(gt_vals))

        recovered_train_samples = self.decoder(train_latent_space_samples).detach()
        recovered_test_samples = self.decoder(test_latent_space_samples)

        # Classify the test samples with a modified knn classifier
        positive_score_mat = 1 / torch.cdist(recovered_test_samples, gt_vals, p=1)
        negative_score_mat = 1 / torch.cdist(
            recovered_test_samples, recovered_train_samples, p=1
        )
        positive_scores = torch.topk(positive_score_mat, self.k, dim=-1).values.mean(-1)
        negative_scores = torch.topk(negative_score_mat, self.k, dim=-1).values.mean(-1)

        losses = torch.relu(negative_scores - positive_scores)
        coverage_loss = self.loss_function(losses, torch.zeros_like(losses)).mean()

        return coverage_loss


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
