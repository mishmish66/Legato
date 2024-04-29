import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat


class Perceptron(nn.Module):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super().__init__()
        layer_sizes = [input_dim] + layer_sizes + [output_dim]
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class TransitionModel(nn.Module):
    def __init__(
        self,
        act_dim,
        state_dim,
        latent_dim,
        n_heads=4,
        n_layers=3,
        pe_wavelength_range=[1, 1024],
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.sa_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(latent_dim, self.n_heads, batch_first=True)
                for _ in range(self.n_layers)
            ]
        )
        self.up_scales = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim * 4) for _ in range(self.n_layers)]
        )
        self.down_scales = nn.ModuleList(
            [nn.Linear(latent_dim * 4, latent_dim) for _ in range(self.n_layers)]
        )
        self.up_scale = nn.Linear(state_dim + act_dim, latent_dim)
        self.down_scale = nn.Linear(latent_dim, state_dim)

        self.pe_wavelength_range = pe_wavelength_range

    def forward(self, initial_state, actions, start_indices=None, return_mask=False):
        # Concatenate actions to initial_state
        x = torch.cat(
            [repeat(initial_state, "... e -> ... r e", r=actions.shape[-2]), actions],
            dim=-1,
        )
        x = torch.relu(self.up_scale(x))

        if start_indices is None:
            start_indices = torch.zeros(
                initial_state.shape[0], dtype=torch.long, device=initial_state.device
            )

        embed_dim = x.shape[-1]
        # Do a log range of frequencies
        pe_freqs = 1 / torch.logspace(
            np.log(self.pe_wavelength_range[0]),
            np.log(self.pe_wavelength_range[1]),
            embed_dim // 2,
            base=2,
            device=x.device,
        )
        batch_size = x.shape[0]
        seq_len = actions.shape[-2]
        # Compute the positional encoding
        times = torch.arange(seq_len, device=x.device) - start_indices[..., None]
        pe_freq_mat = einsum(pe_freqs, times, "freq, ... time -> ... time freq")
        pe = torch.cat([torch.sin(pe_freq_mat), torch.cos(pe_freq_mat)], dim=-1)
        x = x + pe

        # Make a mask out to mask out the past
        mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
        mask[torch.arange(batch_size), start_indices] = True
        mask = ~(mask.cumsum(dim=-1) > 0)

        big_mask = repeat(
            mask,
            "i seq ... -> (i heads) seq_also seq ...",
            heads=self.n_heads,
            seq_also=seq_len,
        )

        for up_scale, sa_layer, down_scale in zip(
            self.up_scales, self.sa_layers, self.down_scales
        ):
            x = x + sa_layer(x, x, x, attn_mask=big_mask, need_weights=False)[0]
            z = torch.relu(up_scale(x))
            x = x + down_scale(z)

        x = self.down_scale(x)

        if return_mask:
            return x, mask
        else:
            return x


class ActorPolicy(nn.Module):

    def __init__(
        self,
        action_dim,
        action_space_size,
        state_encoder,
        transition_model,
        action_decoder,
        horizon=128,
        iters=64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_space_size = action_space_size
        self.state_encoder = state_encoder
        self.transition_model = transition_model
        self.action_decoder = action_decoder
        self.horizon = horizon
        self.iters = iters

    def forward(self, state, target_state, prev_latent_action_plan=None):
        def gen_latent_actions(leading_dim):
            new_actions = torch.randn(
                state.shape[0], leading_dim, self.action_dim, device=state.device
            )
            new_actions = new_actions / torch.norm(
                new_actions, p=1, dim=-1, keepdim=True
            )
            new_actions = new_actions * torch.rand(
                (*new_actions.shape[:-1], 1), device=state.device
            )
            new_actions = new_actions * self.action_space_size
            return new_actions

        if prev_latent_action_plan is None:
            prev_latent_action_plan = gen_latent_actions(self.horizon)

        latent_action_plan = torch.nn.Parameter(
            prev_latent_action_plan.clone().detach()
        )

        latent_state = self.state_encoder(state).detach()
        latent_target_state = self.state_encoder(target_state).detach()

        optim = torch.optim.Adam([latent_action_plan], lr=0.025)

        state_mse = nn.MSELoss()

        for i in range(self.iters):
            optim.zero_grad()
            latent_fut_states = self.transition_model(latent_state, latent_action_plan)
            loss = state_mse(latent_fut_states, latent_target_state)
            loss.backward()
            optim.step()

            if i % 16 == 0:
                print(loss.item())

        next_action = self.action_decoder(
            torch.cat([latent_action_plan[..., 0, :], latent_state], dim=-1)
        )

        # Pop the first action and append a new one
        new_end_action = gen_latent_actions(1)
        latent_action_plan = torch.cat(
            [latent_action_plan[..., 1:, :], new_end_action], dim=-2
        )

        return next_action.detach(), latent_action_plan.detach()
