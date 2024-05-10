import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat

from legato.sampler import PBallSampler


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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class FreqLayer(nn.Module):
    def __init__(self, freqs):
        super().__init__()
        self.freqs = freqs

    @property
    def device(self):
        return self.freqs.device

    def forward(self, x):
        x_freq = einsum(x, self.freqs, "... d, ... w -> ... w d")
        sines = torch.sin(x_freq)
        cosines = torch.cos(x_freq)
        y = rearrange([sines, cosines], "f ... w d -> ... (w d f)")
        return y


class Freqceptron(Perceptron):
    def __init__(self, input_dim, layer_sizes, output_dim, freqs):
        super().__init__(input_dim * 2 * len(freqs), layer_sizes, output_dim)
        self.freq_layer = FreqLayer(freqs)

    def forward(self, x):
        x = self.freq_layer(x)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class DoublePerceptron(Perceptron):
    def __init__(self, input_dim_a, input_dim_b, layer_sizes, output_dim):
        super().__init__(input_dim_a + input_dim_b, layer_sizes, output_dim)

    def forward(self, inputs):
        a, b = inputs
        x = torch.cat([a, b], dim=-1)
        return super().forward(x)


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

    @property
    def device(self):
        return next(self.parameters()).device


class ActorPolicy(nn.Module):

    def __init__(
        self,
        action_dim,
        action_space_size,
        state_encoder,
        transition_model,
        state_decoder,
        action_decoder,
        optim_factory=torch.optim.SGD,
        loss_func=nn.L1Loss(),
        lr=0.01,
        decay=0.01,
        horizon=128,
        iters=64,
        tail_states=None,
        discount=1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_space_size = action_space_size
        self.state_encoder = state_encoder
        self.transition_model = transition_model
        self.state_decoder = state_decoder
        self.action_decoder = action_decoder
        self.action_sampler = PBallSampler(
            action_dim, 1, action_space_size, device=action_decoder.device
        )
        self.optim_factory = optim_factory
        self.loss_func = loss_func
        self.lr = lr
        self.decay = decay
        self.horizon = horizon
        self.iters = iters
        self.tail_states = tail_states
        self.discount = discount

    def forward(
        self,
        state,
        target_state,
        horizons,
        prev_latent_action_plan=None,
        return_curve=False,
    ):

        max_horizon = horizons.max()
        if prev_latent_action_plan is None:
            n_samples = int(state.shape[0] * max_horizon)
            prev_latent_action_plan = rearrange(
                self.action_sampler(n_samples), "(n h) d -> n h d", n=state.shape[0]
            )
        else:
            prev_latent_action_plan = prev_latent_action_plan.clone().detach()

        max_horizon = max(max_horizon, prev_latent_action_plan.shape[-2])

        n_samples = int(state.shape[0] * max_horizon)
        latent_action_plan = rearrange(
            self.action_sampler(n_samples), "(n h) d -> n h d", n=state.shape[0]
        )
        latent_action_plan[..., : prev_latent_action_plan.shape[-2], :] = (
            prev_latent_action_plan
        )

        latent_action_plan = torch.nn.Parameter(latent_action_plan.clone().detach())

        latent_state = self.state_encoder(state).detach()
        latent_target_state = self.state_encoder(target_state).detach()

        optim = self.optim_factory([latent_action_plan], lr=self.lr)
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, self.decay ** (1 / self.iters)
        )

        use_action = (
            torch.arange(0, max_horizon, device="cuda")[None] < horizons[..., None]
        )

        loss_curve = []

        for i in range(self.iters):
            optim.zero_grad()
            latent_fut_states = self.transition_model(latent_state, latent_action_plan)
            fut_states = self.state_decoder(latent_fut_states)
            fut_states_filt = torch.where(
                use_action[..., None], fut_states, target_state[..., None, :]
            )[..., :2]
            target_broad = repeat(
                target_state, "... d -> ... t d", t=fut_states_filt.shape[-2]
            )
            # Use just the tail states if set
            if self.tail_states is not None:
                n_states = min(fut_states_filt.shape[-2], self.tail_states)
                target_broad = target_broad[..., -n_states:, :]
                fut_states_filt = fut_states_filt[..., -n_states:, :]

            losses = self.loss_func(fut_states_filt, target_broad[..., :2])
            times = torch.arange(0, max_horizon, device="cuda")
            times = times[-losses.shape[-1] :]
            discounts = torch.pow(self.discount, times)
            discounted_losses = einsum(losses, discounts, "... e t, t -> ... e t")
            loss = discounted_losses.mean()

            loss.backward()
            optim.step()
            lr_sched.step()

            loss_curve.append(loss.item())

            if i == self.iters - 1:
                print(loss.item())

        next_action = self.action_decoder((latent_action_plan[..., 0, :], latent_state))

        # Pop the first action and append a new one
        # new_end_action = latent_action_plan[..., -1:, :]  # gen_latent_actions(1)
        latent_action_plan = latent_action_plan[..., 1:, :]  # torch.cat(
        #     [latent_action_plan[..., 1:, :], new_end_action], dim=-2
        # )

        if return_curve:
            return next_action.detach(), latent_action_plan.detach(), loss_curve
        else:
            return next_action.detach(), latent_action_plan.detach()
