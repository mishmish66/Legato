import torch


class PBallSampler:
    def __init__(self, d, p=1, sample_factor=1.5, device="cpu"):
        self.d = d
        self.p = p
        self.sample_factor = sample_factor
        self.device = device

    def __call__(self, n_samples):
        # Use rejection sampling to sample from the p-ball

        sample_count = int(n_samples * self.sample_factor)

        def gen_samples():
            new_samples = torch.rand(sample_count, self.d, device=self.device) * 2 - 1
            valids = torch.norm(new_samples, p=self.p, dim=-1) <= 1
            return new_samples[valids]

        # Initialize the samples
        result = torch.ones(n_samples, self.d, device=self.device) * torch.nan

        while True:
            nans = torch.isnan(result).any(dim=-1)
            if not nans.any():
                return result

            new_samples = gen_samples()
            n_samples_to_add = min(nans.sum(), len(new_samples))
            indices_to_fill = nans
            indices_to_fill[(~nans).sum() :][n_samples_to_add:] = False
            result[indices_to_fill] = new_samples[:n_samples_to_add]


if __name__ == "__main__":
    sampler = PBallSampler(2)
    samples = sampler(100)
    print(samples)
