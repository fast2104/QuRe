import math

import torch
import torch.nn as nn


class InferenceBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class PlanarFlow(nn.Module):
    def forward(self, latent, u, w, bias):
        latent_column = latent.unsqueeze(-1)
        activation = torch.tanh(torch.bmm(w, latent_column) + bias)
        transformed = (latent_column + u * activation).squeeze(-1)
        psi = w * (1 - activation.square())
        determinant = 1 + torch.bmm(psi, u)
        log_abs_det_jacobian = torch.log(determinant.abs() + 1e-5)
        return transformed, log_abs_det_jacobian.squeeze(-1).squeeze(-1)


class PlanarPromptFlow(nn.Module):
    """Generate image-conditioned Q-Former prompts with Bayes-PFL planar flows."""

    def __init__(self, vision_dim, hidden_dim, num_prompts, num_flows=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts
        self.num_flows = num_flows
        self.condition = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Softplus(),
        )
        self.mean = InferenceBlock(hidden_dim, hidden_dim, hidden_dim)
        self.log_variance = InferenceBlock(hidden_dim, hidden_dim, hidden_dim)
        self.amor_u = nn.Linear(hidden_dim, num_flows * hidden_dim)
        self.amor_w = nn.Linear(hidden_dim, num_flows * hidden_dim)
        self.amor_b = nn.Linear(hidden_dim, num_flows)
        self.flows = nn.ModuleList(PlanarFlow() for _ in range(num_flows))
        self.prompt_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, num_prompts * hidden_dim),
        )
        nn.init.normal_(self.prompt_decoder[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.prompt_decoder[-1].bias)

    def forward(self, image_features):
        if image_features.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward(image_features.float())
        return self._forward(image_features.float())

    def _forward(self, image_features):
        condition = self.condition(image_features)
        mean = self.mean(condition)
        log_variance = self.log_variance(condition).clamp(min=-10.0, max=10.0)

        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        initial_latent = latent

        batch_size = condition.shape[0]
        u = self.amor_u(condition).reshape(
            batch_size, self.num_flows, self.hidden_dim, 1
        )
        w = self.amor_w(condition).reshape(
            batch_size, self.num_flows, 1, self.hidden_dim
        )
        bias = self.amor_b(condition).reshape(batch_size, self.num_flows, 1, 1)

        log_det_jacobian = torch.zeros(
            batch_size, device=condition.device, dtype=torch.float32
        )
        for index, flow in enumerate(self.flows):
            latent, log_det = flow(
                latent,
                u[:, index].to(dtype=latent.dtype),
                w[:, index].to(dtype=latent.dtype),
                bias[:, index].to(dtype=latent.dtype),
            )
            log_det_jacobian = log_det_jacobian + log_det.float()

        prompts = self.prompt_decoder(latent).reshape(
            batch_size, self.num_prompts, self.hidden_dim
        )

        log_variance = log_variance.float()
        initial_latent = initial_latent.float()
        mean = mean.float()
        final_latent = latent.float()
        log_two_pi = math.log(2.0 * math.pi)
        log_q = -0.5 * (
            log_two_pi
            + log_variance
            + (initial_latent - mean).square() * torch.exp(-log_variance)
        ).sum(dim=-1)
        log_p = -0.5 * (log_two_pi + final_latent.square()).sum(dim=-1)
        flow_kl = (log_q - log_p - log_det_jacobian).mean()

        return prompts, flow_kl
