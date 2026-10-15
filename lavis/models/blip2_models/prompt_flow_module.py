import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceBlock(nn.Module):
    """
    MLP inference block with Softplus activations from Bayes-PFL.
    """
    def __init__(self, input_units, d_theta, output_units):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        return self.module(inps)


class Encoder(nn.Module):
    """
    Encoder predicting mean and log-variance for base Gaussian distribution (from Bayes-PFL).
    """
    def __init__(self, input_units=768, d_theta=384, output_units=768):
        super().__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(
            input_units, d_theta, output_units
        )

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        weight_std = torch.exp(0.5 * torch.clamp(weight_log_variance, min=-10.0, max=10.0))
        return weight_mean, weight_std


class Decoder(nn.Module):
    """
    Decoder mapping transformed latents back to query token dimension (from Bayes-PFL).
    """
    def __init__(self, input_units=768, d_theta=384, output_units=768):
        super().__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        return self.weight_mean(inps)


class Planar(nn.Module):
    """
    Planar normalizing flow layer from Bayes-PFL (models/flows.py).
    Computes: z' = z + u * tanh(w^T z + b)
    """
    def __init__(self):
        super().__init__()
        self.h = nn.Tanh()

    def forward(self, z, u, w, b):
        """
        Input shapes:
        z: (N, z_size)
        u: (N, z_size, 1)
        w: (N, 1, z_size)
        b: (N, 1, 1)
        """
        z_unsqueezed = z.unsqueeze(2)  # (N, z_size, 1)
        prod = torch.bmm(w, z_unsqueezed) + b  # (N, 1, 1)
        h_prod = self.h(prod)
        f_z = z_unsqueezed + u * h_prod  # (N, z_size, 1)
        f_z = f_z.squeeze(2)  # (N, z_size)

        # logdetJ
        psi = w * (1.0 - h_prod ** 2)  # (N, 1, z_size)
        psi_u = torch.bmm(psi, u)  # (N, 1, 1)
        log_det_jacobian = torch.log(
            torch.abs(1.0 + psi_u) + 1e-5
        )
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        return f_z, log_det_jacobian


class PromptFlow(nn.Module):
    """
    Prompt Flow Module adapted from Bayes-PFL (CVPR 2025) for QuRe query tokens.

    Models query tokens using a learned non-Gaussian distribution via
    Planar Normalizing Flows:
        1. Encodes base query tokens Q_pre into base distribution q(z_0) = N(mu(Q_pre), diag(sigma(Q_pre)^2)).
        2. Generates amortized flow parameters u, w, b from Q_pre.
        3. Samples z_0 via reparameterization trick.
        4. Warps z_0 through K planar normalizing flow steps: z_0 -> z_1 -> ... -> z_K.
        5. Decodes z_K to delta Q_phi.
        6. Produces perturbed query tokens residually:
           Q_new = Q_pre + gamma * delta Q_phi
    """
    def __init__(
        self,
        embed_dim=768,
        num_query_tokens=32,
        num_flows=4,
        gamma=0.1,
        eval_mode="mean",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        self.num_flows = num_flows
        self.gamma = gamma
        self.eval_mode = eval_mode  # "mean", "sample", or "fixed"

        hidden_dim = embed_dim // 2
        self.encoder = Encoder(input_units=embed_dim, d_theta=hidden_dim, output_units=embed_dim)
        self.decoder = Decoder(input_units=embed_dim, d_theta=hidden_dim, output_units=embed_dim)

        # Amortized flow parameters
        self.amor_u = nn.Linear(embed_dim, num_flows * embed_dim)
        self.amor_w = nn.Linear(embed_dim, num_flows * embed_dim)
        self.amor_b = nn.Linear(embed_dim, num_flows)

        for k in range(num_flows):
            self.add_module(f"flow_{k}", Planar())

        self._cached_eval_delta = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_eval_cache(self):
        self._cached_eval_delta = None

    def forward(self, base_query_tokens, batch_size=1):
        """
        Args:
            base_query_tokens: Tensor of shape (1, num_query_tokens, embed_dim) or (num_query_tokens, embed_dim)
            batch_size: int, desired batch size for output query tokens
        Returns:
            q_new: Tensor of shape (batch_size, num_query_tokens, embed_dim)
            delta_q: Tensor of shape (batch_size, num_query_tokens, embed_dim)
        """
        device = base_query_tokens.device
        dtype = base_query_tokens.dtype

        if base_query_tokens.dim() == 3:
            q_in = base_query_tokens.squeeze(0)  # (num_query_tokens, embed_dim)
        else:
            q_in = base_query_tokens

        # If in eval and using fixed cached delta
        if not self.training and self.eval_mode == "fixed" and self._cached_eval_delta is not None:
            delta_q = self._cached_eval_delta.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
            q_base = base_query_tokens.expand(batch_size, -1, -1)
            return q_base + self.gamma * delta_q, delta_q

        # 1. Base distribution parameters from Encoder
        mu, std = self.encoder(q_in)  # (num_query_tokens, embed_dim)

        # 2. Amortized flow parameters
        u = self.amor_u(q_in).view(self.num_query_tokens, self.num_flows, self.embed_dim, 1)
        w = self.amor_w(q_in).view(self.num_query_tokens, self.num_flows, 1, self.embed_dim)
        b = self.amor_b(q_in).view(self.num_query_tokens, self.num_flows, 1, 1)

        # 3. Sampling via Reparameterization Trick
        if self.training or self.eval_mode == "sample":
            eps = torch.randn(batch_size, self.num_query_tokens, self.embed_dim, device=device, dtype=dtype)
            z_0 = mu.unsqueeze(0) + std.unsqueeze(0) * eps
        else:
            # eval_mode == "mean" or deterministic evaluation
            z_0 = mu.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. Planar Normalizing Flows: z_0 -> z_1 -> ... -> z_K
        total_tokens = batch_size * self.num_query_tokens
        z = z_0.contiguous().view(total_tokens, self.embed_dim)

        for k in range(self.num_flows):
            flow_k = getattr(self, f"flow_{k}")
            u_k = u[:, k, :, :].unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().view(total_tokens, self.embed_dim, 1)
            w_k = w[:, k, :, :].unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().view(total_tokens, 1, self.embed_dim)
            b_k = b[:, k, :, :].unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().view(total_tokens, 1, 1)
            z, _ = flow_k(z, u_k, w_k, b_k)

        # 5. Decode transformed latent back to query token space
        z_k = z.view(batch_size, self.num_query_tokens, self.embed_dim)
        delta_q = self.decoder(z_k)  # (batch_size, num_query_tokens, embed_dim)

        if not self.training and self.eval_mode == "fixed":
            self._cached_eval_delta = delta_q[:1].detach().clone()

        # 6. Residual addition: Q_new = Q_pre + gamma * delta_Q
        q_base = base_query_tokens.expand(batch_size, -1, -1)
        q_new = q_base + self.gamma * delta_q
        return q_new, delta_q


class PFL(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.z_size = args.embed_dim
        self.input_size = args.embed_dim
        self.input_dim = args.embed_dim
        self.encoder_dim = args.embed_dim
        self.decoder_dim = args.embed_dim
        self.is_cuda = True
        self.L = getattr(args, "sample_num", 1)

        self.p_mu = nn.Sequential(
            nn.Linear(self.decoder_dim, self.input_dim),
        )
        self.log_det_j = 0.0

    def reparameterize(self, mu, var, mode=None):
        std = var.sqrt()
        if mode == "train":
            eps = torch.randn_like(std)
            return eps * std + mu

        sample_bias_list = []
        for _ in range(self.L):
            eps = torch.randn_like(std)
            sample_bias_list.append(eps * std + mu)
        return torch.stack(sample_bias_list, dim=0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder(z)
        x_mean = self.p_mu(h)
        return x_mean.view(-1, self.input_size)

    def forward(self, x):
        z_mu, z_var = self.encode(x)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)
        return x_mean, z_mu, z_var, self.log_det_j, z, z


class PlanarPFL(PFL):
    """Image-specific prompt distribution with amortized planar flows from Bayes-PFL."""

    def __init__(self, encoder, decoder, args):
        super().__init__(encoder, decoder, args)
        self.log_det_j = 0.0
        self.num_flows = args.num_flows

        self.amor_u = nn.Linear(
            self.encoder_dim, self.num_flows * self.z_size
        )
        self.amor_w = nn.Linear(
            self.encoder_dim, self.num_flows * self.z_size
        )
        self.amor_b = nn.Linear(self.encoder_dim, self.num_flows)

        for k in range(self.num_flows):
            self.add_module("flow_" + str(k), Planar())

    def encode(self, x):
        batch_size = x.size(0)
        mu, var = self.encoder(x)
        u = self.amor_u(x).view(
            batch_size, self.num_flows, self.z_size, 1
        )
        w = self.amor_w(x).view(
            batch_size, self.num_flows, 1, self.z_size
        )
        b = self.amor_b(x).view(batch_size, self.num_flows, 1, 1)
        return mu, var, u, w, b

    def forward(self, x, mode=None):
        self.log_det_j = torch.zeros(x.shape[0], device=x.device)
        z_mu, z_var, u, w, b = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var, mode=mode)

        if mode == "train":
            log_det_j = self.log_det_j
            z_list = [z_0.clone()]
            for k in range(self.num_flows):
                flow_k = getattr(self, "flow_" + str(k))
                z_k, log_det_jacobian = flow_k(
                    z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
                )
                z_list.append(z_k)
                log_det_j = log_det_j + log_det_jacobian

            x_mean = self.decode(z_list[-1])
            return x_mean, z_mu, z_var, log_det_j, z_list[0], z_list[-1]

        zk_list = []
        for i in range(z_0.shape[0]):
            z_list = [z_0[i, :, :].clone()]
            for k in range(self.num_flows):
                flow_k = getattr(self, "flow_" + str(k))
                z_k, _ = flow_k(
                    z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
                )
                z_list.append(z_k)
            zk_list.append(z_list[-1])

        z_k_final = torch.cat(zk_list, dim=0)
        x_mean = self.decode(z_list[-1])
        return x_mean, z_k_final, z_k_final, z_k_final, z_k_final, z_k_final


class PlanarPFL_state(PFL):
    """Image-agnostic prompt distribution with a learnable free vector from Bayes-PFL."""

    def __init__(self, encoder, decoder, args):
        super().__init__(encoder, decoder, args)
        self.log_det_j = 0.0
        self.num_flows = args.num_flows

        self.amor_u = nn.Parameter(
            torch.randn(1, self.num_flows, self.z_size, 1)
        )
        self.amor_w = nn.Parameter(
            torch.randn(1, self.num_flows, 1, self.z_size)
        )
        self.amor_b = nn.Parameter(torch.randn(1, self.num_flows, 1, 1))
        self.state = nn.Parameter(torch.randn(1, self.encoder_dim))

        for k in range(self.num_flows):
            self.add_module("flow_" + str(k), Planar())

    def encode(self, x):
        mu, var = self.encoder(x)
        return mu, var, self.amor_u, self.amor_w, self.amor_b

    def forward(self, x, mode=None):
        x = self.state
        self.log_det_j = torch.zeros(x.shape[0], device=x.device)
        z_mu, z_var, u, w, b = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var, mode=mode)

        if mode == "train":
            log_det_j = self.log_det_j
            z_list = [z_0.clone()]
            for k in range(self.num_flows):
                flow_k = getattr(self, "flow_" + str(k))
                z_k, log_det_jacobian = flow_k(
                    z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
                )
                z_list.append(z_k)
                log_det_j = log_det_j + log_det_jacobian

            x_mean = self.decode(z_list[-1])
            return x_mean, z_mu, z_var, log_det_j, z_list[0], z_list[-1]

        zk_list = []
        for i in range(z_0.shape[0]):
            z_list = [z_0[i, :, :].clone()]
            for k in range(self.num_flows):
                flow_k = getattr(self, "flow_" + str(k))
                z_k, _ = flow_k(
                    z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
                )
                z_list.append(z_k)
            zk_list.append(z_list[-1])

        z_k_final = torch.cat(zk_list, dim=0)
        x_mean = self.decode(z_list[-1])
        return x_mean, z_k_final, z_k_final, z_k_final, z_k_final, z_k_final


def compute_query_token_diversity(query_tokens):
    """
    Computes query-token diversity D_Q across samples in a batch.
    D_Q = 1 - [2 / (B(B-1))] sum_{i<j} cos(Q^(i), Q^(j))
    """
    B = query_tokens.shape[0]
    if B <= 1:
        return 0.0
    flat_tokens = query_tokens.reshape(B, -1)
    norm = flat_tokens.norm(dim=-1, keepdim=True) + 1e-8
    normalized = flat_tokens / norm
    sim_matrix = torch.mm(normalized, normalized.t())
    indices = torch.triu_indices(B, B, offset=1)
    mean_cos = sim_matrix[indices[0], indices[1]].mean().item()
    return float(max(0.0, 1.0 - mean_cos))


def compute_feature_diversity(features):
    """
    Computes feature diversity D_Z across samples in a batch after Q-Former.
    D_Z = 1 - [2 / (B(B-1))] sum_{i<j} cos(z_q^(i), z_q^(j))
    """
    B = features.shape[0]
    if B <= 1:
        return 0.0
    flat_feat = features.reshape(B, -1)
    norm = flat_feat.norm(dim=-1, keepdim=True) + 1e-8
    normalized = flat_feat / norm
    sim_matrix = torch.mm(normalized, normalized.t())
    indices = torch.triu_indices(B, B, offset=1)
    mean_cos = sim_matrix[indices[0], indices[1]].mean().item()
    return float(max(0.0, 1.0 - mean_cos))
