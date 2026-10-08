import torch
import torch.nn as nn


class InferenceBlock(nn.Module):
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
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super().__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(
            input_units, d_theta, output_units
        )

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, torch.exp(0.5 * weight_log_variance)


class Decoder(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super().__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        return self.weight_mean(inps)


class Planar(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Tanh()

    def forward(self, z, u, w, b):
        """Compute z' = z + u * tanh(w^T z + b)."""
        z = z.unsqueeze(2)
        prod = torch.bmm(w, z) + b
        f_z = z + u * self.h(prod)
        f_z = f_z.squeeze(2)

        psi = w * (1 - self.h(prod) ** 2)
        log_det_jacobian = torch.log(
            torch.abs(1 + torch.bmm(psi, u)) + 1e-5
        )
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        return f_z, log_det_jacobian


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
        self.L = args.sample_num

        self.p_mu = nn.Sequential(
            nn.Linear(self.decoder_dim, self.input_dim),
        )
        self.log_det_j = 0.0

    def reparameterize(self, mu, var, mode=None):
        if mode == "train":
            std = var.sqrt()
            eps = torch.randn_like(std)
            return eps * std + mu

        sample_bias_list = []
        for _ in range(self.L):
            std = var.sqrt()
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
    """Image-specific prompt distribution with amortized planar flows."""

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
    """Image-agnostic prompt distribution with a learnable free vector."""

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
