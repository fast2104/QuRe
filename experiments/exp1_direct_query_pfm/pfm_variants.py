"""Generate reference queries directly from image-conditioned latent flow."""
import torch
from torch import nn
from torch.nn import functional as F
from model import PositiveQure


class DirectPromptFlow(nn.Module):
    def __init__(self, vision_width, num_queries, query_width, hidden=64, num_flows=10):
        super().__init__()
        self.num_queries = num_queries
        self.query_width = query_width
        self.hidden = hidden
        self.num_flows = num_flows
        self.context = nn.Sequential(nn.LayerNorm(vision_width),
                                     nn.Linear(vision_width, hidden), nn.SiLU())
        self.posterior = nn.Linear(hidden, 2 * hidden)
        self.flow_params = nn.Linear(hidden, num_flows * (2 * hidden + 1))
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128), nn.SiLU(),
            nn.Linear(128, num_queries * query_width))
        self.output_norm = nn.LayerNorm(query_width)
        # Fresh initialization: no pretrained query values are copied here.
        nn.init.normal_(self.posterior.weight, std=0.01)
        nn.init.zeros_(self.posterior.bias)
        nn.init.normal_(self.flow_params.weight, std=0.005)
        nn.init.zeros_(self.flow_params.bias)
        nn.init.normal_(self.decoder[-1].weight, std=0.01)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.constant_(self.output_norm.weight, 0.02)
        nn.init.zeros_(self.output_norm.bias)

    @staticmethod
    def planar_step(z, u, w, bias):
        dot = (w * u).sum(-1, keepdim=True)
        desired = -1.0 + F.softplus(dot)
        u_hat = u + (desired - dot) * w / w.square().sum(-1, keepdim=True).clamp_min(1e-8)
        activation = torch.tanh((w * z).sum(-1, keepdim=True) + bias)
        transformed = z + u_hat * activation
        determinant = 1 + (1 - activation.square()) * (w * u_hat).sum(-1, keepdim=True)
        return transformed, determinant.clamp_min(1e-8).log().squeeze(-1)

    def forward(self, visual):
        with torch.autocast(device_type=visual.device.type, enabled=False):
            context = self.context(visual.float().mean(1))
            mu, log_var = self.posterior(context).chunk(2, -1)
            log_var = log_var.clamp(-8, 4)
            eps = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
            initial = mu + torch.exp(0.5 * log_var) * eps
            z = initial
            params = self.flow_params(context).reshape(
                visual.shape[0], self.num_flows, 2 * self.hidden + 1)
            log_det = torch.zeros_like(z[:, 0])
            for index in range(self.num_flows):
                u, w, bias = torch.split(params[:, index], [self.hidden, self.hidden, 1], -1)
                z, ld = self.planar_step(z, u, w, bias)
                log_det = log_det + ld
            log_q = -0.5 * (log_var + (initial - mu).square() * torch.exp(-log_var)).sum(-1)
            log_p = -0.5 * z.square().sum(-1)
            # Monte Carlo KL in train; normalize per latent dimension.
            # In eval this diagnostic is not used as an unbiased KL estimate.
            kl = (log_q - log_p - log_det).mean() / self.hidden
            decoded = self.decoder(z).reshape(-1, self.num_queries, self.query_width)
            queries = self.output_norm(decoded)
            return queries, kl, queries.square().mean().sqrt().detach()


class PromptFlowQure(PositiveQure):
    def __init__(self, *args, pfm_config, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment != 'exp1' or pfm_config['variant'] != 'direct_planar_image':
            raise ValueError('This implementation requires Exp1/direct_planar_image')
        with torch.random.fork_rng():
            self.ref_flow = DirectPromptFlow(
                self.encoder_width, self.num_query_tokens, self.hidden_size,
                hidden=pfm_config['hidden'], num_flows=pfm_config['num_flows'])
        self.beta = pfm_config['kl_weight']
        self.warmup_epochs = pfm_config['kl_warmup_epochs']
        self.epoch = 0

    def _query_hidden(self, reference_images, captions):
        text = self._tokenize(captions)
        visual = self._vision(reference_images)
        queries, self.flow_kl, self.query_rms = self.ref_flow(visual)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.long, device=self.device)
        hidden = self._qformer(
            query_embeds=queries, input_ids=text.input_ids,
            attention_mask=torch.cat((query_mask, text.attention_mask), dim=1),
            encoder_hidden_states=visual,
            encoder_attention_mask=self._memory_mask(visual))
        return hidden[:, :self.num_query_tokens, :]

    def forward(self, *args, **kwargs):
        losses = super().forward(*args, **kwargs)
        coefficient = self.beta * min(1.0, (self.epoch + 1) / max(1, self.warmup_epochs))
        losses['loss'] = losses['loss'] + coefficient * self.flow_kl
        losses['flow_kl'] = self.flow_kl.detach()
        losses['query_rms'] = self.query_rms
        losses['kl_coefficient'] = losses['loss'].new_tensor(coefficient)
        return losses
