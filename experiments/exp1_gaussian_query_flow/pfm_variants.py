"""Sample reference query tokens in Q-Former space, then transform each with planar flow."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model import PositiveQure


class GaussianQueryFlow(nn.Module):
    def __init__(self, slots, width, init_std, *, samples, steps, context_width, eval_seed):
        super().__init__()
        if samples < 1 or steps < 0 or init_std <= 0:
            raise ValueError("Require samples >= 1, steps >= 0 and init_std > 0")
        self.samples = samples
        self.steps = steps
        self.width = width
        self.init_std = init_std
        self.slot_mean = nn.Parameter(torch.zeros(slots, width))
        self.slot_log_std = nn.Parameter(torch.zeros(slots, width))
        self.text_context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, context_width), nn.SiLU())
        self.gaussian = nn.Linear(context_width, 2 * width)
        nn.init.zeros_(self.gaussian.weight)
        nn.init.zeros_(self.gaussian.bias)
        if steps:
            self.planar = nn.Linear(context_width, steps * (2 * width + 1))
        generator = torch.Generator(device="cpu").manual_seed(eval_seed)
        self.register_buffer("eval_eps", torch.randn(samples, slots, width, generator=generator))

    def forward(self, caption_embedding):
        with torch.autocast(device_type=caption_embedding.device.type, enabled=False):
            c = self.text_context(caption_embedding.float())
            mean_offset, log_scale = self.gaussian(c).chunk(2, -1)
            mean = self.slot_mean + mean_offset[:, None]
            std = (self.slot_log_std + log_scale[:, None]).clamp(-5, 3).exp()
            shape = (len(c), self.samples, *self.slot_mean.shape)
            eps = torch.randn(shape, device=c.device) if self.training else self.eval_eps[None]
            # Work in units of the original initializer std for numerical conditioning.
            start = mean[:, None] + std[:, None] * eps
            z = start
            if self.steps:
                params = self.planar(c).reshape(len(c), self.steps, 2 * self.width + 1)
                for step in params.unbind(1):
                    u, w, bias = step.split([self.width, self.width, 1], -1)
                    w = F.normalize(w, dim=-1)
                    dot = (u * w).sum(-1, keepdim=True)
                    u = u + (F.softplus(dot) - 1 - dot) * w
                    h = torch.tanh((z * w[:, None, None]).sum(-1, keepdim=True) + bias[:, None, None])
                    z = z + u[:, None, None] * h
            return start * self.init_std, z * self.init_std


def sample_spread(banks):
    """RMS of per-coordinate standard deviation across samples of the same query."""
    return banks.detach().float().var(dim=1, unbiased=False).mean().sqrt()


class PromptFlowQure(PositiveQure):
    def __init__(self, *args, pfm_config, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment != "exp1" or self.use_variance:
            raise ValueError("This notebook implements the standard Exp1 in-batch objective")
        with torch.random.fork_rng(devices=[]):
            self.ref_flow = GaussianQueryFlow(
                self.num_query_tokens, self.hidden_size, self.base.Qformer.config.initializer_range,
                samples=pfm_config["samples"], steps=pfm_config["num_flows"],
                context_width=pfm_config["hidden"], eval_seed=pfm_config["eval_seed"])

    def _encode_bank(self, bank, visual, ids, attention_mask):
        hidden = self._qformer(query_embeds=bank, input_ids=ids, attention_mask=attention_mask,
                               encoder_hidden_states=visual, encoder_attention_mask=self._memory_mask(visual))
        projected = self.base.text_proj(hidden[:, :self.num_query_tokens])
        return F.normalize(projected.mean(dim=1), dim=-1)

    def _sample_features(self, images, captions):
        visual = self._vision(images)
        text = self._tokenize(captions)
        words = self.base.Qformer.bert.embeddings.word_embeddings(text.input_ids)
        mask = text.attention_mask.unsqueeze(-1)
        pooled = (words * mask).sum(1) / mask.sum(1).clamp_min(1)
        start, banks = self.ref_flow(pooled)
        query_mask = text.attention_mask.new_ones((len(images), self.num_query_tokens))
        attention = torch.cat([query_mask, text.attention_mask], dim=1)
        features = []
        for bank in banks.unbind(dim=1):
            args = (bank, visual, text.input_ids, attention)
            if torch.is_grad_enabled():
                feature = checkpoint(self._encode_bank, *args, use_reentrant=False)
            else:
                feature = self._encode_bank(*args)
            features.append(feature)
        return torch.stack(features, dim=1), start, banks

    def query_features(self, reference_images, captions):
        features, _, _ = self._sample_features(reference_images, captions)
        return features

    @staticmethod
    def sample_scores(features, targets):
        # Each sample sees all target tokens. Average only after the QuRe max.
        return torch.einsum("bsd,jkd->bsjk", features, targets).max(dim=-1).values.mean(1)

    def forward(self, reference_images, target_images, captions, target_names):
        features, start, banks = self._sample_features(reference_images, captions)
        targets = self.target_features(target_images)
        logits = self.sample_scores(features, targets) / self.base.temp
        mask = self._positive_mask(target_names, logits.device)
        positive = torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)
        loss = (torch.logsumexp(logits, dim=1) - positive).mean()
        return {"loss": loss, "loss_contrastive": loss, "loss_var": loss.new_zeros(()),
                "pre_flow_spread": sample_spread(start), "post_flow_spread": sample_spread(banks),
                "feature_spread": sample_spread(features)}

    @torch.no_grad()
    def score(self, query_features, gallery_features, chunk_size=256):
        rows = []
        for queries in query_features.split(32):
            scores = [self.sample_scores(queries, targets.to(queries.device)).cpu()
                      for targets in gallery_features.split(chunk_size)]
            rows.append(torch.cat(scores, dim=1))
        return torch.cat(rows)
