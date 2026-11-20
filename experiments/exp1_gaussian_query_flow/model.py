"""Positive-only QuRe variants with isolated training objectives.

Unchanged BLIP-2 modules are loaded from the author's QuRe checkout.  This file
only supplies the new query flows and objective.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_qure_to_path(qure_root: str | Path):
    root = str(Path(qure_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_qure(backbone: str, device: torch.device, qure_root: str | Path):
    root = Path(qure_root).resolve()
    add_qure_to_path(root)
    # Code-ablation (the existing server QuRe checkout) keeps the pretrained
    # BLIP-2 checkpoint at <qure-root>/checkpoints. Its YAML reads this env var;
    # set it automatically so a sibling experiment does not download/copy it.
    checkpoint_root = root / "checkpoints"
    if checkpoint_root.is_dir() and "QURe_CHECKPOINT_ROOT" not in os.environ:
        os.environ["QURe_CHECKPOINT_ROOT"] = str(checkpoint_root)
    pretrained_path = Path(os.environ.get("QURe_CHECKPOINT_ROOT", checkpoint_root)) / "blip2_pretrained_vitL.pth"
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            "Cannot find the QuRe BLIP-2 pretrained checkpoint: "
            f"{pretrained_path}"
        )
    print(f"[PRETRAINED] Loading BLIP-2 weights from: {pretrained_path}", flush=True)
    # The author's registry only knows this architecture after this module has
    # been imported.  We import it, but do not alter its source code.
    import lavis.models.blip2_models.blip2_qure  # noqa: F401
    from lavis.models import load_model_and_preprocess
    return load_model_and_preprocess(name="blip2_qure", model_type=backbone, is_eval=False, device=device)


class PositiveQure(nn.Module):
    def __init__(self, base, experiment: str, max_txt_len: int, use_variance: bool,
                 lambda_var: float, var_gamma: float, var_epsilon: float):
        super().__init__()
        if experiment not in {"exp1", "exp2", "exp3"}:
            raise ValueError(f"Unknown experiment: {experiment}")
        self.base = base
        self.experiment = experiment
        self.max_txt_len = max_txt_len
        self.num_query_tokens = base.query_tokens.shape[1]
        self.hidden_size = base.Qformer.config.hidden_size
        self.encoder_width = base.Qformer.config.encoder_width
        self.use_variance = use_variance
        self.lambda_var = lambda_var
        self.var_gamma = var_gamma
        self.var_epsilon = var_epsilon
        if experiment == "exp2":
            self.reference_memory_proj = nn.Linear(self.hidden_size, self.encoder_width)

    @property
    def device(self):
        return next(self.parameters()).device

    def _queries(self, batch_size):
        return self.base.query_tokens.expand(batch_size, -1, -1)

    def _tokenize(self, captions):
        return self.base.tokenizer(
            captions, padding="max_length", truncation=True,
            max_length=self.max_txt_len, return_tensors="pt",
        ).to(self.device)

    def _vision(self, images):
        with self.base.maybe_autocast():
            memory = self.base.ln_vision(self.base.visual_encoder(images))
        return memory.float()

    @staticmethod
    def _memory_mask(memory):
        return torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)

    def _qformer(self, *, query_embeds, input_ids=None, attention_mask=None,
                 encoder_hidden_states=None, encoder_attention_mask=None):
        return self.base.Qformer.bert(
            input_ids=input_ids, query_embeds=query_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state

    def _manual_qformer(self, query_embeds, *, input_ids=None, latent_tokens=None,
                        attention_mask=None, encoder_hidden_states=None,
                        encoder_attention_mask=None):
        """Q-Former execution with two minimal Exp3 extensions.

        - If image memory is None, cross-attention layers are skipped.
        - latent_tokens are text-like 768-D embeddings concatenated after query
          tokens in the self-attention sequence.
        The original Q-Former modules/weights are reused directly.
        """
        bert = self.base.Qformer.bert
        emb = bert.embeddings
        batch_size = query_embeds.shape[0]
        if latent_tokens is not None and input_ids is not None:
            raise ValueError("Use either input_ids or latent_tokens, not both")
        if latent_tokens is not None:
            length = latent_tokens.shape[1]
            positions = emb.position_ids[:, :length].to(latent_tokens.device)
            text_embed = latent_tokens + emb.position_embeddings(positions)
            embeddings = torch.cat([query_embeds, text_embed], dim=1)
            embeddings = emb.dropout(emb.LayerNorm(embeddings))
        else:
            embeddings = emb(input_ids=input_ids, query_embeds=query_embeds)
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device)
        extended_mask = bert.get_extended_attention_mask(attention_mask, embeddings.shape[:2], embeddings.device, False)
        if encoder_hidden_states is None:
            extended_encoder_mask = None
        else:
            if encoder_attention_mask is None:
                encoder_attention_mask = self._memory_mask(encoder_hidden_states)
            extended_encoder_mask = bert.invert_attention_mask(encoder_attention_mask)

        hidden = embeddings
        query_length = query_embeds.shape[1]
        for layer in bert.encoder.layer:
            self_outputs = layer.attention(hidden, extended_mask, None, output_attentions=False)
            attention_output = self_outputs[0]
            query_output = attention_output[:, :query_length, :]
            if layer.has_cross_attention and encoder_hidden_states is not None:
                query_output = layer.crossattention(
                    query_output, extended_mask, None, encoder_hidden_states,
                    extended_encoder_mask, output_attentions=False,
                )[0]
            query_output = layer.feed_forward_chunk_query(query_output)
            if attention_output.shape[1] > query_length:
                text_output = layer.feed_forward_chunk(attention_output[:, query_length:, :])
                hidden = torch.cat([query_output, text_output], dim=1)
            else:
                hidden = query_output
        return hidden

    def _image_hidden(self, images):
        memory = self._vision(images)
        queries = self._queries(memory.shape[0])
        hidden = self._qformer(
            query_embeds=queries, encoder_hidden_states=memory,
            encoder_attention_mask=self._memory_mask(memory),
        )
        return hidden[:, :self.num_query_tokens, :]

    def _query_hidden(self, reference_images, captions):
        text = self._tokenize(captions)
        visual = self._vision(reference_images)
        batch_size = reference_images.shape[0]
        queries = self._queries(batch_size)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.long, device=self.device)

        if self.experiment == "exp1":
            mask = torch.cat([query_mask, text.attention_mask], dim=1)
            hidden = self._qformer(
                query_embeds=queries, input_ids=text.input_ids, attention_mask=mask,
                encoder_hidden_states=visual, encoder_attention_mask=self._memory_mask(visual),
            )
            return hidden[:, :self.num_query_tokens, :]

        if self.experiment == "exp2":
            reference_hidden = self._qformer(
                query_embeds=queries, encoder_hidden_states=visual,
                encoder_attention_mask=self._memory_mask(visual),
            )[:, :self.num_query_tokens, :]
            reference_memory = self.reference_memory_proj(reference_hidden)
            mask = torch.cat([query_mask, text.attention_mask], dim=1)
            hidden = self._qformer(
                query_embeds=queries, input_ids=text.input_ids, attention_mask=mask,
                encoder_hidden_states=reference_memory,
                encoder_attention_mask=self._memory_mask(reference_memory),
            )
            return hidden[:, :self.num_query_tokens, :]

        # Exp3 stage 1: raw text + shared queries, cross-attention skipped.
        stage1_mask = torch.cat([query_mask, text.attention_mask], dim=1)
        text_hidden = self._manual_qformer(
            queries, input_ids=text.input_ids, attention_mask=stage1_mask,
        )[:, :self.num_query_tokens, :]
        # Exp3 stage 2: [same shared queries, T] self-attention; only queries
        # cross-attend to the reference visual memory.
        stage2_mask = torch.ones((batch_size, self.num_query_tokens * 2), dtype=torch.long, device=self.device)
        hidden = self._manual_qformer(
            queries, latent_tokens=text_hidden, attention_mask=stage2_mask,
            encoder_hidden_states=visual, encoder_attention_mask=self._memory_mask(visual),
        )
        return hidden[:, :self.num_query_tokens, :]

    def query_features(self, reference_images, captions, return_pre_norm=False):
        hidden = self._query_hidden(reference_images, captions)
        tokens_pre = self.base.text_proj(hidden)
        pre = tokens_pre.mean(dim=1)
        feature = F.normalize(pre, dim=-1)
        return (feature, pre) if return_pre_norm else feature

    def target_features(self, target_images, return_pre_norm=False):
        hidden = self._image_hidden(target_images)
        pre = self.base.vision_proj(hidden)
        feature = F.normalize(pre, dim=-1)
        return (feature, pre) if return_pre_norm else feature

    @staticmethod
    def _best_target(query, target_tokens, target_pre):
        similarities = torch.bmm(target_tokens, query.unsqueeze(-1)).squeeze(-1)
        indices = similarities.argmax(dim=1)
        batch = torch.arange(query.shape[0], device=query.device)
        return target_tokens[batch, indices], target_pre[batch, indices]

    def variance_loss(self, features):
        std = torch.sqrt(features.var(dim=0, unbiased=False) + self.var_epsilon)
        return F.relu(self.var_gamma - std).mean()

    @staticmethod
    def _positive_mask(target_names, device):
        """Repeated FashionIQ target IDs are multiple valid positives."""
        return torch.tensor(
            [[left == right for right in target_names] for left in target_names],
            dtype=torch.bool,
            device=device,
        )

    def forward(self, reference_images, target_images, captions, target_names):
        q, q_pre = self.query_features(reference_images, captions, return_pre_norm=True)
        target, target_pre = self.target_features(target_images, return_pre_norm=True)
        # Exact QuRe retrieval score for every query i and every target image j:
        # S[i, j] = max_k(q_i dot target_j,k). Other images in the same batch
        # are implicit negatives; no negative-image branch/sampling is used.
        scores = torch.einsum("id,jkd->ijk", q, target).max(dim=-1).values
        logits = scores / self.base.temp
        positive_mask = self._positive_mask(target_names, logits.device)
        positive_logit = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
        loss_contrastive = -(positive_logit - torch.logsumexp(logits, dim=1)).mean()
        if self.use_variance:
            # Contrastive learning already prevents global collapse. This is an
            # optional extra regularizer on one pre-normalized feature/image.
            loss_var = self.variance_loss(q_pre) + self.variance_loss(target_pre.mean(dim=1))
        else:
            loss_var = loss_contrastive.new_zeros(())
        return {
            "loss": loss_contrastive + self.lambda_var * loss_var,
            "loss_contrastive": loss_contrastive,
            "loss_var": loss_var,
        }

    @torch.no_grad()
    def score(self, query_features, gallery_features, chunk_size=2048):
        rows = []
        for start in range(0, gallery_features.shape[0], chunk_size):
            gallery = gallery_features[start:start + chunk_size].to(query_features.device)
            rows.append(torch.einsum("qd,ikd->qik", query_features, gallery).max(dim=-1).values.cpu())
        return torch.cat(rows, dim=1)


class PairwiseKLQure(PositiveQure):
    """Exp4: Exp1 fusion with a separate QuRe-style pairwise KL objective.

    This class deliberately does not alter ``PositiveQure.forward``.  Exp1--3
    therefore retain their existing in-batch InfoNCE implementation exactly.
    Negatives are other target images already in the batch; no negative-image
    branch, sampling, or hard-negative reranking is introduced.
    """

    def pairwise_kl_preference_loss(self, scores, target_names):
        """Average QuRe binary-preference KL over every valid in-batch pair.

        For query ``i``, every target with the same FashionIQ target ID is a
        positive, and every different target ID is a negative.  Each valid
        positive/negative pair uses KL(log_softmax([s_pos, s_neg]), [1, 0]).
        """
        logits = scores / self.base.temp
        positive_mask = self._positive_mask(target_names, logits.device)
        negative_mask = ~positive_mask
        batch_size = logits.shape[1]

        positive_logits = logits.unsqueeze(2).expand(-1, -1, batch_size)
        negative_logits = logits.unsqueeze(1).expand(-1, batch_size, -1)
        valid_pairs = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)
        pair_logits = torch.stack((positive_logits, negative_logits), dim=-1)[valid_pairs]
        if pair_logits.numel() == 0:
            raise ValueError("Pairwise KL needs at least one in-batch negative per query.")

        log_probs = F.log_softmax(pair_logits, dim=-1)
        preference_targets = torch.zeros_like(log_probs)
        preference_targets[:, 0] = 1.0
        return F.kl_div(log_probs, preference_targets, reduction="batchmean")

    def forward(self, reference_images, target_images, captions, target_names):
        q, q_pre = self.query_features(reference_images, captions, return_pre_norm=True)
        target, target_pre = self.target_features(target_images, return_pre_norm=True)
        scores = torch.einsum("id,jkd->ijk", q, target).max(dim=-1).values
        loss_pairwise_kl = self.pairwise_kl_preference_loss(scores, target_names)
        if self.use_variance:
            loss_var = self.variance_loss(q_pre) + self.variance_loss(target_pre.mean(dim=1))
        else:
            loss_var = loss_pairwise_kl.new_zeros(())
        return {
            "loss": loss_pairwise_kl + self.lambda_var * loss_var,
            "loss_pairwise_kl": loss_pairwise_kl,
            "loss_var": loss_var,
        }


class Exp5TextAuxQure(PositiveQure):
    """Exp5: Exp1 retrieval objective plus a text-only contrastive objective.

    Exp1--3 are intentionally untouched: this class owns the extra text-only
    Q-Former pass and its auxiliary loss.  It still reuses the exact same
    ``base.Qformer``, ``base.query_tokens``, ``text_proj``, ``vision_proj``,
    and target branch as the main Exp1 path.
    """

    def __init__(self, *args, lambda_text: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment != "exp1":
            raise ValueError("Exp5TextAuxQure must use the Exp1 query architecture.")
        if lambda_text < 0:
            raise ValueError("lambda_text must be non-negative.")
        self.lambda_text = float(lambda_text)

    def text_only_features(self, captions, return_pre_norm=False):
        """Encode raw text plus the shared query tokens without image memory."""
        text = self._tokenize(captions)
        batch_size = text.input_ids.shape[0]
        queries = self._queries(batch_size)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.long, device=self.device)
        mask = torch.cat([query_mask, text.attention_mask], dim=1)
        hidden = self._manual_qformer(
            queries, input_ids=text.input_ids, attention_mask=mask,
        )[:, :self.num_query_tokens, :]
        tokens_pre = self.base.text_proj(hidden)
        pre = tokens_pre.mean(dim=1)
        feature = F.normalize(pre, dim=-1)
        return (feature, pre) if return_pre_norm else feature

    def _inbatch_contrastive_loss(self, scores, target_names):
        logits = scores / self.base.temp
        positive_mask = self._positive_mask(target_names, logits.device)
        positive_logit = torch.logsumexp(
            logits.masked_fill(~positive_mask, -torch.inf), dim=1
        )
        return -(positive_logit - torch.logsumexp(logits, dim=1)).mean()

    def forward(self, reference_images, target_images, captions, target_names):
        q_reference_text, q_reference_text_pre = self.query_features(
            reference_images, captions, return_pre_norm=True
        )
        q_text, q_text_pre = self.text_only_features(captions, return_pre_norm=True)
        target, target_pre = self.target_features(target_images, return_pre_norm=True)

        reference_text_scores = torch.einsum(
            "id,jkd->ijk", q_reference_text, target
        ).max(dim=-1).values
        text_scores = torch.einsum("id,jkd->ijk", q_text, target).max(dim=-1).values
        loss_contrastive_reference_text = self._inbatch_contrastive_loss(
            reference_text_scores, target_names
        )
        loss_contrastive_text = self._inbatch_contrastive_loss(text_scores, target_names)
        if self.use_variance:
            loss_var = (
                self.variance_loss(q_reference_text_pre)
                + self.variance_loss(q_text_pre)
                + self.variance_loss(target_pre.mean(dim=1))
            )
        else:
            loss_var = loss_contrastive_reference_text.new_zeros(())
        return {
            "loss": (
                loss_contrastive_reference_text
                + self.lambda_text * loss_contrastive_text
                + self.lambda_var * loss_var
            ),
            "loss_contrastive_reference_text": loss_contrastive_reference_text,
            "loss_contrastive_text": loss_contrastive_text,
            "loss_var": loss_var,
        }


class VICRegQure(PositiveQure):
    """Exp6: Exp1 encoder with a train-only VICReg projector and FIFO queue.

    Retrieval remains exactly Exp1's normalised ``q`` / ``T`` max-token score.
    Raw paired features feed a shared MLP projector, and VICReg is applied only
    to its outputs.  A detached FIFO queue supplies stable cross-batch
    statistics for variance/covariance without retaining Q-Former graphs.
    """

    def __init__(self, *args, lambda_align: float = 25.0,
                 lambda_v: float = 25.0, lambda_c: float = 1.0,
                 projector_hidden_dim: int = 512, projector_output_dim: int = 128,
                 queue_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment != "exp1":
            raise ValueError("VICRegQure must use the Exp1 query architecture.")
        if min(lambda_align, lambda_v, lambda_c) < 0:
            raise ValueError("VICReg coefficients must be non-negative.")
        if projector_hidden_dim < 1 or projector_output_dim < 1 or queue_size < 0:
            raise ValueError("VICReg projector dimensions and queue_size must be positive.")
        feature_dim = self.base.text_proj.out_features
        if feature_dim != self.base.vision_proj.out_features:
            raise ValueError("VICReg requires query and target projection dimensions to match.")
        self.vicreg_projector = nn.Sequential(
            nn.Linear(feature_dim, projector_hidden_dim, bias=False),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_output_dim, bias=False),
        )
        self.register_buffer("vicreg_queue_q", torch.empty(0, projector_output_dim))
        self.register_buffer("vicreg_queue_p", torch.empty(0, projector_output_dim))
        self._pending_queue_q = None
        self._pending_queue_p = None
        self.lambda_align = float(lambda_align)
        self.lambda_v = float(lambda_v)
        self.lambda_c = float(lambda_c)
        self.vicreg_queue_size = int(queue_size)

    def _vicreg_variance_loss(self, z_q, z_p):
        std_q = torch.sqrt(z_q.var(dim=0, unbiased=False) + self.var_epsilon)
        std_p = torch.sqrt(z_p.var(dim=0, unbiased=False) + self.var_epsilon)
        return 0.5 * (
            F.relu(self.var_gamma - std_q).mean()
            + F.relu(self.var_gamma - std_p).mean()
        )

    @staticmethod
    def _covariance_loss(features):
        batch_size, feature_dim = features.shape
        if batch_size < 2:
            raise ValueError("VICReg covariance loss needs a batch of at least two samples.")
        centered = features - features.mean(dim=0)
        covariance = centered.T @ centered / (batch_size - 1)
        off_diagonal = covariance.flatten()[:-1].view(feature_dim - 1, feature_dim + 1)[:, 1:].flatten()
        return off_diagonal.square().sum() / feature_dim

    def _features_for_statistics(self, current, queue):
        """Concatenate only detached history; gradients flow through current z."""
        current = current.float()
        if queue.numel() == 0:
            return current
        return torch.cat([queue, current], dim=0)

    @torch.no_grad()
    def update_vicreg_queue(self):
        """FIFO update called after optimizer.step(), never retaining a graph."""
        if self._pending_queue_q is None:
            return
        if self.vicreg_queue_size == 0:
            self._pending_queue_q = None
            self._pending_queue_p = None
            return
        self.vicreg_queue_q = torch.cat(
            [self.vicreg_queue_q, self._pending_queue_q.float()], dim=0
        )[-self.vicreg_queue_size:]
        self.vicreg_queue_p = torch.cat(
            [self.vicreg_queue_p, self._pending_queue_p.float()], dim=0
        )[-self.vicreg_queue_size:]
        self._pending_queue_q = None
        self._pending_queue_p = None

    def forward(self, reference_images, target_images, captions, target_names):
        del target_names  # Exp6 is paired-only and does not form a negative set.
        q, h_q = self.query_features(reference_images, captions, return_pre_norm=True)
        target, target_pre = self.target_features(target_images, return_pre_norm=True)
        # Select h_p with Exp1's exact inference-time max_k(q dot t_k) rule.
        _, h_p = self._best_target(q, target, target_pre)
        z_q = self.vicreg_projector(h_q)
        z_p = self.vicreg_projector(h_p)
        loss_align = F.mse_loss(z_q, z_p)
        stats_q = self._features_for_statistics(z_q, self.vicreg_queue_q)
        stats_p = self._features_for_statistics(z_p, self.vicreg_queue_p)
        loss_var = self._vicreg_variance_loss(stats_q, stats_p)
        loss_cov = self._covariance_loss(stats_q) + self._covariance_loss(stats_p)
        self._pending_queue_q = z_q.detach()
        self._pending_queue_p = z_p.detach()
        loss = (
            self.lambda_align * loss_align
            + self.lambda_v * loss_var
            + self.lambda_c * loss_cov
        )
        return {
            "loss": loss,
            "loss_align": loss_align,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
        }
