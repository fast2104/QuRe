"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import numpy as np
import logging
from tqdm import tqdm
import PIL
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from data.utils import targetpad_transform

preprocess = targetpad_transform(1.25, 224)

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2_qure")
class Blip2QuRe(Blip2Base):
    """
    QuRe with independent Q-Formers for the reference image, relative text,
    and target image. Reference-image and text features are fused into the
    composed query representation.

    Supported model types:
        - pretrained: pretrained model
        - coco: fine-tuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_qure", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            print("freeze vision encoder")
        else:
            print("train vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.ref_Qformer, self.ref_query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.text_Qformer, self.text_query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )

        self._prepare_qformer(self.Qformer)
        self._prepare_qformer(self.ref_Qformer)
        self._prepare_qformer(self.text_Qformer)

        hidden_size = self.Qformer.config.hidden_size
        self.vision_proj = nn.Linear(hidden_size, embed_dim)
        self.ref_proj = nn.Linear(hidden_size, embed_dim)
        self.text_proj = nn.Linear(hidden_size, embed_dim)
        self.query_fusion = nn.Linear(embed_dim * 2, embed_dim)
        self._initialize_separate_qformers()

        self.itm_head = nn.Linear(hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def _prepare_qformer(self, qformer):
        qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = qformer.state_dict()
        for name, param in qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

    @torch.no_grad()
    def _initialize_separate_qformers(self):
        self.ref_Qformer.load_state_dict(self.Qformer.state_dict())
        self.text_Qformer.load_state_dict(self.Qformer.state_dict())
        self.ref_query_tokens.copy_(self.query_tokens)
        self.text_query_tokens.copy_(self.query_tokens)
        self.ref_proj.load_state_dict(self.vision_proj.state_dict())

        embed_dim = self.query_fusion.out_features
        self.query_fusion.weight.zero_()
        self.query_fusion.weight[:, :embed_dim].copy_(
            torch.eye(embed_dim, device=self.query_fusion.weight.device) * 0.5
        )
        self.query_fusion.weight[:, embed_dim:].copy_(
            torch.eye(embed_dim, device=self.query_fusion.weight.device) * 0.5
        )
        self.query_fusion.bias.zero_()

    def load_from_pretrained(self, url_or_filename, **kwargs):
        message = super().load_from_pretrained(url_or_filename, **kwargs)
        self._initialize_separate_qformers()
        return message

    def _encode_image_queries(self, images, qformer, query_tokens):
        with self.maybe_autocast():
            image_embedding = self.ln_vision(self.visual_encoder(images))
        image_embedding = image_embedding.float()
        image_atts = torch.ones(
            image_embedding.size()[:-1], dtype=torch.long, device=images.device
        )
        expanded_query_tokens = query_tokens.expand(image_embedding.shape[0], -1, -1)
        query_output = qformer.bert(
            query_embeds=expanded_query_tokens,
            encoder_hidden_states=image_embedding,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state

    def _encode_query_features(self, ref_images, sentences, device, normalize=True):
        text_tokens = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        ref_query_embeds = self._encode_image_queries(
            ref_images, self.ref_Qformer, self.ref_query_tokens
        )
        ref_features = self.ref_proj(ref_query_embeds).mean(dim=1)

        text_output = self.text_Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_mask = text_tokens.attention_mask.unsqueeze(-1).to(
            text_output.last_hidden_state.dtype
        )
        text_hidden = (text_output.last_hidden_state * text_mask).sum(dim=1)
        text_hidden = text_hidden / text_mask.sum(dim=1).clamp_min(1)
        text_features = self.text_proj(text_hidden)

        query_features = self.query_fusion(
            torch.cat([ref_features, text_features], dim=-1)
        )
        if normalize:
            query_features = F.normalize(query_features, p=2, dim=-1)
        return query_features

    def _encode_target_features(self, images):
        target_query_embeds = self._encode_image_queries(
            images, self.Qformer, self.query_tokens
        )
        target_features = self.vision_proj(target_query_embeds)
        return F.normalize(target_features, p=2, dim=-1)

    def forward(self, ref_images, tar_images, sentences, negative_images, use_temp):
        device = ref_images.device
        projected_query = self._encode_query_features(
            ref_images, sentences, device, normalize=True
        )
        tar_projected_image = self._encode_target_features(tar_images)
        neg_projected_image = self._encode_target_features(negative_images)

        scores_1 = torch.bmm(tar_projected_image, projected_query.unsqueeze(-1))
        score_1, _ = torch.max(scores_1, dim=1)

        scores_2 = torch.bmm(neg_projected_image, projected_query.unsqueeze(-1))
        score_2, _ = torch.max(scores_2, dim=1)

        if use_temp:
            score_1 = score_1 / self.temp
            score_2 = score_2 / self.temp

        scores = torch.stack([score_1, score_2], dim=1).squeeze(-1)
        return scores

    @torch.no_grad()
    def extract_target_features(self, dataloader, use_temp, device):
        index_features = torch.empty(
            (0, self.query_tokens.shape[1], self.vision_proj.out_features),
            device=device,
        )
        index_names = []

        for images, names in tqdm(dataloader, desc="Extract Sample Features"):
            images = images.to(device, non_blocking=True)
            batch_features = self._encode_target_features(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
        return index_features.cpu(), index_names

    @torch.no_grad()
    def extract_query_features_fiq(self, dataloader, use_temp, txt_processors, device):
        predicted_features = torch.empty(
            (0, self.vision_proj.out_features), device=device
        )
        target_names = []

        for ref_images, sentences, target_name in tqdm(
            dataloader, desc="Extract Query Features"
        ):
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            ref_images = ref_images.to(device, non_blocking=True)
            query_features = self._encode_query_features(
                ref_images, sentences, device, normalize=True
            )
            predicted_features = torch.vstack((predicted_features, query_features))
            target_names.extend(target_name)
        return predicted_features.cpu(), target_names

    @torch.no_grad()
    def extract_query_features_cirr(self, dataloader, use_temp, txt_processors, device):
        predicted_features = torch.empty(
            (0, self.vision_proj.out_features), device=device
        )
        group_members = []
        reference_names = []
        pairs_ids = []

        for ref_names, ref_images, sentences, pair_id, batch_group_members in tqdm(
            dataloader, desc="Extract Query Features"
        ):
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            batch_group_members = np.array(batch_group_members).tolist()
            ref_images = ref_images.to(device, non_blocking=True)
            query_features = self._encode_query_features(
                ref_images, sentences, device, normalize=use_temp
            )
            predicted_features = torch.vstack((predicted_features, query_features))
            reference_names.extend(ref_names)
            group_members.extend(batch_group_members)
            pairs_ids.extend(pair_id)

        return predicted_features.cpu(), reference_names, group_members, pairs_ids

    @torch.no_grad()
    def extract_query_features_circo(self, dataloader, use_temp, txt_processors, device):
        predicted_features = torch.empty(
            (0, self.vision_proj.out_features), device=device
        )
        query_ids_list = []

        for batch in tqdm(dataloader, desc="Extract Query Features"):
            ref_images = batch['reference_img']
            relative_captions = batch['relative_caption']
            query_ids = batch['query_id']

            sentences = [f"a photo of $ that {caption}" for caption in relative_captions]
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            ref_images = ref_images.to(device, non_blocking=True)
            query_features = self._encode_query_features(
                ref_images, sentences, device, normalize=use_temp
            )
            predicted_features = torch.vstack((predicted_features, query_features))
            query_ids_list.extend(query_ids)
        return predicted_features.cpu(), query_ids_list

    @torch.no_grad()
    def extract_query_features_cirr_val(self, dataloader, use_temp, txt_processors, device):
        predicted_features = torch.empty(
            (0, self.vision_proj.out_features), device=device
        )
        group_members = []
        reference_names = []
        pairs_ids = []
        target_names = []

        for ref_names, ref_images, targ_names, sentences, pair_id, batch_group_members in tqdm(
            dataloader, desc="Extract Query Features"
        ):
            sentences = [txt_processors["eval"](caption) for caption in sentences]
            batch_group_members = np.array(batch_group_members).tolist()
            ref_images = ref_images.to(device, non_blocking=True)
            query_features = self._encode_query_features(
                ref_images, sentences, device, normalize=use_temp
            )
            predicted_features = torch.vstack((predicted_features, query_features))
            reference_names.extend(ref_names)
            target_names.extend(targ_names)
            group_members.extend(batch_group_members)

        return predicted_features.cpu(), reference_names, target_names, group_members, pairs_ids

    @torch.no_grad()
    def score(self, query_features, target_features):
        scoress = torch.einsum('ik,jlk->ijl', query_features, target_features)
        scores, _ = scoress.max(-1)
        # scores = scores / self.temp.cpu()
        return scores

    @torch.no_grad()
    def extract_target_image_features_fiq(self, dataloader, device):
        index_features = torch.empty(
            (0, self.query_tokens.shape[1], self.vision_proj.out_features),
            device=device,
        )
        target_name_list = []
        for ref_images, sentences, target_names in dataloader:  # Load data
            target_images = []
            for target_name in target_names:
                target_image_path = '-' + '/images' + f"/{target_name}.png"
                target_image = preprocess(PIL.Image.open(target_image_path)).unsqueeze(0)
                target_images.append(target_image)
            target_image = torch.cat(target_images, dim=0)
            target_image = target_image.to(device, non_blocking=True)

            batch_features = self._encode_target_features(target_image)
            index_features = torch.vstack((index_features, batch_features))
            target_name_list.extend(target_names)

        return index_features.cpu(), target_name_list

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model




