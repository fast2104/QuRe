"""Generate Exp1 reference query embeddings with the 3Qformer+pfm module."""
import torch

from model import PositiveQure
from prompt_flow import PlanarPromptFlow


class ReferencePFMQuRe(PositiveQure):
    def __init__(self, *args, pfm_config, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment != "exp1" or pfm_config["variant"] != "reference_3qpfm":
            raise ValueError("This runner requires Exp1/reference_3qpfm.")
        self.ref_flow = PlanarPromptFlow(
            condition_dim=self.encoder_width + self.hidden_size,
            hidden_dim=pfm_config["hidden_dim"],
            num_prompts=self.num_query_tokens * self.hidden_size,
            num_flows=pfm_config["num_flows"],
        )
        # The original decoder has zero output weights. Give its 32 slots
        # distinct initial vectors and recover Exp1 exactly at initialization.
        with torch.no_grad():
            self.ref_flow.prompt_decoder[-1].bias.copy_(self.base.query_tokens.detach().reshape(-1))

    def _query_hidden(self, reference_images, captions):
        text = self._tokenize(captions)
        visual = self._vision(reference_images)
        words = self.base.Qformer.bert.embeddings.word_embeddings(text.input_ids)
        mask = text.attention_mask.unsqueeze(-1)
        caption_mean = (words * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        condition = torch.cat((visual.mean(dim=1), caption_mean), dim=-1)
        flat_queries, _ = self.ref_flow(condition)
        queries = flat_queries.reshape(visual.shape[0], self.num_query_tokens, self.hidden_size)
        query_mask = text.attention_mask.new_ones(queries.shape[:2])
        hidden = self._qformer(query_embeds=queries, input_ids=text.input_ids,
                               attention_mask=torch.cat((query_mask, text.attention_mask), dim=1),
                               encoder_hidden_states=visual, encoder_attention_mask=self._memory_mask(visual))
        return hidden[:, :self.num_query_tokens]
