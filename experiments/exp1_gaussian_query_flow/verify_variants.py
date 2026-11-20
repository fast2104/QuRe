"""CPU checks of query sampling, joint reference encoding and the retrieval score."""
from types import SimpleNamespace

import torch
from torch import nn
from pfm_variants import GaussianQueryFlow, PromptFlowQure, sample_spread


def check():
    torch.manual_seed(1000)
    flow = GaussianQueryFlow(32, 768, 0.02, samples=10, steps=4, context_width=64, eval_seed=101).eval()
    context = torch.randn(2, 768)
    with torch.no_grad():
        start, banks = flow(context)
        assert banks.shape == (2, 10, 32, 768)
        torch.testing.assert_close(start[0], 0.02 * flow.eval_eps)
        torch.testing.assert_close(banks[:1], flow(context[:1])[1], atol=2e-6, rtol=1e-4)
        assert sample_spread(banks) > 0 and not torch.equal(banks, start)
        assert not torch.equal(banks[0], banks[1])
        print(f"Eval on synthetic captions: Gaussian spread={sample_spread(start):.6f}, flow spread={sample_spread(banks):.6f}")
        flow.train()
        assert not torch.equal(flow(context)[1], flow(context)[1])
        control = GaussianQueryFlow(32, 768, 0.02, samples=10, steps=0, context_width=64, eval_seed=101).eval()
        a, b = control(context)
        torch.testing.assert_close(a, b)

    class ToyJoint(PromptFlowQure):
        def _tokenize(self, captions):
            return SimpleNamespace(input_ids=captions, attention_mask=torch.ones_like(captions))

        def _vision(self, images):
            return images

        def _qformer(self, *, query_embeds, encoder_hidden_states, input_ids=None, **kwargs):
            fused = query_embeds + encoder_hidden_states.mean(1, keepdim=True)
            if input_ids is not None:
                fused = fused + self.base.Qformer.bert.embeddings.word_embeddings(input_ids).mean(1, keepdim=True)
            return fused.tanh()

    base = nn.Module()
    base.query_tokens = nn.Parameter(torch.randn(1, 3, 8))
    base.Qformer = nn.Module()
    base.Qformer.config = SimpleNamespace(initializer_range=0.02, hidden_size=8, encoder_width=8)
    base.Qformer.bert = nn.Module()
    base.Qformer.bert.embeddings = nn.Module()
    base.Qformer.bert.embeddings.word_embeddings = nn.Embedding(20, 8)
    base.text_proj, base.vision_proj = nn.Linear(8, 5), nn.Linear(8, 5)
    base.temp = nn.Parameter(torch.tensor(0.07))
    model = ToyJoint(base, "exp1", 4, False, 0, 1, 1e-4,
                     pfm_config=dict(samples=10, num_flows=4, hidden=6, eval_seed=101))
    refs, target_images = torch.randn(3, 5, 8), torch.randn(3, 5, 8)
    ids = torch.randint(0, 20, (3, 4))
    model.eval()
    q = model.query_features(refs, ids)
    t = model.target_features(target_images)
    assert q.shape == (3, 10, 5) and t.shape == (3, 3, 5)
    assert torch.autograd.grad(q.sum(), base.query_tokens, allow_unused=True)[0] is None
    expected = torch.stack([torch.einsum("bd,jkd->bjk", item, t).max(-1).values for item in q.unbind(1)]).mean(0)
    torch.testing.assert_close(model.score(q, t, 2), expected)
    model.train()
    loss = model(refs, target_images, ids, ["a", "a", "b"])["loss"]
    loss.backward()
    for parameter in (model.ref_flow.gaussian.weight, model.ref_flow.planar.weight,
                      model.ref_flow.slot_mean, model.ref_flow.slot_log_std, base.query_tokens):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    print("PASS Gaussian control, fixed eval, stochastic train, score aggregation and gradients.")


if __name__ == "__main__":
    check()
