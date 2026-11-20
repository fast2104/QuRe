"""A real GPU forward/backward through Exp1, with the user's dataset."""
import json
import sys
from pathlib import Path
import torch
from model import PositiveQure, load_qure
from pfm_variants import PromptFlowQure
from train_pfm import seed_everything, load_local_data_module

config = json.loads(Path(sys.argv[1]).read_text())
qure_root, data_root = Path(sys.argv[2]), Path(sys.argv[3])
seed_everything(config['seed'])
base, _, processors = load_qure(config['backbone'], torch.device('cuda'), qure_root)
args = (base, 'exp1', config['max_txt_len'], False, 0, config['var_gamma'], config['var_epsilon'])
baseline = PositiveQure(*args).cuda().eval()
variant = config['pfm']['variant']
model = baseline if variant == 'baseline' else PromptFlowQure(*args, pfm_config=config['pfm']).cuda().eval()
from data.utils import targetpad_transform
dataset = load_local_data_module().FashionIQPositiveTrain(
    data_root, targetpad_transform(config['target_ratio'], config['image_size']))
samples = [dataset[i] for i in range(2)]
references = torch.stack([s[0] for s in samples]).cuda()
targets = torch.stack([s[1] for s in samples]).cuda()
captions = [processors['eval'](s[2]) for s in samples]
names = [s[3] for s in samples]
with torch.no_grad():
    expected = baseline.target_features(targets)
    torch.testing.assert_close(model.target_features(targets), expected)
    q_before = model.query_features(references, captions)
    token_backup = base.query_tokens.detach().clone()
    base.query_tokens.add_(torch.randn_like(base.query_tokens))
    try:
        q_after = model.query_features(references, captions)
        torch.testing.assert_close(q_before, q_after)
    finally:
        base.query_tokens.copy_(token_backup)
# Reference features must have no autograd dependency on original tokens.
reference_features = model.query_features(references, captions)
probe = (reference_features * torch.randn_like(reference_features)).sum()
gradient = torch.autograd.grad(probe, base.query_tokens, allow_unused=True)[0]
assert gradient is None, 'Reference still depends on pretrained query tokens'
del reference_features, probe, gradient
model.zero_grad(set_to_none=True)
model.train()
with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
    losses = model(references, targets, captions, names)
assert torch.isfinite(losses['loss'])
losses['loss'].backward()
if variant != 'baseline':
    gradients = [p.grad for p in model.ref_flow.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
print('PASS actual Exp1 GPU forward/backward:', variant)
