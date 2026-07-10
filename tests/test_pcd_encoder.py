import torch

from src.pcd.encoder import TopKEncoder


def _tiny(d=4, m=8, k=2, **kw):
    return TopKEncoder(d_model=d, n_concepts=m, k=k, **kw)


def test_paper_initialization():
    enc = _tiny(d=4, m=8)
    row_norms = enc.W_enc.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones(8), atol=1e-5)
    assert torch.allclose(enc.W_emb, enc.W_enc.t(), atol=1e-6)
    assert torch.count_nonzero(enc.b_enc) == 0
    assert torch.count_nonzero(enc.tokens_since_active) == 0


def test_encode_shapes_and_leading_dims():
    enc = _tiny(d=4, m=8, k=3)
    a = torch.randn(2, 5, 4)
    out = enc.encode(a)
    assert out.soft_tokens.shape == (2, 5, 4)
    assert out.topk_indices.shape == (2, 5, 3)
    assert out.topk_values.shape == (2, 5, 3)
    assert out.pre_acts.shape == (10, 8)


def test_topk_selects_largest_preacts():
    enc = _tiny(d=4, m=4, k=2)
    with torch.no_grad():
        enc.W_enc.copy_(torch.eye(4))
        enc.b_enc.zero_()
        enc.W_emb.copy_(torch.eye(4))
    a = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
    out = enc.encode(a)
    assert set(out.topk_indices[0].tolist()) == {0, 2}
    assert torch.allclose(out.soft_tokens[0], torch.tensor([3.0, 0.0, 2.0, 0.0]))


def test_aux_loss_zero_when_nothing_dead():
    enc = _tiny(d=4, m=8, dead_window_tokens=1000)
    a = torch.randn(6, 4)
    out = enc.encode(a)
    assert float(enc.aux_loss(out.pre_acts)) == 0.0


def test_aux_loss_fires_on_dead_concepts():
    enc = _tiny(d=4, m=8, k=2, aux_k=3, aux_coef=1.0, dead_window_tokens=10)
    with torch.no_grad():
        enc.tokens_since_active.fill_(100)
    a = torch.randn(4, 4)
    out = enc.encode(a)
    loss = enc.aux_loss(out.pre_acts)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert enc.W_enc.grad is not None and enc.W_enc.grad.abs().sum() > 0
    assert enc.b_enc.grad is None or float(enc.b_enc.grad.abs().sum()) == 0.0


def test_activity_update_and_dead_detection():
    enc = _tiny(d=4, m=8, dead_window_tokens=50)
    active = torch.zeros(8, dtype=torch.bool)
    active[[1, 3]] = True
    enc.update_activity(active, n_tokens=100)
    assert enc.tokens_since_active[1] == 0 and enc.tokens_since_active[3] == 0
    assert enc.tokens_since_active[0] == 100
    stats = enc.activity_stats()
    assert stats["n_alive"] == 2 and stats["n_dead"] == 6


def test_batch_active_mask():
    enc = _tiny(d=4, m=8, k=2)
    topk = torch.tensor([[0, 5], [5, 7]])
    mask = enc.batch_active_mask(topk)
    assert mask.sum() == 3 and mask[0] and mask[5] and mask[7]
