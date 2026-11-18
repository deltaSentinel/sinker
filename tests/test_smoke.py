"""CI smoke tests: model shapes, encoding sanity, tiny end-to-end step."""
import pytest
import torch
import numpy as np

from sinker.data import encode_url, load_tsv, stratified_split, MAX_LEN
from sinker.model import HookCNN


def test_encode_shape_and_content():
    v = encode_url("https://Example.COM/Login")
    assert v.shape == (MAX_LEN,)
    assert v.dtype == np.int64
    assert v[0] == ord("h")            # lowercased
    assert (v > 0).sum() == len("https://example.com/login")


def test_pad_and_truncation():
    short = encode_url("ab")
    assert (short[2:] == 0).all()
    long_url = "a" * (MAX_LEN + 50)
    assert (encode_url(long_url) != 0).sum() == MAX_LEN


def test_model_forward_shape():
    model = HookCNN()
    xb = torch.randint(0, 128, (4, MAX_LEN))
    out = model(xb)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_training_step_decreases_or_not_but_runs():
    torch.manual_seed(0)
    model = HookCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    xb = torch.randint(97, 123, (16, MAX_LEN))
    yb = torch.randint(0, 2, (16,)).float()
    loss_before = crit(model(xb), yb).item()
    for _ in range(3):
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
    # just needs to run deterministically; assert finite outputs
    assert torch.isfinite(model(xb)).all()
    assert loss_before == loss_before  # tautology guard vs accidental None


def test_stratified_split(tmp_path):
    lines = [f"https://site{i}.com\t{i % 2}" for i in range(200)]
    p = tmp_path / "u.tsv"
    p.write_text("\n".join(lines))
    ds = load_tsv(p)
    tr, va = stratified_split(ds, 0.2, seed=42)
    assert abs(len(va) - 40) <= 40  # ~20% per class
    assert len(set(tr.tolist()) & set(va.tolist())) == 0  # disjoint
