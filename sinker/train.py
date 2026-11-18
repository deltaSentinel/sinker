"""
Stage 1 training entrypoint for the `hook` (URL) signal.

    uv run python -m sinker.train --data data/raw/urls.tsv

Outputs (under train/out/):
  hook.pt         model checkpoint (state_dict + config)
  metrics.json    val metrics for this run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import set_seed
from .data import load_tsv, make_tensor_ds, stratified_split
from .model import HookCNN

OUT_DIR = Path("train") / "out"
BEST_TSV_EPOCHS = 30       # bump freely; early stopping guards runtime
EARLY_STOP_PATIENCE = 5


@torch.no_grad()
def evaluate(model, loader, device, thresh: float = 0.5):
    model.eval()
    probs_all, ys_all = [], []
    loss_total, n = 0.0, 0
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_total += crit(logits, yb).item()
        n += len(yb)
        probs_all.append(torch.sigmoid(logits).cpu())
        ys_all.append(yb.cpu())
    probs = torch.cat(probs_all)
    ys = torch.cat(ys_all)
    preds = (probs >= thresh).float()
    tp = ((preds == 1) & (ys == 1)).sum().item()
    fp = ((preds == 1) & (ys == 0)).sum().item()
    fn = ((preds == 0) & (ys == 1)).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "loss": loss_total / n,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": _auc(ys.numpy(), probs.numpy()),
    }


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (handles ties); no sklearn dependency needed yet."""
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    pos = y == 1
    n_pos = pos.sum()
    n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sinker.train")
    ap.add_argument("--data", required=True, help="TSV: url<TAB>label")
    ap.add_argument("--epochs", type=int, default=BEST_TSV_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_tsv(args.data)
    tr_idx, va_idx = stratified_split(ds, args.val_frac, args.seed)
    print(f"[split] train={len(tr_idx)} val={len(va_idx)}")
    train_dl = DataLoader(make_tensor_ds(ds, tr_idx), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(make_tensor_ds(ds, va_idx), batch_size=args.batch_size)

    model = HookCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        metrics = evaluate(model, val_dl, device)
        print(
            f"[epoch {epoch:02d}] train_loss={running / nb:.4f} "
            f"val_loss={metrics['loss']:.4f} val_f1={metrics['f1']:.4f} "
            f"val_auc={metrics['auc']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1, best_state, patience = metrics["f1"], model.state_dict(), 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[early-stop] no val F1 gain for {patience} epochs")
                break

    assert best_state is not None
    torch.save({"state_dict": best_state, "config": {}}, OUT_DIR / "hook.pt")
    final = evaluate(model, val_dl, device)  # NB: reports last epoch; best kept in ckpt
    record = {
        "run_ts": dt.datetime.now().isoformat(timespec="seconds"),
        "data": str(args.data),
        "seed": args.seed,
        "epochs_run": epoch,
        "best_val_f1": round(best_f1, 4),
        "last_epoch_metrics": {k: round(v, 4) for k, v in final.items()},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"[done] best val F1={best_f1:.4f}; wrote {OUT_DIR/'hook.pt'}")


if __name__ == "__main__":
    main()

