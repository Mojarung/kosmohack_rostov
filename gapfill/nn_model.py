"""Нейросеть по ежедневной сетке сезона: дилатированные свёртки + трансформер, обучение с маскированием.

Запуск: uv run python -m gapfill.nn_model --epochs 300 --out nn_v1
Валидация — та же маска, что у LightGBM (val_seed 777), поэтому предсказания можно усреднять.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from gapfill.config import ARTIFACTS_DIR, RANDOM_SEED, SENSOR_CODE, TARGET
from gapfill.data import gap_score, load_all, make_mask, polygon_kinds, rmse, testlike_rmse
from gapfill.nn_data import N_CHANNELS, EpochInputs, SeasonTensors, assemble, build_tensors, loo_residual_array

DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8)
KERNEL = 5


class ResBlock(nn.Module):
    """Свёрточный блок с дилатацией и остаточной связью."""

    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        pad = (KERNEL - 1) // 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, KERNEL, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(hidden, hidden, 1)
        self.norm = nn.GroupNorm(8, hidden)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.act(self.conv(self.norm(x))))
        return x + self.drop(h)


class SeasonNet(nn.Module):
    """Вход (B, L, C) → предсказание primary_ndvi на каждый день (B, L)."""

    def __init__(self, n_channels: int = N_CHANNELS, hidden: int = 128, dropout: float = 0.1, n_tf: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(n_channels, hidden, 1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d, dropout) for d in DILATIONS])
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, n_tf)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(x.transpose(1, 2))).transpose(1, 2)
        h = self.tf(h)
        return self.head(h).squeeze(-1)


def query_days_from_obs(t: SeasonTensors, query_obs: np.ndarray) -> np.ndarray:
    """Булева матрица дней-запросов (n, 213) по маске точек obs."""
    q = np.zeros(t.known.shape, bool)
    ok = t.idx >= 0
    q[ok] = query_obs[t.idx[ok]]
    return q


def to_batches(data: dict, device: torch.device, batch: int, shuffle: bool, rng: np.random.Generator):
    """Итератор мини-батчей по образцам, у которых есть хотя бы один запрос."""
    keep = np.flatnonzero(data["query"].any(1))
    if shuffle:
        keep = rng.permutation(keep)
    for i in range(0, len(keep), batch):
        sel = keep[i:i + batch]
        yield (torch.from_numpy(data["x"][sel]).to(device), torch.from_numpy(data["y"][sel]).to(device),
               torch.from_numpy(data["query"][sel]).to(device), sel)


def predict(model: nn.Module, data: dict, device: torch.device) -> np.ndarray:
    """Предсказания (n, 213) для всех образцов."""
    model.eval()
    out = np.zeros(data["y"].shape, np.float32)
    with torch.no_grad():
        for i in range(0, len(data["x"]), 256):
            x = torch.from_numpy(data["x"][i:i + 256]).to(device)
            out[i:i + 256] = model(x).float().cpu().numpy()
    return out


def val_metrics(pred_days: np.ndarray, t: SeasonTensors, meta: pd.DataFrame, query_obs: np.ndarray) -> tuple[dict, np.ndarray]:
    """Собирает предсказания по строкам obs (маска запросов) и считает метрики."""
    pred = np.full(len(query_obs), np.nan, np.float32)
    ok = t.idx >= 0
    pred[t.idx[ok]] = pred_days[ok]
    sel = query_obs
    y, p = meta[TARGET].to_numpy()[sel], pred[sel]
    err = p - y
    m = {"rmse": rmse(y, p), "rmse_testlike": testlike_rmse(err, meta["poly_kind"].to_numpy()[sel], meta["is_2025"].to_numpy()[sel])}
    m["gap_score_testlike"] = gap_score(m["rmse_testlike"])
    for name, code in SENSOR_CODE.items():
        s = meta["sensor"].to_numpy()[sel] == code
        m[f"rmse_{name}"] = rmse(y[s], p[s])
    return m, pred


def train(args: argparse.Namespace) -> dict:
    """Цикл обучения с новой глобальной маской каждую эпоху и отбором лучшей эпохи по валидации."""
    torch.manual_seed(RANDOM_SEED + args.seed)
    rng = np.random.default_rng(RANDOM_SEED + args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, grid, _ = load_all()
    meta = obs.assign(poly_kind=obs["pid"].map(polygon_kinds(obs)).astype(str), is_2025=obs["year"].eq(2025))
    t = build_tensors(obs, grid)
    val_mask = make_mask(obs, seed=args.val_seed)
    context = ~val_mask
    ep = EpochInputs(t, obs, loo_residual_array(obs, context))
    val_data = assemble(t, ep, context, query_days_from_obs(t, val_mask))
    model = SeasonNet(hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * 25, pct_start=0.1)
    best, best_state, history = {"rmse_testlike": np.inf}, None, []
    ctx_idx = np.flatnonzero(context)
    for epoch in range(args.epochs):
        t0 = time.time()
        m = np.zeros(len(obs), bool)
        m[rng.choice(ctx_idx, size=int(0.15 * len(ctx_idx)), replace=False)] = True
        data = assemble(t, ep, context & ~m, query_days_from_obs(t, m))
        model.train()
        losses = []
        for x, y, q, _ in to_batches(data, device, args.batch, True, rng):
            pred = model(x)
            err = (pred - y.clamp(args.clip[0], args.clip[1]))[q]
            loss = (err ** 2).mean() if args.loss == "mse" else nn.functional.huber_loss(pred[q], y[q], delta=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            losses.append(loss.item())
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics, _ = val_metrics(predict(model, val_data, device), t, meta, val_mask)
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
            print(f"эпоха {epoch + 1}: loss {np.mean(losses):.5f} val {metrics['rmse']:.4f} "
                  f"testlike {metrics['rmse_testlike']:.4f} ({time.time() - t0:.1f} с)", flush=True)
            if metrics["rmse_testlike"] < best["rmse_testlike"]:
                best = metrics | {"epoch": epoch + 1}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, pred = val_metrics(predict(model, val_data, device), t, meta, val_mask)
    out_dir = ARTIFACTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "nn_val.pt")
    meta.loc[val_mask, ["pid", "date", "year", "sensor", TARGET, "poly_kind", "is_2025"]].assign(
        pred=pred[val_mask]).reset_index(drop=True).to_parquet(out_dir / "val_pred.parquet")
    (out_dir / "result.json").write_text(json.dumps({"best": best, "history": history, "args": vars(args)},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Нейросеть по сезонной сетке для восстановления primary_ndvi")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--clip", type=float, nargs=2, default=(-0.2, 1.1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=777)
    parser.add_argument("--out", type=str, default="nn_v1")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))
