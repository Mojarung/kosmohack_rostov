"""Обучение сезонной сети на Apple MPS/CUDA с независимой валидацией и маскированием погоды."""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import load_all, make_mask, polygon_kinds, rmse
from gapfill.nn_data import N_CHANNELS, EpochInputs, assemble, build_tensors, loo_residual_array
from gapfill.nn_model import ResBlock, SeasonNet, gap_query_days, query_days_from_obs, to_batches, val_metrics
from gapfill.research_data import hidden_grid


def interpolate_channels(t, visible):
    """Интерполяции и расстояния по каждому сенсору, вычисленные только по видимым датам."""
    day = np.arange(t.known.shape[1])
    vd = np.zeros_like(t.known)
    ok = t.idx >= 0
    vd[ok] = visible[t.idx[ok]]
    out = np.zeros((*t.known.shape, 12), np.float32)
    for j, channel in enumerate((0, 3, 6)):
        for i in range(len(t.pid)):
            mask = vd[i] & (t.present[i, :, channel] > 0)
            ix = np.flatnonzero(mask)
            if not len(ix):
                out[i, :, j] = .3
                out[i, :, 3 + 2 * j:5 + 2 * j] = 1.
                continue
            val = t.vals[i, ix, channel].clip(-.1, 1.)
            out[i, :, j] = np.interp(day, ix, val)
            pos = np.searchsorted(ix, day)
            left = np.where(pos > 0, day - ix[np.maximum(pos - 1, 0)], 100)
            right = np.where(pos < len(ix), ix[np.minimum(pos, len(ix) - 1)] - day, 100)
            out[i, :, 3 + 2 * j] = np.clip(left / 60., 0., 1.)
            out[i, :, 4 + 2 * j] = np.clip(right / 60., 0., 1.)
            period = (5, 8, 16)[j]
            residue = (day[None, :] - ix[:, None]) % period == 0
            local = np.abs(day[None, :] - ix[:, None]) <= 40
            out[i, :, 9 + j] = np.minimum((residue & local).sum(0), 6) / 6.
    return out


class ResidualSeasonNet(nn.Module):
    """Нейросеть поправок к интерполяции с отдельным выбором скрытого сенсора."""

    def __init__(self, hidden=96, dropout=.25):
        super().__init__()
        self.inp = nn.Conv1d(N_CHANNELS + 13, hidden, 1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d, dropout) for d in (1, 2, 4, 8, 16, 32, 1, 2)])
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 6))

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2))).transpose(1, 2)
        out = self.head(self.tf(h))
        values = x[:, :, N_CHANNELS:N_CHANNELS + 3] + .15 * out[:, :, :3]
        logits = out[:, :, 3:]
        return (values * logits.softmax(-1)).sum(-1), logits


def inputs(t, ep, visible, query, residual):
    """Скрытые дни теряют погоду и спутниковые значения; контекст не содержит holdout."""
    data = assemble(t, ep, visible, query)
    hidden = t.known.copy()
    ok = t.idx >= 0
    hidden[ok] = ~visible[t.idx[ok]]
    data["x"][:, :, 16:19] *= (~hidden)[:, :, None]
    if residual:
        year = np.broadcast_to(((t.year - 2017) / 10.)[:, None, None], (*t.known.shape, 1))
        data["x"] = np.concatenate([data["x"], interpolate_channels(t, visible), year], axis=2).astype("float32")
    return data


def inference(model, data, device, batch=64):
    """Ограниченный размер батча удерживает использование памяти ускорителя."""
    model.eval()
    out = np.zeros(data["y"].shape, np.float32)
    keep = np.flatnonzero(data["query"].any(1))
    with torch.inference_mode():
        for start in range(0, len(keep), batch):
            idx = keep[start:start + batch]
            pred = model(torch.from_numpy(data["x"][idx]).to(device))
            out[idx] = (pred[0] if isinstance(pred, tuple) else pred).cpu().numpy()
    return out


def parse_args():
    """Параметры исследовательского и финального обучения."""
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["seasonnet", "residual"], default="residual")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--dropout", type=float, default=.25)
    p.add_argument("--out", default="nn_residual777")
    p.add_argument("--final", action="store_true")
    p.add_argument("--schedule-epochs", type=int)
    p.add_argument("--fixed-epoch", type=int, help="Заранее выбранная эпоха для независимого подтверждения")
    return p.parse_args()


def train_epoch(model, opt, t, obs, context, rng, args, device):
    """Одна эпоха с новой маской и пересчитанными остатками видимого контекста."""
    selected = np.flatnonzero(context)
    mask = np.zeros(len(obs), bool)
    mask[rng.choice(selected, int(.15 * len(selected)), replace=False)] = True
    visible = context & ~mask
    ep_train = EpochInputs(t, obs, loo_residual_array(obs, visible))
    data = inputs(t, ep_train, visible, query_days_from_obs(t, mask), args.kind == "residual")
    model.train()
    losses = []
    for x, y, q, idx in to_batches(data, device, args.batch, True, rng):
        pred = model(x)
        if args.kind == "residual":
            pred, logits = pred
            sensor = torch.from_numpy(t.sensor[idx].astype("int64")).to(device)
            auxiliary = nn.functional.cross_entropy(logits[q], sensor[q]) * .003
        else:
            auxiliary = 0.
        loss = ((pred - y.clamp(-.1, 1.))[q] ** 2).mean() + auxiliary
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        losses.append(loss.detach().item())
    return float(np.mean(losses))


def checkpoint(model, vd, device, t, meta, val, args, out, epoch, start, loss, history, best):
    """Записывает метрику и выбранную эпоху, не меняя правило отбора на подтверждающей маске."""
    if args.final:
        print(f"Эпоха {epoch}: loss={loss:.6f}, время={time.monotonic() - start:.0f} с", flush=True)
        torch.save(model.state_dict(), out / "model.pt")
        return best
    pred_days = inference(model, vd, device, args.batch)
    metrics, pred_obs = val_metrics(pred_days, t, meta, val)
    sel = val & meta.split.eq("test").to_numpy()
    metrics["rmse_target"] = rmse(meta.loc[sel, TARGET], pred_obs[sel])
    metrics.update(epoch=epoch, seconds=time.monotonic() - start, loss=loss)
    history.append(metrics)
    save = epoch == args.fixed_epoch if args.fixed_epoch else metrics["rmse"] < best
    if save:
        best = metrics["rmse"]
        torch.save(model.state_dict(), out / "model.pt")
        meta.loc[val].assign(pred=pred_obs[val]).to_parquet(out / "val_pred.parquet")
        (out / "result.json").write_text(json.dumps(metrics | {"args": vars(args)}, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(metrics), flush=True)
    return best


def main():
    """Обучение с сохранением checkpoint и метрики на точных полигонах текущего test."""
    args = parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(42 + args.seed)
    rng = np.random.default_rng(42 + args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    obs, grid, gaps = load_all()
    val = np.zeros(len(obs), bool) if args.final else make_mask(obs, seed=args.val_seed)
    context = ~val
    t = build_tensors(obs, hidden_grid(grid, obs.loc[val]))
    ep = EpochInputs(t, obs, loo_residual_array(obs, context))
    residual = args.kind == "residual"
    model = (ResidualSeasonNet(args.hidden, args.dropout) if residual else
             SeasonNet(hidden=args.hidden, dropout=args.dropout)).to(device)
    if args.final:
        vquery, si, pos = gap_query_days(t, gaps)
    else:
        vquery = query_days_from_obs(t, val)
    vd = inputs(t, ep, context, vquery, residual)
    meta = obs.assign(poly_kind=obs.pid.map(polygon_kinds(obs)), is_2025=obs.year.eq(2025))
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.schedule_epochs or args.epochs,
                                                         eta_min=args.lr / 50)
    best, history = np.inf, []
    print(f"Устройство {device}, модель {args.kind}, сезонов {len(t.pid)}, эпох {args.epochs}", flush=True)
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, opt, t, obs, context, rng, args, device)
        schedule.step()
        if epoch % 10 and epoch != args.epochs:
            continue
        best = checkpoint(model, vd, device, t, meta, val, args, out, epoch, start, loss, history, best)
    if args.final:
        pred_days = inference(model, vd, device, args.batch)
        gaps[["pid", "date"]].assign(pred=pred_days[si, pos]).to_parquet(out / "gap_pred.parquet")
        (out / "result.json").write_text(json.dumps({"args": vars(args), "seconds": time.monotonic() - start},
                                                  indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
