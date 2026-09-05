"""Смесь независимо обученных моделей и калибровка на опубликованных агрегатах."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import gap_score, load_all, make_mask, rmse
from gapfill.nn_data import EpochInputs, build_tensors, loo_residual_array
from gapfill.nn_model import SeasonNet, gap_query_days
from gapfill.research_calibrate import calibrate, get_predictions, with_uncertainty
from gapfill.research_constraints import published_norms
from gapfill.research_data import hidden_grid
from gapfill.research_nn import ResidualSeasonNet, inference, inputs


def neural_prior(directory, obs, grid, held, targets, device):
    """Прогнозирует все скрытые даты одной маски, используя только обученный без них checkpoint."""
    info = json.loads((directory / "result.json").read_text())
    args = info["args"]
    if bool(args["final"]) != (len(held) == 0):
        raise ValueError("Нельзя оценивать финальную сеть на её обучающих значениях")
    keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
    visible = ~pd.MultiIndex.from_frame(obs[["pid", "date"]]).isin(keys)
    t = build_tensors(obs, hidden_grid(grid, held))
    ep = EpochInputs(t, obs, loo_residual_array(obs, visible))
    query, si, pos = gap_query_days(t, targets)
    residual = args["kind"] == "residual"
    data = inputs(t, ep, visible, query, residual)
    model = (ResidualSeasonNet(args["hidden"], args["dropout"]) if residual else
             SeasonNet(hidden=args["hidden"], dropout=args["dropout"])).to(device)
    model.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=True))
    return inference(model, data, device)[si, pos]


def metric(frame, column):
    """Полный RMSE без обрезания неудобных наблюдений."""
    out = {}
    for name, sel in {"all": frame.is_val, "target": frame.is_val & frame.split.eq("test")}.items():
        out[name] = rmse(frame.loc[sel, TARGET], frame.loc[sel, column])
    return out


def main():
    """Сохраняет воспроизводимые смеси и их независимые контрольные ошибки."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--nn", nargs="+", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--weights", type=float, nargs="+", default=[.5, .7, .85])
    p.add_argument("--sigma", type=float, default=.06)
    p.add_argument("--calibration", default="robust")
    p.add_argument("--out", default="blend777")
    p.add_argument("--final", action="store_true")
    p.add_argument("--uncertainty")
    args = p.parse_args()
    torch.set_num_threads(3)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    obs, grid, _ = load_all()
    held = obs.iloc[:0] if args.final else obs.loc[make_mask(obs, seed=args.val_seed)]
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    base = get_predictions(obs, grid, held, args.model, out)
    if args.uncertainty:
        base = with_uncertainty(base, obs, grid, held, args.uncertainty)
    predictions = []
    for name in args.nn:
        directory = ARTIFACTS_DIR / "research" / name
        info = json.loads((directory / "result.json").read_text())
        if not args.final and info["args"]["val_seed"] != args.val_seed:
            raise ValueError("Неверная валидационная маска нейросети")
        pred = neural_prior(directory, obs, grid, held, base, device)
        predictions.append(pred)
    nn_mean = np.mean(predictions, axis=0)
    base.assign(neural=nn_mean).to_parquet(out / "components.parquet")
    norms = published_norms(held)
    report = {"args": vars(args), "results": []}
    if not args.final:
        report["lgb"] = metric(base, "prior")
        report["neural"] = metric(base.assign(pred=nn_mean), "pred")
    for weight in args.weights:
        source = base.assign(prior=weight * base.prior + (1 - weight) * nn_mean)
        calibrated, evidence = calibrate(source, grid, norms, args.calibration, args.sigma)
        calibrated.to_parquet(out / f"blend_{weight}.parquet")
        entry = {"weight_lgb": weight}
        if not args.final:
            entry.update(raw=metric(source, "prior"), calibrated=metric(calibrated, "pred"))
            entry["gap_score_target"] = gap_score(entry["calibrated"]["target"])
        report["results"].append(entry)
        print(json.dumps(entry), flush=True)
    (out / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
