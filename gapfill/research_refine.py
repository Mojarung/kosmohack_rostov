"""Повторное прогнозирование после однозначного восстановления части данных из агрегатов."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.linalg import lstsq

from gapfill.config import ARTIFACTS_DIR, INDEX_COLS, SENSOR_CODE, TARGET, WEATHER_COLS
from gapfill.data import load_all, make_mask, rmse
from gapfill.research_calibrate import calibrate
from gapfill.research_constraints import published_norms, system
from gapfill.research_data import examples, features, hidden_grid
from gapfill.train import holdout_groups
from gapfill.dataset import groups_of
from gapfill.nn_data import EpochInputs, build_tensors, loo_residual_array
from gapfill.nn_model import gap_query_days
from gapfill.research_nn import ResidualSeasonNet, inference, inputs


def refined_neural(model_dir, context, grid, targets):
    """Нейросеть получает только исходно известный и однозначно восстановленный контекст."""
    args = json.loads((model_dir / "result.json").read_text())["args"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(3)
    tensors = build_tensors(context, grid)
    visible = np.ones(len(context), bool)
    epoch = EpochInputs(tensors, context, loo_residual_array(context, visible))
    query, si, pos = gap_query_days(tensors, targets)
    data = inputs(tensors, epoch, visible, query, True)
    model = ResidualSeasonNet(args["hidden"], args["dropout"]).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    return inference(model, data, device)[si, pos]


def exact_recovery(grid, held, norms):
    """Возвращает лишь значения, одинаковые у всех решений системы опубликованных средних."""
    keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
    rows = grid.loc[grid[TARGET].notna() | grid.is_gap].copy()
    rows["unknown"] = rows.is_gap | pd.MultiIndex.from_frame(rows[["pid", "date"]]).isin(keys)
    pieces = []
    for pid, group in rows.groupby("pid"):
        mask = group.unknown.to_numpy()
        ng = norms.loc[norms.pid.eq(pid)]
        if not mask.any() or ng.empty:
            continue
        a, b, _, _ = system(group, ng, mask)
        u, singular, vt = np.linalg.svd(a, full_matrices=False)
        rank = int((singular > 1e-8).sum())
        solution = vt[:rank].T @ (u[:, :rank].T @ b / singular[:rank])
        identifiable = (vt[:rank] ** 2).sum(0) > 1 - 1e-8
        if np.max(np.abs(a @ solution - b)) > 1e-7:
            raise ValueError(f"Несогласованные опубликованные средние для {pid}")
        part = group.loc[mask, ["pid", "date"]].copy()
        part[TARGET] = solution
        pieces.append(part.loc[identifiable])
    return pd.concat(pieces, ignore_index=True)


def sensor_model(x, meta, directory):
    """Классификатор сенсора обучается без внешнего holdout, как основной регрессор."""
    path = directory / "sensor.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    es = holdout_groups(groups_of(meta), .1, 42)
    params = {"objective": "multiclass", "num_class": 3, "learning_rate": .05,
              "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": .8,
              "num_threads": 4, "verbosity": -1, "seed": 42, "force_col_wise": True}
    data = lgb.Dataset(x.loc[~es], meta.loc[~es, "sensor"])
    valid = lgb.Dataset(x.loc[es], meta.loc[es, "sensor"], reference=data)
    model = lgb.train(params, data, num_boost_round=1500, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    model.save_model(str(path))
    return model


def enrich(obs, grid, held, recovered, sensor):
    """Создаёт контекст из видимых строк и доказуемо восстановленных NDVI; прочие колонки скрыты."""
    key = ["pid", "date"]
    held_keys = pd.MultiIndex.from_frame(held[key])
    context = obs.loc[~pd.MultiIndex.from_frame(obs[key]).isin(held_keys)].copy()
    clean = hidden_grid(grid, held)
    clean.loc[pd.MultiIndex.from_frame(clean[key]).isin(held_keys), "is_gap"] = True
    extra = clean.merge(recovered[key + [TARGET]], on=key, suffixes=("_old", ""))
    extra = extra.drop(columns=[f"{TARGET}_old"])
    extra[INDEX_COLS + WEATHER_COLS] = np.nan
    extra["sensor"] = sensor
    for name, code in SENSOR_CODE.items():
        sel = extra.sensor.eq(code)
        extra.loc[sel, f"{name}_ndvi"] = extra.loc[sel, TARGET].clip(-1., 1.)
    extra["is_gap"] = False
    positions = pd.MultiIndex.from_frame(clean[key]).get_indexer(pd.MultiIndex.from_frame(extra[key]))
    clean.loc[positions, [TARGET, *INDEX_COLS, "is_gap"]] = extra[[TARGET, *INDEX_COLS, "is_gap"]].to_numpy()
    return pd.concat([context, extra[context.columns]], ignore_index=True).sort_values(key), clean


def main():
    """Проверяет, помогает ли точная часть восстановленного контекста оставшимся пропускам."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--uncertainty", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--out", default="refine777")
    p.add_argument("--nn")
    p.add_argument("--sensor-model")
    args = p.parse_args()
    obs, grid, _ = load_all()
    held = obs.loc[make_mask(obs, seed=args.val_seed)]
    norms = published_norms(held)
    recovered = exact_recovery(grid, held, norms)
    directory = ARTIFACTS_DIR / "research" / args.out
    directory.mkdir(parents=True, exist_ok=True)
    if args.sensor_model:
        classifier = lgb.Booster(model_file=args.sensor_model)
    else:
        x, meta, _, _ = examples(obs, grid, args.val_seed, 12)
        classifier = sensor_model(x, meta, directory)
    keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
    targets = grid.loc[grid.is_gap | pd.MultiIndex.from_frame(grid[["pid", "date"]]).isin(keys)].reset_index(drop=True)
    targets["is_val"] = pd.MultiIndex.from_frame(targets[["pid", "date"]]).isin(keys)
    selected = pd.MultiIndex.from_frame(targets[["pid", "date"]]).isin(pd.MultiIndex.from_frame(recovered[["pid", "date"]]))
    targets_recovered = targets.loc[selected].reset_index(drop=True)
    context = obs.loc[~pd.MultiIndex.from_frame(obs[["pid", "date"]]).isin(keys)]
    xr = features(targets_recovered, context, hidden_grid(grid, held))
    probs = classifier.predict(xr, num_threads=4)
    recovered = targets_recovered[["pid", "date"]].merge(recovered, on=["pid", "date"])
    new_context, new_grid = enrich(obs, grid, held, recovered, probs.argmax(1))
    unresolved = targets.loc[~selected].reset_index(drop=True)
    xu = features(unresolved, new_context, new_grid)
    model = lgb.Booster(model_file=args.model)
    variance = lgb.Booster(model_file=args.uncertainty)
    unresolved["prior"] = model.predict(xu[model.feature_name()], num_threads=4)
    if args.nn:
        neural = refined_neural(Path(args.nn), new_context.reset_index(drop=True), new_grid, unresolved)
        unresolved["prior"] = .6 * unresolved.prior + .4 * neural
    unresolved["sigma"] = np.clip(.5 * variance.predict(xu[variance.feature_name()], num_threads=4) + .5 * .035, .015, .15)
    calibrated, evidence = calibrate(unresolved, new_grid, norms, "robust", .04)
    exact = targets_recovered.assign(prior=recovered[TARGET].to_numpy(), pred=recovered[TARGET].to_numpy())
    prediction = pd.concat([calibrated, exact], ignore_index=True)
    prediction.to_parquet(directory / "predictions.parquet")
    report = {"args": vars(args), "n_exact": len(recovered), "results": {}}
    for name, sel in {"all": prediction.is_val, "target": prediction.is_val & prediction.split.eq("test")}.items():
        report["results"][name] = {c: rmse(prediction.loc[sel, TARGET], prediction.loc[sel, c]) for c in ("prior", "pred")}
    (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
