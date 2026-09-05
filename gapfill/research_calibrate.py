"""Оценка калибровки модельного прогноза по опубликованным историческим mean/std."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.optimize import least_squares
from scipy.special import ive, ndtri, logsumexp
from scipy.stats import qmc

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import gap_score, load_all, make_mask, rmse
from gapfill.research_constraints import linear_projection, published_norms, system
from gapfill.research_data import features, hidden_grid


def get_predictions(obs, grid, held, model_path, directory):
    """Прогноз одновременно реальных и валидационных пропусков без подглядывания в holdout."""
    target_keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
    is_val = pd.MultiIndex.from_frame(grid[["pid", "date"]]).isin(target_keys)
    targets = grid.loc[grid.is_gap | is_val].copy().reset_index(drop=True)
    targets["is_val"] = pd.MultiIndex.from_frame(targets[["pid", "date"]]).isin(target_keys)
    context = obs.loc[~pd.MultiIndex.from_frame(obs[["pid", "date"]]).isin(target_keys)]
    cache = directory / "prior.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    model = lgb.Booster(model_file=str(model_path))
    kind = "baseline" if all(not c.startswith(("raw_", "sp_")) for c in model.feature_name()) else "all"
    if any(c.startswith("an_") for c in model.feature_name()):
        kind = "analog"
    if any(c.startswith("kg_") for c in model.feature_name()):
        kind = "kriging"
    x = features(targets, context, hidden_grid(grid, held), kind)
    targets["prior"] = model.predict(x[model.feature_name()], num_threads=4)
    targets.to_parquet(cache)
    return targets


def with_uncertainty(priors, obs, grid, held, model_path):
    """Оценивает масштаб ошибки без истинных значений скрытых дат."""
    keys = pd.MultiIndex.from_frame(held[["pid", "date"]])
    context = obs.loc[~pd.MultiIndex.from_frame(obs[["pid", "date"]]).isin(keys)]
    model = lgb.Booster(model_file=str(model_path))
    kind = "analog" if any(c.startswith("an_") for c in model.feature_name()) else "all"
    if any(c.startswith("kg_") for c in model.feature_name()):
        kind = "kriging"
    x = features(priors, context, hidden_grid(grid, held), kind)
    scale = model.predict(x[model.feature_name()], num_threads=4)
    return priors.assign(sigma=np.clip(.5 * scale + .5 * .035, .015, .15))


def moment_projection(prior, a, b, c, weight=1e-3):
    """Суммы соблюдаются точно; суммы квадратов — с регуляризацией к модельному прогнозу."""
    u, s, vt = np.linalg.svd(a, full_matrices=True)
    rank = int((s > 1e-9).sum())
    if rank == 0:
        return prior.copy()
    basis = vt[:rank]
    sums = u[:, :rank].T @ b / s[:rank]
    squares = u[:, :rank].T @ c / s[:rank]
    origin = prior + basis.T @ (sums - basis @ prior)
    null = vt[rank:].T
    if null.shape[1] == 0:
        return origin

    def fun(z):
        x = origin + null @ z
        return np.r_[basis @ (x * x) - squares, np.sqrt(weight) * z]

    def jac(z):
        x = origin + null @ z
        return np.vstack([(2 * basis * x[None, :]) @ null, np.sqrt(weight) * np.eye(len(z))])

    fit = least_squares(fun, np.zeros(null.shape[1]), jac=jac, max_nfev=120,
                        ftol=1e-8, xtol=1e-8, gtol=1e-8)
    return origin + null @ fit.x


@lru_cache(maxsize=32)
def sphere_directions(n):
    """Квазислучайные направления в подпространстве нулевой суммы для интегрирования апостериора."""
    z = ndtri(qmc.Sobol(n, scramble=True, seed=42).random_base2(15).clip(1e-9, 1 - 1e-9))
    z -= z.mean(1, keepdims=True)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def log_likelihood(error, sigma, density=None):
    """Плотность ошибки: Student-t либо обученная асимметричная смесь с защитой тяжёлых хвостов."""
    z = error / sigma
    if density is None or density.get("family") == "student":
        df = 3. if density is None else density["df"]
        return -.5 * (df + 1.) * np.log1p(z * z / df)
    components = []
    for weight, mean, std in zip(density["weights"], density["means"], density["stds"]):
        components.append(np.log(.998 * weight / (std * np.sqrt(2 * np.pi))) - .5 * ((z - mean) / std) ** 2)
    components.append(np.log(.002 / (10 * np.pi)) - np.log1p((z / 10) ** 2))
    return logsumexp(np.stack(components), axis=0)


def robust_group_mean(prior, center, radius, sigma, density=None):
    """Усредняет все допустимые варианты; тяжёлые хвосты не навязывают выброс конкретной дате."""
    x = center + radius * sphere_directions(len(prior))
    lp = log_likelihood(x - prior, sigma, density).sum(1)
    weights = np.exp(lp - lp.max())
    weights /= weights.sum()
    ess = 1. / (weights @ weights)
    estimate = weights @ x
    # Если интеграл покрыт слишком малым числом направлений, сохраняем более устойчивое среднее.
    trust = min(1., ess / 30.)
    return trust * estimate + (1 - trust) * (prior - prior.mean() + center)


def grouped_moments(prior, a, b, c, sigma=.06, max_group=100, robust=False, scale=None, density=None,
                    use_posterior=False):
    """Условное среднее для групп с однозначно восстановленными суммой и суммой квадратов.

    Две даты с одинаковыми колонками A неразличимы по агрегатам: усредняем две возможные
    перестановки по вероятностям Student-t вместо произвольного выбора одной из них.
    """
    if use_posterior:
        from gapfill.calibration_posterior import posterior
        pred, _ = posterior(prior, a, b, scale=scale)
    else:
        pred = linear_projection(prior, a, b, scale=scale)
    patterns, inverse = np.unique(a.T, axis=0, return_inverse=True)
    g = patterns.T
    pinv = np.linalg.pinv(g, rcond=1e-9)
    sums, squares = pinv @ b, pinv @ c
    identified = np.diag(pinv @ g) > 1 - 1e-7
    for j in np.flatnonzero(identified):
        select = inverse == j
        n = int(select.sum())
        if n > max_group:
            continue
        center = sums[j] / n
        radius2 = max(squares[j] - sums[j] ** 2 / n, 0.)
        if n == 1:
            pred[select] = center
            continue
        p = prior[select]
        local_sigma = sigma if scale is None else scale[select]
        if n == 2:
            delta = np.sqrt(radius2 / 2.)
            candidates = np.array([[center - delta, center + delta], [center + delta, center - delta]])
            loglike = log_likelihood(candidates - p, local_sigma, density).sum(1)
            w = np.exp(loglike - loglike.max())
            pred[select] = (candidates * (w / w.sum())[:, None]).sum(0)
        elif robust:
            pred[select] = robust_group_mean(p, center, np.sqrt(radius2), local_sigma, density)
        else:
            direction = p - p.mean()
            norm = np.linalg.norm(direction)
            radius = np.sqrt(radius2)
            kappa = radius * norm / (sigma ** 2)
            dim = n - 1
            shrink = float(ive(dim / 2., kappa) / ive(dim / 2. - 1., kappa)) if kappa > 1e-6 else 0.
            pred[select] = center + shrink * radius * direction / max(norm, 1e-9)
    return pred


def calibrate(priors, grid, norms, mode, weight, density=None):
    """Калибрует только 20 целевых полигонов; на остальных оставляет чистый модельный прогноз."""
    out = priors.copy()
    out["pred"] = out.prior
    target_pids = set(grid.loc[grid.split.eq("test"), "pid"])
    all_rows = grid.loc[grid[TARGET].notna() | grid.is_gap].copy()
    evidence = []
    for pid, group in all_rows.groupby("pid"):
        if pid not in target_pids:
            continue
        sel = out.pid.eq(pid)
        unknown_keys = pd.MultiIndex.from_frame(out.loc[sel, ["pid", "date"]])
        mask = pd.MultiIndex.from_frame(group[["pid", "date"]]).isin(unknown_keys)
        unknown = group.loc[mask, ["pid", "date"]].merge(out.loc[sel], on=["pid", "date"], how="left")
        a, b, c, count = system(group, norms.loc[norms.pid.eq(pid)], mask)
        active = a.sum(0) > 0
        prior = unknown.prior.to_numpy()
        scale = unknown["sigma"].to_numpy() if "sigma" in unknown else None
        pred = prior.copy()
        if mode == "linear":
            pred[active] = linear_projection(prior[active], a[:, active], b, strength=weight,
                                             scale=scale[active] if scale is not None else None)
        elif mode == "moments":
            pred[active] = moment_projection(prior[active], a[:, active], b, c, weight=weight)
        elif mode in ("groups", "pairs", "robust", "posterior"):
            pred[active] = grouped_moments(prior[active], a[:, active], b, c, sigma=weight,
                                           max_group=2 if mode == "pairs" else 100,
                                           robust=mode in ("robust", "posterior"),
                                           scale=scale[active] if scale is not None else None, density=density,
                                           use_posterior=mode == "posterior")
        else:
            raise ValueError(mode)
        predicted = pd.Series(pred, index=pd.MultiIndex.from_frame(unknown[["pid", "date"]]))
        out.loc[sel, "pred"] = predicted.reindex(unknown_keys).to_numpy()
        evidence.append({"pid": pid, "n_unknown": int(active.sum()),
                         "mean_constraint_rmse": float(np.sqrt(np.mean((a @ pred - b) ** 2))),
                         "square_constraint_rmse": float(np.sqrt(np.mean((a @ (pred ** 2) - c) ** 2)))})
    return out, evidence


def main():
    """Сравнивает калибровку с тем же самым некалиброванным прогнозом."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--final", action="store_true")
    p.add_argument("--mode", choices=["linear", "moments", "groups", "pairs", "robust", "posterior"], default="linear")
    p.add_argument("--weight", type=float, default=1.)
    p.add_argument("--out", default="calibrate777")
    p.add_argument("--uncertainty")
    p.add_argument("--density")
    args = p.parse_args()
    obs, grid, _ = load_all()
    held = obs.iloc[:0] if args.final else obs.loc[make_mask(obs, seed=args.val_seed)]
    directory = ARTIFACTS_DIR / "research" / args.out
    directory.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    info_path = directory / "prior_info.json"
    identity = {"model_sha256": model_hash, "val_seed": None if args.final else args.val_seed}
    if info_path.exists() and json.loads(info_path.read_text()) != identity:
        raise ValueError("Папка кэша содержит прогноз другой модели/маски; задайте новый --out")
    info_path.write_text(json.dumps(identity), encoding="utf-8")
    priors = get_predictions(obs, grid, held, args.model, directory)
    if args.uncertainty:
        priors = with_uncertainty(priors, obs, grid, held, args.uncertainty)
    density = json.loads(Path(args.density).read_text()) if args.density else None
    pred, evidence = calibrate(priors, grid, published_norms(held), args.mode, args.weight, density)
    label = f"{args.mode}_{args.weight}" + ("_weighted" if args.uncertainty else "") + ("_mixture" if density else "")
    pred.to_parquet(directory / f"{label}.parquet")
    report = {"args": vars(args), "evidence": evidence}
    if not args.final:
        for name, sel in {"all": pred.is_val, "target": pred.is_val & pred.split.eq("test")}.items():
            report[f"rmse_{name}"] = rmse(pred.loc[sel, TARGET], pred.loc[sel, "pred"])
            report[f"prior_rmse_{name}"] = rmse(pred.loc[sel, TARGET], pred.loc[sel, "prior"])
        report["gap_score_target"] = gap_score(report["rmse_target"])
    (directory / f"{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "evidence"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
