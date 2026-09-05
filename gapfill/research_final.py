"""Финальное обучение выбранной табличной конфигурации на всех известных значениях."""

import argparse
import gzip
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import load_all
from gapfill.dataset import groups_of
from gapfill.research_data import examples, features
from gapfill.train import LGB_PARAMS, holdout_groups


def fit_uncertainty(x, meta, params, rounds, out):
    """Отдельный внутренний holdout обеспечивает честные обучающие ошибки для оценки масштаба."""
    es = holdout_groups(groups_of(meta), .1, 42)
    y = meta[TARGET].to_numpy().clip(-.1, 1.)
    train = lgb.Dataset(x.loc[~es], y[~es])
    valid = lgb.Dataset(x.loc[es], y[es], reference=train)
    model = lgb.train(params, train, num_boost_round=rounds, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)])
    error = np.abs(meta.loc[es, TARGET] - model.predict(x.loc[es], num_threads=4)).clip(.003, .3)
    settings = {"objective": "regression", "learning_rate": .025, "num_leaves": 15,
                "min_data_in_leaf": 200, "lambda_l2": 20., "feature_fraction": .7,
                "num_threads": 4, "verbosity": -1, "seed": 42, "force_col_wise": True}
    uncertainty = lgb.train(settings, lgb.Dataset(x.loc[es], error), num_boost_round=700)
    uncertainty.save_model(str(out / "uncertainty.txt"))
    print(f"Оценка неопределённости обучена на {es.sum()} независимых внутренних примерах", flush=True)


def main():
    """Сохраняет веса и прогнозы отдельно от исходного models/ и submission.csv."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="result.json выбранного валидационного запуска")
    p.add_argument("--n-masks", type=int, default=30)
    p.add_argument("--rounds", type=int, default=8500)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", default="final_lgb")
    p.add_argument("--uncertainty", action="store_true")
    p.add_argument("--only-uncertainty", action="store_true")
    args = p.parse_args()
    reference = json.loads(Path(args.source).read_text())
    settings = reference["args"]
    obs, grid, _ = load_all()
    kriging = settings.get("kriging", False)
    analog = settings.get("analog", False) or kriging
    x, meta, _, _ = examples(obs, grid, n_masks=args.n_masks, share=settings.get("share", .15),
                             final=True, analog=analog, kriging=kriging)
    params = LGB_PARAMS | {"num_threads": args.threads, "force_col_wise": True,
                          "num_leaves": settings["leaves"], "min_data_in_leaf": settings["min_leaf"],
                          "learning_rate": settings["lr"], "feature_fraction": settings["ff"],
                          "lambda_l2": settings["l2"]}
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.only_uncertainty:
        fit_uncertainty(x, meta, params, args.rounds, out)
        return
    targets = grid.loc[grid.is_gap].copy().reset_index(drop=True)
    xg = features(targets, obs, grid, "kriging" if kriging else "analog" if analog else "all")
    y = meta[TARGET].to_numpy().clip(*settings["clip"])
    for seed in args.seeds:
        path = out / f"lgb_seed{seed}.txt.gz"
        if path.exists():
            continue
        start = time.monotonic()
        model = lgb.train(params | {"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed},
                          lgb.Dataset(x, y), num_boost_round=args.rounds,
                          callbacks=[lambda env: print(f"seed={seed}, деревьев={env.iteration + 1}", flush=True)
                                     if (env.iteration + 1) % 1000 == 0 else None])
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(model.model_to_string())
        targets.assign(pred=model.predict(xg, num_threads=4)).to_parquet(out / f"gap_pred_seed{seed}.parquet")
        print(f"Финальная модель seed={seed}: {time.monotonic() - start:.0f} с", flush=True)
    if args.uncertainty and not (out / "uncertainty.txt").exists():
        fit_uncertainty(x, meta, params, args.rounds, out)
    (out / "result.json").write_text(json.dumps({"args": vars(args), "source": reference,
                                                "n_train": len(x), "n_known": len(obs), "final": True},
                                               indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
