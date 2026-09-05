"""Последовательные эксперименты LightGBM с сохранением всех предсказаний и параметров."""

import argparse
import json
import time

import lightgbm as lgb

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import gap_score, load_all, rmse
from gapfill.dataset import groups_of
from gapfill.research.data import examples
from gapfill.train import LGB_PARAMS, evaluate, holdout_groups, importance


def selection(x, mode):
    """Наборы признаков для определения вклада каждой новой группы."""
    if mode == "baseline":
        return [c for c in x if not c.startswith(("raw_", "sp_"))]
    if mode == "raw":
        return [c for c in x if not c.startswith("sp_")]
    if mode == "spatial":
        return [c for c in x if not c.startswith("raw_")]
    return list(x)


def fit_one(x, meta, xv, mv, args, mode):
    """Ранняя остановка использует только внутренние группы; внешний holdout — для оценки."""
    out = ARTIFACTS_DIR / "research" / args.out / mode
    out.mkdir(parents=True, exist_ok=True)
    cols = selection(x, mode)
    y = meta[TARGET].to_numpy().clip(args.clip[0], args.clip[1])
    es = holdout_groups(groups_of(meta), .1, 42)
    params = LGB_PARAMS | {"num_threads": args.threads, "learning_rate": args.lr,
                          "num_leaves": args.leaves, "min_data_in_leaf": args.min_leaf,
                          "feature_fraction": args.ff, "lambda_l2": args.l2,
                          "objective": args.objective, "seed": args.seed, "force_col_wise": True}
    train = lgb.Dataset(x.loc[~es, cols], y[~es])
    valid = lgb.Dataset(x.loc[es, cols], y[es], reference=train)
    start = time.monotonic()
    model = lgb.train(params, train, num_boost_round=args.rounds, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)])
    pred = model.predict(xv[cols], num_threads=args.threads)
    result = evaluate(pred, mv, xv)
    target = mv.split.eq("test").to_numpy()
    result.update(rmse_target=rmse(mv.loc[target, TARGET], pred[target]),
                  rounds=model.best_iteration, n_features=len(cols), seconds=time.monotonic() - start,
                  args=vars(args), feature_mode=mode, importance=importance(model, 60))
    result["gap_score_target"] = gap_score(result["rmse_target"])
    mv.assign(pred=pred).to_parquet(out / "val_pred.parquet")
    model.save_model(str(out / "model.txt"))
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("args", "importance")},
                     ensure_ascii=False, indent=2), flush=True)


def main():
    """Ограниченная серия моделей на переиспользуемом наборе масок."""
    p = argparse.ArgumentParser()
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--n-masks", type=int, default=12)
    p.add_argument("--share", type=float, default=.15)
    p.add_argument("--modes", nargs="+", default=["baseline", "raw", "spatial", "all"])
    p.add_argument("--out", default="screen777")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--rounds", type=int, default=6500)
    p.add_argument("--leaves", type=int, default=63)
    p.add_argument("--min-leaf", type=int, default=40)
    p.add_argument("--lr", type=float, default=.03)
    p.add_argument("--ff", type=float, default=.6)
    p.add_argument("--l2", type=float, default=2.)
    p.add_argument("--clip", type=float, nargs=2, default=[-.1, 1.])
    p.add_argument("--objective", default="regression")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--analog", action="store_true")
    p.add_argument("--kriging", action="store_true")
    args = p.parse_args()
    obs, grid, _ = load_all()
    x, meta, xv, mv = examples(obs, grid, args.val_seed, args.n_masks, args.share,
                              analog=args.analog or args.kriging, kriging=args.kriging)
    print(f"Обучение {x.shape}, валидация {xv.shape}", flush=True)
    for mode in args.modes:
        fit_one(x, meta, xv, mv, args, mode)


if __name__ == "__main__":
    main()
