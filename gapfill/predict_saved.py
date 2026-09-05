"""Инференс из готовых артефактов без обучения: models/ (LightGBM .txt.gz + SeasonNet .pt) → submission.csv.

Запуск: uv run python -m gapfill.predict_saved --input data/test_features_new.csv --output submission.csv --models models
Контекст для контрольных точек — все известные точки train, входного файла и файлов --extra; признаки считаются так же,
как при обучении (gapfill.features). Смесь: 0.5 · среднее LightGBM + 0.5 · среднее SeasonNet.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from gapfill.config import EXTRA_PATHS, ROOT, SUBMISSION_PATH, TEST_PATH, TRAIN_PATH
from gapfill.data import load_all
from gapfill.nn_data import EpochInputs, assemble, build_tensors, loo_residual_array
from gapfill.nn_model import SeasonNet, gap_query_days, predict
from gapfill.predict import gap_features, write_submission

WEIGHT_LGB = 0.5


def predict_lgb(models_dir: Path, obs, grid, gaps) -> np.ndarray:
    """Среднее по всем LightGBM-моделям из папки (файлы lgb_seed*.txt.gz)."""
    X = gap_features(obs, grid, gaps)
    preds = []
    for path in sorted(models_dir.glob("lgb_seed*.txt.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            booster = lgb.Booster(model_str=f.read())
        preds.append(booster.predict(X[booster.feature_name()]))
    if not preds:
        raise FileNotFoundError(f"нет lgb_seed*.txt.gz в {models_dir}")
    return np.mean(preds, axis=0)


def predict_nn(models_dir: Path, obs, grid, gaps, hidden: int = 128, n_tf: int = 2) -> np.ndarray:
    """Среднее по всем SeasonNet-весам из папки (файлы nn_final_seed*.pt)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = build_tensors(obs, grid)
    visible = np.ones(len(obs), dtype=bool)
    ep = EpochInputs(t, obs, loo_residual_array(obs, visible))
    gap_q, si, pos = gap_query_days(t, gaps)
    data = assemble(t, ep, visible, gap_q)
    preds = []
    for path in sorted(models_dir.glob("nn_final_seed*.pt")):
        model = SeasonNet(hidden=hidden, n_tf=n_tf).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        preds.append(predict(model, data, device)[si, pos])
    if not preds:
        raise FileNotFoundError(f"нет nn_final_seed*.pt в {models_dir}")
    return np.mean(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Инференс из сохранённых моделей")
    parser.add_argument("--input", type=str, default=str(TEST_PATH), help="private_features.csv")
    parser.add_argument("--train", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--extra", type=str, nargs="*", default=[str(p) for p in EXTRA_PATHS],
                        help="файлы с дополнительными известными точками (первая версия test); --extra без значений — не использовать")
    parser.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--models", type=str, default=str(ROOT / "models"))
    args = parser.parse_args()
    obs, grid, gaps = load_all(args.train, args.input, args.extra)
    models_dir = Path(args.models)
    p_lgb = predict_lgb(models_dir, obs, grid, gaps)
    p_nn = predict_nn(models_dir, obs, grid, gaps)
    sub = write_submission(gaps, WEIGHT_LGB * p_lgb + (1 - WEIGHT_LGB) * p_nn, args.output)
    print(f"{args.output}: {len(sub)} строк, среднее {sub['primary_ndvi_pred'].mean():.4f}, "
          f"corr LGB/NN {np.corrcoef(p_lgb, p_nn)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
