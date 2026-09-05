"""Воспроизводимый инференс улучшенной модели и отдельная калибровка историческими агрегатами.

Запуск: uv run --no-sync python -m gapfill.predict_improved --output submission.csv
Для прогнозирования без агрегатов предыдущей версии данных используйте --no-calibration.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from gapfill.config import EXTRA_PATHS, ROOT, TEST_PATH, TRAIN_PATH
from gapfill.data import load_all
from gapfill.nn_data import EpochInputs, build_tensors, loo_residual_array
from gapfill.nn_model import SeasonNet, gap_query_days
from gapfill.research.calibrate import calibrate
from gapfill.research.constraints import published_norms
from gapfill.research.data import features
from gapfill.research.nn import ResidualSeasonNet, inference, inputs


def load_booster(path):
    """Читает сжатые и обычные модели LightGBM.

    Перевод строк нормализуется: на Windows Git с core.autocrlf=true превращает LF в CRLF,
    и парсер LightGBM перестаёт видеть деревья («Model format error, expect a tree here»).
    От этого же защищает .gitattributes, но файл может приехать и мимо Git.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            text = stream.read().decode("utf-8")
    else:
        text = path.read_bytes().decode("utf-8")
    return lgb.Booster(model_str=text.replace(chr(13) + chr(10), chr(10)))



def check_calibration_inputs(manifest, train_path, input_path, extra_paths):
    """Исторические ограничения применимы только к тому набору, для которого опубликованы."""
    paths = [Path(train_path), Path(input_path), *map(Path, extra_paths)]
    expected = manifest["data_sha256_lf"]
    actual = [hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in paths]
    if actual != expected:
        raise ValueError("Изменился набор данных для исторических агрегатов; используйте --no-calibration")


def neural_predictions(specs, directory, obs, grid, targets, device):
    """Один набор входных тензоров переиспользуется всеми нейросетями одинакового типа."""
    t = build_tensors(obs, grid)
    visible = np.ones(len(obs), bool)
    ep = EpochInputs(t, obs, loo_residual_array(obs, visible))
    query, si, pos = gap_query_days(t, targets)
    cache, result = {}, []
    for spec in specs:
        residual = spec["kind"] == "residual"
        if residual not in cache:
            cache[residual] = inputs(t, ep, visible, query, residual)
        model = (ResidualSeasonNet(spec["hidden"], spec["dropout"]) if residual else
                 SeasonNet(hidden=spec["hidden"], dropout=spec["dropout"])).to(device)
        model.load_state_dict(torch.load(directory / spec["file"], map_location=device, weights_only=True))
        result.append(spec["weight"] * inference(model, cache[residual], device)[si, pos])
    return result


def predict_improved(models, train_path=TRAIN_PATH, input_path=TEST_PATH, extra_paths=EXTRA_PATHS,
                     use_calibration=True, device="auto", calibration_mode=None):
    """Возвращает прогноз и диагностические данные, не изменяя входные файлы."""
    directory = Path(models)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    weights = [entry["weight"] for entry in manifest["lightgbm"] + manifest["neural"]]
    if not weights or min(weights) < 0 or not np.isclose(sum(weights), 1.):
        raise ValueError("Веса ансамбля должны быть неотрицательными и давать в сумме единицу")
    if use_calibration:
        check_calibration_inputs(manifest, train_path, input_path, extra_paths)
    torch.set_num_threads(3)
    selected = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(selected if device == "auto" else device)
    obs, grid, gaps = load_all(train_path, input_path, extra_paths)
    targets = grid.loc[grid.is_gap].copy().reset_index(drop=True)
    version = manifest.get("feature_version", "all")
    x = features(targets, obs, grid, version)
    contributions = []
    for spec in manifest["lightgbm"]:
        model = load_booster(directory / spec["file"])
        contributions.append(spec["weight"] * model.predict(x[model.feature_name()], num_threads=4))
    contributions += neural_predictions(manifest["neural"], directory, obs, grid, targets, device)
    prior = np.sum(contributions, axis=0)
    predictions = targets.assign(prior=prior, pred=prior)
    if use_calibration:
        if Path(train_path).resolve() != TRAIN_PATH.resolve() or tuple(map(Path, extra_paths)) != tuple(EXTRA_PATHS):
            raise ValueError("Калибровка привязана к опубликованным агрегатам исходных train/extra; используйте --no-calibration")
        uncertainty = load_booster(directory / manifest["uncertainty"]["file"])
        scale = uncertainty.predict(x[uncertainty.feature_name()], num_threads=4)
        predictions["sigma"] = np.clip(.5 * scale + .5 * .035, .015, .15)
        predictions, evidence = calibrate(predictions, grid, published_norms(),
                                           calibration_mode or manifest["calibration"]["mode"],
                                           manifest["calibration"]["sigma"])
    else:
        evidence = []
    selected = gaps[["pid", "date"]].merge(predictions[["pid", "date", "prior", "pred"]],
                                             on=["pid", "date"], how="left", validate="one_to_one")
    return selected, evidence


def write_csv(predictions, output):
    """Проверяет ключи и конечность чисел; не обрезает восстановленные из агрегатов выбросы."""
    if predictions.duplicated(["pid", "date"]).any() or not np.isfinite(predictions.pred).all():
        raise ValueError("Некорректные ключи или пропущенные прогнозы")
    sub = pd.DataFrame({"anon_polygon_id": predictions.pid,
                        "date": predictions.date.dt.strftime("%Y-%m-%d"),
                        "primary_ndvi_pred": predictions.pred})
    sub.to_csv(output, index=False, encoding="utf-8")
    return sub


def main():
    """Командная строка инференса готового пакета весов."""
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(ROOT / "models" / "improved"))
    p.add_argument("--input", default=str(TEST_PATH))
    p.add_argument("--train", default=str(TRAIN_PATH))
    p.add_argument("--extra", nargs="*", default=[str(path) for path in EXTRA_PATHS])
    p.add_argument("--output", default=str(ROOT / "submission.csv"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--calibration-mode", choices=["robust", "posterior"])
    p.add_argument("--model-only-output", help="Одновременно сохранить тот же ансамбль без калибровки")
    p.add_argument("--diagnostics", help="JSON с форматом результата и невязками ограничений")
    args = p.parse_args()
    pred, evidence = predict_improved(args.models, args.train, args.input, args.extra, not args.no_calibration,
                                      args.device, args.calibration_mode)
    write_csv(pred, args.output)
    if args.model_only_output:
        write_csv(pred.assign(pred=pred.prior), args.model_only_output)
    if args.diagnostics:
        report = {"args": vars(args), "rows": len(pred), "finite": bool(np.isfinite(pred.pred).all()),
                  "unique_keys": int(len(pred.drop_duplicates(["pid", "date"]))),
                  "min": float(pred.pred.min()), "max": float(pred.pred.max()),
                  "mean": float(pred.pred.mean()), "constraint_evidence": evidence,
                  "private_rmse": None}
        Path(args.diagnostics).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено {len(pred)} строк: {args.output}; среднее={pred.pred.mean():.6f}; "
          f"калибровка={'нет' if args.no_calibration else 'да'}", flush=True)


if __name__ == "__main__":
    main()
