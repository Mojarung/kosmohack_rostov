"""Сборка самостоятельного пакета улучшенных весов без перезаписи исходной модели."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from gapfill.config import ARTIFACTS_DIR, EXTRA_PATHS, ROOT, TEST_PATH, TRAIN_PATH


def digest(path, normalize=False):
    """Отпечаток модели либо CSV с нормализованными окончаниями строк."""
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalize else data).hexdigest()


def copy_model(source, destination):
    """Не перезаписывает отличающиеся веса уже собранного пакета."""
    if destination.exists() and digest(source) != digest(destination):
        raise ValueError(f"Файл уже содержит другую модель: {destination}")
    shutil.copy2(source, destination)
    return {"file": destination.name, "sha256": digest(destination)}


def main():
    """Сохраняет спецификацию инференса, источники данных и метаданные обучения."""
    p = argparse.ArgumentParser()
    p.add_argument("--lgb", default="final_kriging")
    p.add_argument("--nn", nargs="+", default=["final_nn_s0"])
    p.add_argument("--output", default=str(ROOT / "models" / "improved"))
    p.add_argument("--calibration", choices=["robust", "posterior"], default="posterior")
    args = p.parse_args()
    root = ARTIFACTS_DIR / "research"
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    tree_info = json.loads((root / args.lgb / "result.json").read_text())
    setting = tree_info["source"]["args"]
    version = "kriging" if setting.get("kriging") else "analog" if setting.get("analog") else "all"
    seeds = tree_info["args"]["seeds"]
    manifest = {"format_version": 1, "feature_version": version, "lightgbm": [], "neural": [],
                "calibration": {"mode": args.calibration, "sigma": .04,
                                "warning": "Агрегаты старой версии содержат информацию о скрытых ответах."},
                "data_sha256_lf": [digest(path, True) for path in [TRAIN_PATH, TEST_PATH, *EXTRA_PATHS]],
                "training": {"lightgbm": tree_info, "neural": []}}
    for seed in seeds:
        filename = f"lgb_seed{seed}.txt.gz"
        spec = copy_model(root / args.lgb / filename, directory / filename)
        manifest["lightgbm"].append(spec | {"weight": .6 / len(seeds)})
    for i, name in enumerate(args.nn):
        info = json.loads((root / name / "result.json").read_text())
        if not info["args"]["final"]:
            raise ValueError(f"Нейросеть {name} не обучена в финальном режиме")
        spec = copy_model(root / name / "model.pt", directory / f"residual_seed{i}.pt")
        manifest["neural"].append(spec | {k: info["args"][k] for k in ["kind", "hidden", "dropout"]}
                                   | {"weight": .4 / len(args.nn)})
        manifest["training"]["neural"].append(info)
    manifest["uncertainty"] = copy_model(root / args.lgb / "uncertainty.txt", directory / "uncertainty.txt")
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готовый пакет: {directory}; бустингов {len(seeds)}, нейросетей {len(args.nn)}", flush=True)


if __name__ == "__main__":
    main()
