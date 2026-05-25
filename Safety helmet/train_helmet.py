#!/usr/bin/env python3
"""Train a YOLO detector for safety helmet detection."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from dataset_probe import SCORE_CLASSES, build_report, is_image, normalize_class


SPLIT_SEED = 2026


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Please run: pip install -r requirements.txt") from exc
    return YOLO


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def split_items(items: list[Any], val_ratio: float, test_ratio: float) -> dict[str, list[Any]]:
    rng = random.Random(SPLIT_SEED)
    items = list(items)
    rng.shuffle(items)
    if len(items) == 1:
        return {"train": items, "val": items, "test": []}

    n_total = len(items)
    n_test = int(n_total * test_ratio)
    n_val = max(1, int(n_total * val_ratio))
    return {
        "test": items[:n_test],
        "val": items[n_test : n_test + n_val],
        "train": items[n_test + n_val :],
    }


def find_image_by_name(root: Path, file_name: str, image_index: dict[str, list[Path]]) -> Path | None:
    direct = root / file_name
    if direct.exists():
        return direct
    matches = image_index.get(Path(file_name).name, [])
    return matches[0] if matches else None


def parse_float(node: ET.Element | None, name: str, default: float = 0.0) -> float:
    if node is None:
        return default
    try:
        return float(node.findtext(name, str(default)))
    except (TypeError, ValueError):
        return default


def prepare_voc(root: Path, output_dir: Path, copy_mode: str, val_ratio: float, test_ratio: float) -> Path:
    if yaml is None:
        raise SystemExit("Missing PyYAML. Please run: pip install -r requirements.txt")

    image_index: dict[str, list[Path]] = defaultdict(list)
    for image_path in root.rglob("*"):
        if image_path.is_file() and is_image(image_path):
            image_index[image_path.name].append(image_path)

    class_to_idx = {name: idx for idx, name in enumerate(SCORE_CLASSES)}
    records: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []
    kept_objects = 0

    for xml_path in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue

        node = tree.getroot()
        filename = node.findtext("filename") or f"{xml_path.stem}.jpg"
        image_path = find_image_by_name(xml_path.parent, filename, image_index)
        if image_path is None:
            continue

        size = node.find("size")
        width = parse_float(size, "width", 1.0)
        height = parse_float(size, "height", 1.0)
        if width <= 0 or height <= 0:
            continue

        objects: list[tuple[int, float, float, float, float]] = []
        for obj in node.findall("object"):
            name = normalize_class(obj.findtext("name") or "")
            if name not in class_to_idx:
                continue

            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = max(0.0, parse_float(box, "xmin"))
            ymin = max(0.0, parse_float(box, "ymin"))
            xmax = min(width, parse_float(box, "xmax"))
            ymax = min(height, parse_float(box, "ymax"))
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            if w < 1 or h < 1:
                continue

            cx = (xmin + w / 2.0) / width
            cy = (ymin + h / 2.0) / height
            objects.append((class_to_idx[name], cx, cy, w / width, h / height))

        if objects:
            kept_objects += len(objects)
            records.append((image_path, objects))

    if not records:
        raise SystemExit("No usable VOC annotations were found for safety helmet classes.")

    print(f"[prepare_voc] {len(records)} images, {kept_objects} objects across {len(class_to_idx)} classes", flush=True)

    reset_dir(output_dir)
    splits = split_items(records, val_ratio, test_ratio)
    for split, split_records in splits.items():
        for image_path, objects in split_records:
            link_or_copy(image_path, output_dir / "images" / split / image_path.name, copy_mode)
            lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in objects]
            label_path = output_dir / "labels" / split / f"{image_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(SCORE_CLASSES),
        "names": SCORE_CLASSES,
    }
    if splits.get("test"):
        data["test"] = "images/test"

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    print(f"[prepare_voc] YOLO dataset ready: {data_yaml}", flush=True)
    print(f"[prepare_voc] splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}", flush=True)
    return data_yaml


def train_yolo(
    data_yaml: Path,
    model_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: Path,
    name: str,
    device: str | None,
    workers: int,
    save_period: int,
    model_output: Path,
    cache: bool,
    patience: int,
    close_mosaic: int,
) -> None:
    YOLO = require_ultralytics()
    model = YOLO(model_path)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "workers": workers,
        "save_period": save_period,
        "patience": patience,
        "close_mosaic": close_mosaic,
        "cos_lr": True,
        "pretrained": True,
        "plots": False,
        "verbose": True,
    }
    if device is not None:
        train_kwargs["device"] = device
    if cache:
        train_kwargs["cache"] = True

    print(f"[train_yolo] starting training: epochs={epochs}, imgsz={imgsz}, batch={batch}", flush=True)
    model.train(**train_kwargs)
    print("[train_yolo] training complete", flush=True)


def save_model_artifacts(project: Path, name: str, output_dir: Path, report: dict) -> None:
    run_dir = project / name
    if not run_dir.exists():
        candidates = sorted(project.glob(f"{name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            run_dir = candidates[0]

    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    source_pt = best_pt if best_pt.exists() else last_pt
    if not source_pt.exists():
        raise SystemExit(f"Training finished, but no weights were found under: {weights_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pt, output_dir / "model.pt")
    if best_pt.exists():
        shutil.copy2(best_pt, output_dir / "best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, output_dir / "last.pt")

    metadata = {
        "task_type": "detect",
        "model_file": "model.pt",
        "source_run_dir": str(run_dir),
        "recommended_task": report.get("recommended_task"),
        "dataset_root": report.get("dataset_root"),
        "class_names": SCORE_CLASSES,
        "alert_classes": ["head"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model artifacts to: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train safety helmet detection.")
    parser.add_argument("--data", default="/home/data/831", help="Dataset root")
    parser.add_argument("--model", default="yolov8n.pt", help="Local YOLO weight path")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0, or 0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoint every N epochs. Use -1 to disable.")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--cache", action="store_true", help="Cache training images in RAM/disk when the platform has enough memory.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--workdir", default="/project/train/work/helmet_detection_dataset")
    parser.add_argument("--project", default="/project/train/runs")
    parser.add_argument("--name", default="helmet_detection")
    parser.add_argument("--model-output", default="/project/train/models/your_model")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Warning: ignoring unsupported arguments: {' '.join(unknown_args)}", flush=True)

    root = Path(args.data).expanduser().resolve()
    report = build_report(root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["exists"]:
        raise SystemExit(f"Dataset path does not exist: {root}")

    data_yaml = prepare_voc(root, Path(args.workdir) / "yolo_from_voc", args.copy_mode, args.val_ratio, args.test_ratio)
    train_yolo(
        data_yaml,
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        Path(args.project),
        args.name,
        args.device,
        args.workers,
        args.save_period,
        Path(args.model_output),
        args.cache,
        args.patience,
        args.close_mosaic,
    )
    save_model_artifacts(Path(args.project), args.name, Path(args.model_output), report)


if __name__ == "__main__":
    main()
