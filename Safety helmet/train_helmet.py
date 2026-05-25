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
DEFAULT_CONF_GRID = "0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40"
DEFAULT_INFER_CONF = 0.18
DEFAULT_INFER_IOU = 0.45
DEFAULT_INFER_IMGSZ = 640
DEFAULT_MAX_DET = 250
DEFAULT_MIN_LONG_SIDE = 8
HEAD_CLASS_NAME = "head"


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


def prepare_voc(
    root: Path,
    output_dir: Path,
    copy_mode: str,
    val_ratio: float,
    test_ratio: float,
    head_oversample: int,
) -> Path:
    if yaml is None:
        raise SystemExit("Missing PyYAML. Please run: pip install -r requirements.txt")

    image_index: dict[str, list[Path]] = defaultdict(list)
    for image_path in root.rglob("*"):
        if image_path.is_file() and is_image(image_path):
            image_index[image_path.name].append(image_path)

    class_to_idx = {name: idx for idx, name in enumerate(SCORE_CLASSES)}
    records: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []
    kept_objects = 0
    negative_images = 0

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
        else:
            negative_images += 1
        records.append((image_path, objects))

    if not records or kept_objects == 0:
        raise SystemExit("No usable VOC annotations were found for safety helmet classes.")

    reset_dir(output_dir)
    splits = split_items(records, val_ratio, test_ratio)
    head_class_id = class_to_idx[HEAD_CLASS_NAME]
    for split, split_records in splits.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        expanded_records: list[tuple[Path, list[tuple[int, float, float, float, float]], int]] = []
        for image_path, objects in split_records:
            repeat = 1
            if split == "train" and head_oversample > 1 and any(obj[0] == head_class_id for obj in objects):
                repeat = head_oversample
            for repeat_index in range(repeat):
                expanded_records.append((image_path, objects, repeat_index))

        for image_path, objects, repeat_index in expanded_records:
            if repeat_index:
                image_name = f"{image_path.stem}_headx{repeat_index}{image_path.suffix}"
                label_stem = f"{image_path.stem}_headx{repeat_index}"
            else:
                image_name = image_path.name
                label_stem = image_path.stem
            link_or_copy(image_path, output_dir / "images" / split / image_name, copy_mode)
            lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in objects]
            label_path = output_dir / "labels" / split / f"{label_stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

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
    split_counts = {split: len(split_records) for split, split_records in splits.items()}
    train_effective = len(list((output_dir / "images" / "train").glob("*")))
    print(
        "[prepare_voc] "
        f"images={len(records)}, positive_images={len(records) - negative_images}, "
        f"negative_images={negative_images}, objects={kept_objects}, splits={split_counts}, "
        f"head_oversample={head_oversample}, train_effective={train_effective}",
        flush=True,
    )
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

    def on_model_save(ultra_trainer: Any) -> None:
        sync_current_checkpoint(ultra_trainer, model_output)

    try:
        model.add_callback("on_model_save", on_model_save)
    except AttributeError:
        print("[checkpoint] warning: this ultralytics version does not support add_callback", flush=True)

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
        "optimizer": "auto",
        "cos_lr": True,
        "amp": True,
        "pretrained": True,
        "mosaic": 1.0,
        "mixup": 0.05,
        "degrees": 3.0,
        "translate": 0.10,
        "scale": 0.55,
        "fliplr": 0.50,
        "hsv_h": 0.015,
        "hsv_s": 0.65,
        "hsv_v": 0.35,
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


def sync_current_checkpoint(ultra_trainer: Any, output_dir: Path) -> None:
    """Keep the platform model directory populated during long platform runs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_attr(name: str) -> Path | None:
        value = getattr(ultra_trainer, name, None)
        return Path(value) if value else None

    last_pt = checkpoint_attr("last")
    best_pt = checkpoint_attr("best")
    save_dir = getattr(ultra_trainer, "save_dir", None)
    if save_dir:
        weights_dir = Path(save_dir) / "weights"
        if last_pt is None or not last_pt.exists():
            last_pt = weights_dir / "last.pt"
        if best_pt is None or not best_pt.exists():
            best_pt = weights_dir / "best.pt"

    source_pt = best_pt if best_pt is not None and best_pt.exists() else last_pt
    if source_pt is None or not source_pt.exists() or not source_pt.is_file():
        return

    shutil.copy2(source_pt, output_dir / "model.pt")
    if best_pt is not None and best_pt.exists() and best_pt.is_file():
        shutil.copy2(best_pt, output_dir / "best.pt")
    if last_pt is not None and last_pt.exists() and last_pt.is_file():
        shutil.copy2(last_pt, output_dir / "last.pt")
    epoch = getattr(ultra_trainer, "epoch", None)
    epoch_text = f"epoch {int(epoch) + 1}" if isinstance(epoch, int) else "current epoch"
    print(f"[checkpoint] synced {epoch_text} checkpoint to: {output_dir / 'model.pt'}", flush=True)


def find_final_checkpoint(project: Path, name: str) -> Path:
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
    return source_pt


def parse_conf_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        try:
            value = float(item.strip())
        except ValueError:
            continue
        if 0.0 < value < 1.0:
            values.append(value)
    return sorted(set(values)) or [DEFAULT_INFER_CONF]


def yolo_label_to_xyxy(label_path: Path, image_path: Path) -> list[dict[str, Any]]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size

    boxes: list[dict[str, Any]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, bw, bh = (float(value) for value in parts[1:])
        except ValueError:
            continue
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append({"class_id": class_id, "box": (x1, y1, x2, y2), "matched": False})
    return boxes


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_predictions(
    predictions_by_image: dict[str, list[dict[str, Any]]],
    ground_truth_by_image: dict[str, list[dict[str, Any]]],
    thresholds_by_class: dict[int, float],
    match_iou: float = 0.5,
    class_id: int | None = None,
) -> dict[str, float]:
    tp = fp = fn = 0
    for image_key, gt_boxes in ground_truth_by_image.items():
        gt = [
            dict(item, matched=False)
            for item in gt_boxes
            if class_id is None or item["class_id"] == class_id
        ]
        preds = []
        for pred in predictions_by_image.get(image_key, []):
            if class_id is not None and pred["class_id"] != class_id:
                continue
            threshold = thresholds_by_class.get(pred["class_id"], DEFAULT_INFER_CONF)
            if pred["confidence"] >= threshold:
                preds.append(pred)
        preds.sort(key=lambda item: item["confidence"], reverse=True)
        for pred in preds:
            best_index = -1
            best_iou = 0.0
            for index, gt_item in enumerate(gt):
                if gt_item["matched"] or gt_item["class_id"] != pred["class_id"]:
                    continue
                current_iou = box_iou(pred["box"], gt_item["box"])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = index
            if best_index >= 0 and best_iou >= match_iou:
                gt[best_index]["matched"] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for item in gt if not item["matched"])

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def default_infer_config(imgsz: int, conf: float = DEFAULT_INFER_CONF) -> dict[str, Any]:
    class_conf = {name: conf for name in SCORE_CLASSES}
    return {
        "conf": conf,
        "class_conf": class_conf,
        "alert_conf": class_conf[HEAD_CLASS_NAME],
        "iou": DEFAULT_INFER_IOU,
        "imgsz": imgsz or DEFAULT_INFER_IMGSZ,
        "max_det": DEFAULT_MAX_DET,
        "min_long_side": DEFAULT_MIN_LONG_SIDE,
        "class_names": SCORE_CLASSES,
        "alert_classes": [HEAD_CLASS_NAME],
    }


def calibrate_infer_config(
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    device: str | None,
    conf_grid: list[float],
) -> dict[str, Any]:
    val_dir = data_yaml.parent / "images" / "val"
    image_paths = sorted(path for path in val_dir.iterdir() if path.is_file() and is_image(path))
    if not image_paths:
        print("[calibrate] no validation images found; using default inference config", flush=True)
        return default_infer_config(imgsz)

    ground_truth_by_image: dict[str, list[dict[str, Any]]] = {}
    for image_path in image_paths:
        label_path = data_yaml.parent / "labels" / "val" / f"{image_path.stem}.txt"
        ground_truth_by_image[str(image_path.resolve())] = yolo_label_to_xyxy(label_path, image_path)

    YOLO = require_ultralytics()
    model = YOLO(str(model_path))
    predict_kwargs: dict[str, Any] = {
        "source": [str(path) for path in image_paths],
        "imgsz": imgsz,
        "conf": min(conf_grid),
        "iou": DEFAULT_INFER_IOU,
        "max_det": DEFAULT_MAX_DET,
        "verbose": False,
        "stream": True,
    }
    if device:
        predict_kwargs["device"] = device

    predictions_by_image: dict[str, list[dict[str, Any]]] = {key: [] for key in ground_truth_by_image}
    for result in model.predict(**predict_kwargs):
        image_key = str(Path(result.path).resolve())
        preds: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes.cpu():
                class_id = int(box.cls[0].item())
                if class_id < 0 or class_id >= len(SCORE_CLASSES):
                    continue
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append(
                    {
                        "class_id": class_id,
                        "confidence": confidence,
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                    }
                )
        predictions_by_image[image_key] = preds

    global_scores = []
    for threshold in conf_grid:
        thresholds = {class_id: threshold for class_id in range(len(SCORE_CLASSES))}
        global_scores.append((threshold, score_predictions(predictions_by_image, ground_truth_by_image, thresholds)))
    best_global_threshold, best_global_score = max(
        global_scores,
        key=lambda item: (item[1]["f1"], item[1]["precision"], item[1]["recall"], item[0]),
    )
    if best_global_score["f1"] <= 0.0:
        print("[calibrate] validation scan found no positive F1; using default inference config", flush=True)
        return default_infer_config(imgsz)

    class_thresholds: dict[int, float] = {}
    class_scores: dict[str, dict[str, float]] = {}
    for class_id, class_name in enumerate(SCORE_CLASSES):
        per_class_scores = []
        for threshold in conf_grid:
            per_class_scores.append(
                (
                    threshold,
                    score_predictions(
                        predictions_by_image,
                        ground_truth_by_image,
                        {class_id: threshold},
                        class_id=class_id,
                    ),
                )
            )
        best_threshold, best_score = max(
            per_class_scores,
            key=lambda item: (item[1]["f1"], item[1]["precision"], item[1]["recall"], item[0]),
        )
        class_thresholds[class_id] = best_threshold if best_score["f1"] > 0.0 else best_global_threshold
        class_scores[class_name] = best_score

    head_class_id = SCORE_CLASSES.index(HEAD_CLASS_NAME)
    class_conf = {SCORE_CLASSES[class_id]: threshold for class_id, threshold in class_thresholds.items()}
    alert_conf = min(class_conf[HEAD_CLASS_NAME], class_thresholds[head_class_id])
    class_conf[HEAD_CLASS_NAME] = alert_conf
    class_aware_score = score_predictions(predictions_by_image, ground_truth_by_image, class_thresholds)
    base_conf = min([best_global_threshold, alert_conf, *class_conf.values(), min(conf_grid)])
    config = default_infer_config(imgsz, base_conf)
    config.update(
        {
            "conf": base_conf,
            "class_conf": class_conf,
            "alert_conf": alert_conf,
            "calibration": {
                "source": "validation_class_conf_scan",
                "match_iou": 0.5,
                "thresholds": conf_grid,
                "global_threshold": best_global_threshold,
                "global_best": best_global_score,
                "class_best": class_scores,
                "class_aware_score": class_aware_score,
            },
        }
    )
    print(
        "[calibrate] selected "
        f"base_conf={base_conf:.3f}, class_conf={class_conf}, "
        f"val f1={class_aware_score['f1']:.4f}, precision={class_aware_score['precision']:.4f}, "
        f"recall={class_aware_score['recall']:.4f}",
        flush=True,
    )
    return config


def save_model_artifacts(
    project: Path,
    name: str,
    output_dir: Path,
    report: dict,
    infer_config: dict[str, Any],
    train_config: dict[str, Any],
) -> None:
    run_dir = project / name
    if not run_dir.exists():
        candidates = sorted(project.glob(f"{name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            run_dir = candidates[0]

    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    source_pt = find_final_checkpoint(project, name)

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
        "infer_config": infer_config,
        "train_config": train_config,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "infer_config.json").write_text(json.dumps(infer_config, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--conf-grid", default=DEFAULT_CONF_GRID, help="Comma-separated thresholds to scan on validation data.")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip validation confidence calibration.")
    parser.add_argument("--head-oversample", type=int, default=2, help="Repeat training images containing head N times.")
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

    data_yaml = prepare_voc(
        root,
        Path(args.workdir) / "yolo_from_voc",
        args.copy_mode,
        args.val_ratio,
        args.test_ratio,
        max(1, args.head_oversample),
    )
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
    final_checkpoint = find_final_checkpoint(Path(args.project), args.name)
    infer_config = default_infer_config(args.imgsz)
    if not args.no_calibrate:
        try:
            infer_config = calibrate_infer_config(
                final_checkpoint,
                data_yaml,
                args.imgsz,
                args.device,
                parse_conf_grid(args.conf_grid),
            )
        except Exception as exc:
            print(f"[calibrate] warning: calibration failed, using defaults: {exc}", flush=True)
    train_config = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "head_oversample": max(1, args.head_oversample),
        "conf_grid": parse_conf_grid(args.conf_grid),
        "augment_profile": "helmet_small_head_v1",
    }
    save_model_artifacts(Path(args.project), args.name, Path(args.model_output), report, infer_config, train_config)


if __name__ == "__main__":
    main()
