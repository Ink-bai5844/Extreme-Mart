#!/usr/bin/env python3
"""Probe the safety-helmet VOC dataset."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCORE_CLASSES = ["person", "hat", "head"]
CLASS_ALIASES = {
    "helmet": "hat",
    "safety_helmet": "hat",
    "safetyhelmet": "hat",
    "no_helmet": "head",
    "nohelmet": "head",
    "bare_head": "head",
    "barehead": "head",
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def normalize_class(name: str) -> str:
    clean = name.strip().lower()
    return CLASS_ALIASES.get(clean, clean)


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def probe_voc(root: Path) -> dict[str, Any]:
    xml_files = sorted(root.rglob("*.xml")) if root.exists() else []
    objects: Counter[str] = Counter()
    examples: list[str] = []
    parsed = 0

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        object_nodes = tree.getroot().findall("object")
        if not object_nodes:
            continue
        parsed += 1
        for obj in object_nodes:
            raw_name = obj.findtext("name") or ""
            name = normalize_class(raw_name)
            if name:
                objects[name] += 1
        if len(examples) < 10:
            examples.append(safe_rel(xml_path, root))

    return {
        "xml_files": len(xml_files),
        "voc_xml_files": parsed,
        "objects": dict(objects),
        "score_class_counts": {name: objects.get(name, 0) for name in SCORE_CLASSES},
        "examples": examples,
    }


def build_report(dataset_root: Path) -> dict[str, Any]:
    images = sorted([p for p in dataset_root.rglob("*") if p.is_file() and is_image(p)]) if dataset_root.exists() else []
    voc = probe_voc(dataset_root)
    warnings: list[str] = []
    if not dataset_root.exists():
        warnings.append(f"Dataset path does not exist: {dataset_root}")
    if not images:
        warnings.append("No image files were found.")
    if voc["voc_xml_files"] < max(1, int(len(images) * 0.5)):
        warnings.append("VOC XML annotations are missing for many images.")
    missing_score = [name for name, count in voc["score_class_counts"].items() if count == 0]
    if missing_score:
        warnings.append(f"No samples found for score classes: {missing_score}")

    return {
        "dataset_root": str(dataset_root),
        "exists": dataset_root.exists(),
        "image_count": len(images),
        "image_extensions": dict(Counter(p.suffix.lower() for p in images)),
        "sample_images": [safe_rel(p, dataset_root) for p in images[:20]],
        "formats": {"voc": voc},
        "score_classes": SCORE_CLASSES,
        "recommended_task": "detection_voc" if images and voc["voc_xml_files"] else "missing_or_empty",
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe safety-helmet data.")
    parser.add_argument("--data", default="/home/data/831", help="Dataset root")
    parser.add_argument("--output", default="dataset_report.json", help="Where to save the JSON report")
    args = parser.parse_args()

    report = build_report(Path(args.data).expanduser().resolve())
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\nReport saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
