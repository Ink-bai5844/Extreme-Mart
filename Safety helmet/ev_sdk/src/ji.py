import json
import importlib
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL_PATH = "/project/train/models/your_model/model.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 300
DEFAULT_MIN_LONG_SIDE = 10
CACHE_CLEAN_INTERVAL = 64
ALERT_CLASSES = {"head"}
ALL_CLASSES = {"person", "hat", "head"}
DEFAULT_DEPS_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
RUNTIME_MODULES = ("ultralytics", "cv2", "numpy", "PIL", "yaml")
RUNTIME_PACKAGE_BY_MODULE = {
    "ultralytics": "ultralytics==8.2.103",
    "cv2": "opencv-python-headless==4.10.0.84",
    "numpy": "numpy==1.26.4",
    "PIL": "Pillow==10.4.0",
    "yaml": "PyYAML==6.0.2",
}


def _float_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _module_import_errors(module_names):
    errors = {}
    for module_name in module_names:
        try:
            __import__(module_name)
        except Exception as exc:
            errors[module_name] = repr(exc)
    return errors


def _ensure_runtime_deps():
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    import_errors = _module_import_errors(RUNTIME_MODULES)
    if not import_errors:
        return
    if not _bool_env("SDK_INSTALL_DEPS", True):
        raise ModuleNotFoundError(
            "Missing Python dependencies: "
            + ", ".join(import_errors)
            + ". Set SDK_INSTALL_DEPS=1 or preinstall runtime packages."
        )

    packages = [RUNTIME_PACKAGE_BY_MODULE[name] for name in import_errors]
    index_url = os.getenv("DEPS_INDEX_URL", DEFAULT_DEPS_INDEX_URL)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-i",
        index_url,
    ] + packages
    print(f"[ji.py] installing: {' '.join(packages)}", flush=True)
    subprocess.check_call(cmd)
    importlib.invalidate_caches()


def init():
    """Initialize model."""
    _ensure_runtime_deps()
    from ultralytics import YOLO

    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not Path(model_path).exists():
        alt = Path("/project/train/models/your_model/best.pt")
        if alt.exists():
            model_path = str(alt)

    conf = _float_env("CONF_THRESHOLD", DEFAULT_CONF)
    iou = _float_env("IOU_THRESHOLD", DEFAULT_IOU)
    imgsz = _int_env("INFER_IMGSZ", DEFAULT_IMGSZ)
    max_det = _int_env("MAX_DET", DEFAULT_MAX_DET)

    model = YOLO(model_path)
    print(f"[ji.py] model loaded: {model_path}", flush=True)
    print(f"[ji.py] conf={conf}, iou={iou}, imgsz={imgsz}, max_det={max_det}", flush=True)

    return {
        "model": model,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "seen": 0,
    }


def _clip_box(x1, y1, x2, y2, shape):
    h, w = shape[:2]
    x = max(0, int(round(x1)))
    y = max(0, int(round(y1)))
    bw = max(1, min(int(round(x2 - x1)), w - x))
    bh = max(1, min(int(round(y2 - y1)), h - y))
    return x, y, bw, bh


def _normalize_name(name):
    name = name.strip().lower()
    aliases = {
        "helmet": "hat",
        "safety_helmet": "hat",
        "no_helmet": "head",
        "bare_head": "head",
    }
    return aliases.get(name, name)


def _class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def process_image(handle=None, input_image=None, args=None, **kwargs):
    """Do inference to analysis input_image and get output."""
    import gc
    import torch

    model = handle["model"]
    conf = handle["conf"]
    iou = handle["iou"]
    imgsz = handle["imgsz"]
    max_det = handle["max_det"]
    min_long_side = _int_env("MIN_LONG_SIDE", DEFAULT_MIN_LONG_SIDE)

    if args:
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
            conf = float(parsed.get("conf_threshold", conf))
            iou = float(parsed.get("iou_threshold", iou))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    objects = []
    try:
        results = model.predict(
            source=input_image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )
        names = results[0].names if results else {}
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = _clip_box(x1, y1, x2, y2, input_image.shape)
                name = _normalize_name(_class_name(names, class_id))
                if name not in ALL_CLASSES:
                    continue
                if max(w, h) < min_long_side:
                    continue
                objects.append(
                    {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "confidence": confidence,
                        "name": name,
                    }
                )
    finally:
        handle["seen"] = handle.get("seen", 0) + 1
        if handle["seen"] % CACHE_CLEAN_INTERVAL == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    objects.sort(key=lambda item: item["confidence"], reverse=True)
    target_info = [obj for obj in objects if obj["name"] in ALERT_CLASSES]
    output = {
        "algorithm_data": {
            "is_alert": len(target_info) > 0,
            "target_count": len(target_info),
            "target_info": target_info,
        },
        "model_data": {
            "objects": objects,
        },
    }
    return json.dumps(output, separators=(",", ":"))
