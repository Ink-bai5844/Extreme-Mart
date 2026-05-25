import json
import importlib
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

DEFAULT_MODEL_PATH = "/project/train/models/your_model/model.pt"
DEFAULT_CONF = 0.18
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 250
DEFAULT_MIN_LONG_SIDE = 8
CACHE_CLEAN_INTERVAL = 16
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


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _config_int(name, config_value, default):
    if os.getenv(name) is not None:
        return _int_env(name, default)
    return _coerce_int(config_value, default)


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
    import_errors_after_install = _module_import_errors(RUNTIME_MODULES)
    if import_errors_after_install:
        raise ModuleNotFoundError(
            "Missing Python dependencies after install: "
            + "; ".join(f"{name} -> {error}" for name, error in import_errors_after_install.items())
        )


def _candidate_model_paths():
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        yield Path(env_model)

    model_dir = Path(os.getenv("MODEL_DIR", "/project/train/models/your_model"))
    for name in ("model.pt", "best.pt", "last.pt"):
        yield model_dir / name

    for base in (
        Path("/project/train/models/your_model"),
        Path("/project/train/models"),
        Path("/project/train"),
        Path("/usr/local/ev_sdk"),
        Path("/project/ev_sdk"),
        Path(__file__).resolve().parents[1],
    ):
        for name in ("model.pt", "best.pt", "last.pt"):
            yield base / name


def _find_model_path():
    seen = set()
    for path in _candidate_model_paths():
        resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            return resolved
    checked = ", ".join(str(path) for path in list(_candidate_model_paths())[:8])
    raise FileNotFoundError(f"No trained safety-helmet model was found. Checked: {checked}")


def _load_infer_config(model_path):
    candidates = []
    env_config = os.getenv("INFER_CONFIG_PATH")
    if env_config:
        candidates.append(Path(env_config))
    candidates.extend(
        [
            model_path.parent / "infer_config.json",
            model_path.parent / "metadata.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ji.py] warning: failed to read inference config {path}: {exc}", flush=True)
            continue
        if path.name == "metadata.json":
            data = data.get("infer_config", {})
        if isinstance(data, dict):
            print(f"[ji.py] loaded inference config: {path}", flush=True)
            return data
    return {}


def _class_conf_from_config(infer_config, default_conf):
    raw = infer_config.get("class_conf", {}) if isinstance(infer_config, dict) else {}
    class_conf = {name: default_conf for name in ALL_CLASSES}
    if isinstance(raw, dict):
        for name, value in raw.items():
            normalized = _normalize_name(str(name))
            if normalized in class_conf:
                class_conf[normalized] = _coerce_float(value, default_conf)
    alert_conf = _coerce_float(infer_config.get("alert_conf"), class_conf.get("head", default_conf))
    class_conf["head"] = min(class_conf.get("head", default_conf), alert_conf)
    return class_conf, alert_conf


def init():
    """Initialize model."""
    _ensure_runtime_deps()
    from ultralytics import YOLO
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(_int_env("TORCH_NUM_THREADS", 1))

    model_path = _find_model_path()
    infer_config = _load_infer_config(model_path)
    conf = _float_env("CONF_THRES", _float_env("CONF_THRESHOLD", _coerce_float(infer_config.get("conf"), DEFAULT_CONF)))
    iou = _float_env("IOU_THRES", _float_env("IOU_THRESHOLD", _coerce_float(infer_config.get("iou"), DEFAULT_IOU)))
    if os.getenv("IMGSZ") is not None:
        imgsz = _int_env("IMGSZ", DEFAULT_IMGSZ)
    elif os.getenv("INFER_IMGSZ") is not None:
        imgsz = _int_env("INFER_IMGSZ", DEFAULT_IMGSZ)
    else:
        imgsz = _coerce_int(infer_config.get("imgsz"), DEFAULT_IMGSZ)
    max_det = _config_int("MAX_DET", infer_config.get("max_det"), DEFAULT_MAX_DET)
    min_long_side = _int_env("MIN_LONG_SIDE", _coerce_int(infer_config.get("min_long_side"), DEFAULT_MIN_LONG_SIDE))
    class_conf, alert_conf = _class_conf_from_config(infer_config, conf)

    model = YOLO(str(model_path))
    device = os.getenv("DEVICE") or ("0" if torch.cuda.is_available() else "cpu")
    try:
        model.fuse()
    except Exception as exc:
        print(f"[ji.py] warning: model fuse skipped: {exc}", flush=True)
    use_half = torch.cuda.is_available() and _bool_env("HALF", True)
    print(f"[ji.py] model loaded: {model_path} exists={model_path.exists()}", flush=True)
    print(
        f"[ji.py] conf={conf}, class_conf={class_conf}, alert_conf={alert_conf}, "
        f"iou={iou}, imgsz={imgsz}, max_det={max_det}, device={device}, half={use_half}",
        flush=True,
    )

    return {
        "model": model,
        "device": device,
        "conf": conf,
        "class_conf": class_conf,
        "alert_conf": alert_conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "min_long_side": min_long_side,
        "augment": _bool_env("AUGMENT", False),
        "half": use_half,
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

    if handle is None:
        handle = init()
    if input_image is None:
        output = {
            "algorithm_data": {"is_alert": False, "target_count": 0, "target_info": []},
            "model_data": {"objects": [], "object_data": []},
        }
        return json.dumps(output, separators=(",", ":"))

    model = handle["model"]
    conf = handle["conf"]
    iou = handle["iou"]
    imgsz = handle["imgsz"]
    max_det = handle["max_det"]
    min_long_side = handle.get("min_long_side", DEFAULT_MIN_LONG_SIDE)
    class_conf = handle.get("class_conf", {name: conf for name in ALL_CLASSES})
    alert_conf = float(handle.get("alert_conf", class_conf.get("head", conf)))

    if args:
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
            conf = float(parsed.get("conf_threshold", conf))
            iou = float(parsed.get("iou_threshold", iou))
            alert_conf = float(parsed.get("alert_conf", alert_conf))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    objects = []
    raw_alerts = []
    names = getattr(model, "names", {})
    try:
        with torch.inference_mode():
            results_iter = model.predict(
                source=input_image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                half=handle.get("half", False),
                device=handle.get("device"),
                augment=handle.get("augment", False),
                verbose=False,
                stream=True,
                batch=1,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show=False,
                plots=False,
            )

        for result in results_iter:
            if result.boxes is None:
                continue
            names = getattr(result, "names", names)
            data = result.boxes.data.detach().cpu().tolist()
            for row in data:
                if len(row) < 6:
                    continue
                x1, y1, x2, y2, confidence, class_id = row[:6]
                class_id = int(class_id)
                confidence = float(confidence)
                x, y, w, h = _clip_box(x1, y1, x2, y2, input_image.shape)
                name = _normalize_name(_class_name(names, class_id))
                if name not in ALL_CLASSES:
                    continue
                if max(w, h) < min_long_side:
                    continue
                obj = {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": confidence,
                    "name": name,
                }
                threshold = float(class_conf.get(name, conf))
                if name in ALERT_CLASSES and confidence >= alert_conf:
                    raw_alerts.append(obj)
                if name in ALERT_CLASSES:
                    threshold = min(threshold, alert_conf)
                if confidence >= threshold:
                    objects.append(obj)
            del data, result
    finally:
        handle["seen"] = handle.get("seen", 0) + 1
        if handle["seen"] % CACHE_CLEAN_INTERVAL == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    objects.sort(key=lambda item: item["confidence"], reverse=True)
    raw_alerts.sort(key=lambda item: item["confidence"], reverse=True)
    target_info = raw_alerts
    output = {
        "algorithm_data": {
            "is_alert": len(target_info) > 0,
            "target_count": len(target_info),
            "target_info": target_info,
        },
        "model_data": {
            "objects": objects,
            "object_data": objects,
        },
    }
    return json.dumps(output, separators=(",", ":"))
