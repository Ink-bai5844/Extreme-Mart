# Safety Helmet Detection (安全帽识别)

基于 YOLOv8 的安全帽佩戴检测算法，用于工地/工厂安全生产场景。训练流程会自动把 VOC 标注转换为 YOLO 数据集，并在训练后用验证集校准推理阈值，生成 `infer_config.json` 给平台 SDK 直接读取。

## 算法说明

- **目标**：识别人员是否佩戴安全帽（红色、黄色、蓝色、白色、橘色）
- **检测类别**：`person`（人）、`hat`（佩戴安全帽的头）、`head`（未佩戴安全帽的头）
- **报警逻辑**：检测到 `head`（未佩戴安全帽）时触发报警

## 目录结构

```
Safety helmet/
├── 831/                        # 样例数据集 (VOC格式: .jpg + .xml)
├── dataset_probe.py            # 数据集探测工具
├── train_helmet.py             # 训练脚本
├── ev_sdk/src/ji.py            # 平台推理接口
├── start_train.sh              # Linux 训练入口
├── start_local_train.ps1       # Windows 本地训练脚本
├── requirements.txt            # 在线依赖
├── requirements-offline.txt    # 离线依赖
├── yolov8n.pt                  # 预训练权重 (需自行放置)
└── README.md
```

## 本地训练

```powershell
# Windows
.\start_local_train.ps1 -Epochs 3 -Batch 4 -ImgSz 640
```

## 平台训练

训练代码位于 `/project/train/src_repo/`，平台执行：
```bash
bash start_train.sh
```

模型保存至 `/project/train/models/your_model/model.pt`，并同步输出：

- `best.pt` / `last.pt`：YOLO 原始权重备份
- `metadata.json`：训练配置、类别、报警类别
- `infer_config.json`：验证集自动选择的 `conf`、`class_conf`、`alert_conf`、`imgsz`、`max_det`

常用覆盖参数：

```bash
EPOCHS=180 IMGSZ=640 BATCH=8 bash start_train.sh
CONF_GRID=0.08,0.10,0.12,0.15,0.20,0.25 bash start_train.sh
HEAD_OVERSAMPLE=3 bash start_train.sh
NO_CALIBRATE=1 bash start_train.sh
INSTALL_DEPS=0 bash start_train.sh   # 依赖已预装时跳过安装
```

默认会优先使用 `wheelhouse/` 的离线包；没有离线包时默认从清华源安装 `requirements.txt`。如需换源可设置 `DEPS_INDEX_URL`。

## 平台测试

测试代码位于 `/project/ev_sdk/src/ji.py`，实现接口：
- `init()` — 加载模型
- `process_image(handle, input_image, args)` — 推理并返回 JSON 结果

`ji.py` 会优先查找 `MODEL_PATH`，然后查找 `/project/train/models/your_model/{model.pt,best.pt,last.pt}`，并自动读取同目录的 `infer_config.json` 或 `metadata.json`。GPU 环境下默认启用 `model.fuse()`、半精度推理和 `stream=True`，用于提升 FPS；如需调试精度可用 `CONF_THRES`、`IOU_THRES`、`IMGSZ`、`MAX_DET`、`MIN_LONG_SIDE` 覆盖。

## 输出格式

```json
{
    "algorithm_data": {
        "is_alert": true,
        "target_count": 1,
        "target_info": [
            {
                "x": 693,
                "y": 733,
                "width": 51,
                "height": 35,
                "confidence": 0.954,
                "name": "head"
            }
        ]
    },
    "model_data": {
        "objects": [
            {
                "x": 693,
                "y": 733,
                "width": 51,
                "height": 35,
                "confidence": 0.954,
                "name": "head"
            }
        ],
        "object_data": [
            {
                "x": 693,
                "y": 733,
                "width": 51,
                "height": 35,
                "confidence": 0.954,
                "name": "head"
            }
        ]
    }
}
```

## 评分标准

| 指标 | 说明 | 权重 |
|------|------|------|
| 算法精度 | F1 Score = 2PR/(P+R) | 0.7 |
| 算法性能 | fps/100 | 0.3 |
