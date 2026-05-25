param(
    [string]$DataDir = "",
    [int]$Epochs = 3,
    [int]$ImgSz = 640,
    [int]$Batch = 4,
    [int]$SavePeriod = 1,
    [string]$Device = "",
    [string]$Model = "yolov8n.pt"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $DataDir) {
    $DataDir = Join-Path $ScriptDir "831"
}

$DataDir = (Resolve-Path $DataDir).Path
$LocalTrainDir = Join-Path $ScriptDir "local_train"
$env:YOLO_CONFIG_DIR = Join-Path $LocalTrainDir "ultralytics_config"
$WorkDir = Join-Path $LocalTrainDir "work\helmet_detection_dataset"
$ProjectDir = Join-Path $LocalTrainDir "runs"
$ModelOutput = Join-Path $LocalTrainDir "models\your_model"
$Wheelhouse = Join-Path $ScriptDir "wheelhouse"

Set-Location $ScriptDir

if ($env:USE_WHEELHOUSE -eq "1" -and (Test-Path $Wheelhouse)) {
    $WheelFiles = Get-ChildItem -Path $Wheelhouse -File -Include *.whl,*.tar.gz
    if ($WheelFiles.Count -gt 0) {
        python -m pip install --no-index --find-links $Wheelhouse --no-deps -r requirements-offline.txt
    } else {
        python -m pip install -r requirements.txt
    }
} else {
    python -m pip install -r requirements.txt
}
python dataset_probe.py --data $DataDir --output dataset_report_local.json

$TrainArgs = @(
    "train_helmet.py",
    "--data", $DataDir,
    "--epochs", "$Epochs",
    "--imgsz", "$ImgSz",
    "--batch", "$Batch",
    "--workers", "0",
    "--save-period", "$SavePeriod",
    "--copy-mode", "copy",
    "--workdir", $WorkDir,
    "--project", $ProjectDir,
    "--name", "helmet_detection_local",
    "--model", $Model,
    "--model-output", $ModelOutput
)

if ($Device) {
    $TrainArgs += @("--device", $Device)
}

python @TrainArgs
