# CropGuard-AI

CropGuard-AI is a YOLOv8-based computer-vision project for detecting Helicoverpa-Armigera and simulating a precision spraying decision. The project has one dataset class: `0 = Helicoverpa-Armigera`.

## Installation

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
```

## Dataset and models

The dataset is stored under `dataset/` with `train/`, `valid/`, `test/`, and `collected/` folders. `dataset/data.yaml` uses paths relative to the YAML file.

The pretrained YOLOv8 nano model is `models/pretrained/yolov8n.pt`. After training, place the best checkpoint at `models/trained/cropguard_yolov8n_best.pt` for evaluation, test prediction, and simulation.

## Commands

```powershell
python -m src.training.train
python -m src.training.evaluate
python -m src.training.test_model
python -m src.simulation.spray_simulation
```

Training, evaluation, prediction, and simulation outputs are stored under `results/`. Camera capture and preprocessing scripts remain under `src/camera/` and `src/preprocessing/`.

## Spray decision

The simulation uses a confidence threshold of `0.60`. A detection must meet or exceed that threshold for 3 consecutive frames before the overlay changes to `SPRAY`; otherwise it shows `DON'T SPRAY`. A missing or low-confidence detection resets the counter.

This is software simulation only. It does not activate GPIO, relays, pumps, pesticide systems, or other physical hardware.

## Structure

```text
CropGuard/
├── dataset/
├── models/
│   ├── pretrained/
│   └── trained/
├── src/
│   ├── camera/
│   ├── preprocessing/
│   ├── detection/
│   ├── training/
│   ├── decision/
│   └── simulation/
├── results/
└── tests/
```
