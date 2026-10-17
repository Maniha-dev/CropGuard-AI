# CropGuard-AI

CropGuard-AI is a YOLOv8-based computer-vision project for detecting Helicoverpa-Armigera and simulating a precision spraying decision. The project has one dataset class: `0 = Helicoverpa-Armigera`.

## Installation

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
```

## Dataset and models

The dataset is stored under `dataset/` with `train/`, `valid/`, `test/`, and `collected/` folders. `dataset/data.yaml` uses paths relative to the YAML file.

The pretrained YOLOv8 nano model is `models/pretrained/yolov8n.pt`. The trained checkpoint is `models/trained/best.pt` and is used for evaluation, test prediction, simulation, and the Streamlit app.

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

The snapshot Streamlit app is kept in `app.py`. The live browser-camera app is `app_live.py`; it runs model inference every third video frame.

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Select **Create app**.
4. Choose `Maniha-dev/CropGuard-AI`, branch `main`, and main file path `app_live.py`.
5. In **Advanced settings**, select Python `3.12`.
6. Select **Deploy**.

The first load can be slow because Streamlit Cloud needs to install PyTorch and load the model. The free tier has limited RAM, which may constrain model loading and image processing.

The live camera tries Google STUN and the public Metered OpenRelay demo TURN server. That public relay is best-effort and may be slow, rate-limited, or unavailable. For a dependable deployment, get TURN URLs and temporary credentials from a TURN provider, then add them under **App settings > Secrets** in Streamlit Cloud. Do not commit provider credentials. Use this format:

```toml
[turn]
urls = ["turn:your-provider.example:3478?transport=udp", "turns:your-provider.example:443?transport=tcp"]
username = "provider-issued-username"
credential = "provider-issued-credential"
```

The app keeps STUN enabled and prefers your configured TURN server when all three values are valid. TURN credentials are passed to the browser as part of WebRTC, so use provider-issued temporary/session credentials, not a permanent account password or API key. Without working TURN credentials, camera access may still fail on restrictive networks. The Linux deployment installs headless OpenCV; Windows installs regular OpenCV so local camera-window tools remain available.

Run locally from the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
streamlit run app_live.py
python -m pytest tests -q
```

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
