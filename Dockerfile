FROM python:3.12-slim

# System libraries that OpenCV needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch (the default CUDA build is several GB)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user . .

# Fail the BUILD (not the running app) if the trained model is missing or broken
RUN python -c "from ultralytics import YOLO; m = YOLO('models/trained/best.pt'); print('model ok:', m.names)"

EXPOSE 7860
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
