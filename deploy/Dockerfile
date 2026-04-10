FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

COPY . .

# Pre-download models at build time
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')" && \
    python3 -c "import clip; clip.load('ViT-L/14', device='cpu')"

EXPOSE 8000 8002

CMD ["python3", "main.py"]
