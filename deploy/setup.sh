#!/bin/bash
# HiO v2 — One-step deployment
set -e

echo "=== HiO Intelligence Stream v2 Setup ==="

# 1. Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Download YOLO model
echo "Downloading YOLO pose model..."
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"

# 5. Pre-download CLIP model
echo "Downloading CLIP model..."
python -c "import clip; clip.load('ViT-L/14', device='cpu')"

# 6. Pre-download Qwen model
echo "Downloading Qwen2.5-VL-3B model (this may take a while)..."
python -c "
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype='auto')
print('Qwen model downloaded successfully')
"

# 7. Create .env from example if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it with your API keys"
fi

# 8. Create data directories
mkdir -p data/clips data/events data/thumbnails data/logs

echo ""
echo "=== Setup Complete ==="
echo "Start model server:    python main.py"
echo "Start frontend:        cd frontend_server && python main.py"
