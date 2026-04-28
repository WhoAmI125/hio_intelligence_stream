# HIO v3 — AWS g4dn.2xlarge Spot Deployment

4 camera deployment guide for Ubuntu 22.04 on AWS EC2 g4dn.2xlarge spot
(T4 15GB + 8 vCPU + 32GB) with NVIDIA driver + CUDA 12.1.

## One-time Setup

1. Attach a persistent gp3 EBS volume (500GB+) at `/srv/hio-data` for `data/`.
2. Clone the repo to `/opt/hio_intelligence_stream_v3`.
3. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` etc.
4. Create venv and install deps:
   ```bash
   cd /opt/hio_intelligence_stream_v3
   python3 -m venv .venv
   ./.venv/bin/pip install --upgrade pip
   ./.venv/bin/pip install -r requirements_gpu.txt -r requirements.txt
   ./.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
5. Pre-download HF models to warm local cache:
   ```bash
   ./.venv/bin/python - <<'PY'
   from transformers import AutoImageProcessor, SiglipForImageClassification, AutoModel
   for m in [
       "google/siglip-base-patch16-224",
       "prithivMLmods/Fire-Detection-Siglip2",
       "prithivMLmods/Human-Action-Recognition",
   ]:
       AutoImageProcessor.from_pretrained(m)
       try:
           SiglipForImageClassification.from_pretrained(m)
       except Exception:
           AutoModel.from_pretrained(m)
       print("OK", m)
   PY
   ```
6. Symlink `data/` to EBS mount:
   ```bash
   mkdir -p /srv/hio-data
   ln -s /srv/hio-data /opt/hio_intelligence_stream_v3/data
   mkdir -p /var/log/hio && chown ubuntu:ubuntu /var/log/hio
   ```

## Systemd Units

```bash
sudo cp deploy/vlm-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vlm-db vlm-model vlm-frontend vlm-spot-watcher
```

Logs:

```bash
tail -f /var/log/hio/*.log
journalctl -u vlm-model -f
```

## NVDEC Verification

After start, confirm NVDEC is active:

```bash
nvidia-smi --query-gpu=utilization.decoder --format=csv -l 1
```

Decoder util should be > 0% when cameras stream. If 0%, FFmpeg was not built
with `--enable-cuda --enable-cuvid` or `RTSP_HWACCEL_DECODER` is unsupported.

## Spot Termination Behavior

`vlm-spot-watcher.service` polls `169.254.169.254/latest/meta-data/spot/instance-action`
every 5s. On interruption notice it stops `vlm-model / vlm-db / vlm-frontend`
(via systemd) to trigger graceful SIGTERM with `TimeoutStopSec=120`. Services
have `Restart=always` so when EC2 replaces the spot node, systemd auto-starts
them after boot. `AUTO_RESTORE_CAMERAS_ON_BOOT=true` brings cameras back
automatically.

## Rollback

```bash
sudo systemctl stop vlm-model vlm-db vlm-frontend vlm-spot-watcher
cd /opt/hio_intelligence_stream_v3
git reset --hard v3-prod-4cam-baseline-pre
sudo systemctl start vlm-db vlm-model vlm-frontend vlm-spot-watcher
```

`.env` is not tracked by git, so rollback preserves the operator config.
