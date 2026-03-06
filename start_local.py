"""
Start all 3 servers locally (Model Server, DB Server, Frontend).

Usage:
    python start_local.py          → starts all 3
    python start_local.py model    → only model server
    python start_local.py db       → only db server
    python start_local.py frontend → only frontend server
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = str(BASE_DIR / "venv" / "Scripts" / "python.exe")

if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = str(BASE_DIR / "venv" / "bin" / "python")

SERVERS = {
    "model": {
        "name": "Model Server",
        "module": "model_server.main:app",
        "port": int(os.getenv("MODEL_SERVER_PORT", 8000)),
        "color": "\033[36m",  # cyan
        "reload_dir": "model_server",
    },
    "db": {
        "name": "DB Server",
        "module": "db_server.main:app",
        "port": int(os.getenv("DB_SERVER_PORT", 8001)),
        "color": "\033[33m",  # yellow
        "reload_dir": "db_server",
    },
    "frontend": {
        "name": "Frontend Server",
        "module": "frontend_server.main:app",
        "port": int(os.getenv("FRONTEND_SERVER_PORT", 8002)),
        "color": "\033[32m",  # green
        "reload_dir": "frontend_server",
    },
}


def start_server(key: str, config: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        VENV_PYTHON, "-m", "uvicorn",
        config["module"],
        "--host", "0.0.0.0",
        "--port", str(config["port"]),
        "--reload",
        "--reload-dir", config.get("reload_dir", key),
    ]

    print(f"{config['color']}[{config['name']}] Starting on port {config['port']}...\033[0m")

    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
    )
    return proc


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(SERVERS.keys())

    print("=" * 60)
    print("  Intelligent CCTV System - Local Launcher")
    print("=" * 60)

    procs = {}
    for key in targets:
        if key not in SERVERS:
            print(f"Unknown server: {key}. Choose from: {list(SERVERS.keys())}")
            continue
        procs[key] = start_server(key, SERVERS[key])
        time.sleep(0.5)

    if not procs:
        print("No servers started.")
        return

    print()
    print("=" * 60)
    for key, config in SERVERS.items():
        if key in procs:
            print(f"  {config['name']:20s} → http://localhost:{config['port']}")
    print()
    print("  Dashboard → http://localhost:8002/dashboard")
    print("  API Docs  → http://localhost:8000/docs (model)")
    print("              http://localhost:8001/docs (db)")
    print("=" * 60)
    print("  Press Ctrl+C to stop all servers")
    print("=" * 60)

    try:
        for proc in procs.values():
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for key, proc in procs.items():
            print(f"  Stopping {SERVERS[key]['name']}...")
            proc.terminate()
        for key, proc in procs.items():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  Force killing {SERVERS[key]['name']}...")
                proc.kill()
                proc.wait(timeout=3)
        print("All servers stopped.")


if __name__ == "__main__":
    main()
