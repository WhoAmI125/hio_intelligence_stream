"""
Frontend Server — FastAPI entry point with Jinja2 templates.

Serves the monitoring UI and proxies API calls to model_server / db_server.

Endpoints:
    GET /                    → redirect to /monitor/adhoc
    GET /monitor/adhoc       → real-time CCTV monitoring UI
    GET /monitor/shadow      → shadow agent review UI
    GET /api/proxy/cameras   → proxy to db_server for camera configs
    GET /api/proxy/events    → proxy to db_server for event list
    GET /api/proxy/stats     → proxy to db_server for stats
    GET /api/proxy/status    → proxy to model_server for status
"""

import logging
import os
import sys
import json
import subprocess
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    import psutil
except Exception:
    psutil = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("frontend_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")
DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8001")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
VLM_TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, "vlm_pipeline")


def _mask_rtsp_url(rtsp_url: str) -> str:
    """Mask credentials in RTSP URL for safe logging."""
    raw = str(rtsp_url or "").strip()
    if not raw:
        return raw
    try:
        parsed = urlparse(raw)
        if not parsed.hostname:
            return raw
        auth = ""
        if parsed.username:
            auth = parsed.username
            if parsed.password is not None:
                auth += ":***"
            auth += "@"
        host = parsed.hostname
        if parsed.port:
            host += f":{parsed.port}"
        netloc = f"{auth}{host}"
        return parsed._replace(netloc=netloc).geturl()
    except Exception:
        return raw


def _validate_rtsp_url(rtsp_url: str) -> tuple[bool, str]:
    raw = str(rtsp_url or "").strip()
    if not raw:
        return False, "rtsp_url is required"
    try:
        parsed = urlparse(raw)
    except Exception:
        return False, "Invalid RTSP URL format"
    if (parsed.scheme or "").lower() not in {"rtsp", "rtsps"}:
        return False, "RTSP URL must start with rtsp:// or rtsps://"
    if not parsed.hostname:
        return False, "RTSP URL must include a valid host"
    return True, ""


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Frontend Server ready.")
    logger.info(f"  Model Server: {MODEL_SERVER_URL}")
    logger.info(f"  DB Server:    {DB_SERVER_URL}")
    yield
    logger.info("Frontend Server stopped.")


app = FastAPI(
    title="Intelligent CCTV Frontend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates — directory set to 'templates/' so {% extends "vlm_pipeline/base_public.html" %} works
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# ---------------------------------------------------------------------------
# Page endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/monitor/adhoc")


@app.get("/monitor/adhoc", response_class=HTMLResponse)
async def adhoc_monitor(request: Request):
    """Real-time CCTV monitoring UI — rendered via Jinja2."""
    try:
        return templates.TemplateResponse(
            "vlm_pipeline/adhoc_rtsp.html",
            {"request": request, "active_page": "cctv"},
        )
    except Exception as e:
        logger.error(f"Template render error: {e}")
        return HTMLResponse(
            content=f"<h1>Template error</h1><pre>{e}</pre>",
            status_code=500,
        )


@app.get("/monitor/shadow", response_class=HTMLResponse)
async def shadow_monitor(request: Request):
    """Shadow agent review UI — rendered via Jinja2."""
    try:
        return templates.TemplateResponse(
            "vlm_pipeline/monitor_shadow.html",
            {"request": request, "active_page": "shadow"},
        )
    except Exception as e:
        logger.error(f"Template render error: {e}")
        return HTMLResponse(
            content=f"<h1>Template error</h1><pre>{e}</pre>",
            status_code=500,
        )


@app.get("/monitor/gemini-logs", response_class=HTMLResponse)
async def gemini_logs_monitor(request: Request):
    """Gemini validation log UI rendered via Jinja2."""
    try:
        return templates.TemplateResponse(
            "vlm_pipeline/gemini_logs.html",
            {"request": request, "active_page": "gemini"},
        )
    except Exception as e:
        logger.error(f"Template render error: {e}")
        return HTMLResponse(
            content=f"<h1>Template error</h1><pre>{e}</pre>",
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Dashboard (simple status page when templates aren't available)
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple dashboard that shows system status."""
    html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent CCTV Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 20px; }
        h1 { color: #00d4ff; margin-bottom: 20px; font-size: 1.8em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
        .card { background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px; }
        .card h2 { color: #00d4ff; font-size: 1.1em; margin-bottom: 12px; }
        .stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; }
        .stat:last-child { border-bottom: none; }
        .label { color: #888; }
        .value { color: #00ff88; font-weight: bold; }
        .error { color: #ff4444; }
        .loading { color: #888; }
        #refresh-btn { background: #00d4ff; color: #0a0a1a; border: none; padding: 8px 20px;
                       border-radius: 6px; cursor: pointer; font-weight: bold; margin-bottom: 16px; }
        #refresh-btn:hover { background: #00b4dd; }
    </style>
</head>
<body>
    <h1>Intelligent CCTV System</h1>
    <button id="refresh-btn" onclick="loadAll()">Refresh</button>
    <div class="grid">
        <div class="card" id="model-card">
            <h2>Model Server</h2>
            <div class="loading">Loading...</div>
        </div>
        <div class="card" id="db-card">
            <h2>DB Server</h2>
            <div class="loading">Loading...</div>
        </div>
        <div class="card" id="events-card">
            <h2>Recent Events</h2>
            <div class="loading">Loading...</div>
        </div>
        <div class="card" id="system-card">
            <h2>System</h2>
            <div class="loading">Loading...</div>
        </div>
    </div>
    <script>
        async function loadAll() {
            // Model Server status
            try {
                const r = await fetch('/api/proxy/status');
                const d = await r.json();
                if (d.error) throw new Error(d.error);
                const streams = Object.keys(d.streams || {});
                const agents = (d.agents_loaded || []).join(', ');
                document.getElementById('model-card').innerHTML = `
                    <h2>Model Server</h2>
                    <div class="stat"><span class="label">Florence</span><span class="value">${d.florence_loaded ? 'Loaded' : 'Not loaded'}</span></div>
                    <div class="stat"><span class="label">Agents</span><span class="value">${agents}</span></div>
                    <div class="stat"><span class="label">Active Streams</span><span class="value">${streams.length}</span></div>
                    <div class="stat"><span class="label">Shadow Agents</span><span class="value">${Object.keys(d.shadow_agents || {}).length}</span></div>
                `;
            } catch(e) {
                document.getElementById('model-card').innerHTML = '<h2>Model Server</h2><div class="error">Offline: ' + e.message + '</div>';
            }

            // DB Server stats
            try {
                const r = await fetch('/api/proxy/stats');
                const d = await r.json();
                if (d.error) throw new Error(d.error);
                document.getElementById('db-card').innerHTML = `
                    <h2>DB Server</h2>
                    <div class="stat"><span class="label">Total Events</span><span class="value">${d.total_events}</span></div>
                    <div class="stat"><span class="label">Detected</span><span class="value">${d.total_detected}</span></div>
                    ${(d.by_scenario || []).map(s =>
                        `<div class="stat"><span class="label">${s.scenario || 'unknown'}</span><span class="value">${s.cnt} (avg ${(s.avg_conf||0).toFixed(2)})</span></div>`
                    ).join('')}
                `;
            } catch(e) {
                document.getElementById('db-card').innerHTML = '<h2>DB Server</h2><div class="error">Offline: ' + e.message + '</div>';
            }

            // Recent events
            try {
                const r = await fetch('/api/proxy/events?per_page=5');
                const d = await r.json();
                if (d.error) throw new Error(d.error);
                const rows = (d.events || []).map(e =>
                    `<div class="stat"><span class="label">${e.scenario} (tier ${e.tier})</span><span class="value">${(e.confidence||0).toFixed(2)} @ ${e.created_at || ''}</span></div>`
                ).join('');
                document.getElementById('events-card').innerHTML = `<h2>Recent Events (${d.total})</h2>` + (rows || '<div class="loading">No events yet</div>');
            } catch(e) {
                document.getElementById('events-card').innerHTML = '<h2>Recent Events</h2><div class="error">Cannot load</div>';
            }

            // System metrics (CPU/RAM/GPU/VRAM)
            try {
                const r = await fetch('/api/proxy/system');
                const d = await r.json();
                if (d.error) throw new Error(d.error);
                const cpu = d.cpu_percent ?? 0;
                const ram = d.ram || {};
                const gpu = d.gpu || {};
                const vramPct = (gpu.vram_percent ?? 0).toFixed(1);
                const gpuUtil = (gpu.utilization_percent ?? 0).toFixed(1);
                const ramPct = (ram.percent ?? 0).toFixed(1);

                document.getElementById('system-card').innerHTML = `
                    <h2>System</h2>
                    <div class="stat"><span class="label">CPU</span><span class="value">${cpu.toFixed(1)}%</span></div>
                    <div class="stat"><span class="label">RAM</span><span class="value">${ram.used_gb ?? 0} / ${ram.total_gb ?? 0} GB (${ramPct}%)</span></div>
                    <div class="stat"><span class="label">GPU</span><span class="value">${gpu.name || 'N/A'} (${gpuUtil}%)</span></div>
                    <div class="stat"><span class="label">VRAM</span><span class="value">${gpu.vram_used_mb ?? 0} / ${gpu.vram_total_mb ?? 0} MB (${vramPct}%)</span></div>
                `;
            } catch(e) {
                document.getElementById('system-card').innerHTML = '<h2>System</h2><div class="error">Cannot load: ' + e.message + '</div>';
            }
        }

        loadAll();
        setInterval(loadAll, 5000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Proxy endpoints (frontend → backend APIs)
# ---------------------------------------------------------------------------
@app.get("/api/proxy/status")
async def proxy_model_status():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{MODEL_SERVER_URL}/status", timeout=5.0)
            return r.json()
        except Exception as e:
            return {"error": str(e)}


@app.get("/api/proxy/events")
async def proxy_events(
    page: int = 1,
    per_page: int = 20,
    scenario: str | None = None,
):
    async with httpx.AsyncClient() as client:
        try:
            params = {"page": page, "per_page": per_page}
            if scenario:
                params["scenario"] = scenario
            r = await client.get(f"{DB_SERVER_URL}/api/events", params=params, timeout=5.0)
            return r.json()
        except Exception as e:
            return {"error": str(e)}


@app.get("/api/proxy/stats")
async def proxy_stats():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{DB_SERVER_URL}/api/stats", timeout=5.0)
            return r.json()
        except Exception as e:
            return {"error": str(e)}


@app.get("/api/proxy/cameras")
async def proxy_cameras():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{DB_SERVER_URL}/api/cameras", timeout=8.0)
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return {"error": str(e)}


@app.get("/api/proxy/cameras/{camera_id}")
async def proxy_get_camera(camera_id: str):
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{DB_SERVER_URL}/api/cameras/{camera_id}", timeout=8.0)
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return {"error": str(e)}


@app.post("/api/proxy/cameras")
async def proxy_create_camera(request: Request):
    async with httpx.AsyncClient() as client:
        try:
            body = await request.body()
            r = await client.post(
                f"{DB_SERVER_URL}/api/cameras",
                content=body,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
                timeout=8.0,
            )
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return {"error": str(e)}


@app.put("/api/proxy/cameras/{camera_id}")
async def proxy_update_camera(camera_id: str, request: Request):
    async with httpx.AsyncClient() as client:
        try:
            body = await request.body()
            r = await client.put(
                f"{DB_SERVER_URL}/api/cameras/{camera_id}",
                content=body,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
                timeout=8.0,
            )
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return {"error": str(e)}


@app.delete("/api/proxy/cameras/{camera_id}")
async def proxy_delete_camera(camera_id: str):
    async with httpx.AsyncClient() as client:
        try:
            r = await client.delete(f"{DB_SERVER_URL}/api/cameras/{camera_id}", timeout=8.0)
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except Exception as e:
            return {"error": str(e)}


@app.get("/api/proxy/system")
async def proxy_system_metrics():
    try:
        cpu_percent = None
        ram = {}
        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            ram = {
                "used_gb": round(vm.used / (1024 ** 3), 2),
                "total_gb": round(vm.total / (1024 ** 3), 2),
                "percent": vm.percent,
            }

        gpu = {
            "name": "N/A",
            "utilization_percent": 0.0,
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "vram_percent": 0.0,
        }
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True, timeout=2).strip().splitlines()
            if out:
                first = [x.strip() for x in out[0].split(",")]
                if len(first) >= 4:
                    vram_used = float(first[2])
                    vram_total = float(first[3]) if float(first[3]) > 0 else 0.0
                    gpu = {
                        "name": first[0],
                        "utilization_percent": float(first[1]),
                        "vram_used_mb": int(vram_used),
                        "vram_total_mb": int(vram_total),
                        "vram_percent": round((vram_used / vram_total) * 100, 1) if vram_total else 0.0,
                    }
        except Exception:
            pass

        return {
            "cpu_percent": cpu_percent if cpu_percent is not None else 0.0,
            "ram": ram,
            "gpu": gpu,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# /api/vlm/* Reverse Proxy → Model Server
# The adhoc_rtsp.html sends all API calls to the same origin (frontend),
# so we proxy them to the model_server.
# ---------------------------------------------------------------------------
from fastapi.responses import Response as FastAPIResponse

@app.api_route("/api/vlm/{path:path}", methods=["GET", "POST"])
async def vlm_proxy(request: Request, path: str):
    """Reverse proxy all /api/vlm/* requests to Model Server."""
    target_url = f"{MODEL_SERVER_URL}/api/vlm/{path}"

    # Preserve query string
    if request.url.query:
        target_url += f"?{request.url.query}"

    # For MJPEG streaming (/api/vlm/video/), the httpx client must stay alive
    # for the entire duration of the streaming response. We cannot use
    # 'async with' here because the context manager would close the client
    # before StreamingResponse finishes iterating.
    if request.method == "GET" and "video" in path:
        try:
            client = httpx.AsyncClient(timeout=httpx.Timeout(None))
            req = client.build_request("GET", target_url)
            resp = await client.send(req, stream=True)

            async def stream_mjpeg():
                try:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        yield chunk
                except (httpx.ReadError, httpx.RemoteProtocolError):
                    # Upstream stream dropped (e.g., model_server restart/crash).
                    # End stream gracefully so frontend endpoint doesn't throw 500.
                    return
                except Exception:
                    return
                finally:
                    await resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                stream_mjpeg(),
                media_type=resp.headers.get(
                    "content-type",
                    "multipart/x-mixed-replace; boundary=frame",
                ),
            )
        except (httpx.ConnectError, httpx.ReadError):
            return JSONResponse(
                {"success": False, "error": "Model Server offline"},
                status_code=502,
            )
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )

    # Non-streaming requests (POST, GET JSON)
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if request.method == "POST":
                body = await request.body()
                path_norm = path.strip("/").lower()
                if path_norm == "start":
                    try:
                        payload = json.loads(body.decode("utf-8")) if body else {}
                    except Exception:
                        return JSONResponse(
                            {"success": False, "error": "Invalid JSON body"},
                            status_code=400,
                        )
                    ok, err = _validate_rtsp_url(payload.get("rtsp_url", ""))
                    if not ok:
                        return JSONResponse({"success": False, "error": err}, status_code=400)
                    logger.info(
                        "[Frontend Proxy] /api/vlm/start camera_id=%s rtsp=%s -> %s",
                        str(payload.get("camera_id", "")),
                        _mask_rtsp_url(str(payload.get("rtsp_url", ""))),
                        MODEL_SERVER_URL,
                    )
                resp = await client.post(
                    target_url,
                    content=body,
                    headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
                )
            else:
                resp = await client.get(target_url)

            return FastAPIResponse(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except httpx.ConnectError:
            return JSONResponse(
                {"success": False, "error": "Model Server offline"},
                status_code=502,
            )
        except Exception as e:
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )
