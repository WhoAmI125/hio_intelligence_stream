"""Frontend Server — Jinja2 UI + reverse proxy to model server."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

app = FastAPI(title="HiO v2 Frontend")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MODEL_SERVER = f"http://127.0.0.1:{config.SERVER_PORT}"

# Serve media files (clips, thumbnails) directly from data dir
from fastapi.staticfiles import StaticFiles
app.mount("/media", StaticFiles(directory=str(config.DATA_DIR), check_dir=False), name="media")


# ── Page routes ────────────────────────────────────────────────────
@app.get("/", response_class=RedirectResponse)
async def index():
    return RedirectResponse("/monitor")


@app.get("/monitor", response_class=HTMLResponse)
async def monitor(request: Request):
    return templates.TemplateResponse(
        request, "hio_v2/monitor.html",
        context={"active_page": "monitor"},
    )


@app.get("/events", response_class=HTMLResponse)
async def events(request: Request):
    return templates.TemplateResponse(
        request, "hio_v2/events.html",
        context={"active_page": "events"},
    )


@app.get("/tier1-logs", response_class=HTMLResponse)
async def tier1_logs(request: Request):
    return templates.TemplateResponse(
        request, "hio_v2/tier1_logs.html",
        context={
            "active_page": "tier1",
            "cash_threshold": config.CASH_WRIST_THRESHOLD_PX,
            "fire_threshold": config.CLIP_FIRE_THRESHOLD,
            "violence_threshold": config.CLIP_VIOLENCE_THRESHOLD,
        },
    )


@app.get("/tier2-logs", response_class=HTMLResponse)
async def tier2_logs(request: Request):
    return templates.TemplateResponse(
        request, "hio_v2/tier2_logs.html",
        context={"active_page": "tier2"},
    )


@app.get("/clip-review", response_class=HTMLResponse)
async def clip_review(request: Request):
    return templates.TemplateResponse(
        request, "hio_v2/clip_review.html",
        context={"active_page": "clip_review"},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request, "hio_v2/base_public.html",
        context={"active_page": "dashboard"},
    )


# ── Specific proxies (MUST be before generic proxy) ────────────────
@app.get("/api/proxy/snapshot/{cam_id}")
async def proxy_snapshot(cam_id: str):
    url = f"{MODEL_SERVER}/api/cameras/{cam_id}/snapshot"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return StreamingResponse(iter([resp.content]), media_type="image/jpeg")
            return JSONResponse({"error": "snapshot failed"}, status_code=resp.status_code)
    except Exception:
        return JSONResponse({"error": "snapshot failed"}, status_code=502)


@app.get("/api/proxy/clip-reviews")
async def proxy_clip_reviews():
    url = f"{MODEL_SERVER}/api/clip-reviews"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception:
        return JSONResponse({"reviews": []}, status_code=502)


@app.post("/api/proxy/clip-review")
async def proxy_clip_review(request: Request):
    """Forward multipart clip upload to model server."""
    url = f"{MODEL_SERVER}/api/clip-review"
    try:
        form = await request.form()
        clip_file = form.get("clip")
        scenario = str(form.get("scenario", "cash"))

        if clip_file is None:
            return JSONResponse({"error": "no clip file"}, status_code=400)

        content = await clip_file.read()
        filename = getattr(clip_file, "filename", "clip.mp4") or "clip.mp4"

        async with httpx.AsyncClient(timeout=600) as client:  # 10min — 3 scenarios × Qwen + Gemini
            resp = await client.post(
                url,
                files={"clip": (filename, content, "video/mp4")},
                data={"scenario": scenario},
            )
            try:
                return JSONResponse(resp.json(), status_code=resp.status_code)
            except Exception:
                return JSONResponse({"error": resp.text[:500]}, status_code=resp.status_code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/proxy/cameras/{cam_id}/tier1")
async def proxy_tier1(cam_id: str):
    url = f"{MODEL_SERVER}/api/cameras/{cam_id}/tier1"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception:
        return JSONResponse({"status": "no_data"}, status_code=502)


@app.get("/api/proxy/cameras/{cam_id}/tier1/history")
async def proxy_tier1_history(cam_id: str, limit: int = 50):
    url = f"{MODEL_SERVER}/api/cameras/{cam_id}/tier1/history?limit={limit}"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception:
        return JSONResponse({"logs": []}, status_code=502)


@app.get("/api/proxy/mjpeg/{cam_id}")
async def proxy_mjpeg(cam_id: str):
    """Non-buffering MJPEG stream proxy."""
    url = f"{MODEL_SERVER}/api/cameras/{cam_id}/mjpeg"
    try:
        client = httpx.AsyncClient(timeout=None)
        req = client.build_request("GET", url)
        resp = await client.send(req, stream=True)

        if resp.status_code != 200:
            await resp.aclose()
            await client.aclose()
            return JSONResponse({"error": "mjpeg unavailable"}, status_code=resp.status_code)

        async def stream_body():
            try:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(stream_body(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception:
        return JSONResponse({"error": "mjpeg proxy failed"}, status_code=502)


# ── Generic reverse proxy ─────────────────────────────────────────
@app.api_route("/api/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    """Proxy API requests to the model server."""
    url = f"{MODEL_SERVER}/api/{path}"
    params = dict(request.query_params)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if request.method == "GET":
                resp = await client.get(url, params=params)
            elif request.method == "POST":
                body = await request.body()
                resp = await client.post(url, params=params, content=body, headers={"Content-Type": "application/json"})
            elif request.method == "PUT":
                body = await request.body()
                resp = await client.put(url, params=params, content=body, headers={"Content-Type": "application/json"})
            elif request.method == "DELETE":
                resp = await client.delete(url, params=params)
            else:
                return JSONResponse({"error": "method not supported"}, status_code=405)

            return JSONResponse(resp.json(), status_code=resp.status_code)
    except httpx.ConnectError:
        return JSONResponse({"error": "Model server not reachable"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── WebSocket proxy ────────────────────────────────────────────────
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """Proxy WebSocket to model server, or hold connection if unavailable."""
    await ws.accept()

    model_ws_url = f"ws://127.0.0.1:{config.SERVER_PORT}/ws/events"

    try:
        import websockets

        async with websockets.connect(model_ws_url) as upstream:
            async def forward_upstream():
                async for msg in upstream:
                    await ws.send_text(msg)

            async def forward_client():
                while True:
                    try:
                        await ws.receive_text()
                    except WebSocketDisconnect:
                        break

            await asyncio.gather(forward_upstream(), forward_client())
    except Exception:
        # Model server WS not available — keep connection alive, just idle
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.FRONTEND_PORT,
        log_level="info",
    )
