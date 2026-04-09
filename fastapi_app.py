"""
FastAPI service + WebSocket live updates for compressor stability inference.

Run:
    uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

Dashboard:
    http://localhost:8001/

Environment:
    COMPRESSOR_MODEL   — path to model.pkl (default: ./data/model.pkl)
    COMPRESSOR_DEMO    — optional path to .xlsx/.csv for demo replay ticks
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from compressor_inference import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    CompressorInference,
    SensorStreamBuffer,
)

_ROOT = Path(__file__).resolve().parent

_engine: CompressorInference | None = None
_buffer: SensorStreamBuffer | None = None
_history: list[dict[str, Any]] = []
_history_max = 600
_ws_clients: set[WebSocket] = set()
_lock = asyncio.Lock()

_demo_df: pd.DataFrame | None = None
_demo_idx: int = 0


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            return obj
    return obj


def _augment_result(result: dict, engine: CompressorInference) -> dict:
    """Move frontend computation logic to the backend."""
    cur = result.get("current_sensors", {})
    pred = result.get("predicted_sensors", {})
    
    fad_now = cur.get("fad_cfm")
    ss_ref = engine.stable_ref.get("fad_cfm", None) if getattr(engine, "stable_ref", None) is not None else None
    
    spec_min = None
    margin_pct = None
    if ss_ref is not None:
        spec_min = ss_ref * 0.94
        if fad_now is not None:
            margin_pct = ((fad_now - spec_min) / spec_min) * 100
            
    result["readiness"] = {
        "spec_min": spec_min,
        "margin_pct": margin_pct,
        "ss_ref": ss_ref
    }
    
    g, a, r = 0, 0, 0
    red_names = []
    current_vs_ref = {}
    forecast_delta = {}
    
    for s in engine.sensors:
        val = cur.get(s)
        ref = engine.stable_ref.get(s) if getattr(engine, "stable_ref", None) is not None else None
        std = engine.stable_std.get(s) if getattr(engine, "stable_std", None) is not None else None
        
        # Standard deviation (sigma) and param status
        sig = None
        if val is not None and ref is not None and std is not None and std > 0:
            sig = (val - ref) / std
            dev = abs(sig)
            if dev < 1:
                g += 1
            elif dev < 2:
                a += 1
            else:
                r += 1
                red_names.append(s)
        elif val is not None:
            g += 1 # Default to green if no std available
            
        current_vs_ref[s] = sig
        
        # Forecast delta
        f = pred.get(s)
        d = None
        if f is not None and val is not None:
            d = f - val
        forecast_delta[s] = d
        
    result["param_status"] = {
        "green": g,
        "amber": a,
        "red": r,
        "red_names": red_names
    }
    result["current_vs_ref"] = current_vs_ref
    result["forecast_delta"] = forecast_delta
    
    return result



async def _broadcast(payload: dict) -> None:
    raw = json.dumps(_json_safe(payload))
    stale: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(raw)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _ws_clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _buffer, _demo_df, _demo_idx
    raw = os.environ.get("COMPRESSOR_MODEL", str(DEFAULT_MODEL_PATH))
    model_path = Path(raw)
    if not model_path.is_file():
        legacy = _ROOT / "data" / "quantile_models.pkl"
        if legacy.is_file():
            model_path = legacy
    if not model_path.is_file():
        raise RuntimeError(
            f"Model not found: {raw}. Train with compressor_train.py "
            f"(writes data/model.pkl) or set COMPRESSOR_MODEL."
        )
    _engine = CompressorInference(str(model_path))
    _buffer = SensorStreamBuffer(_engine.sensors)

    demo_path = os.environ.get("COMPRESSOR_DEMO", "").strip()
    if not demo_path and DEFAULT_DATA_PATH.is_file():
        demo_path = str(DEFAULT_DATA_PATH)
    if demo_path and Path(demo_path).is_file():
        _demo_df = (
            pd.read_excel(demo_path)
            if demo_path.endswith(".xlsx")
            else pd.read_csv(demo_path)
        )
        _demo_df = _demo_df.sort_values("elapsed_time_min").reset_index(drop=True)
        _demo_idx = 0
    else:
        _demo_df = None
        _demo_idx = 0

    yield
    _ws_clients.clear()


app = FastAPI(title="Compressor Stability API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    """One multivariate sensor snapshot from PLCs / edge gateway."""

    elapsed_time_min: float = Field(..., ge=0, description="Minutes since test start")
    values: dict[str, float] = Field(..., description="Sensor column name → value")
    phase: str | None = Field(None, description="Optional run phase label")


STATIC_DIR = _ROOT / "static"
_assets = STATIC_DIR / "assets"
if _assets.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


@app.get("/")
async def serve_dashboard():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(
            404,
            "static/index.html missing. Create the dashboard file next to fastapi_app.py.",
        )
    return FileResponse(index)


@app.get("/api/v1/meta")
async def api_meta():
    assert _engine is not None
    ref = {k: float(_engine.stable_ref[k]) for k in _engine.sensors}
    std = {k: float(_engine.stable_std[k]) for k in _engine.sensors}
    return {
        "sensors": list(_engine.sensors),
        "stable_onset_min": float(_engine.stable_time),
        "stable_ref": ref,
        "stable_std": std,
        "baseline_test_min": 180,
        "min_readings": 20,
        "min_elapsed_min": 15.0,
        "model_mae_min": float(_engine.cv_mae) if _engine.cv_mae is not None else None,
        "demo_loaded": _demo_df is not None,
        "demo_rows": int(len(_demo_df)) if _demo_df is not None else 0,
    }


@app.post("/api/v1/session/reset")
async def session_reset():
    async with _lock:
        assert _buffer is not None
        _buffer.clear()
        global _history, _demo_idx
        _history = []
        if _demo_df is not None:
            _demo_idx = 0
    return {"ok": True}


@app.post("/api/v1/ingest")
async def ingest(body: IngestRequest):
    async with _lock:
        assert _engine is not None and _buffer is not None
        try:
            window = _buffer.append(
                body.elapsed_time_min,
                body.values,
                phase=body.phase,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        result = _engine.predict(window)
        result = _augment_result(result, _engine)
        row = {"elapsed_min": result.get("elapsed_min"), "result": result}
        _history.append(row)
        if len(_history) > _history_max:
            del _history[: len(_history) - _history_max]

    await _broadcast({"type": "prediction", "payload": result})
    return result


@app.get("/api/v1/status")
async def status():
    async with _lock:
        assert _buffer is not None
        last = _history[-1]["result"] if _history else None
        return {
            "buffer_rows": len(_buffer),
            "history_points": len(_history),
            "last": last,
        }


@app.get("/api/v1/history")
async def history(limit: int = 200):
    limit = max(1, min(limit, _history_max))
    async with _lock:
        return {"series": _history[-limit:]}


@app.post("/api/v1/demo/tick")
async def demo_tick():
    """Append the next row from COMPRESSOR_DEMO file (for UI testing without real sensors)."""
    async with _lock:
        if _demo_df is None or _engine is None or _buffer is None:
            raise HTTPException(
                400,
                "Demo file not loaded. Set env COMPRESSOR_DEMO to a .xlsx/.csv path and restart.",
            )
        global _demo_idx
        if _demo_idx >= len(_demo_df):
            return {"done": True, "message": "End of demo file"}
        row = _demo_df.iloc[_demo_idx]
        _demo_idx += 1
        values = {s: float(row[s]) for s in _engine.sensors}
        phase = str(row["phase"]) if "phase" in row and pd.notna(row["phase"]) else None
        window = _buffer.append(float(row["elapsed_time_min"]), values, phase=phase)
        result = _engine.predict(window)
        result = _augment_result(result, _engine)
        _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
        if len(_history) > _history_max:
            del _history[: len(_history) - _history_max]

    await _broadcast({"type": "prediction", "payload": result})
    return result


@app.get("/api/v1/demo/progress")
async def demo_progress():
    if _demo_df is None:
        return {"enabled": False}
    return {"enabled": True, "index": _demo_idx, "total": len(_demo_df)}


@app.post("/api/v1/demo/auto")
async def demo_auto(speed_ms: int = 300):
    """Stream all remaining demo rows with a configurable delay between each."""
    if _demo_df is None or _engine is None or _buffer is None:
        raise HTTPException(400, "Demo file not loaded.")
    global _demo_idx
    results = []
    while _demo_idx < len(_demo_df):
        async with _lock:
            if _demo_idx >= len(_demo_df):
                break
            row = _demo_df.iloc[_demo_idx]
            _demo_idx += 1
            values = {s: float(row[s]) for s in _engine.sensors}
            phase = str(row["phase"]) if "phase" in row and pd.notna(row["phase"]) else None
            window = _buffer.append(float(row["elapsed_time_min"]), values, phase=phase)
            result = _engine.predict(window)
            result = _augment_result(result, _engine)
            _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
            if len(_history) > _history_max:
                del _history[: len(_history) - _history_max]
        await _broadcast({"type": "prediction", "payload": result})
        results.append(result)
        await asyncio.sleep(speed_ms / 1000.0)
    return {"done": True, "ticks": len(results)}


@app.get("/api/v1/checkpoints")
async def checkpoints():
    """Prediction accuracy at standard time checkpoints (30, 60, 90, 120, 150, 180 min).

    For each checkpoint, return the prediction made at that elapsed time
    alongside the actual sensor values at the end of the test (last known reading).
    """
    async with _lock:
        if not _history:
            return {"checkpoints": []}
        cp_minutes = [30, 60, 90, 120, 150, 180]
        result_list = []
        last_result = _history[-1]["result"] if _history else None
        actual_sensors = last_result.get("current_sensors", {}) if last_result else {}
        max_elapsed = last_result.get("elapsed_min", 0) if last_result else 0
        for cp in cp_minutes:
            if max_elapsed < cp - 0.5:
                result_list.append({"checkpoint_min": cp, "available": False})
                continue
            best = None
            best_dist = float("inf")
            for h in _history:
                em = h.get("elapsed_min") or 0
                dist = abs(em - cp)
                if dist < best_dist:
                    best_dist = dist
                    best = h["result"]
            if best is None:
                result_list.append({"checkpoint_min": cp, "available": False})
                continue
            predicted = best.get("predicted_sensors", {})
            errors = {}
            for s, pv in predicted.items():
                av = actual_sensors.get(s)
                if av and abs(av) > 1e-9:
                    errors[s] = round(abs(pv - av) / abs(av) * 100, 2)
                else:
                    errors[s] = None
            result_list.append({
                "checkpoint_min": cp,
                "available": True,
                "elapsed_min": best.get("elapsed_min"),
                "predicted_sensors": predicted,
                "actual_sensors": actual_sensors,
                "error_pct": errors,
                "action": best.get("action"),
                "confidence_pct": best.get("confidence_pct"),
                "time_to_stability_min": best.get("time_to_stability_min"),
            })
        return {"checkpoints": result_list}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        async with _lock:
            snap = _history[-1]["result"] if _history else None
        await ws.send_json(
            {
                "type": "hello",
                "payload": snap,
                "meta": {
                    "sensors": list(_engine.sensors) if _engine else [],
                },
            }
        )
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


if __name__ == "__main__":
    import uvicorn
    import socket
    # host = os.environ.get("HOST", "0.0.0.0")
    # port = int(os.environ.get("PORT", "8000"))
    # uvicorn.run("app:app", host=host, port=port, reload=False)


    def find_free_port(start=8000, max_tries=20):
        for port in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        raise RuntimeError("No free ports available")

    host = os.environ.get("HOST", "0.0.0.0")

    # Try env PORT first, else find free one
    port = int(os.environ.get("PORT", 0))
    if port == 0:
        port = find_free_port()

    print(f"🚀 Starting server on port {port}")

    uvicorn.run("app:app", host=host, port=port, reload=False)

    
