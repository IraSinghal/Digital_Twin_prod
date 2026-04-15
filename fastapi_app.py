# # """
# # FastAPI service + WebSocket live updates for compressor stability inference.

# # Run:
# #     uvicorn fastapi_app:app --reload --host 127.0.0.1 --port 8000
# #     # OR simply: python fastapi_app.py

# # Dashboard:
# #     http://localhost:8000/

# # Environment:
# #     COMPRESSOR_MODEL   — path to model.pkl (default: ./data/quantile_models.pkl)
# #     COMPRESSOR_DEMO    — optional path to .xlsx/.csv for demo replay ticks
# # """

# # from __future__ import annotations

# # import asyncio
# # import json
# # import os
# # import webbrowser
# # from contextlib import asynccontextmanager
# # from pathlib import Path
# # from typing import Any

# # import pandas as pd
# # from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import FileResponse
# # from fastapi.staticfiles import StaticFiles
# # from pydantic import BaseModel, Field

# # from compressor_inference import (
# #     DEFAULT_DATA_PATH,
# #     DEFAULT_MODEL_PATH,
# #     CompressorInference,
# #     SensorStreamBuffer,
# # )

# # _ROOT = Path(__file__).resolve().parent

# # # ── runtime state ──
# # _engine: CompressorInference | None = None
# # _buffer: SensorStreamBuffer | None = None
# # _extra_cols: list[str] = []
# # _demo_df: pd.DataFrame | None = None
# # _demo_idx: int = 0
# # _history: list[dict[str, Any]] = []
# # _history_max = 600
# # _ws_clients: set[WebSocket] = set()
# # _lock = asyncio.Lock()


# # # ── helpers ──

# # def _json_safe(obj: Any) -> Any:
# #     """Recursively convert numpy scalars / tuples so json.dumps won't choke."""
# #     if isinstance(obj, dict):
# #         return {k: _json_safe(v) for k, v in obj.items()}
# #     if isinstance(obj, (list, tuple)):
# #         return [_json_safe(v) for v in obj]
# #     if isinstance(obj, float):
# #         return obj
# #     if hasattr(obj, "item"):
# #         try:
# #             return float(obj.item())
# #         except Exception:
# #             return obj
# #     return obj


# # def _augment_result(result: dict, engine: CompressorInference) -> dict:
# #     """Attach param_status, readiness, and forecast_delta computed server-side."""
# #     cur = result.get("current_sensors", {})
# #     pred = result.get("predicted_sensors", {})
# #     extras = result.get("extra_columns", {})

# #     fad_now = cur.get("fad_cfm")
# #     rated_fad = extras.get("rated_fad_cfm") or engine.stable_ref.get("fad_cfm", 0.0)

# #     tolerance_pct = (
# #         extras.get("tolerance_flow_pct")
# #         or extras.get("flow_tolerance_pct(%)")
# #         or 6.0
# #     )
# #     tolerance_factor = (100.0 - tolerance_pct) / 100.0
# #     spec_min = rated_fad * tolerance_factor if rated_fad else None
# #     margin_pct = None
# #     if spec_min and fad_now is not None and spec_min > 0:
# #         margin_pct = round(((fad_now - spec_min) / spec_min) * 100, 2)

# #     result["readiness"] = {
# #         "spec_min": round(spec_min, 2) if spec_min else None,
# #         "margin_pct": margin_pct,
# #         "ss_ref": round(rated_fad, 2) if rated_fad else None,
# #         "tolerance_pct": tolerance_pct,
# #     }

# #     g, a, r = 0, 0, 0
# #     red_names: list[str] = []
# #     current_vs_ref: dict[str, float | None] = {}
# #     forecast_delta: dict[str, float | None] = {}

# #     for s in engine.sensors:
# #         val = cur.get(s)
# #         ref = engine.stable_ref.get(s)
# #         std = engine.stable_std.get(s)
# #         sig = None
# #         if val is not None and ref is not None and std and std > 0:
# #             sig = round((val - ref) / std, 4)
# #             dev = abs(sig)
# #             if dev < 1:
# #                 g += 1
# #             elif dev < 2:
# #                 a += 1
# #             else:
# #                 r += 1
# #                 red_names.append(s)
# #         elif val is not None:
# #             g += 1
# #         current_vs_ref[s] = sig

# #         fv = pred.get(s)
# #         forecast_delta[s] = round(fv - val, 4) if fv is not None and val is not None else None

# #     result["param_status"] = {"green": g, "amber": a, "red": r, "red_names": red_names}
# #     result["current_vs_ref"] = current_vs_ref
# #     result["forecast_delta"] = forecast_delta
# #     return result


# # async def _broadcast(payload: dict) -> None:
# #     raw = json.dumps(_json_safe(payload))
# #     stale: list[WebSocket] = []
# #     for ws in _ws_clients:
# #         try:
# #             await ws.send_text(raw)
# #         except Exception:
# #             stale.append(ws)
# #     for ws in stale:
# #         _ws_clients.discard(ws)


# # def _identify_extra_columns(engine: CompressorInference) -> list[str]:
# #     """Return the non-sensor, non-time columns that feature engineering needs.

# #     These come from spec_columns in the bundle plus the hard-coded derived
# #     feature source columns used in compressor_train.engineer_features_batch.
# #     """
# #     needed: set[str] = set()
# #     for c in getattr(engine, "spec_columns", []):
# #         needed.add(c)
# #     needed.update(["rated_fad_cfm", "rated_motor_output_kw"])
# #     return sorted(needed)


# # def _build_window(demo_row: pd.Series, engine: CompressorInference,
# #                   buffer: SensorStreamBuffer, extra_cols: list[str]) -> pd.DataFrame:
# #     """Append one demo row to the buffer AND return a window with extra columns.

# #     SensorStreamBuffer only stores sensor + elapsed columns. The spec / rated
# #     columns that compressor_inference._engineer_features needs are carried
# #     in a parallel list and merged into the DataFrame before returning.
# #     """
# #     values = {s: float(demo_row[s]) for s in engine.sensors}
# #     phase = str(demo_row["phase"]) if "phase" in demo_row.index and pd.notna(demo_row.get("phase")) else None

# #     extra_vals: dict[str, float] = {}
# #     for c in extra_cols:
# #         if c in demo_row.index and pd.notna(demo_row.get(c)):
# #             try:
# #                 extra_vals[c] = float(demo_row[c])
# #             except (ValueError, TypeError):
# #                 extra_vals[c] = 0.0
# #         else:
# #             extra_vals[c] = 0.0

# #     window = buffer.append(float(demo_row["elapsed_time_min"]), values, phase=phase)

# #     for c in extra_cols:
# #         if c not in window.columns:
# #             window[c] = extra_vals[c]
# #         else:
# #             window.loc[window.index[-1], c] = extra_vals[c]

# #     return window, extra_vals


# # # ── lifespan ──

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global _engine, _buffer, _extra_cols, _demo_df, _demo_idx

# #     raw = os.environ.get("COMPRESSOR_MODEL", str(DEFAULT_MODEL_PATH))
# #     model_path = Path(raw)
# #     if not model_path.is_file():
# #         legacy = _ROOT / "data" / "quantile_models.pkl"
# #         if legacy.is_file():
# #             model_path = legacy
# #     if not model_path.is_file():
# #         raise RuntimeError(
# #             f"Model not found: {raw}. Train with compressor_train.py "
# #             f"(writes data/quantile_models.pkl) or set COMPRESSOR_MODEL."
# #         )

# #     _engine = CompressorInference(str(model_path))
# #     _buffer = SensorStreamBuffer(_engine.sensors)
# #     _extra_cols = _identify_extra_columns(_engine)

# #     demo_path = os.environ.get("COMPRESSOR_DEMO", "").strip()
# #     if not demo_path and DEFAULT_DATA_PATH.is_file():
# #         demo_path = str(DEFAULT_DATA_PATH)
# #     if demo_path and Path(demo_path).is_file():
# #         _demo_df = (
# #             pd.read_excel(demo_path) if demo_path.endswith(".xlsx")
# #             else pd.read_csv(demo_path)
# #         )
# #         _demo_df.columns = [c.strip() for c in _demo_df.columns]
# #         _demo_df = _demo_df.sort_values("elapsed_time_min").reset_index(drop=True)
# #         _demo_idx = 0
# #     else:
# #         _demo_df = None
# #         _demo_idx = 0


    

# #     yield
# #     _ws_clients.clear()


# # # ── app ──

# # app = FastAPI(title="Compressor Stability API", lifespan=lifespan)
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# # class IngestRequest(BaseModel):
# #     """One multivariate sensor snapshot from PLCs / edge gateway."""
# #     elapsed_time_min: float = Field(..., ge=0, description="Minutes since test start")
# #     values: dict[str, float] = Field(..., description="Sensor column name → value")
# #     extras: dict[str, float] = Field(default_factory=dict,
# #                                      description="Spec / rated columns (rated_fad_cfm, etc.)")
# #     phase: str | None = Field(None, description="Optional run phase label")


# # STATIC_DIR = _ROOT / "static"
# # _assets = STATIC_DIR / "assets"
# # if _assets.is_dir():
# #     app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


# # # ── routes ──

# # @app.get("/")
# # async def serve_dashboard():
# #     index = STATIC_DIR / "index.html"
# #     if not index.is_file():
# #         raise HTTPException(404, "static/index.html missing.")
# #     return FileResponse(index)

# # @app.get("/api/health")
# # def health_check():
# #     return {"message": "API running"}


# # @app.get("/api/v1/meta")
# # async def api_meta():
# #     assert _engine is not None
# #     return {
# #         "sensors": _engine.sensors,
# #         "stable_onset_min": _engine.stable_time,
# #         "stable_ref": _engine.stable_ref,
# #         "stable_std": _engine.stable_std,
# #         "demo_loaded": _demo_df is not None,
# #     }


# # @app.post("/api/v1/session/reset")
# # async def session_reset():
# #     async with _lock:
# #         assert _buffer is not None
# #         _buffer.clear()
# #         global _history, _demo_idx
# #         _history = []
# #         if _demo_df is not None:
# #             _demo_idx = 0
# #     return {"ok": True}


# # @app.post("/api/v1/ingest")
# # async def ingest(body: IngestRequest):
# #     """Accept a single sensor snapshot (real-time PLC / edge gateway path)."""
# #     async with _lock:
# #         assert _engine is not None and _buffer is not None

# #         try:
# #             window = _buffer.append(body.elapsed_time_min, body.values, phase=body.phase)
# #         except ValueError as e:
# #             raise HTTPException(400, str(e)) from e

# #         for c, v in body.extras.items():
# #             if c not in window.columns:
# #                 window[c] = v
# #             else:
# #                 window.loc[window.index[-1], c] = v

# #         result = _engine.predict(window)
# #         result["extra_columns"] = body.extras
# #         result = _augment_result(result, _engine)
# #         _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
# #         if len(_history) > _history_max:
# #             del _history[: len(_history) - _history_max]

# #     await _broadcast({"type": "prediction", "payload": result})
# #     return result


# # @app.get("/api/v1/status")
# # async def status():
# #     async with _lock:
# #         assert _buffer is not None
# #         last = _history[-1]["result"] if _history else None
# #         return {
# #             "buffer_rows": len(_buffer),
# #             "history_points": len(_history),
# #             "last": last,
# #         }


# # @app.get("/api/v1/history")
# # async def history(limit: int = 200):
# #     limit = max(1, min(limit, _history_max))
# #     async with _lock:
# #         return {"series": _history[-limit:]}


# # @app.post("/api/v1/demo/tick")
# # async def demo_tick():
# #     """Append the next row from the demo file."""
# #     async with _lock:
# #         if _demo_df is None or _engine is None or _buffer is None:
# #             raise HTTPException(400, "Demo file not loaded.")
# #         global _demo_idx
# #         if _demo_idx >= len(_demo_df):
# #             return {"done": True, "message": "End of demo file"}

# #         row = _demo_df.iloc[_demo_idx]
# #         _demo_idx += 1

# #         window, extra_vals = _build_window(row, _engine, _buffer, _extra_cols)
# #         result = _engine.predict(window)
# #         result["extra_columns"] = extra_vals
# #         result = _augment_result(result, _engine)

# #         _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
# #         if len(_history) > _history_max:
# #             del _history[: len(_history) - _history_max]

# #     await _broadcast({"type": "prediction", "payload": result})
# #     return result


# # @app.get("/api/v1/demo/progress")
# # async def demo_progress():
# #     if _demo_df is None:
# #         return {"enabled": False}
# #     return {"enabled": True, "index": _demo_idx, "total": len(_demo_df)}


# # @app.post("/api/v1/demo/auto")
# # async def demo_auto(speed_ms: int = 300):
# #     """Stream all remaining demo rows with a configurable delay."""
# #     if _demo_df is None or _engine is None or _buffer is None:
# #         raise HTTPException(400, "Demo file not loaded.")
# #     global _demo_idx
# #     results = []
# #     while _demo_idx < len(_demo_df):
# #         async with _lock:
# #             if _demo_idx >= len(_demo_df):
# #                 break
# #             row = _demo_df.iloc[_demo_idx]
# #             _demo_idx += 1

# #             window, extra_vals = _build_window(row, _engine, _buffer, _extra_cols)
# #             result = _engine.predict(window)
# #             result["extra_columns"] = extra_vals
# #             result = _augment_result(result, _engine)

# #             _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
# #             if len(_history) > _history_max:
# #                 del _history[: len(_history) - _history_max]

# #         await _broadcast({"type": "prediction", "payload": result})
# #         results.append(result)
# #         await asyncio.sleep(speed_ms / 1000.0)
# #     return {"done": True, "ticks": len(results)}


# # @app.get("/api/v1/checkpoints")
# # async def checkpoints():
# #     """Prediction accuracy at standard time checkpoints.

# #     Only returns data for checkpoints the test has actually reached.
# #     """
# #     async with _lock:
# #         if not _history:
# #             return {"checkpoints": []}

# #         cp_minutes = [30, 60, 90, 120, 150, 180]
# #         last_result = _history[-1]["result"]
# #         actual_sensors = last_result.get("current_sensors", {})
# #         max_elapsed = last_result.get("elapsed_min", 0)
# #         result_list = []

# #         for cp in cp_minutes:
# #             if max_elapsed < cp - 0.5:
# #                 result_list.append({"checkpoint_min": cp, "available": False})
# #                 continue

# #             best = None
# #             best_dist = float("inf")
# #             for h in _history:
# #                 em = h.get("elapsed_min") or 0
# #                 dist = abs(em - cp)
# #                 if dist < best_dist:
# #                     best_dist = dist
# #                     best = h["result"]

# #             if best is None:
# #                 result_list.append({"checkpoint_min": cp, "available": False})
# #                 continue

# #             predicted = best.get("predicted_sensors", {})
# #             errors: dict[str, float | None] = {}
# #             for s, pv in predicted.items():
# #                 av = actual_sensors.get(s)
# #                 if av and abs(av) > 1e-9:
# #                     errors[s] = round(abs(pv - av) / abs(av) * 100, 2)
# #                 else:
# #                     errors[s] = None

# #             result_list.append({
# #                 "checkpoint_min": cp,
# #                 "available": True,
# #                 "elapsed_min": best.get("elapsed_min"),
# #                 "predicted_sensors": predicted,
# #                 "actual_sensors": actual_sensors,
# #                 "error_pct": errors,
# #                 "action": best.get("action"),
# #                 "confidence_pct": best.get("confidence_pct"),
# #                 "time_to_stability_min": best.get("time_to_stability_min"),
# #             })

# #         return {"checkpoints": result_list}


# # @app.websocket("/ws/live")
# # async def ws_live(ws: WebSocket):
# #     await ws.accept()
# #     _ws_clients.add(ws)
# #     try:
# #         async with _lock:
# #             snap = _history[-1]["result"] if _history else None
# #         await ws.send_json({
# #             "type": "hello",
# #             "payload": snap,
# #             "meta": {"sensors": list(_engine.sensors) if _engine else []},
# #         })
# #         while True:
# #             await ws.receive_text()
# #     except WebSocketDisconnect:
# #         pass
# #     finally:
# #         _ws_clients.discard(ws)


# # if __name__ == "__main__":
# #     import os
# #     import socket
# #     import uvicorn

# #     def find_free_port(start: int = 8000, max_tries: int = 50) -> int:
# #         for port in range(start, start + max_tries):
# #             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
# #                 try:
# #                     s.bind(("127.0.0.1", port))  # better than connect_ex
# #                     return port
# #                 except OSError:
# #                     continue
# #         raise RuntimeError("No free ports available")

# #     host = os.environ.get("HOST", "127.0.0.1")

# #     # If PORT is set → use it, else auto-find
# #     port_env = os.environ.get("PORT")
# #     if port_env:
# #         port = int(port_env)
# #     else:
# #         port = find_free_port()

# #     print(f"🚀 Server starting on {host}:{port}")
# #     print(f"👉 Open in browser: http://localhost:{port}/")

# #     uvicorn.run(
# #         "fastapi_app:app",
# #         host=host,
# #         port=port,
# #         reload=True  # turn ON for development
# #     )



# """
# FastAPI service + WebSocket live updates for compressor stability inference.

# Run:
#     uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
#     # OR simply: python fastapi_app.py

# Dashboard:
#     http://localhost:8000/

# Environment:
#     COMPRESSOR_MODEL   — path to model.pkl (default: ./data/quantile_models.pkl)
#     COMPRESSOR_DEMO    — optional path to .xlsx/.csv for demo replay ticks
# """

# from __future__ import annotations

# import asyncio
# import json
# import os
# import subprocess
# import webbrowser
# from contextlib import asynccontextmanager
# from pathlib import Path
# from typing import Any

# import pandas as pd
# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel, Field

# from compressor_inference import (
#     DEFAULT_DATA_PATH,
#     DEFAULT_MODEL_PATH,
#     CompressorInference,
#     SensorStreamBuffer,
# )

# _ROOT = Path(__file__).resolve().parent

# # ── runtime state ──
# _engine: CompressorInference | None = None
# _buffer: SensorStreamBuffer | None = None
# _extra_cols: list[str] = []
# _demo_df: pd.DataFrame | None = None
# _demo_idx: int = 0
# _history: list[dict[str, Any]] = []
# _history_max = 600
# _ws_clients: set[WebSocket] = set()
# _lock = asyncio.Lock()

# # ── health-score runtime state ──
# _health_ema: float | None = None
# _health_cfg: dict[str, Any] | None = None
# _health_forecasts: list[dict[str, Any]] = []

# # Only score health on these signals (per user requirement).
# _HEALTH_SENSORS = (
#     "airend_discharge_temp_c",
#     "fad_cfm",
#     "motor_output_power_kw",
# )


# # ── helpers ──

# def _json_safe(obj: Any) -> Any:
#     """Recursively convert numpy scalars / tuples so json.dumps won't choke."""
#     if isinstance(obj, dict):
#         return {k: _json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_json_safe(v) for v in obj]
#     if isinstance(obj, float):
#         return obj
#     if hasattr(obj, "item"):
#         try:
#             return float(obj.item())
#         except Exception:
#             return obj
#     return obj


# def _augment_result(result: dict, engine: CompressorInference) -> dict:
#     """Attach param_status, readiness, and forecast_delta computed server-side."""
#     cur = result.get("current_sensors", {})
#     pred = result.get("predicted_sensors", {})
#     extras = result.get("extra_columns", {})

#     fad_now = cur.get("fad_cfm")
#     rated_fad = extras.get("rated_fad_cfm") or engine.stable_ref.get("fad_cfm", 0.0)

#     tolerance_pct = (
#         extras.get("tolerance_flow_pct")
#         or extras.get("flow_tolerance_pct(%)")
#         or 6.0
#     )
#     tolerance_factor = (100.0 - tolerance_pct) / 100.0
#     spec_min = rated_fad * tolerance_factor if rated_fad else None
#     margin_pct = None
#     if spec_min and fad_now is not None and spec_min > 0:
#         margin_pct = round(((fad_now - spec_min) / spec_min) * 100, 2)

#     motor_now = cur.get("motor_output_power_kw")
#     rated_motor = extras.get("rated_motor_output_kw") or engine.stable_ref.get(
#         "motor_output_power_kw", 0.0
#     )
#     motor_tolerance_pct = (
#         extras.get("(motor_power_tolerance_pct(%))")
#         or extras.get("motor_power_tolerance_pct")
#         or extras.get("motor_power_tolerance_pct(%)")
#         or 6.0
#     )
#     motor_tol_factor = (100.0 - float(motor_tolerance_pct)) / 100.0
#     motor_spec_min = rated_motor * motor_tol_factor if rated_motor else None
#     motor_margin_pct = None
#     if motor_spec_min and motor_now is not None and motor_spec_min > 0:
#         motor_margin_pct = round(((motor_now - motor_spec_min) / motor_spec_min) * 100, 2)

#     result["readiness"] = {
#         "spec_min": round(spec_min, 2) if spec_min else None,
#         "margin_pct": margin_pct,
#         "ss_ref": round(rated_fad, 2) if rated_fad else None,
#         "tolerance_pct": tolerance_pct,
#         "motor_spec_min": round(motor_spec_min, 2) if motor_spec_min else None,
#         "motor_margin_pct": motor_margin_pct,
#     }

#     g, a, r = 0, 0, 0
#     red_names: list[str] = []
#     current_vs_ref: dict[str, float | None] = {}
#     forecast_delta: dict[str, float | None] = {}

#     for s in engine.sensors:
#         val = cur.get(s)
#         ref = engine.stable_ref.get(s)
#         std = engine.stable_std.get(s)
#         sig = None
#         if val is not None and ref is not None and std and std > 0:
#             sig = round((val - ref) / std, 4)
#             dev = abs(sig)
#             if dev < 1:
#                 g += 1
#             elif dev < 2:
#                 a += 1
#             else:
#                 r += 1
#                 red_names.append(s)
#         elif val is not None:
#             g += 1
#         current_vs_ref[s] = sig

#         fv = pred.get(s)
#         forecast_delta[s] = round(fv - val, 4) if fv is not None and val is not None else None

#     result["param_status"] = {"green": g, "amber": a, "red": r, "red_names": red_names}
#     result["current_vs_ref"] = current_vs_ref
#     result["forecast_delta"] = forecast_delta

#     # Health score (deviation-based, explainable)
#     _attach_health_score(result, engine)
#     return result


# def _load_health_cfg(engine: CompressorInference) -> dict[str, Any]:
#     """
#     Load/calc health scoring configuration.

#     Override via env:
#       - COMPRESSOR_HEALTH_WEIGHTS : JSON object of {sensor_name: weight}
#       - COMPRESSOR_HEALTH_SCALING_FACTOR : float (default 20.0)
#       - COMPRESSOR_HEALTH_EMA_ALPHA : float in (0,1] (default 0.2)
#       - COMPRESSOR_HEALTH_EXPECTED_MODE : "forecast_matured" (default) | "stable_ref" | "forecast"
#     """
#     global _health_cfg
#     if _health_cfg is not None:
#         return _health_cfg

#     # Lock scoring to the agreed 3 features only (even if engine.sensors expands).
#     engine_sensors = set(getattr(engine, "sensors", []) or [])
#     sensors = [s for s in _HEALTH_SENSORS if s in engine_sensors]

#     # Default weights by semantic group, mapped to your current sensor set.
#     default_weights = {
#         "airend_discharge_temp_c": 0.40,  # temperature
#         "fad_cfm": 0.30,                  # flow
#         "motor_output_power_kw": 0.30,    # power
#     }

#     raw = os.environ.get("COMPRESSOR_HEALTH_WEIGHTS", "").strip()
#     w: dict[str, float] = {}
#     if raw:
#         try:
#             parsed = json.loads(raw)
#             if isinstance(parsed, dict):
#                 for k, v in parsed.items():
#                     try:
#                         w[str(k)] = float(v)
#                     except Exception:
#                         continue
#         except Exception:
#             w = {}

#     if not w:
#         w = {s: float(default_weights.get(s, 0.0)) for s in sensors}

#     # Keep only allowed sensors and normalise to sum=1.
#     w = {k: float(v) for k, v in w.items() if k in sensors and v is not None and float(v) > 0}
#     if not w:
#         # Fallback: uniform across sensors
#         w = {s: 1.0 for s in sensors}
#     w_sum = sum(w.values()) or 1.0
#     w = {k: v / w_sum for k, v in w.items()}

#     def _fenv(name: str, default: float) -> float:
#         try:
#             return float(os.environ.get(name, str(default)).strip())
#         except Exception:
#             return float(default)

#     scaling_factor = _fenv("COMPRESSOR_HEALTH_SCALING_FACTOR", 20.0)
#     alpha = _fenv("COMPRESSOR_HEALTH_EMA_ALPHA", 0.2)
#     if not (0.0 < alpha <= 1.0):
#         alpha = 0.2

#     expected_mode = os.environ.get("COMPRESSOR_HEALTH_EXPECTED_MODE", "forecast_matured").strip().lower()
#     if expected_mode not in ("forecast_matured", "stable_ref", "forecast"):
#         expected_mode = "forecast_matured"

#     _health_cfg = {
#         "weights": w,
#         "scaling_factor": float(scaling_factor),
#         "ema_alpha": float(alpha),
#         "expected_mode": expected_mode,
#         "epsilon": 1e-9,
#     }
#     return _health_cfg


# def _attach_health_score(result: dict, engine: CompressorInference) -> None:
#     """
#     Attach:
#       - health_score (0-100)
#       - health_score_smoothed (EMA)
#       - health_band: Healthy/Warning/Critical
#       - health_deviation: per-sensor normalised deviation
#       - health_risk_contrib: per-sensor contribution (points deducted)
#     """
#     global _health_ema
#     global _health_forecasts

#     cur = result.get("current_sensors") or {}
#     if not cur:
#         return

#     cfg = _load_health_cfg(engine)
#     w = cfg["weights"]
#     eps = float(cfg["epsilon"])
#     sf = float(cfg["scaling_factor"])
#     alpha = float(cfg["ema_alpha"])
#     expected_mode = str(cfg.get("expected_mode") or "stable_ref")

#     pred = result.get("predicted_sensors") or {}
#     ref = getattr(engine, "stable_ref", {}) or {}
#     n_readings = result.get("n_readings")

#     # Maintain a lightweight "matured forecast" queue so we can compare
#     # forecasted sensors made at time t vs actual sensors at time t+h.
#     try:
#         fh = int(getattr(engine, "future_horizon", 10))
#     except Exception:
#         fh = 10

#     if isinstance(n_readings, int) and pred:
#         _health_forecasts.append({"due_n": int(n_readings) + fh, "pred": dict(pred)})
#         if len(_health_forecasts) > 1200:
#             _health_forecasts = _health_forecasts[-800:]

#     matured_pred: dict[str, Any] | None = None
#     if expected_mode == "forecast_matured" and isinstance(n_readings, int):
#         due = [f for f in _health_forecasts if int(f.get("due_n", 0)) <= int(n_readings)]
#         if due:
#             due.sort(key=lambda x: int(x.get("due_n", 0)))
#             matured_pred = due[0].get("pred") if isinstance(due[0].get("pred"), dict) else None
#             cut_due_n = int(due[0].get("due_n", 0))
#             _health_forecasts = [f for f in _health_forecasts if int(f.get("due_n", 0)) > cut_due_n]

#     deviations: dict[str, float | None] = {}
#     contrib: dict[str, float | None] = {}
#     total_points = 0.0
#     valid_terms = 0

#     for s, weight in w.items():
#         av = cur.get(s)
#         expected = None
#         if expected_mode == "forecast_matured":
#             expected = matured_pred.get(s) if matured_pred else None
#         elif expected_mode == "forecast":
#             expected = pred.get(s)
#         else:
#             expected = ref.get(s)
#         scale = None
#         try:
#             scale = float(getattr(engine, "stable_std", {}).get(s))  # type: ignore[union-attr]
#         except Exception:
#             scale = None
#         if scale is None or not (scale > 0):
#             scale = 1.0

#         if av is None or expected is None:
#             deviations[s] = None
#             contrib[s] = None
#             continue

#         d = abs(float(av) - float(expected)) / (eps + float(scale))
#         deviations[s] = round(float(d), 6)
#         c = float(weight) * float(d) * sf
#         contrib[s] = round(c, 4)
#         total_points += c
#         valid_terms += 1

#     if valid_terms == 0:
#         return

#     raw_score = 100.0 - total_points
#     score = max(0.0, min(100.0, raw_score))

#     if _health_ema is None:
#         _health_ema = score
#     else:
#         _health_ema = alpha * score + (1.0 - alpha) * _health_ema

#     smoothed = max(0.0, min(100.0, float(_health_ema)))
#     band = "Healthy" if smoothed > 80 else "Warning" if smoothed >= 60 else "Critical"

#     result["health_score"] = round(score, 2)
#     result["health_score_smoothed"] = round(smoothed, 2)
#     result["health_band"] = band
#     result["health_deviation"] = deviations
#     result["health_risk_contrib"] = contrib


# async def _broadcast(payload: dict) -> None:
#     raw = json.dumps(_json_safe(payload))
#     stale: list[WebSocket] = []
#     for ws in _ws_clients:
#         try:
#             await ws.send_text(raw)
#         except Exception:
#             stale.append(ws)
#     for ws in stale:
#         _ws_clients.discard(ws)


# def _identify_extra_columns(engine: CompressorInference) -> list[str]:
#     """Return the non-sensor, non-time columns that feature engineering needs."""
#     needed: set[str] = set()
#     for c in getattr(engine, "spec_columns", []):
#         needed.add(c)
#     needed.update(["rated_fad_cfm", "rated_motor_output_kw"])
#     return sorted(needed)


# def _build_window(demo_row: pd.Series, engine: CompressorInference,
#                   buffer: SensorStreamBuffer, extra_cols: list[str]) -> pd.DataFrame:
#     """Append one demo row to the buffer AND return a window with extra columns."""
#     values = {s: float(demo_row[s]) for s in engine.sensors}
#     phase = str(demo_row["phase"]) if "phase" in demo_row.index and pd.notna(demo_row.get("phase")) else None

#     extra_vals: dict[str, float] = {}
#     for c in extra_cols:
#         if c in demo_row.index and pd.notna(demo_row.get(c)):
#             try:
#                 extra_vals[c] = float(demo_row[c])
#             except (ValueError, TypeError):
#                 extra_vals[c] = 0.0
#         else:
#             extra_vals[c] = 0.0

#     window = buffer.append(float(demo_row["elapsed_time_min"]), values, phase=phase)

#     for c in extra_cols:
#         if c not in window.columns:
#             window[c] = extra_vals[c]
#         else:
#             window.loc[window.index[-1], c] = extra_vals[c]

#     return window, extra_vals


# # ── lifespan ──

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global _engine, _buffer, _extra_cols, _demo_df, _demo_idx

#     raw = os.environ.get("COMPRESSOR_MODEL", str(DEFAULT_MODEL_PATH))
#     model_path = Path(raw)
#     if not model_path.is_file():
#         legacy = _ROOT / "data" / "quantile_models.pkl"
#         if legacy.is_file():
#             model_path = legacy
#     if not model_path.is_file():
#         raise RuntimeError(
#             f"Model not found: {raw}. Train with compressor_train.py "
#             f"(writes data/quantile_models.pkl) or set COMPRESSOR_MODEL."
#         )

#     _engine = CompressorInference(str(model_path))
#     _buffer = SensorStreamBuffer(_engine.sensors)
#     _extra_cols = _identify_extra_columns(_engine)

#     demo_path = os.environ.get("COMPRESSOR_DEMO", "").strip()
#     if not demo_path and DEFAULT_DATA_PATH.is_file():
#         demo_path = str(DEFAULT_DATA_PATH)
#     if demo_path and Path(demo_path).is_file():
#         _demo_df = (
#             pd.read_excel(demo_path) if demo_path.endswith(".xlsx")
#             else pd.read_csv(demo_path)
#         )
#         _demo_df.columns = [c.strip() for c in _demo_df.columns]
#         _demo_df = _demo_df.sort_values("elapsed_time_min").reset_index(drop=True)
#         _demo_idx = 0
#     else:
#         _demo_df = None
#         _demo_idx = 0

#     # ── auto-launch dashboard in browser ──
#     _port = int(os.environ.get("PORT", 8000))
#     # Prefer 127.0.0.1 over localhost to avoid edge DNS/proxy issues on Windows.
#     _url = f"http://127.0.0.1:{_port}/"

#     def _try_open_url(url: str) -> None:
#         """
#         Best-effort open of the dashboard URL.
#         On Windows, `start` is typically more reliable than `webbrowser.open()`.
#         """
#         try:
#             if os.name == "nt":
#                 # Use cmd's start to open default browser; empty title prevents URL-as-title bug.
#                 subprocess.Popen(
#                     ["cmd", "/c", "start", "", url],
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                     shell=False,
#                 )
#                 return
#         except Exception:
#             pass
#         try:
#             webbrowser.open_new_tab(url)
#         except Exception:
#             pass

#     async def _open_browser():
#         # When running with `uvicorn --reload`, lifespan may execute multiple times.
#         # Use a simple env guard so we only try once per terminal session.
#         if os.environ.get("COMPRESSOR_BROWSER_OPENED") == "1":
#             return
#         os.environ["COMPRESSOR_BROWSER_OPENED"] = "1"
#         await asyncio.sleep(1.2)   # give uvicorn time to finish binding
#         _try_open_url(_url)

#     asyncio.ensure_future(_open_browser())

#     yield
#     _ws_clients.clear()


# # ── app ──

# app = FastAPI(title="Compressor Stability API", lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class IngestRequest(BaseModel):
#     """One multivariate sensor snapshot from PLCs / edge gateway."""
#     elapsed_time_min: float = Field(..., ge=0, description="Minutes since test start")
#     values: dict[str, float] = Field(..., description="Sensor column name → value")
#     extras: dict[str, float] = Field(default_factory=dict,
#                                      description="Spec / rated columns (rated_fad_cfm, etc.)")
#     phase: str | None = Field(None, description="Optional run phase label")


# STATIC_DIR = _ROOT / "static"
# _assets = STATIC_DIR / "assets"
# if _assets.is_dir():
#     app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


# # ── routes ──

# @app.get("/")
# async def serve_dashboard():
#     index = STATIC_DIR / "index.html"
#     if not index.is_file():
#         raise HTTPException(404, "static/index.html missing.")
#     return FileResponse(index)


# @app.get("/api/health")
# def health_check():
#     return {"message": "API running"}


# @app.get("/api/v1/meta")
# async def api_meta():
#     assert _engine is not None
#     return {
#         "sensors": _engine.sensors,
#         "stable_onset_min": _engine.stable_time,
#         "stable_ref": _engine.stable_ref,
#         "stable_std": _engine.stable_std,
#         "demo_loaded": _demo_df is not None,
#     }


# @app.post("/api/v1/session/reset")
# async def session_reset():
#     async with _lock:
#         assert _buffer is not None
#         _buffer.clear()
#         global _history, _demo_idx
#         _history = []
#         if _demo_df is not None:
#             _demo_idx = 0
#         global _health_ema, _health_forecasts
#         _health_ema = None
#         _health_forecasts = []
#     return {"ok": True}


# @app.post("/api/v1/ingest")
# async def ingest(body: IngestRequest):
#     """Accept a single sensor snapshot (real-time PLC / edge gateway path)."""
#     async with _lock:
#         assert _engine is not None and _buffer is not None

#         try:
#             window = _buffer.append(body.elapsed_time_min, body.values, phase=body.phase)
#         except ValueError as e:
#             raise HTTPException(400, str(e)) from e

#         for c, v in body.extras.items():
#             if c not in window.columns:
#                 window[c] = v
#             else:
#                 window.loc[window.index[-1], c] = v

#         result = _engine.predict(window)
#         result["extra_columns"] = body.extras
#         result = _augment_result(result, _engine)
#         _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
#         if len(_history) > _history_max:
#             del _history[: len(_history) - _history_max]

#     await _broadcast({"type": "prediction", "payload": result})
#     return result


# @app.get("/api/v1/status")
# async def status():
#     async with _lock:
#         assert _buffer is not None
#         last = _history[-1]["result"] if _history else None
#         return {
#             "buffer_rows": len(_buffer),
#             "history_points": len(_history),
#             "last": last,
#         }


# @app.get("/api/v1/history")
# async def history(limit: int = 200):
#     limit = max(1, min(limit, _history_max))
#     async with _lock:
#         return {"series": _history[-limit:]}


# @app.post("/api/v1/demo/tick")
# async def demo_tick():
#     """Append the next row from the demo file."""
#     async with _lock:
#         if _demo_df is None or _engine is None or _buffer is None:
#             raise HTTPException(400, "Demo file not loaded.")
#         global _demo_idx
#         if _demo_idx >= len(_demo_df):
#             return {"done": True, "message": "End of demo file"}

#         row = _demo_df.iloc[_demo_idx]
#         _demo_idx += 1

#         window, extra_vals = _build_window(row, _engine, _buffer, _extra_cols)
#         result = _engine.predict(window)
#         result["extra_columns"] = extra_vals
#         result = _augment_result(result, _engine)

#         _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
#         if len(_history) > _history_max:
#             del _history[: len(_history) - _history_max]

#     await _broadcast({"type": "prediction", "payload": result})
#     return result


# @app.get("/api/v1/demo/progress")
# async def demo_progress():
#     if _demo_df is None:
#         return {"enabled": False}
#     return {"enabled": True, "index": _demo_idx, "total": len(_demo_df)}


# @app.post("/api/v1/demo/auto")
# async def demo_auto(speed_ms: int = 300):
#     """Stream all remaining demo rows with a configurable delay."""
#     if _demo_df is None or _engine is None or _buffer is None:
#         raise HTTPException(400, "Demo file not loaded.")
#     global _demo_idx
#     results = []
#     while _demo_idx < len(_demo_df):
#         async with _lock:
#             if _demo_idx >= len(_demo_df):
#                 break
#             row = _demo_df.iloc[_demo_idx]
#             _demo_idx += 1

#             window, extra_vals = _build_window(row, _engine, _buffer, _extra_cols)
#             result = _engine.predict(window)
#             result["extra_columns"] = extra_vals
#             result = _augment_result(result, _engine)

#             _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
#             if len(_history) > _history_max:
#                 del _history[: len(_history) - _history_max]

#         await _broadcast({"type": "prediction", "payload": result})
#         results.append(result)
#         await asyncio.sleep(speed_ms / 1000.0)
#     return {"done": True, "ticks": len(results)}


# @app.get("/api/v1/checkpoints")
# async def checkpoints():
#     """Prediction accuracy at standard time checkpoints."""
#     async with _lock:
#         if not _history:
#             return {"checkpoints": []}

#         cp_minutes = [30, 60, 90, 120, 150, 180]
#         last_result = _history[-1]["result"]
#         max_elapsed = last_result.get("elapsed_min", 0)
#         result_list = []

#         for cp in cp_minutes:
#             if max_elapsed < cp - 0.5:
#                 result_list.append({"checkpoint_min": cp, "available": False})
#                 continue

#             best = None
#             best_dist = float("inf")
#             for h in _history:
#                 em = h.get("elapsed_min") or 0
#                 dist = abs(em - cp)
#                 if dist < best_dist:
#                     best_dist = dist
#                     best = h["result"]

#             if best is None:
#                 result_list.append({"checkpoint_min": cp, "available": False})
#                 continue

#             predicted = best.get("predicted_sensors", {})
#             actual_at_cp = best.get("current_sensors", {})
#             errors: dict[str, float | None] = {}
#             for s, pv in predicted.items():
#                 av = actual_at_cp.get(s)
#                 if av and abs(av) > 1e-9:
#                     errors[s] = round(abs(pv - av) / abs(av) * 100, 2)
#                 else:
#                     errors[s] = None

#             result_list.append({
#                 "checkpoint_min": cp,
#                 "available": True,
#                 "elapsed_min": best.get("elapsed_min"),
#                 "predicted_sensors": predicted,
#                 "actual_sensors": actual_at_cp,
#                 "error_pct": errors,
#                 "action": best.get("action"),
#                 "confidence_pct": best.get("confidence_pct"),
#                 "time_to_stability_min": best.get("time_to_stability_min"),
#             })

#         return {"checkpoints": result_list}


# @app.websocket("/ws/live")
# async def ws_live(ws: WebSocket):
#     await ws.accept()
#     _ws_clients.add(ws)
#     try:
#         async with _lock:
#             snap = _history[-1]["result"] if _history else None
#         await ws.send_json({
#             "type": "hello",
#             "payload": snap,
#             "meta": {"sensors": list(_engine.sensors) if _engine else []},
#         })
#         while True:
#             await ws.receive_text()
#     except WebSocketDisconnect:
#         pass
#     finally:
#         _ws_clients.discard(ws)


# if __name__ == "__main__":
#     import socket
#     import uvicorn

#     def find_free_port(start: int = 8000, max_tries: int = 50) -> int:
#         for port in range(start, start + max_tries):
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 try:
#                     s.bind(("127.0.0.1", port))
#                     return port
#                 except OSError:
#                     continue
#         raise RuntimeError("No free ports available")

#     host = os.environ.get("HOST", "0.0.0.0")

#     port_env = os.environ.get("PORT")
#     if port_env:
#         port = int(port_env)
#     else:
#         port = find_free_port()

#     print(f" Server starting on {host}:{port}")
#     print(f" Open in browser: http://localhost:{port}/")

#     uvicorn.run(
#         "fastapi_app:app",
#         host=host,
#         port=port,
#         reload=False,  # keep False — reload causes double browser open
#     )






"""
fastapi_app.py — Compressor Stability & Health Monitoring API
=============================================================

Health score v2 design (deviation-based, explainable, production-grade):
  - Anchored scaling: 2sigma weighted deviation → Warning band (~75)
  - Two-tier sensor scale: stable_std if trusted (>=3% of ref), else 5% of ref
  - Smooth baseline blending: forecast_matured ramps toward forecast over time
  - Derived metrics: efficiency ratio (FAD/power), thermal load (temp/power)
  - Temporal awareness: rolling drift detection (15-60 tick windows)
  - Combined score: 70% instantaneous + 30% drift-adjusted
  - Noise deadband (0.15 sigma) + deviation clamping (8 sigma max)
  - Confidence scoring based on sensor coverage + data maturity
  - Persistence logic: bands only change after N consecutive ticks
  - Explainability: top driver, human-readable interpretation, rule-based causes
  - Degradation vs anomaly classification per channel
  - EMA smoothing with configurable alpha

Environment overrides:
  COMPRESSOR_MODEL               : path to model .pkl
  COMPRESSOR_DEMO                : path to demo CSV/XLSX
  COMPRESSOR_HEALTH_WEIGHTS      : JSON object {"sensor": weight, ...}
  COMPRESSOR_HEALTH_EMA_ALPHA    : float in (0,1], default 0.2
  COMPRESSOR_HEALTH_EXPECTED_MODE: forecast_matured | stable_ref | forecast
  PORT                           : HTTP port (default 8000)
  HOST                           : bind host (default 0.0.0.0)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import webbrowser
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
from src.health_scorer import CompressorHealthScorer

_ROOT = Path(__file__).resolve().parent

# ── runtime state ──────────────────────────────────────────────────────────────
_engine:     CompressorInference       | None = None
_buffer:     SensorStreamBuffer        | None = None
_scorer:     CompressorHealthScorer    | None = None
_extra_cols: list[str]                        = []
_demo_df:    pd.DataFrame              | None = None
_demo_idx:   int                              = 0
_history:    list[dict[str, Any]]             = []
_history_max = 600
_ws_clients: set[WebSocket]                   = set()
_lock = asyncio.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH SCORING — delegated to CompressorHealthScorer
# ══════════════════════════════════════════════════════════════════════════════

def _attach_health_score(result: dict, engine: CompressorInference) -> None:
    """Compute and merge all health fields into *result* via the scorer."""
    if _scorer is None:
        return
    try:
        fh = int(getattr(engine, "future_horizon", 10))
    except (TypeError, ValueError):
        fh = 10

    health = _scorer.score(result, future_horizon=fh)
    if health is not None:
        result.update(health)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERAL helpers
# ══════════════════════════════════════════════════════════════════════════════

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / tuples so json.dumps won't choke."""
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
    """
    Attach param_status, readiness, current_vs_ref, forecast_delta,
    and health score fields to a raw inference result dict.
    """
    cur    = result.get("current_sensors", {})
    pred   = result.get("predicted_sensors", {})
    extras = result.get("extra_columns", {})

    # ── FAD readiness ──────────────────────────────────────────────────────────
    fad_now   = cur.get("fad_cfm")
    rated_fad = extras.get("rated_fad_cfm") or engine.stable_ref.get("fad_cfm", 0.0)

    tolerance_pct = (
        extras.get("tolerance_flow_pct")
        or extras.get("flow_tolerance_pct(%)")
        or 6.0
    )
    tolerance_factor = (100.0 - float(tolerance_pct)) / 100.0
    spec_min         = rated_fad * tolerance_factor if rated_fad else None
    margin_pct: float | None = None
    if spec_min and fad_now is not None and spec_min > 0:
        margin_pct = round(((fad_now - spec_min) / spec_min) * 100, 2)

    # ── Motor readiness ────────────────────────────────────────────────────────
    motor_now   = cur.get("motor_output_power_kw")
    rated_motor = extras.get("rated_motor_output_kw") or engine.stable_ref.get(
        "motor_output_power_kw", 0.0
    )
    motor_tolerance_pct = float(
        extras.get("(motor_power_tolerance_pct(%))")
        or extras.get("motor_power_tolerance_pct")
        or extras.get("motor_power_tolerance_pct(%)")
        or 6.0
    )
    motor_tol_factor  = (100.0 - motor_tolerance_pct) / 100.0
    motor_spec_min    = rated_motor * motor_tol_factor if rated_motor else None
    motor_margin_pct: float | None = None
    if motor_spec_min and motor_now is not None and motor_spec_min > 0:
        motor_margin_pct = round(
            ((motor_now - motor_spec_min) / motor_spec_min) * 100, 2
        )

    result["readiness"] = {
        "spec_min":        round(spec_min, 2)       if spec_min       else None,
        "margin_pct":      margin_pct,
        "ss_ref":          round(rated_fad, 2)      if rated_fad      else None,
        "tolerance_pct":   tolerance_pct,
        "motor_spec_min":  round(motor_spec_min, 2) if motor_spec_min else None,
        "motor_margin_pct": motor_margin_pct,
    }

    # ── Parameter status (σ-based RAG) ─────────────────────────────────────────
    g, a, r      = 0, 0, 0
    red_names:   list[str]              = []
    current_vs_ref: dict[str, float | None] = {}
    forecast_delta: dict[str, float | None] = {}

    for s in engine.sensors:
        val = cur.get(s)
        ref = engine.stable_ref.get(s)
        std = engine.stable_std.get(s)
        sig: float | None = None

        if val is not None and ref is not None and std and std > 0:
            sig = round((val - ref) / std, 4)
            dev = abs(sig)
            if dev < 1:
                g += 1
            elif dev < 2:
                a += 1
            else:
                r += 1
                red_names.append(s)
        elif val is not None:
            g += 1

        current_vs_ref[s] = sig

        fv = pred.get(s)
        forecast_delta[s] = (
            round(fv - val, 4) if fv is not None and val is not None else None
        )

    result["param_status"]    = {"green": g, "amber": a, "red": r, "red_names": red_names}
    result["current_vs_ref"]  = current_vs_ref
    result["forecast_delta"]  = forecast_delta

    # ── Health score (must be last — uses fields set above) ────────────────────
    _attach_health_score(result, engine)

    return result


def _identify_extra_columns(engine: CompressorInference) -> list[str]:
    """Return the non-sensor, non-time columns that feature engineering needs."""
    needed: set[str] = set()
    for c in getattr(engine, "spec_columns", []):
        needed.add(c)
    needed.update(["rated_fad_cfm", "rated_motor_output_kw"])
    return sorted(needed)


def _build_window(
    demo_row:   pd.Series,
    engine:     CompressorInference,
    buffer:     SensorStreamBuffer,
    extra_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Append one demo row to the buffer and return a window with extra columns."""
    values = {s: float(demo_row[s]) for s in engine.sensors}
    phase  = (
        str(demo_row["phase"])
        if "phase" in demo_row.index and pd.notna(demo_row.get("phase"))
        else None
    )

    extra_vals: dict[str, float] = {}
    for c in extra_cols:
        if c in demo_row.index and pd.notna(demo_row.get(c)):
            try:
                extra_vals[c] = float(demo_row[c])
            except (ValueError, TypeError):
                extra_vals[c] = 0.0
        else:
            extra_vals[c] = 0.0

    window = buffer.append(float(demo_row["elapsed_time_min"]), values, phase=phase)

    for c in extra_cols:
        if c not in window.columns:
            window[c] = extra_vals[c]
        else:
            window.loc[window.index[-1], c] = extra_vals[c]

    return window, extra_vals


async def _broadcast(payload: dict) -> None:
    """Send a JSON payload to all connected WebSocket clients."""
    raw    = json.dumps(_json_safe(payload))
    stale: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(raw)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _ws_clients.discard(ws)


# ══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _buffer, _scorer, _extra_cols, _demo_df, _demo_idx

    # ── Load model ─────────────────────────────────────────────────────────────
    raw        = os.environ.get("COMPRESSOR_MODEL", str(DEFAULT_MODEL_PATH))
    model_path = Path(raw)
    if not model_path.is_file():
        legacy = _ROOT / "data" / "quantile_models.pkl"
        if legacy.is_file():
            model_path = legacy
    if not model_path.is_file():
        raise RuntimeError(
            f"Model not found: {raw}. "
            "Train with compressor_train.py (writes data/quantile_models.pkl) "
            "or set COMPRESSOR_MODEL."
        )

    _engine     = CompressorInference(str(model_path))
    _buffer     = SensorStreamBuffer(_engine.sensors)
    _extra_cols = _identify_extra_columns(_engine)

    # ── Instantiate health scorer ──────────────────────────────────────────────
    _scorer = CompressorHealthScorer(
        stable_ref=dict(_engine.stable_ref),
        stable_std=dict(_engine.stable_std),
        sensors=list(_engine.sensors),
    )

    # ── Load demo data ─────────────────────────────────────────────────────────
    demo_path = os.environ.get("COMPRESSOR_DEMO", "").strip()
    if not demo_path and DEFAULT_DATA_PATH.is_file():
        demo_path = str(DEFAULT_DATA_PATH)

    if demo_path and Path(demo_path).is_file():
        _demo_df = (
            pd.read_excel(demo_path)
            if demo_path.endswith(".xlsx")
            else pd.read_csv(demo_path)
        )
        _demo_df.columns = [c.strip() for c in _demo_df.columns]
        _demo_df = _demo_df.sort_values("elapsed_time_min").reset_index(drop=True)
        _demo_idx = 0
    else:
        _demo_df  = None
        _demo_idx = 0

    # ── Auto-launch dashboard ──────────────────────────────────────────────────
    _port = int(os.environ.get("PORT", 8000))
    _url  = f"http://127.0.0.1:{_port}/"

    def _try_open_url(url: str) -> None:
        try:
            if os.name == "nt":
                subprocess.Popen(
                    ["cmd", "/c", "start", "", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                return
        except Exception:
            pass
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    async def _open_browser() -> None:
        if os.environ.get("COMPRESSOR_BROWSER_OPENED") == "1":
            return
        os.environ["COMPRESSOR_BROWSER_OPENED"] = "1"
        await asyncio.sleep(1.2)
        _try_open_url(_url)

    asyncio.ensure_future(_open_browser())

    yield

    _ws_clients.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════

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
    values:  dict[str, float] = Field(...,                  description="Sensor column → value")
    extras:  dict[str, float] = Field(default_factory=dict, description="Spec / rated columns")
    phase:   str | None       = Field(None,                 description="Optional run-phase label")


STATIC_DIR = _ROOT / "static"
_assets    = STATIC_DIR / "assets"
if _assets.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def serve_dashboard():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "static/index.html missing.")
    return FileResponse(
        index,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/health")
def health_check():
    return {"message": "API running"}


@app.get("/api/v1/meta")
async def api_meta():
    assert _engine is not None
    health_cfg: dict[str, Any] = {}
    if _scorer is not None:
        health_cfg = {
            "weights":       _scorer.weights,
            "ema_alpha":     _scorer.ema_alpha,
            "expected_mode": _scorer.expected_mode,
            "scored_channels": _scorer.scored_channels,
            "scoring_info":  "2sigma → Warning (~75); 3sigma → Critical; drift + persistence filtering",
        }
    return {
        "sensors":          _engine.sensors,
        "stable_onset_min": _engine.stable_time,
        "stable_ref":       _engine.stable_ref,
        "stable_std":       _engine.stable_std,
        "demo_loaded":      _demo_df is not None,
        "health_config":    health_cfg,
    }


@app.post("/api/v1/session/reset")
async def session_reset():
    """Reset buffer, history, and health scorer state."""
    async with _lock:
        assert _buffer is not None
        _buffer.clear()

        global _history, _demo_idx
        _history = []
        if _demo_df is not None:
            _demo_idx = 0

        if _scorer is not None:
            _scorer.reset()

    return {"ok": True}


@app.post("/api/v1/ingest")
async def ingest(body: IngestRequest):
    """Accept a single sensor snapshot (real-time PLC / edge-gateway path)."""
    async with _lock:
        assert _engine is not None and _buffer is not None

        try:
            window = _buffer.append(body.elapsed_time_min, body.values, phase=body.phase)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        for c, v in body.extras.items():
            if c not in window.columns:
                window[c] = v
            else:
                window.loc[window.index[-1], c] = v

        result                  = _engine.predict(window)
        result["extra_columns"] = body.extras
        result                  = _augment_result(result, _engine)

        _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
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
            "buffer_rows":     len(_buffer),
            "history_points":  len(_history),
            "last":            last,
        }


@app.get("/api/v1/history")
async def history(limit: int = 200):
    limit = max(1, min(limit, _history_max))
    async with _lock:
        return {"series": _history[-limit:]}


@app.get("/api/v1/history/health")
async def history_health(limit: int = 200):
    """
    Lightweight endpoint — returns the full health-score time-series
    including new fields: confidence, top driver, interpretations, drift,
    issue types, and derived metrics.
    """
    limit = max(1, min(limit, _history_max))
    _HEALTH_KEYS = (
        "health_score", "health_score_instant", "health_score_drift",
        "health_score_smoothed", "health_band", "health_band_raw",
        "health_confidence", "health_low_confidence",
        "health_deviation", "health_signed_deviation",
        "health_risk_contrib", "health_valid_sensors",
        "health_top_driver", "health_explanation",
        "health_interpretations", "health_issue_types",
        "health_channel_slopes", "health_score_trend",
        "health_derived_metrics",
    )
    async with _lock:
        series = []
        for h in _history[-limit:]:
            r = h["result"]
            entry: dict[str, Any] = {"elapsed_min": h.get("elapsed_min")}
            for k in _HEALTH_KEYS:
                entry[k] = r.get(k)
            series.append(entry)
    return {"series": series}


@app.post("/api/v1/demo/tick")
async def demo_tick():
    """Append the next row from the demo file."""
    async with _lock:
        if _demo_df is None or _engine is None or _buffer is None:
            raise HTTPException(400, "Demo file not loaded.")

        global _demo_idx
        if _demo_idx >= len(_demo_df):
            return {"done": True, "message": "End of demo file"}

        row       = _demo_df.iloc[_demo_idx]
        _demo_idx += 1

        window, extra_vals      = _build_window(row, _engine, _buffer, _extra_cols)
        result                  = _engine.predict(window)
        result["extra_columns"] = extra_vals
        result                  = _augment_result(result, _engine)

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
    """Stream all remaining demo rows with a configurable inter-tick delay."""
    if _demo_df is None or _engine is None or _buffer is None:
        raise HTTPException(400, "Demo file not loaded.")

    global _demo_idx
    results: list[dict] = []

    while _demo_idx < len(_demo_df):
        async with _lock:
            if _demo_idx >= len(_demo_df):
                break

            row       = _demo_df.iloc[_demo_idx]
            _demo_idx += 1

            window, extra_vals      = _build_window(row, _engine, _buffer, _extra_cols)
            result                  = _engine.predict(window)
            result["extra_columns"] = extra_vals
            result                  = _augment_result(result, _engine)

            _history.append({"elapsed_min": result.get("elapsed_min"), "result": result})
            if len(_history) > _history_max:
                del _history[: len(_history) - _history_max]

        await _broadcast({"type": "prediction", "payload": result})
        results.append(result)
        await asyncio.sleep(speed_ms / 1000.0)

    return {"done": True, "ticks": len(results)}


@app.get("/api/v1/checkpoints")
async def checkpoints():
    """Prediction accuracy at standard time checkpoints (30, 60 … 180 min)."""
    async with _lock:
        if not _history:
            return {"checkpoints": []}

        cp_minutes  = [30, 60, 90, 120, 150, 180]
        last_result = _history[-1]["result"]
        max_elapsed = last_result.get("elapsed_min", 0)
        result_list = []

        for cp in cp_minutes:
            if max_elapsed < cp - 0.5:
                result_list.append({"checkpoint_min": cp, "available": False})
                continue

            best: dict | None = None
            best_dist         = float("inf")
            for h in _history:
                em   = h.get("elapsed_min") or 0
                dist = abs(em - cp)
                if dist < best_dist:
                    best_dist = dist
                    best      = h["result"]

            if best is None:
                result_list.append({"checkpoint_min": cp, "available": False})
                continue

            predicted    = best.get("predicted_sensors", {})
            actual_at_cp = best.get("current_sensors", {})
            errors: dict[str, float | None] = {}
            for s, pv in predicted.items():
                av = actual_at_cp.get(s)
                if av and abs(av) > 1e-9:
                    errors[s] = round(abs(pv - av) / abs(av) * 100, 2)
                else:
                    errors[s] = None

            result_list.append({
                "checkpoint_min":        cp,
                "available":             True,
                "elapsed_min":           best.get("elapsed_min"),
                "predicted_sensors":     predicted,
                "actual_sensors":        actual_at_cp,
                "error_pct":             errors,
                "action":                best.get("action"),
                "confidence_pct":        best.get("confidence_pct"),
                "time_to_stability_min": best.get("time_to_stability_min"),
                # Include health snapshot at checkpoint.
                "health_score":          best.get("health_score"),
                "health_score_smoothed": best.get("health_score_smoothed"),
                "health_band":           best.get("health_band"),
            })

        return {"checkpoints": result_list}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        async with _lock:
            snap = _history[-1]["result"] if _history else None
        await ws.send_json({
            "type":    "hello",
            "payload": snap,
            "meta":    {"sensors": list(_engine.sensors) if _engine else []},
        })
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import socket
    import uvicorn

    def find_free_port(start: int = 8000, max_tries: int = 50) -> int:
        for port in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No free ports available")

    host = os.environ.get("HOST", "0.0.0.0")

    port_env = os.environ.get("PORT")
    port     = int(port_env) if port_env else find_free_port()

    print(f" Server starting on {host}:{port}")
    print(f" Open in browser: http://localhost:{port}/")

    uvicorn.run(
        "fastapi_app:app",
        host=host,
        port=port,
        reload=False,
    )