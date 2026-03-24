from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from bpy.types import Context

from .preferences import get_preferences
from .runtime_setup import resolve_hf_token, resolved_checkpoint_root, resolved_text_encoder_root


def _summarize_worker_error(log_output: str) -> str:
    lower_log = log_output.lower()
    if "gatedrepoerror" in lower_log or "you are trying to access a gated repo" in lower_log:
        return (
            "A gated Hugging Face model was requested. Beyond Motion now defaults to an open local text encoder, "
            "so this usually means a legacy encoder path is still selected somewhere."
        )
    if "401 unauthorized" in lower_log and "huggingface.co" in lower_log:
        return (
            "Hugging Face rejected the model download. If you switched back to a gated legacy encoder, "
            "check the token and access approval. Otherwise, retry Prepare Runtime Assets."
        )
    if "offline mode is enabled" in lower_log or "local_files_only=true" in lower_log:
        return (
            "Offline Only is enabled, but the required Kimodo checkpoints or text encoder files are not cached locally yet."
        )
    if "mps tensor to float64" in lower_log or "mps framework doesn't support float64" in lower_log:
        if "constraints.py" in lower_log or "load_constraints_lst" in lower_log:
            return (
                "Kimodo hit a float64 tensor while moving constraint data onto Apple Metal. "
                "Reload Beyond Motion or restart Blender so the latest MPS constraint patch is active, then retry."
            )
        if "kimodo_model.py" in lower_log:
            return (
                "Kimodo hit a float64 tensor while moving the Kimodo model onto Apple Metal. "
                "Reload Beyond Motion or restart Blender so the latest MPS model patch is active, then retry."
            )
        return (
            "Kimodo hit an internal float64 tensor on Apple Metal. "
            "Reload Beyond Motion or restart Blender so the latest MPS patch is active, then retry."
        )
    return log_output or "Kimodo worker failed without output."


def run_generation_job(context: Context, request: dict) -> tuple[dict[str, np.ndarray], dict, str]:
    prefs = get_preferences(context)
    if prefs is None:
        raise RuntimeError("Beyond Motion preferences are unavailable.")

    worker_script = Path(__file__).resolve().parent / "worker" / "kimodo_generate.py"
    temp_dir = Path(tempfile.mkdtemp(prefix="beyond_motion_"))
    request_path = temp_dir / "request.json"
    response_path = temp_dir / "response.json"

    request_payload = dict(request)
    request_payload.setdefault("device", prefs.torch_device)
    request_payload.setdefault("enable_mps_fallback", prefs.enable_mps_fallback)
    request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

    env = os.environ.copy()
    text_encoder_mode = prefs.text_encoder_mode
    if prefs.offline_only and text_encoder_mode == "auto":
        text_encoder_mode = "local"
    env["TEXT_ENCODER_MODE"] = text_encoder_mode
    if prefs.text_encoder_url:
        env["TEXT_ENCODER_URL"] = prefs.text_encoder_url
    env["CHECKPOINT_DIR"] = str(resolved_checkpoint_root(prefs.checkpoint_dir))
    text_encoder_root = resolved_text_encoder_root()
    if text_encoder_root.is_dir():
        env["TEXT_ENCODERS_DIR"] = str(text_encoder_root)
    resolved_token, _token_source = resolve_hf_token(prefs.hf_token)
    if resolved_token:
        env["HF_TOKEN"] = resolved_token
    if prefs.offline_only:
        env["LOCAL_CACHE"] = "true"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env["LOCAL_CACHE"] = "false"
        env["HF_HUB_OFFLINE"] = "0"
        env["TRANSFORMERS_OFFLINE"] = "0"

    python_executable = prefs.resolved_python_executable()
    process = subprocess.run(
        [python_executable, str(worker_script), "--request", str(request_path), "--response", str(response_path)],
        capture_output=True,
        text=True,
        timeout=prefs.job_timeout_seconds,
        env=env,
    )

    stdout = process.stdout or ""
    stderr = process.stderr or ""
    log_output = (stdout + ("\n" + stderr if stderr else "")).strip()

    if process.returncode != 0:
        if not prefs.keep_temp_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(_summarize_worker_error(log_output))

    response = json.loads(response_path.read_text(encoding="utf-8"))
    output_npz_path = Path(response["output_npz"])
    with np.load(output_npz_path) as output_data:
        result = {key: output_data[key] for key in output_data.files}

    if not prefs.keep_temp_files:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result, response, log_output
