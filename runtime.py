from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from bpy.types import Context

from .preferences import get_preferences
from .runtime_setup import resolve_hf_token, resolved_checkpoint_root, resolved_text_encoder_root


@dataclass
class GenerationJobHandle:
    process: subprocess.Popen[str]
    temp_dir: Path
    response_path: Path
    keep_temp_files: bool
    reader_thread: threading.Thread | None = None
    log_chunks: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)


_GENERATION_JOB: GenerationJobHandle | None = None
_GENERATION_STATE_LOCK = threading.Lock()
_GENERATION_STATE: dict[str, object] = {
    "active": False,
    "phase": "idle",
    "progress": 0.0,
    "status_text": "",
    "detail_text": "",
    "error_text": "",
    "model_name": "",
    "num_frames": 0,
}

_TQDM_PROGRESS_RE = re.compile(r"(?P<pct>\d{1,3})%\|.*?(?P<current>\d+)/(?P<total>\d+)")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _set_generation_job_state(**updates: object) -> None:
    with _GENERATION_STATE_LOCK:
        _GENERATION_STATE.update(updates)


def get_generation_job_state() -> dict[str, object]:
    with _GENERATION_STATE_LOCK:
        return dict(_GENERATION_STATE)


def update_generation_job_state(**updates: object) -> None:
    _set_generation_job_state(**updates)


def _clean_progress_text(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text).strip()


def _update_progress_from_output(fragment: str) -> None:
    cleaned = _clean_progress_text(fragment)
    if not cleaned:
        return

    state = get_generation_job_state()
    detail_text = cleaned[-220:]
    phase = str(state.get("phase", "idle"))
    progress = float(state.get("progress", 0.0) or 0.0)
    status_text = str(state.get("status_text", ""))

    if "loading weights:" in cleaned.lower():
        match = _TQDM_PROGRESS_RE.search(cleaned)
        if match:
            total = max(int(match.group("total")), 1)
            current = min(int(match.group("current")), total)
            progress = max(progress, 0.15 * (current / float(total)))
            status_text = f"Loading model weights... {match.group('pct')}%"
            phase = "loading"
    else:
        match = _TQDM_PROGRESS_RE.search(cleaned)
        if match:
            total = max(int(match.group("total")), 1)
            current = min(int(match.group("current")), total)
            progress = max(progress, 0.15 + (0.80 * (current / float(total))))
            status_text = f"Generating in-betweens... {match.group('pct')}%"
            phase = "generating"
        elif "generated " in cleaned.lower() and " frames with " in cleaned.lower():
            progress = max(progress, 0.95)
            status_text = "Applying generated motion in Blender..."
            phase = "applying"

    _set_generation_job_state(
        detail_text=detail_text,
        phase=phase,
        progress=min(0.99, progress),
        status_text=status_text or cleaned,
    )


def _consume_generation_output(job: GenerationJobHandle) -> None:
    stream = job.process.stdout
    if stream is None:
        return

    pending = ""
    try:
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            job.log_chunks.append(chunk)
            pending += chunk
            while True:
                split_index = -1
                delimiter_size = 1
                for delimiter in ("\r\n", "\n", "\r"):
                    index = pending.find(delimiter)
                    if index == -1:
                        continue
                    if split_index == -1 or index < split_index:
                        split_index = index
                        delimiter_size = len(delimiter)
                if split_index == -1:
                    break
                fragment = pending[:split_index]
                pending = pending[split_index + delimiter_size:]
                _update_progress_from_output(fragment)
    finally:
        if pending:
            _update_progress_from_output(pending)
        try:
            stream.close()
        except Exception:
            pass


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


def _build_generation_subprocess(context: Context, request: dict) -> tuple[subprocess.Popen[str], Path, Path, bool]:
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
    env["PYTHONUNBUFFERED"] = "1"
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
    process = subprocess.Popen(
        [python_executable, "-u", str(worker_script), "--request", str(request_path), "--response", str(response_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    return process, temp_dir, response_path, bool(prefs.keep_temp_files)


def start_generation_job(context: Context, request: dict) -> dict[str, object]:
    global _GENERATION_JOB
    if _GENERATION_JOB is not None:
        raise RuntimeError("A Beyond Motion generation job is already running.")

    process, temp_dir, response_path, keep_temp_files = _build_generation_subprocess(context, request)
    job = GenerationJobHandle(
        process=process,
        temp_dir=temp_dir,
        response_path=response_path,
        keep_temp_files=keep_temp_files,
    )
    reader_thread = threading.Thread(target=_consume_generation_output, args=(job,), daemon=True)
    job.reader_thread = reader_thread
    _GENERATION_JOB = job
    _set_generation_job_state(
        active=True,
        phase="starting",
        progress=0.0,
        status_text="Starting local generation...",
        detail_text="Launching the Kimodo worker.",
        error_text="",
        model_name=str(request.get("model_name", "")),
        num_frames=int(request.get("num_frames", 0) or 0),
    )
    reader_thread.start()
    return get_generation_job_state()


def generation_job_is_active() -> bool:
    return bool(get_generation_job_state().get("active", False))


def generation_job_timed_out(timeout_seconds: int) -> bool:
    if _GENERATION_JOB is None:
        return False
    return (time.monotonic() - _GENERATION_JOB.started_at) > float(timeout_seconds)


def generation_job_ready_to_collect() -> bool:
    if _GENERATION_JOB is None:
        return False
    reader_thread = _GENERATION_JOB.reader_thread
    return _GENERATION_JOB.process.poll() is not None and (reader_thread is None or not reader_thread.is_alive())


def cancel_generation_job(reason: str) -> None:
    global _GENERATION_JOB
    job = _GENERATION_JOB
    if job is None:
        _set_generation_job_state(
            active=False,
            phase="error",
            progress=0.0,
            status_text=reason,
            error_text=reason,
        )
        return

    try:
        if job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                job.process.kill()
    except Exception:
        pass
    try:
        if job.reader_thread is not None:
            job.reader_thread.join(timeout=1.0)
    except Exception:
        pass
    if not job.keep_temp_files:
        shutil.rmtree(job.temp_dir, ignore_errors=True)
    _GENERATION_JOB = None
    _set_generation_job_state(
        active=False,
        phase="error",
        progress=0.0,
        status_text=reason,
        detail_text="",
        error_text=reason,
    )


def collect_generation_job_result() -> tuple[dict[str, np.ndarray], dict, str]:
    global _GENERATION_JOB
    job = _GENERATION_JOB
    if job is None:
        raise RuntimeError("No Beyond Motion generation job is running.")
    if not generation_job_ready_to_collect():
        raise RuntimeError("Beyond Motion generation is still running.")

    return_code = job.process.poll()
    log_output = "".join(job.log_chunks).strip()
    if return_code != 0:
        if not job.keep_temp_files:
            shutil.rmtree(job.temp_dir, ignore_errors=True)
        _GENERATION_JOB = None
        error_text = _summarize_worker_error(log_output)
        _set_generation_job_state(
            active=False,
            phase="error",
            progress=0.0,
            status_text=error_text,
            detail_text="",
            error_text=error_text,
        )
        raise RuntimeError(error_text)

    response = json.loads(job.response_path.read_text(encoding="utf-8"))
    output_npz_path = Path(response["output_npz"])
    with np.load(output_npz_path) as output_data:
        result = {key: output_data[key] for key in output_data.files}

    if not job.keep_temp_files:
        shutil.rmtree(job.temp_dir, ignore_errors=True)
    _GENERATION_JOB = None
    _set_generation_job_state(
        active=True,
        phase="applying",
        progress=max(float(get_generation_job_state().get("progress", 0.0) or 0.0), 0.96),
        status_text="Applying generated motion in Blender...",
        detail_text="Writing the generated motion back to the rig.",
        error_text="",
    )
    return result, response, log_output


def complete_generation_job(success_text: str) -> None:
    _set_generation_job_state(
        active=False,
        phase="complete",
        progress=1.0,
        status_text=success_text,
        detail_text="",
        error_text="",
    )


def fail_generation_job(error_text: str) -> None:
    _set_generation_job_state(
        active=False,
        phase="error",
        progress=0.0,
        status_text=error_text,
        detail_text="",
        error_text=error_text,
    )


def run_generation_job(context: Context, request: dict) -> tuple[dict[str, np.ndarray], dict, str]:
    start_generation_job(context, request)
    prefs = get_preferences(context)
    timeout_seconds = prefs.job_timeout_seconds if prefs is not None else 600
    while not generation_job_ready_to_collect():
        if generation_job_timed_out(timeout_seconds):
            cancel_generation_job("Beyond Motion timed out while generating local motion.")
            raise RuntimeError("Beyond Motion timed out while generating local motion.")
        time.sleep(0.1)
    result, response, log_output = collect_generation_job_result()
    complete_generation_job(
        f"Generated {int(response.get('num_frames', request.get('num_frames', 0) or 0))} frames with {request.get('model_name', 'kimodo')}."
    )
    return result, response, log_output
