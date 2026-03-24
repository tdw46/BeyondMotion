from __future__ import annotations

import os
import threading
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

try:
    import bpy
except ImportError:  # pragma: no cover - Blender only
    bpy = None

from .dependency_manager import extension_root


MODEL_REPO_IDS = {
    "kimodo-soma-rp": "nvidia/Kimodo-SOMA-RP-v1",
    "kimodo-soma-seed": "nvidia/Kimodo-SOMA-SEED-v1",
}

MODEL_FOLDER_NAMES = {
    "kimodo-soma-rp": "Kimodo-SOMA-RP-v1",
    "kimodo-soma-seed": "Kimodo-SOMA-SEED-v1",
}

LOCAL_TEXT_ENCODER_REPOS = ("codefuse-ai/F2LLM-v2-8B",)
META_ACCESS_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKENS_URL = "https://huggingface.co/settings/tokens"
META_ACCESS_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"

_RUNTIME_ASSET_JOB_LOCK = threading.Lock()
_RUNTIME_ASSET_JOB_STATE = {
    "active": False,
    "model_name": "",
    "status_text": "",
    "error_text": "",
    "notice_text": "",
    "notice_icon": "INFO",
}


@dataclass
class RuntimeSetupStatus:
    ready: bool
    issues: list[str]
    warnings: list[str]
    model_name: str
    model_ready: bool
    model_path: Path
    text_encoder_mode: str
    text_encoder_service_reachable: bool
    local_text_encoder_ready: bool
    text_encoder_path: Path
    hf_token_available: bool
    hf_token_source: str
    meta_access_approved: bool
    meta_access_known: bool
    pending_meta_access_launch: bool
    next_auth_step: str


def get_runtime_asset_job_state() -> dict:
    with _RUNTIME_ASSET_JOB_LOCK:
        return dict(_RUNTIME_ASSET_JOB_STATE)


def update_runtime_asset_job_state(**changes) -> dict:
    with _RUNTIME_ASSET_JOB_LOCK:
        _RUNTIME_ASSET_JOB_STATE.update(changes)
        return dict(_RUNTIME_ASSET_JOB_STATE)


def clear_runtime_asset_job_notice() -> None:
    update_runtime_asset_job_state(notice_text="", notice_icon="INFO")


def models_root() -> Path:
    return extension_root() / "models"


def checkpoints_root() -> Path:
    return models_root() / "checkpoints"


def text_encoders_root() -> Path:
    return models_root() / "text_encoders"


def ensure_runtime_asset_directories() -> None:
    checkpoints_root().mkdir(parents=True, exist_ok=True)
    text_encoders_root().mkdir(parents=True, exist_ok=True)


def runtime_setup_state_path() -> Path:
    return models_root() / "setup_state.json"


def _resolve_path(path_text: str) -> Path:
    path_text = (path_text or "").strip()
    if path_text.startswith("//") and bpy is not None:
        return Path(bpy.path.abspath(path_text))
    return Path(path_text).expanduser()


def resolved_checkpoint_root(checkpoint_dir_override: str) -> Path:
    if checkpoint_dir_override.strip():
        return _resolve_path(checkpoint_dir_override)
    return checkpoints_root()


def resolved_model_path(model_name: str, checkpoint_dir_override: str) -> Path:
    folder_name = MODEL_FOLDER_NAMES.get(model_name, model_name)
    return resolved_checkpoint_root(checkpoint_dir_override) / folder_name


def resolved_text_encoder_root() -> Path:
    return text_encoders_root()


def _candidate_hf_token_paths() -> list[Path]:
    env_token_path = (os.environ.get("HF_TOKEN_PATH") or "").strip()
    if env_token_path:
        return [Path(env_token_path).expanduser()]

    home = Path.home()
    default_home = home / ".cache"
    hf_home = Path(
        os.path.expanduser(
            os.path.expandvars(
                os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(default_home))) / "huggingface"))
            )
        )
    )
    return [
        hf_home / "token",
        home / ".huggingface" / "token",
    ]


def resolve_hf_token(hf_token: str) -> tuple[str, str]:
    explicit_token = (hf_token or "").strip()
    if explicit_token:
        return explicit_token, "preferences"

    env_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if env_token:
        return env_token, "environment"

    for token_path in _candidate_hf_token_paths():
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token, f"saved login ({token_path})"

    return "", ""


def _token_fingerprint(token: str) -> str:
    if not token:
        return ""
    return sha256(token.encode("utf-8")).hexdigest()[:12]


def _read_setup_state() -> dict:
    path = runtime_setup_state_path()
    if not path.is_file():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_setup_state(data: dict) -> None:
    ensure_runtime_asset_directories()
    import json

    runtime_setup_state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_pending_meta_access_launch(enabled: bool) -> None:
    state = _read_setup_state()
    state["pending_meta_access_launch"] = bool(enabled)
    _write_setup_state(state)


def _probe_meta_access(token: str) -> tuple[bool, bool]:
    if not token:
        return False, False
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    try:
        api.model_info(META_ACCESS_REPO_ID, token=token)
        return True, True
    except GatedRepoError:
        return True, False
    except RepositoryNotFoundError:
        return True, False
    except Exception:
        return False, False


def refresh_auth_setup_state(hf_token: str, offline_only: bool) -> dict:
    state = _read_setup_state()
    resolved_token, token_source = resolve_hf_token(hf_token)
    token_present = bool(resolved_token)
    fingerprint = _token_fingerprint(resolved_token)

    meta_access_known = False
    meta_access_approved = False
    if token_present:
        if offline_only:
            meta_access_known = bool(
                state.get("meta_access_known", False) and state.get("token_fingerprint") == fingerprint
            )
            meta_access_approved = bool(meta_access_known and state.get("meta_access_approved", False))
        else:
            meta_access_known, meta_access_approved = _probe_meta_access(resolved_token)

    if not token_present:
        meta_access_known = False
        meta_access_approved = False

    new_state = {
        "token_fingerprint": fingerprint,
        "hf_token_available": token_present,
        "hf_token_source": token_source,
        "meta_access_known": meta_access_known,
        "meta_access_approved": meta_access_approved,
        "pending_meta_access_launch": bool(state.get("pending_meta_access_launch", False)),
    }
    if meta_access_approved:
        new_state["pending_meta_access_launch"] = False
    _write_setup_state(new_state)
    return new_state


def get_auth_setup_state(hf_token: str, offline_only: bool) -> dict:
    del offline_only
    state = _read_setup_state()
    resolved_token, token_source = resolve_hf_token(hf_token)
    fingerprint = _token_fingerprint(resolved_token)
    token_present = bool(resolved_token)
    if state.get("token_fingerprint") == fingerprint:
        return {
            "token_fingerprint": fingerprint,
            "hf_token_available": token_present,
            "hf_token_source": token_source,
            "meta_access_known": bool(state.get("meta_access_known", False)),
            "meta_access_approved": bool(state.get("meta_access_approved", False)),
            "pending_meta_access_launch": bool(state.get("pending_meta_access_launch", False)),
        }
    return {
        "token_fingerprint": fingerprint,
        "hf_token_available": token_present,
        "hf_token_source": token_source,
        "meta_access_known": False,
        "meta_access_approved": False,
        "pending_meta_access_launch": bool(state.get("pending_meta_access_launch", False)),
    }


def maybe_open_pending_meta_access(hf_token: str, offline_only: bool) -> bool:
    if bpy is None:
        return False
    state = refresh_auth_setup_state(hf_token, offline_only)
    if not state.get("pending_meta_access_launch", False):
        return False
    if not state.get("hf_token_available", False):
        return False
    if state.get("meta_access_approved", False):
        return False
    try:
        bpy.ops.wm.url_open(url=META_ACCESS_URL)
        return True
    except Exception:
        return False


def register_startup_auth_check() -> None:
    if bpy is None:
        return

    def _run():
        try:
            state = _read_setup_state()
            if state.get("pending_meta_access_launch", False):
                state["pending_meta_access_launch"] = False
                _write_setup_state(state)
        except Exception as error:
            print(f"Beyond Motion: startup auth check failed: {error}")
        return None

    bpy.app.timers.register(_run, first_interval=0.5)


def _dir_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.rglob("*"))
    except StopIteration:
        return False
    return True


def model_assets_ready(model_name: str, checkpoint_dir_override: str) -> bool:
    model_path = resolved_model_path(model_name, checkpoint_dir_override)
    return (model_path / "config.yaml").is_file()


def local_text_encoder_assets_ready() -> bool:
    root = resolved_text_encoder_root()
    required_paths = [root / repo_id for repo_id in LOCAL_TEXT_ENCODER_REPOS]
    return all(_dir_has_files(path) for path in required_paths)


def text_encoder_service_reachable(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    try:
        with urlopen(url, timeout=2) as response:  # noqa: S310 - local user-provided URL
            return 200 <= getattr(response, "status", 200) < 500
    except (URLError, ValueError, OSError):
        return False


def get_runtime_setup_status(
    *,
    model_name: str,
    text_encoder_mode: str,
    text_encoder_url: str,
    checkpoint_dir_override: str,
    hf_token: str,
    offline_only: bool,
) -> RuntimeSetupStatus:
    issues: list[str] = []
    warnings: list[str] = []
    model_path = resolved_model_path(model_name, checkpoint_dir_override)
    text_encoder_path = resolved_text_encoder_root()
    model_ready = model_assets_ready(model_name, checkpoint_dir_override)
    service_reachable = text_encoder_service_reachable(text_encoder_url)
    local_text_ready = local_text_encoder_assets_ready()
    mode = (text_encoder_mode or "auto").strip().lower()
    resolved_token, token_source = resolve_hf_token(hf_token)
    token_present = bool(resolved_token)
    auth_state = get_auth_setup_state(hf_token, offline_only)
    meta_access_approved = bool(auth_state.get("meta_access_approved", False))
    meta_access_known = bool(auth_state.get("meta_access_known", False))
    pending_meta_access_launch = bool(auth_state.get("pending_meta_access_launch", False))
    next_auth_step = "ready"

    if not model_ready:
        issues.append(
            "Prepare the selected Kimodo model assets in Generation Setup before creating animation."
        )

    if mode == "api":
        if not text_encoder_url.strip():
            issues.append("Set a local text encoder service URL in the add-on preferences.")
        elif not service_reachable:
            issues.append("The configured local text encoder service is not reachable.")
    elif mode == "local":
        if not local_text_ready:
            if offline_only:
                issues.append("Offline Only is enabled, but the local text encoder assets are not downloaded yet.")
            else:
                next_auth_step = "prepare_runtime"
                issues.append("Prepare the bundled open local text encoder assets in Generation Setup before creating animation.")
    else:
        if service_reachable:
            if not local_text_ready:
                warnings.append("The local text encoder service will be used until bundled text encoder assets are prepared.")
        elif not local_text_ready:
            if offline_only:
                issues.append(
                    "Offline Only is enabled, but no prepared local text encoder assets were found in the extension."
                )
            else:
                next_auth_step = "prepare_runtime"
                issues.append(
                    "The local text encoder service is unavailable. Prepare the bundled open local text encoder assets in Generation Setup."
                )

    if next_auth_step == "ready" and issues:
        next_auth_step = "prepare_runtime"

    return RuntimeSetupStatus(
        ready=not issues,
        issues=issues,
        warnings=warnings,
        model_name=model_name,
        model_ready=model_ready,
        model_path=model_path,
        text_encoder_mode=mode,
        text_encoder_service_reachable=service_reachable,
        local_text_encoder_ready=local_text_ready,
        text_encoder_path=text_encoder_path,
        hf_token_available=token_present,
        hf_token_source=token_source,
        meta_access_approved=meta_access_approved,
        meta_access_known=meta_access_known,
        pending_meta_access_launch=pending_meta_access_launch,
        next_auth_step=next_auth_step,
    )


def prepare_runtime_assets(
    *,
    model_name: str,
    text_encoder_mode: str,
    text_encoder_url: str,
    checkpoint_dir_override: str,
    hf_token: str,
    offline_only: bool,
) -> RuntimeSetupStatus:
    if offline_only:
        raise RuntimeError(
            "Offline Only is enabled. Disable it temporarily to download runtime assets into the extension."
        )

    ensure_runtime_asset_directories()

    from huggingface_hub import snapshot_download

    resolved_token, _token_source = resolve_hf_token(hf_token)
    token = resolved_token or None
    model_repo_id = MODEL_REPO_IDS.get(model_name)
    if model_repo_id is None:
        raise RuntimeError(f"Unknown model '{model_name}'.")

    mode = (text_encoder_mode or "auto").strip().lower()
    service_reachable = text_encoder_service_reachable(text_encoder_url)

    model_path = resolved_model_path(model_name, checkpoint_dir_override)
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_repo_id,
        local_dir=str(model_path),
        token=token,
    )

    if mode == "api":
        if not service_reachable:
            raise RuntimeError("The configured local text encoder service is not reachable.")
    elif mode == "local" or (mode == "auto" and not service_reachable):
        text_root = resolved_text_encoder_root()
        for repo_id in LOCAL_TEXT_ENCODER_REPOS:
            target_dir = text_root / repo_id
            target_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=token,
            )

    return get_runtime_setup_status(
        model_name=model_name,
        text_encoder_mode=text_encoder_mode,
        text_encoder_url=text_encoder_url,
        checkpoint_dir_override=checkpoint_dir_override,
        hf_token=hf_token,
        offline_only=offline_only,
    )
