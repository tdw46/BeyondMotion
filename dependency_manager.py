from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import platform
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import bpy
except ImportError:  # pragma: no cover - used inside Blender
    bpy = None


LOCAL_GENERATION_VRAM_GB = 17
STATE_VERSION = 1

REQUIRED_MODULES = (
    "numpy",
    "torch",
    "scipy",
    "hydra",
    "omegaconf",
    "transformers",
    "urllib3",
    "boto3",
    "peft",
    "einops",
    "tqdm",
    "packaging",
    "pydantic",
    "filelock",
    "gradio_client",
    "huggingface_hub",
    "safetensors",
)

RUNTIME_PACKAGES = (
    "numpy>=1.23",
    "scipy>=1.10",
    "transformers==5.1.0",
    "urllib3>=2.6.3",
    "boto3",
    "peft>=0.18",
    "einops>=0.7",
    "tqdm>=4.0",
    "packaging>=21.0",
    "pydantic>=2.0",
    "filelock>=3.20.3",
    "gradio_client>=1.0",
    "huggingface_hub>=0.34",
    "safetensors>=0.4",
    "pillow>=9.0",
)

HYDRA_STACK_PACKAGES = (
    "hydra-core==1.3.2",
)


@dataclass
class DependencyStatus:
    ready: bool
    missing_modules: list[str]
    vendor_dir: Path
    wheels_dir: Path
    cache_dir: Path
    install_log_path: Path
    state_path: Path
    python_executable: str
    platform_key: str
    requested_device: str
    resolved_backend: str
    installed_backend: str
    last_updated: str
    last_error: str
    install_required: bool


def extension_root() -> Path:
    return Path(__file__).resolve().parent


def vendor_dir() -> Path:
    return extension_root() / "_vendor"


def wheels_dir() -> Path:
    return extension_root() / "wheels"


def wheels_cache_dir() -> Path:
    return wheels_dir() / "cache"


def install_log_path() -> Path:
    return wheels_dir() / "install.log"


def install_state_path() -> Path:
    return wheels_dir() / "state.json"


def ensure_runtime_paths() -> None:
    vendor_path = vendor_dir()
    if vendor_path.is_dir():
        site.addsitedir(str(vendor_path))
    kimodo_path = extension_root() / "vendor" / "kimodo"
    if kimodo_path.is_dir() and str(kimodo_path) not in sys.path:
        sys.path.insert(0, str(kimodo_path))


def ensure_runtime_directories() -> None:
    vendor_dir().mkdir(parents=True, exist_ok=True)
    wheels_cache_dir().mkdir(parents=True, exist_ok=True)


def installer_python_executable() -> str:
    if bpy is not None:
        python_path = getattr(getattr(bpy, "app", None), "binary_path_python", "")
        if python_path:
            return python_path
    return sys.executable


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x86_64")
    if machine == "aarch64":
        machine = "arm64"
    return f"{system}-{machine}"


def has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and bool((result.stdout or "").strip())


def resolve_install_backend(requested_device: str) -> str:
    requested = (requested_device or "auto").strip().lower()
    system = platform.system().lower()
    machine = platform.machine().lower()

    if requested.startswith("cuda") or requested.isdigit():
        return "cuda"
    if requested == "mps":
        return "mps"
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        if system == "darwin" and machine in {"arm64", "aarch64"}:
            return "mps"
        if system in {"linux", "windows"} and has_nvidia_gpu():
            return "cuda"
        return "cpu"
    return "cpu"


def backend_supported(backend: str) -> bool:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if backend == "mps":
        return system == "darwin" and machine in {"arm64", "aarch64"}
    if backend == "cuda":
        return system in {"linux", "windows"}
    return backend == "cpu"


def _read_state() -> dict:
    path = install_state_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(data: dict) -> None:
    ensure_runtime_directories()
    install_state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _module_spec_in_vendor(module_name: str):
    search_paths = [str(vendor_dir())]
    return importlib.machinery.PathFinder.find_spec(module_name, search_paths)


def missing_required_modules() -> list[str]:
    return [module_name for module_name in REQUIRED_MODULES if _module_spec_in_vendor(module_name) is None]


def _compatible_install(installed_backend: str, resolved_backend: str) -> bool:
    if not installed_backend:
        return False
    if installed_backend == resolved_backend:
        return True
    if platform.system().lower() == "darwin" and {installed_backend, resolved_backend} <= {"cpu", "mps"}:
        return True
    return False


def get_dependency_status(requested_device: str) -> DependencyStatus:
    ensure_runtime_paths()
    importlib.invalidate_caches()
    state = _read_state()
    resolved_backend = resolve_install_backend(requested_device)
    missing_modules = missing_required_modules()
    installed_backend = str(state.get("installed_backend", "")).strip().lower()
    if not missing_modules and not installed_backend:
        installed_backend = resolved_backend
        _write_state(
            {
                "state_version": STATE_VERSION,
                "platform_key": platform_key(),
                "requested_device": requested_device,
                "installed_backend": installed_backend,
                "last_updated": str(state.get("last_updated", datetime.now(timezone.utc).isoformat())),
                "last_error": "",
            }
        )
    compatible_install = _compatible_install(installed_backend, resolved_backend)
    ready = not missing_modules and compatible_install
    return DependencyStatus(
        ready=ready,
        missing_modules=missing_modules,
        vendor_dir=vendor_dir(),
        wheels_dir=wheels_dir(),
        cache_dir=wheels_cache_dir(),
        install_log_path=install_log_path(),
        state_path=install_state_path(),
        python_executable=installer_python_executable(),
        platform_key=platform_key(),
        requested_device=requested_device,
        resolved_backend=resolved_backend,
        installed_backend=installed_backend,
        last_updated=str(state.get("last_updated", "")),
        last_error=str(state.get("last_error", "")),
        install_required=bool(missing_modules) or not compatible_install,
    )


def status_message(status: DependencyStatus) -> str:
    if status.ready:
        return (
            f"Generation dependencies are installed for {status.resolved_backend.upper()} on "
            f"{status.platform_key}."
        )
    if status.last_error:
        return f"Dependency install is incomplete. Last error: {status.last_error}"
    if status.installed_backend and status.installed_backend != status.resolved_backend:
        return (
            f"Dependencies were installed for {status.installed_backend.upper()}, but the current device "
            f"selection resolves to {status.resolved_backend.upper()}. Reinstall for this device."
        )
    if status.missing_modules:
        return "Generation dependencies are not installed yet."
    return "Generation dependencies need to be installed."


def dependency_size_estimate(requested_device: str) -> tuple[str, str, str]:
    backend = resolve_install_backend(requested_device)
    if backend == "cuda":
        return (
            backend,
            "Estimated dependency download: about 4-8 GB.",
            "Estimated extension disk usage after install: about 8-14 GB.",
        )
    if backend == "mps":
        return (
            backend,
            "Estimated dependency download: about 0.5-1.5 GB.",
            "Estimated extension disk usage after install: about 1-3 GB.",
        )
    return (
        backend,
        "Estimated dependency download: about 0.5-2 GB.",
        "Estimated extension disk usage after install: about 1-4 GB.",
    )


def _append_log(text: str) -> None:
    ensure_runtime_directories()
    with install_log_path().open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _run_logged_command(command: list[str], env: dict[str, str], label: str) -> None:
    _append_log(f"\n=== {label} ===")
    _append_log("Command: " + " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=7200,
        check=False,
    )
    if result.stdout:
        _append_log(result.stdout)
    if result.stderr:
        _append_log(result.stderr)
    if result.returncode != 0:
        error_tail = (result.stderr or result.stdout or "pip install failed").strip().splitlines()
        concise = error_tail[-1] if error_tail else "pip install failed"
        raise RuntimeError(f"{label} failed: {concise}")


def _torch_install_command(python_executable: str, backend: str) -> list[str]:
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(vendor_dir()),
        "--only-binary=:all:",
        "--cache-dir",
        str(wheels_cache_dir()),
        "torch>=2.0,<3",
    ]
    system = platform.system().lower()
    if backend == "cuda":
        if system not in {"linux", "windows"}:
            raise RuntimeError("CUDA wheel installs are only supported on Windows and Linux.")
        command.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu128"])
    elif backend == "cpu" and system in {"linux", "windows"}:
        command.extend(["--extra-index-url", "https://download.pytorch.org/whl/cpu"])
    elif backend == "mps":
        if not backend_supported("mps"):
            raise RuntimeError("Apple Metal installs require macOS on Apple Silicon.")
    return command


def install_runtime_dependencies(requested_device: str) -> DependencyStatus:
    resolved_backend = resolve_install_backend(requested_device)
    if not backend_supported(resolved_backend):
        raise RuntimeError(
            f"The selected runtime backend '{resolved_backend}' is not supported on {platform_key()}."
        )

    python_executable = installer_python_executable()
    ensure_runtime_directories()
    shutil.rmtree(vendor_dir(), ignore_errors=True)
    vendor_dir().mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    _write_state(
        {
            "state_version": STATE_VERSION,
            "platform_key": platform_key(),
            "requested_device": requested_device,
            "installed_backend": "",
            "last_updated": started_at,
            "last_error": "",
        }
    )

    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    try:
        _run_logged_command(
            [python_executable, "-m", "ensurepip", "--upgrade"],
            env,
            "Bootstrapping pip",
        )
        _run_logged_command(
            _torch_install_command(python_executable, resolved_backend),
            env,
            f"Installing PyTorch for {resolved_backend.upper()}",
        )
        _run_logged_command(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(vendor_dir()),
                "--cache-dir",
                str(wheels_cache_dir()),
                *HYDRA_STACK_PACKAGES,
            ],
            env,
            "Installing Hydra runtime stack",
        )
        _run_logged_command(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(vendor_dir()),
                "--only-binary=:all:",
                "--cache-dir",
                str(wheels_cache_dir()),
                *RUNTIME_PACKAGES,
            ],
            env,
            "Installing Beyond Motion runtime packages",
        )
        ensure_runtime_paths()
        importlib.invalidate_caches()
        _write_state(
            {
                "state_version": STATE_VERSION,
                "platform_key": platform_key(),
                "requested_device": requested_device,
                "installed_backend": resolved_backend,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "last_error": "",
            }
        )
        status = get_dependency_status(requested_device)
        if not status.ready:
            missing = ", ".join(status.missing_modules) if status.missing_modules else "unknown modules"
            raise RuntimeError(f"Dependency install completed but required modules are still missing: {missing}")
        return get_dependency_status(requested_device)
    except Exception as error:
        _write_state(
            {
                "state_version": STATE_VERSION,
                "platform_key": platform_key(),
                "requested_device": requested_device,
                "installed_backend": "",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "last_error": str(error),
            }
        )
        raise
