from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import bpy
from bpy.props import EnumProperty  # type: ignore
from bpy.types import Context, Operator

from .dependency_manager import (
    ensure_runtime_directories,
    get_dependency_status,
    installer_python_executable,
    install_runtime_dependencies,
    install_log_path,
    wheels_dir,
)
from .preferences import get_preferences
from .runtime_setup import (
    HF_TOKENS_URL,
    META_ACCESS_URL,
    checkpoints_root,
    clear_runtime_asset_job_notice,
    ensure_runtime_asset_directories,
    get_runtime_asset_job_state,
    refresh_auth_setup_state,
    get_runtime_setup_status,
    maybe_open_pending_meta_access,
    prepare_runtime_assets,
    set_pending_meta_access_launch,
    text_encoders_root,
    update_runtime_asset_job_state,
)


def _active_model_name(context: Context) -> str:
    active_object = context.active_object
    if active_object and active_object.type == "ARMATURE":
        settings = getattr(active_object.data, "beyond_motion", None)
        model_name = getattr(settings, "model_name", "")
        if isinstance(model_name, str) and model_name:
            return model_name
    return "kimodo-soma-rp"


def _tag_relevant_redraw() -> None:
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "DOPESHEET_EDITOR", "GRAPH_EDITOR", "PREFERENCES"}:
                area.tag_redraw()


def _poll_auth_progress(hf_token: str, offline_only: bool):
    def _run():
        try:
            state = refresh_auth_setup_state(hf_token, offline_only)
            opened = maybe_open_pending_meta_access(hf_token, offline_only)
            _tag_relevant_redraw()
            if state.get("meta_access_approved", False):
                return None
            if opened:
                return 3.0
            return 2.0
        except Exception as error:
            print(f"Beyond Motion: auth progress poll failed: {error}")
            return None

    bpy.app.timers.register(_run, first_interval=2.0)


def _show_popup_message(title: str, icon: str, lines: list[str]) -> None:
    def _draw(menu, _context):
        for line in lines:
            menu.layout.label(text=line)

    bpy.context.window_manager.popup_menu(_draw, title=title, icon=icon)


def _poll_runtime_asset_job():
    state = get_runtime_asset_job_state()
    _tag_relevant_redraw()
    if state.get("active", False):
        return 0.5

    notice_text = str(state.get("notice_text", "")).strip()
    if notice_text:
        _show_popup_message("Beyond Motion", str(state.get("notice_icon", "INFO")), [notice_text])
        clear_runtime_asset_job_notice()
        _tag_relevant_redraw()
    return None


def _start_runtime_asset_job(
    *,
    model_name: str,
    text_encoder_mode: str,
    text_encoder_url: str,
    checkpoint_dir_override: str,
    hf_token: str,
    offline_only: bool,
) -> bool:
    state = get_runtime_asset_job_state()
    if state.get("active", False):
        return False

    update_runtime_asset_job_state(
        active=True,
        model_name=model_name,
        status_text="Downloading assets... Please check back later.",
        error_text="",
        notice_text="",
        notice_icon="INFO",
    )

    def _worker():
        try:
            status = prepare_runtime_assets(
                model_name=model_name,
                text_encoder_mode=text_encoder_mode,
                text_encoder_url=text_encoder_url,
                checkpoint_dir_override=checkpoint_dir_override,
                hf_token=hf_token,
                offline_only=offline_only,
            )
            if status.ready:
                update_runtime_asset_job_state(
                    active=False,
                    model_name=model_name,
                    status_text="Runtime assets are ready for generation!",
                    error_text="",
                    notice_text="Runtime assets are ready for generation!",
                    notice_icon="CHECKMARK",
                )
            else:
                message = status.issues[0] if status.issues else "Runtime assets still need attention."
                update_runtime_asset_job_state(
                    active=False,
                    model_name=model_name,
                    status_text=message,
                    error_text=message,
                    notice_text=message,
                    notice_icon="ERROR",
                )
        except Exception as error:
            message = str(error)
            update_runtime_asset_job_state(
                active=False,
                model_name=model_name,
                status_text=message,
                error_text=message,
                notice_text=message,
                notice_icon="ERROR",
            )
        finally:
            _tag_relevant_redraw()

    thread = threading.Thread(target=_worker, name="BeyondMotionRuntimeAssets", daemon=True)
    thread.start()
    bpy.app.timers.register(_poll_runtime_asset_job, first_interval=0.5)
    return True


def _temp_login_script_path(suffix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix="beyond_motion_hf_login_")) / f"hf_login{suffix}"


def _apply_hf_token_login(token: str) -> tuple[bool, str]:
    clean_token = (token or "").strip()
    if not clean_token:
        return False, "Paste a Hugging Face token before continuing."

    python_executable = installer_python_executable()
    extension_root = Path(__file__).resolve().parent
    vendor_path = extension_root / "_vendor"
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{vendor_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env["BEYOND_MOTION_HF_TOKEN"] = clean_token
    command = [
        str(python_executable),
        "-c",
        (
            "import os; "
            "from huggingface_hub import login; "
            "login(token=os.environ['BEYOND_MOTION_HF_TOKEN'], "
            "add_to_git_credential=False, skip_if_logged_in=False)"
        ),
    ]
    result = subprocess.run(
        command,
        check=False,
        cwd=str(extension_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, ""

    error_output = (result.stderr or result.stdout or "").strip()
    if not error_output:
        error_output = "Hugging Face login could not be completed with the provided token."
    return False, error_output.splitlines()[-1]


def _launch_hf_login_terminal() -> bool:
    python_executable = installer_python_executable()
    extension_root = Path(__file__).resolve().parent
    vendor_path = extension_root / "_vendor"

    if sys.platform == "darwin":
        script_path = _temp_login_script_path(".zsh")
        script_contents = f"""#!/bin/zsh
cd {extension_root!s}
export PYTHONNOUSERSITE=1
export PYTHONPATH="{vendor_path!s}:$PYTHONPATH"
clear
echo "Beyond Motion is opening Hugging Face login."
echo "After login succeeds, this add-on will auto-open Meta model access."
echo
"{python_executable}" -m huggingface_hub.cli.hf auth login
echo
echo "You can close this Terminal window after login completes."
"""
        script_path.write_text(script_contents, encoding="utf-8")
        script_path.chmod(0o755)
        result = subprocess.run(["open", "-a", "Terminal", str(script_path)], check=False)
        return result.returncode == 0

    if os.name == "nt":
        script_path = _temp_login_script_path(".bat")
        script_contents = "\n".join(
            [
                "@echo off",
                f'cd /d "{extension_root}"',
                "set PYTHONNOUSERSITE=1",
                f'set PYTHONPATH={vendor_path};%PYTHONPATH%',
                "cls",
                "echo Beyond Motion is opening Hugging Face login.",
                "echo After login succeeds, this add-on will auto-open Meta model access.",
                "echo.",
                f'"{python_executable}" -m huggingface_hub.cli.hf auth login',
                "echo.",
                "echo You can close this window after login completes.",
                "pause",
            ]
        )
        script_path.write_text(script_contents, encoding="utf-8")
        result = subprocess.run(["cmd.exe", "/c", "start", "", str(script_path)], check=False)
        return result.returncode == 0

    if sys.platform.startswith("linux"):
        script_path = _temp_login_script_path(".sh")
        script_contents = f"""#!/bin/sh
cd {shlex.quote(str(extension_root))}
export PYTHONNOUSERSITE=1
export PYTHONPATH={shlex.quote(str(vendor_path))}:$PYTHONPATH
clear
echo "Beyond Motion is opening Hugging Face login."
echo "After login succeeds, this add-on will auto-open Meta model access."
echo
{shlex.quote(str(python_executable))} -m huggingface_hub.cli.hf auth login
echo
echo "You can close this window after login completes."
printf "Press Enter to close..."
read dummy
"""
        script_path.write_text(script_contents, encoding="utf-8")
        script_path.chmod(0o755)

        terminal_commands = [
            ["x-terminal-emulator", "-e", "sh", str(script_path)],
            ["gnome-terminal", "--", "sh", str(script_path)],
            ["konsole", "-e", "sh", str(script_path)],
            ["xfce4-terminal", "--command", f"sh {shlex.quote(str(script_path))}"],
            ["xterm", "-e", "sh", str(script_path)],
        ]
        for command in terminal_commands:
            if shutil.which(command[0]) is None:
                continue
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                return True
        return False

    return False


class BEYONDMOTION_OT_open_setup_preferences(Operator):
    bl_idname = "beyond_motion.open_setup_preferences"
    bl_label = "Open Beyond Motion Preferences"
    bl_description = "Open the add-on preferences to install generation dependencies"

    def execute(self, context: Context) -> set[str]:
        bpy.ops.screen.userpref_show("INVOKE_DEFAULT")
        try:
            bpy.ops.preferences.addon_show(module=__package__)
        except Exception:
            pass
        return {"FINISHED"}


class BEYONDMOTION_OT_open_external_setup_url(Operator):
    bl_idname = "beyond_motion.open_external_setup_url"
    bl_label = "Open Setup Link"
    bl_description = "Open the relevant Hugging Face setup page in your browser"

    target: EnumProperty(  # type: ignore[valid-type]
        name="Target",
        items=(
            ("TOKENS", "HF Tokens", "Open Hugging Face access token settings"),
            ("META_LLAMA", "Meta Llama Access", "Open the gated Meta-Llama model page"),
        ),
        default="TOKENS",
    )

    def execute(self, context: Context) -> set[str]:
        del context
        url = HF_TOKENS_URL
        if self.target == "META_LLAMA":
            url = META_ACCESS_URL
        bpy.ops.wm.url_open(url=url)
        return {"FINISHED"}


class BEYONDMOTION_OT_begin_hf_local_model_setup(Operator):
    bl_idname = "beyond_motion.begin_hf_local_model_setup"
    bl_label = "Use Token and Continue"
    bl_description = "Save the pasted Hugging Face token as a local login and continue into Meta model access"

    def execute(self, context: Context) -> set[str]:
        prefs = get_preferences(context)
        hf_token = prefs.hf_token if prefs else ""
        offline_only = bool(prefs.offline_only) if prefs else False
        clean_token = (hf_token or "").strip()

        set_pending_meta_access_launch(True)
        if clean_token:
            success, message = _apply_hf_token_login(clean_token)
            if not success:
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
            if prefs is not None:
                prefs.hf_token = ""
            refresh_auth_setup_state("", offline_only)
            maybe_open_pending_meta_access("", offline_only)
            _poll_auth_progress("", offline_only)
            _tag_relevant_redraw()
            self.report({"INFO"}, "Local AI login saved. Beyond Motion will continue into Meta model access.")
            return {"FINISHED"}

        refresh_auth_setup_state(hf_token, offline_only)
        if not _launch_hf_login_terminal():
            bpy.ops.wm.url_open(url=HF_TOKENS_URL)
        _poll_auth_progress(hf_token, offline_only)
        self.report({"INFO"}, "Paste a Hugging Face token first, or complete login in the opened window.")
        return {"FINISHED"}


class BEYONDMOTION_OT_approve_meta_access(Operator):
    bl_idname = "beyond_motion.approve_meta_access"
    bl_label = "Approve Meta Access"
    bl_description = "Open the gated Meta-Llama access page and track completion for the next setup step"

    def execute(self, context: Context) -> set[str]:
        prefs = get_preferences(context)
        hf_token = prefs.hf_token if prefs else ""
        offline_only = bool(prefs.offline_only) if prefs else False
        set_pending_meta_access_launch(True)
        refresh_auth_setup_state(hf_token, offline_only)
        bpy.ops.wm.url_open(url=META_ACCESS_URL)
        _poll_auth_progress(hf_token, offline_only)
        self.report({"INFO"}, "Approve the Meta model access page, then return to Prepare Runtime Assets.")
        return {"FINISHED"}


class BEYONDMOTION_OT_install_generation_dependencies(Operator):
    bl_idname = "beyond_motion.install_generation_dependencies"
    bl_label = "Install Generation Dependencies"
    bl_description = "Download extension-local wheel dependencies for Kimodo generation"

    def execute(self, context: Context) -> set[str]:
        prefs = get_preferences(context)
        if prefs is None:
            self.report({"ERROR"}, "Beyond Motion preferences are unavailable.")
            return {"CANCELLED"}
        existing_status = get_dependency_status(prefs.torch_device)
        if existing_status.ready:
            self.report({"INFO"}, "Generation dependencies are already installed.")
            _tag_relevant_redraw()
            return {"CANCELLED"}

        context.window.cursor_set("WAIT")
        try:
            status = install_runtime_dependencies(prefs.torch_device)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        finally:
            context.window.cursor_set("DEFAULT")

        _tag_relevant_redraw()
        self.report({"INFO"}, f"Installed generation dependencies for {status.resolved_backend.upper()}.")
        return {"FINISHED"}


class BEYONDMOTION_OT_refresh_dependency_status(Operator):
    bl_idname = "beyond_motion.refresh_dependency_status"
    bl_label = "Refresh Dependency Status"
    bl_description = "Refresh the extension-local dependency status"

    def execute(self, context: Context) -> set[str]:
        prefs = get_preferences(context)
        requested_device = prefs.torch_device if prefs else "auto"
        status = get_dependency_status(requested_device)
        if status.ready:
            self.report({"INFO"}, f"Dependencies are installed for {status.resolved_backend.upper()}.")
        else:
            self.report({"WARNING"}, "Dependencies are not installed yet.")
        return {"FINISHED"}


class BEYONDMOTION_OT_prepare_runtime_assets(Operator):
    bl_idname = "beyond_motion.prepare_runtime_assets"
    bl_label = "Prepare Runtime Assets"
    bl_description = "Download and validate the selected Kimodo model and local text encoder assets into this extension"

    def execute(self, context: Context) -> set[str]:
        prefs = get_preferences(context)
        if prefs is None:
            self.report({"ERROR"}, "Beyond Motion preferences are unavailable.")
            return {"CANCELLED"}

        dependency_status = get_dependency_status(prefs.torch_device)
        if not dependency_status.ready:
            self.report({"ERROR"}, "Install generation dependencies before preparing runtime assets.")
            return {"CANCELLED"}

        model_name = _active_model_name(context)
        current_status = get_runtime_setup_status(
            model_name=model_name,
            text_encoder_mode=prefs.text_encoder_mode,
            text_encoder_url=prefs.text_encoder_url,
            checkpoint_dir_override=prefs.checkpoint_dir,
            hf_token=prefs.hf_token,
            offline_only=prefs.offline_only,
        )
        if current_status.ready:
            _show_popup_message("Beyond Motion", "CHECKMARK", ["Runtime assets are ready for generation!"])
            self.report({"INFO"}, "Runtime assets are ready for generation.")
            _tag_relevant_redraw()
            return {"CANCELLED"}

        started = _start_runtime_asset_job(
            model_name=model_name,
            text_encoder_mode=prefs.text_encoder_mode,
            text_encoder_url=prefs.text_encoder_url,
            checkpoint_dir_override=prefs.checkpoint_dir,
            hf_token=prefs.hf_token,
            offline_only=prefs.offline_only,
        )
        if not started:
            self.report({"WARNING"}, "Runtime asset download is already in progress.")
            return {"CANCELLED"}

        _show_popup_message("Beyond Motion", "INFO", ["Downloading assets...", "Please check back later."])
        self.report({"INFO"}, "Downloading assets... Please check back later.")
        _tag_relevant_redraw()
        return {"FINISHED"}


class BEYONDMOTION_OT_refresh_runtime_setup(Operator):
    bl_idname = "beyond_motion.refresh_runtime_setup"
    bl_label = "Refresh Runtime Setup"
    bl_description = "Re-check the selected model and text encoder setup"

    def execute(self, context: Context) -> set[str]:
        prefs = get_preferences(context)
        if prefs is None:
            self.report({"ERROR"}, "Beyond Motion preferences are unavailable.")
            return {"CANCELLED"}
        refresh_auth_setup_state(prefs.hf_token, prefs.offline_only)
        status = get_runtime_setup_status(
            model_name=_active_model_name(context),
            text_encoder_mode=prefs.text_encoder_mode,
            text_encoder_url=prefs.text_encoder_url,
            checkpoint_dir_override=prefs.checkpoint_dir,
            hf_token=prefs.hf_token,
            offline_only=prefs.offline_only,
        )
        if status.ready:
            self.report({"INFO"}, "Runtime assets are ready.")
        else:
            self.report({"WARNING"}, status.issues[0] if status.issues else "Runtime setup is incomplete.")
        _tag_relevant_redraw()
        return {"FINISHED"}


class BEYONDMOTION_OT_open_support_path(Operator):
    bl_idname = "beyond_motion.open_support_path"
    bl_label = "Open Support Path"
    bl_description = "Open the wheels folder or install log"

    target: EnumProperty(  # type: ignore[valid-type]
        name="Target",
        items=(
            ("WHEELS", "Wheels", "Open the extension wheels folder"),
            ("LOG", "Install Log", "Open the dependency install log"),
            ("CHECKPOINTS", "Checkpoints", "Open the extension model checkpoints folder"),
            ("TEXT_ENCODERS", "Text Encoders", "Open the extension local text encoder folder"),
        ),
        default="WHEELS",
    )

    def execute(self, context: Context) -> set[str]:
        ensure_runtime_directories()
        ensure_runtime_asset_directories()
        if self.target == "WHEELS":
            path = wheels_dir()
        elif self.target == "LOG":
            path = install_log_path()
        elif self.target == "CHECKPOINTS":
            path = checkpoints_root()
        else:
            path = text_encoders_root()
        if self.target == "LOG" and not path.exists():
            path.write_text("", encoding="utf-8")
        if self.target == "WHEELS":
            prefs = get_preferences(context)
            status = get_dependency_status(prefs.torch_device if prefs else "auto")
            has_wheel_files = any(path.rglob("*.whl"))
            if not status.ready and not has_wheel_files:
                self.report({"WARNING"}, "! No wheels installed")
        try:
            bpy.ops.wm.path_open(filepath=str(Path(path)))
        except RuntimeError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}
