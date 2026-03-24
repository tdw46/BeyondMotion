from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Context

from .dependency_manager import (
    LOCAL_GENERATION_VRAM_GB,
    dependency_size_estimate,
    get_dependency_status,
    installer_python_executable,
    status_message,
)
from .runtime_setup import (
    get_runtime_asset_job_state,
    get_runtime_setup_status,
    resolved_checkpoint_root,
    resolved_text_encoder_root,
)
from .utils import wrap_text_to_panel


def get_preferences(context: Context | None = None) -> "BeyondMotionPreferences | None":
    ctx = context or bpy.context
    addon = ctx.preferences.addons.get(__package__)
    if not addon:
        return None
    prefs = addon.preferences
    if isinstance(prefs, BeyondMotionPreferences):
        return prefs
    return None


def _active_model_name(context: Context) -> str:
    active_object = context.active_object
    if active_object and active_object.type == "ARMATURE":
        settings = getattr(active_object.data, "beyond_motion", None)
        model_name = getattr(settings, "model_name", "")
        if isinstance(model_name, str) and model_name:
            return model_name
    return "kimodo-soma-rp"


def _needs_guided_token_step(runtime_status) -> bool:
    return runtime_status.next_auth_step == "login"


def _show_token_entry(runtime_status, hf_token: str) -> bool:
    return (not runtime_status.ready) and (_needs_guided_token_step(runtime_status) or bool((hf_token or "").strip()))


def _draw_dependency_setup_box(
    layout,
    context: Context,
    *,
    prefs,
    status,
    backend: str,
    download_estimate: str,
    disk_estimate: str,
    completed_view: bool,
) -> None:
    flow_box = layout.box()
    header_row = flow_box.row()
    header_row.alert = not status.ready
    header_row.label(
        text="Generation Dependencies Ready" if completed_view else "Generation Setup",
        icon="CHECKMARK" if completed_view else ("ERROR" if not status.ready else "IMPORT"),
    )
    flow_text = (
        "Install Beyond Motion's generation dependencies before using Kimodo. "
        "The download stays inside this extension, not Blender globally. "
        f"Local generation needs roughly {LOCAL_GENERATION_VRAM_GB} GB of VRAM, "
        "so use a supported GPU or Apple Metal device with enough memory."
    )
    wrapped_flow = wrap_text_to_panel(flow_text, context, full_width=True)
    for line in wrapped_flow.splitlines():
        line_row = flow_box.row()
        line_row.alert = not status.ready and not completed_view
        line_row.label(text=line)
    size_box = flow_box.box()
    for text in (
        f"Selected runtime backend: {backend.upper()}",
        download_estimate,
        disk_estimate,
        "Model checkpoints download separately after setup.",
    ):
        wrapped_text = wrap_text_to_panel(text, context, full_width=True)
        for line in wrapped_text.splitlines():
            line_row = size_box.row()
            line_row.alert = not status.ready and not completed_view
            line_row.label(text=line)
    flow_box.prop(prefs, "torch_device")
    flow_box.prop(prefs, "enable_mps_fallback")
    wrapped_status = wrap_text_to_panel(status_message(status), context, full_width=True)
    status_icon = "CHECKMARK" if status.ready else "ERROR"
    for index, line in enumerate(wrapped_status.splitlines() or [""]):
        flow_box.label(text=line, icon=status_icon if index == 0 else "BLANK1")
    if status.last_updated:
        flow_box.label(text=f"Last Updated: {status.last_updated}")
    if status.last_error and not completed_view:
        wrapped_error = wrap_text_to_panel(status.last_error, context, full_width=True)
        error_box = flow_box.box()
        error_box.alert = True
        for line in wrapped_error.splitlines():
            error_box.label(text=line)
    actions = flow_box.row(align=True)
    if completed_view:
        actions.operator("beyond_motion.refresh_dependency_status", icon="FILE_REFRESH")
    else:
        actions.operator("beyond_motion.install_generation_dependencies", icon="IMPORT")
        actions.operator("beyond_motion.refresh_dependency_status", icon="FILE_REFRESH")
    paths = flow_box.row(align=True)
    paths.operator("beyond_motion.open_support_path", text="Open Wheels Folder", icon="FILE_FOLDER").target = "WHEELS"
    paths.operator("beyond_motion.open_support_path", text="Open Install Log", icon="TEXT").target = "LOG"


def _draw_runtime_setup_box(
    layout,
    context: Context,
    *,
    prefs,
    model_name: str,
    dependency_ready: bool,
    runtime_status,
    runtime_job: dict,
    completed_view: bool,
) -> None:
    runtime_setup_box = layout.box()
    runtime_setup_header = runtime_setup_box.row()
    runtime_setup_header.alert = not runtime_status.ready and not completed_view
    runtime_setup_header.label(
        text="Runtime Assets Ready" if completed_view else "Runtime Assets Required",
        icon="CHECKMARK" if completed_view else ("ERROR" if not runtime_status.ready else "CHECKMARK"),
    )
    runtime_setup_box.label(text=f"Active Model: {model_name}")
    runtime_setup_box.label(text=f"Checkpoint Cache: {resolved_checkpoint_root(prefs.checkpoint_dir)}")
    runtime_setup_box.label(text=f"Text Encoder Cache: {resolved_text_encoder_root()}")
    runtime_setup_box.label(
        text=(
            f"Hugging Face Login: detected from {runtime_status.hf_token_source}"
            if runtime_status.hf_token_available
            else (
                "Hugging Face Login: not required for the default open local text encoder"
                if runtime_status.next_auth_step != "login"
                else "Hugging Face Login: not detected"
            )
        ),
        icon="LOCKED",
    )
    runtime_setup_box.label(
        text=(
            "Text encoder service reachable."
            if runtime_status.text_encoder_service_reachable
            else "Text encoder service not detected."
        ),
        icon="URL",
    )
    if runtime_job.get("active", False):
        job_box = runtime_setup_box.box()
        job_box.alert = True
        for line in wrap_text_to_panel(str(runtime_job.get("status_text", "")), context, full_width=True).splitlines():
            row = job_box.row()
            row.alert = True
            row.label(text=line, icon="TIME")
    elif runtime_job.get("error_text", "") and not completed_view:
        error_box = runtime_setup_box.box()
        error_box.alert = True
        for line in wrap_text_to_panel(str(runtime_job.get("error_text", "")), context, full_width=True).splitlines():
            row = error_box.row()
            row.alert = True
            row.label(text=line, icon="ERROR")
    for warning in runtime_status.warnings:
        for line in wrap_text_to_panel(warning, context, full_width=True).splitlines():
            runtime_setup_box.label(text=line, icon="INFO")
    if not completed_view:
        for issue in runtime_status.issues:
            for line_index, line in enumerate(wrap_text_to_panel(issue, context, full_width=True).splitlines() or [""]):
                issue_row = runtime_setup_box.row()
                issue_row.alert = True
                issue_row.label(text=line, icon="ERROR" if line_index == 0 else "BLANK1")
        if _show_token_entry(runtime_status, prefs.hf_token):
            token_help = (
                "For secure local AI DL, log in at Hugging Face token settings, create a token, "
                "copy it, and paste it into Blender here. Beyond Motion will save that login "
                "locally and continue setup without requiring console input."
            )
            token_box = runtime_setup_box.box()
            token_box.alert = True
            token_box.label(text="Secure Local AI Login", icon="LOCKED")
            for line in wrap_text_to_panel(token_help, context, full_width=True).splitlines():
                help_row = token_box.row()
                help_row.alert = True
                help_row.label(text=line)
            token_box.prop(prefs, "hf_token", text="HF Token")
            token_actions = token_box.row(align=True)
            token_actions.operator(
                "beyond_motion.open_external_setup_url",
                text="Login and Create Token",
                icon="URL",
            ).target = "TOKENS"
            if prefs.hf_token.strip():
                token_actions.operator(
                    "beyond_motion.begin_hf_local_model_setup",
                    text="Use Token and Continue",
                    icon="LOCKED",
                )
    runtime_actions = runtime_setup_box.row(align=True)
    runtime_actions.enabled = dependency_ready and not runtime_job.get("active", False)
    if completed_view:
        runtime_actions.operator("beyond_motion.refresh_runtime_setup", icon="FILE_REFRESH")
    elif runtime_job.get("active", False):
        runtime_actions.label(text="Downloading assets...", icon="TIME")
    elif _needs_guided_token_step(runtime_status):
        if not prefs.hf_token.strip():
            runtime_actions.enabled = False
            runtime_actions.label(text="Paste an HF token above to continue.", icon="INFO")
    elif runtime_status.next_auth_step == "meta_access":
        runtime_actions.operator(
            "beyond_motion.approve_meta_access",
            text="Approve Meta Access",
            icon="LOCKED",
        )
    else:
        runtime_actions.operator(
            "beyond_motion.prepare_runtime_assets",
            text="Prepare Runtime Assets",
            icon="FILE_REFRESH",
        )
    runtime_paths = runtime_setup_box.row(align=True)
    runtime_paths.operator("beyond_motion.open_support_path", text="Open Checkpoints", icon="FILE_FOLDER").target = "CHECKPOINTS"
    runtime_paths.operator("beyond_motion.open_support_path", text="Open Text Encoders", icon="FILE_FOLDER").target = "TEXT_ENCODERS"


class BeyondMotionPreferences(AddonPreferences):
    bl_idname = __package__

    setup_panel_tab: EnumProperty(  # type: ignore[valid-type]
        name="Setup View",
        items=(
            ("NEXT", "Next Steps", "Show setup steps that still need attention"),
            ("DONE", "Completed", "Show setup steps that are already finished"),
        ),
        default="NEXT",
    )

    python_executable: StringProperty(  # type: ignore[valid-type]
        name="Legacy Python Override",
        subtype="FILE_PATH",
        description="Legacy field kept for compatibility; Beyond Motion now installs generation wheels into the extension and runs with Blender's bundled Python",
    )
    checkpoint_dir: StringProperty(  # type: ignore[valid-type]
        name="Checkpoint Directory",
        subtype="DIR_PATH",
        description="Optional local Kimodo checkpoint cache directory",
    )
    text_encoder_url: StringProperty(  # type: ignore[valid-type]
        name="Text Encoder URL",
        default="http://127.0.0.1:9550/",
        description="Kimodo text encoder service URL. Leave the default to reuse kimodo_textencoder if it is running",
    )
    text_encoder_mode: EnumProperty(  # type: ignore[valid-type]
        name="Text Encoder Mode",
        items=(
            ("auto", "Auto", "Try a local text encoder service first, then fall back to the bundled local encoder"),
            ("local", "Local", "Use the bundled local text encoder only"),
            ("api", "Service URL", "Use only the configured local text encoder service URL"),
        ),
        default="auto",
    )
    hf_token: StringProperty(  # type: ignore[valid-type]
        name="Hugging Face Token",
        subtype="PASSWORD",
        description="Optional Hugging Face token for private or gated Hugging Face downloads; the default open local text encoder does not require it",
    )
    offline_only: BoolProperty(  # type: ignore[valid-type]
        name="Offline Only",
        default=False,
        description="Disallow network access and use cached checkpoints/text-encoder files only",
    )
    torch_device: EnumProperty(  # type: ignore[valid-type]
        name="Torch Device",
        items=(
            ("auto", "Auto", "Prefer CUDA when available, then MPS on Apple Silicon, then CPU"),
            ("mps", "MPS / Metal", "Force Apple's Metal backend when available"),
            ("cpu", "CPU", "Force CPU inference"),
            ("cuda", "CUDA", "Force the default CUDA device"),
            ("cuda:0", "CUDA:0", "Force CUDA device 0"),
        ),
        default="auto",
    )
    enable_mps_fallback: BoolProperty(  # type: ignore[valid-type]
        name="Enable MPS Fallback",
        default=True,
        description="Allow unsupported MPS operations to fall back to CPU when running on Apple Metal",
    )
    job_timeout_seconds: IntProperty(  # type: ignore[valid-type]
        name="Job Timeout",
        default=1800,
        min=30,
        description="Maximum wait time for a generation job",
    )
    keep_temp_files: BoolProperty(  # type: ignore[valid-type]
        name="Keep Temp Files",
        default=False,
        description="Keep generated request and output files for debugging",
    )

    def resolved_python_executable(self) -> str:
        return installer_python_executable()

    def draw(self, context: Context) -> None:
        layout = self.layout
        status = get_dependency_status(self.torch_device)
        backend, download_estimate, disk_estimate = dependency_size_estimate(self.torch_device)
        model_name = _active_model_name(context)
        runtime_job = get_runtime_asset_job_state()
        runtime_status = get_runtime_setup_status(
            model_name=model_name,
            text_encoder_mode=self.text_encoder_mode,
            text_encoder_url=self.text_encoder_url,
            checkpoint_dir_override=self.checkpoint_dir,
            hf_token=self.hf_token,
            offline_only=self.offline_only,
        )
        tabs = layout.row(align=True)
        tabs.prop(self, "setup_panel_tab", expand=True)

        visible_sections = 0
        if self.setup_panel_tab == "NEXT":
            if not status.ready:
                _draw_dependency_setup_box(
                    layout,
                    context,
                    prefs=self,
                    status=status,
                    backend=backend,
                    download_estimate=download_estimate,
                    disk_estimate=disk_estimate,
                    completed_view=False,
                )
                visible_sections += 1
            if not runtime_status.ready or runtime_job.get("active", False):
                _draw_runtime_setup_box(
                    layout,
                    context,
                    prefs=self,
                    model_name=model_name,
                    dependency_ready=status.ready,
                    runtime_status=runtime_status,
                    runtime_job=runtime_job,
                    completed_view=False,
                )
                visible_sections += 1
            if visible_sections == 0:
                done_box = layout.box()
                done_box.label(text="All setup steps are complete.", icon="CHECKMARK")
        else:
            if status.ready:
                _draw_dependency_setup_box(
                    layout,
                    context,
                    prefs=self,
                    status=status,
                    backend=backend,
                    download_estimate=download_estimate,
                    disk_estimate=disk_estimate,
                    completed_view=True,
                )
                visible_sections += 1
            if runtime_status.ready:
                _draw_runtime_setup_box(
                    layout,
                    context,
                    prefs=self,
                    model_name=model_name,
                    dependency_ready=status.ready,
                    runtime_status=runtime_status,
                    runtime_job=runtime_job,
                    completed_view=True,
                )
                visible_sections += 1
            if visible_sections == 0:
                todo_box = layout.box()
                todo_box.label(text="No completed setup steps yet.", icon="INFO")

        runtime_box = layout.box()
        runtime_box.label(text="Runtime", icon="PREFERENCES")
        runtime_box.prop(self, "checkpoint_dir")
        runtime_box.prop(self, "text_encoder_mode")
        runtime_box.prop(self, "text_encoder_url")
        runtime_box.prop(self, "offline_only")
        runtime_box.prop(self, "job_timeout_seconds")
        runtime_box.prop(self, "keep_temp_files")

        info = (
            "Beyond Motion runs Kimodo with Blender's bundled Python and extension-local wheels in "
            "_vendor/. The vendored kimodo repo in this extension is used as the source code for "
            "the worker process. Runtime model assets are prepared into this extension's model cache. "
            "On Apple Silicon, Auto will prefer Metal/MPS after CUDA, and the local Kimodo text encoder "
            "will switch away from bfloat16 automatically when MPS is selected."
        )
        wrapped = wrap_text_to_panel(info, context, full_width=True)
        box = layout.box()
        for line in wrapped.splitlines():
            box.label(text=line)

        if self.text_encoder_mode in {"auto", "local"}:
            token_note = (
                "Beyond Motion now defaults to an open local text encoder, so the usual setup path "
                "does not require gated model approval. Hugging Face login is only needed if you "
                "switch back to a legacy gated encoder workflow."
            )
            token_box = layout.box()
            token_box.alert = runtime_status.next_auth_step == "login"
            for line in wrap_text_to_panel(token_note, context, full_width=True).splitlines():
                token_box.label(text=line)

        python_path = Path(self.resolved_python_executable())
        box.label(text="Blender Python: " + str(python_path), icon="FILE_SCRIPT")
