from __future__ import annotations

import bpy
from bpy.types import Armature, Context, Object, Panel, UILayout

from .dependency_manager import (
    LOCAL_GENERATION_VRAM_GB,
    dependency_size_estimate,
    get_dependency_status,
    status_message,
)
from .human_bones import HumanBoneSpecification, HumanBoneSpecifications
from .preferences import get_preferences
from .runtime_setup import get_runtime_asset_job_state, get_runtime_setup_status
from .utils import selected_keyframes_for_object, wrap_text_to_panel

_PENDING_HUMAN_BONE_INITIALIZATIONS: set[str] = set()


def _active_armature_object(context: Context) -> Object | None:
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    return None


def _settings(context: Context):
    armature_object = _active_armature_object(context)
    if armature_object is None:
        return None
    return armature_object.data.beyond_motion


def _tag_relevant_redraw() -> None:
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "DOPESHEET_EDITOR", "GRAPH_EDITOR"}:
                area.tag_redraw()


def _initialize_human_bones_later(armature_data_name: str):
    def _run():
        _PENDING_HUMAN_BONE_INITIALIZATIONS.discard(armature_data_name)
        armature_data = bpy.data.armatures.get(armature_data_name)
        if armature_data is None:
            return None
        try:
            armature_data.beyond_motion.ensure_human_bones()
        except Exception as error:
            print(f"Beyond Motion: failed to initialize humanoid mapping for {armature_data_name}: {error}")
        _tag_relevant_redraw()
        return None

    if armature_data_name in _PENDING_HUMAN_BONE_INITIALIZATIONS:
        return
    _PENDING_HUMAN_BONE_INITIALIZATIONS.add(armature_data_name)
    bpy.app.timers.register(_run, first_interval=0.01)


def _draw_wrapped_lines(layout: UILayout, context: Context, text: str, *, full_width: bool = True, alert: bool = False, icon: str = "NONE") -> None:
    for index, line in enumerate(wrap_text_to_panel(text, context, full_width=full_width).splitlines() or [""]):
        row = layout.row()
        row.alert = alert
        row.label(text=line, icon=icon if index == 0 else "BLANK1")


def _needs_guided_token_step(runtime_status) -> bool:
    return runtime_status.next_auth_step == "login"


def _show_token_entry(runtime_status, hf_token: str) -> bool:
    return (not runtime_status.ready) and (_needs_guided_token_step(runtime_status) or bool((hf_token or "").strip()))


def _generation_popover_available(context: Context) -> bool:
    armature_object = _active_armature_object(context)
    return armature_object is not None and isinstance(armature_object.data, Armature)


def _draw_setup_box(
    layout: UILayout,
    context: Context,
    *,
    dependency_status,
    runtime_status,
    backend: str,
    download_estimate: str,
    disk_estimate: str,
) -> None:
    setup_box = layout.box()
    prefs = get_preferences(context)
    runtime_job = get_runtime_asset_job_state()
    blocking_issue = not dependency_status.ready or not runtime_status.ready
    setup_box.alert = blocking_issue
    setup_box.label(
        text="Generation Setup Required" if blocking_issue else "Generation Setup Ready",
        icon="ERROR" if blocking_issue else "CHECKMARK",
    )
    alert_text = (
        "Install Beyond Motion's extension-local dependencies and prepare the model runtime assets "
        "before creating animation. Local generation typically needs about "
        f"{LOCAL_GENERATION_VRAM_GB} GB of VRAM."
    )
    _draw_wrapped_lines(setup_box, context, alert_text, alert=blocking_issue)
    for text in (
        f"Selected runtime backend: {backend.upper()}",
        download_estimate,
        disk_estimate,
        f"Selected model cache: {runtime_status.model_path}",
        f"Local text encoder cache: {runtime_status.text_encoder_path}",
        (
            f"Hugging Face login detected from {runtime_status.hf_token_source}"
            if runtime_status.hf_token_available
            else (
                "Hugging Face login not required for the default open local text encoder"
                if runtime_status.next_auth_step != "login"
                else "Hugging Face login not detected"
            )
        ),
    ):
        _draw_wrapped_lines(setup_box, context, text)
    _draw_wrapped_lines(setup_box, context, status_message(dependency_status), icon="INFO")
    if runtime_job.get("active", False):
        _draw_wrapped_lines(
            setup_box,
            context,
            str(runtime_job.get("status_text", "Downloading assets... Please check back later.")),
            alert=True,
            icon="TIME",
        )
    elif runtime_job.get("error_text", ""):
        _draw_wrapped_lines(
            setup_box,
            context,
            str(runtime_job.get("error_text", "")),
            alert=True,
            icon="ERROR",
        )
    for warning in runtime_status.warnings:
        _draw_wrapped_lines(setup_box, context, warning, icon="INFO")
    for issue in runtime_status.issues:
        _draw_wrapped_lines(setup_box, context, issue, alert=True, icon="ERROR")
    if _show_token_entry(runtime_status, prefs.hf_token if prefs else ""):
        token_help = (
            "For secure local AI DL, log in at Hugging Face token settings, create a token, "
            "copy it, and paste it into Blender here. Beyond Motion will save that login "
            "locally and continue setup without requiring console input."
        )
        token_box = setup_box.box()
        token_box.alert = True
        token_box.label(text="Secure Local AI Login", icon="LOCKED")
        _draw_wrapped_lines(token_box, context, token_help, alert=True, icon="ERROR")
        if prefs is not None:
            token_box.prop(prefs, "hf_token", text="HF Token")
        token_actions = token_box.row(align=True)
        token_actions.operator(
            "beyond_motion.open_external_setup_url",
            text="Login and Create Token",
            icon="URL",
        ).target = "TOKENS"
        if prefs is not None and prefs.hf_token.strip():
            token_actions.operator(
                "beyond_motion.begin_hf_local_model_setup",
                text="Use Token and Continue",
                icon="LOCKED",
            )
    actions = setup_box.row(align=True)
    if not dependency_status.ready:
        actions.operator("beyond_motion.open_setup_preferences", icon="PREFERENCES")
        actions.operator("beyond_motion.install_generation_dependencies", icon="IMPORT")
    elif runtime_job.get("active", False):
        actions.enabled = False
        actions.label(text="Downloading assets...", icon="TIME")
    elif _needs_guided_token_step(runtime_status):
        if prefs is None or not prefs.hf_token.strip():
            actions.enabled = False
            actions.label(text="Paste an HF token above to continue.", icon="INFO")
    elif runtime_status.next_auth_step == "meta_access":
        actions.operator(
            "beyond_motion.approve_meta_access",
            text="Approve Meta Access",
            icon="LOCKED",
        )
    elif not runtime_status.ready:
        actions.operator(
            "beyond_motion.prepare_runtime_assets",
            text="Prepare Runtime Assets",
            icon="FILE_REFRESH",
        )
    else:
        actions.enabled = False
        actions.label(text="Setup complete", icon="CHECKMARK")


def _draw_root_settings(layout: UILayout, settings, armature_data: Armature) -> None:
    layout.prop(settings, "root_target_mode")
    if settings.root_target_mode == "MOTION_ROOT":
        layout.prop_search(settings, "motion_root_bone", armature_data, "bones", text="Motion Root")
    layout.prop(settings, "blender_forward_axis")


def _draw_tall_auto_assign_button(layout: UILayout) -> None:
    button_row = layout.row()
    button_row.scale_y = 1.3
    button_row.operator(
        "beyond_motion.auto_assign_human_bones",
        text="Automatic Bone Assignment",
        icon="ARMATURE_DATA",
    )


def _draw_generation_status(
    layout: UILayout,
    context: Context,
    armature_object: Object,
    armature_data: Armature,
    settings,
) -> list[int]:
    generation_box = layout.box()
    generation_box.label(text="Generation", icon="IPO_EASE_IN_OUT")
    selected_frames = selected_keyframes_for_object(armature_object)
    if selected_frames:
        generation_box.label(
            text=(
                f"Selected Keyframes: {len(selected_frames)} "
                f"({selected_frames[0]} to {selected_frames[-1]})"
            ),
            icon="ACTION",
        )
    else:
        empty_row = generation_box.row()
        empty_row.alert = True
        empty_row.label(text="Select at least two keyframes to generate inbetweens.", icon="ERROR")
    _draw_root_settings(generation_box, settings, armature_data)
    return selected_frames


def _draw_prompt_preview(layout: UILayout, context: Context, prompt: str) -> None:
    preview_box = layout.box()
    preview_box.label(text="Prompt Preview", icon="TEXT")
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        preview_box.label(text="Enter a movement prompt to preview it here.", icon="INFO")
        return
    wrapped = wrap_text_to_panel(prompt_text, context, full_width=True, preferred_chars=42)
    for line in (wrapped.splitlines() or [""]):
        preview_box.label(text=line)


def _draw_generation_settings(layout: UILayout, settings) -> None:
    settings_box = layout.box()
    settings_box.label(text="Generation Settings", icon="SETTINGS")
    settings_box.prop(settings, "model_name")
    settings_box.prop(settings, "diffusion_steps")
    settings_box.prop(settings, "cfg_type")
    if settings.cfg_type != "nocfg":
        settings_box.prop(settings, "cfg_text_weight")
        if settings.cfg_type == "separated":
            settings_box.prop(settings, "cfg_constraint_weight")
    settings_box.prop(settings, "seed")
    settings_box.prop(settings, "apply_postprocess")


def _draw_generation_button(layout: UILayout, *, enabled: bool) -> None:
    button_row = layout.row()
    button_row.alert = True
    button_row.scale_y = 1.6
    button_row.enabled = enabled
    button_row.operator_context = "EXEC_DEFAULT"
    button_row.operator("beyond_motion.generate_inbetweens", text="Generate AI In-Betweens", icon="IPO_EASE_IN_OUT")


def _draw_missing_required_bones(layout: UILayout, settings) -> None:
    missing_required = settings.required_bones_missing()
    if not missing_required:
        return
    warning_header, warning_body = layout.panel(
        "beyond_motion_missing_required_bones",
        default_closed=True,
    )
    warning_header.alert = True
    warning_header.label(text="Missing Required Bones", icon="ERROR")
    if warning_body:
        warning_body.alert = True
        warning_body.label(text="Required humanoid assignments still needed:")
        missing_column = warning_body.column(align=True)
        for title in missing_required:
            missing_column.label(text=title)


def _draw_keyframe_context_menu(self, context: Context) -> None:
    layout = self.layout
    layout.separator()
    row = layout.row()
    row.enabled = _generation_popover_available(context)
    row.operator_context = "INVOKE_DEFAULT"
    row.operator("beyond_motion.generate_inbetweens", text="AI Interpolation", icon="IPO_EASE_IN_OUT")


def draw_human_bone_search(
    layout: UILayout,
    armature_data: Armature,
    settings,
    human_bone_specification: HumanBoneSpecification,
) -> None:
    assignment = settings.assignment_for(human_bone_specification.name)
    duplicates = settings.duplicate_bone_names()
    row = layout.row(align=True)
    row.alert = bool(assignment.bone_name and assignment.bone_name in duplicates)
    row.prop_search(
        assignment,
        "bone_name",
        armature_data,
        "bones",
        text="",
        icon=human_bone_specification.icon,
    )


def draw_required_bones_layout(armature: Object, layout: UILayout) -> None:
    armature_data = armature.data
    if not isinstance(armature_data, Armature):
        return
    settings = armature_data.beyond_motion
    split_factor = 0.2

    layout.label(text="Required Human Bones", icon="ARMATURE_DATA")

    row = layout.row(align=True).split(factor=split_factor, align=True)
    label_column = row.column(align=True)
    label_column.label(text=HumanBoneSpecifications.HEAD.label)
    label_column.label(text=HumanBoneSpecifications.SPINE.label)
    label_column.label(text=HumanBoneSpecifications.HIPS.label)

    search_column = row.column(align=True)
    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.HEAD)
    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.SPINE)
    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.HIPS)

    row = layout.row(align=True).split(factor=split_factor, align=True)
    label_column = row.column(align=True)
    label_column.label(text="")
    label_column.label(text=HumanBoneSpecifications.LEFT_UPPER_ARM.label_no_left_right)
    label_column.label(text=HumanBoneSpecifications.LEFT_LOWER_ARM.label_no_left_right)
    label_column.label(text=HumanBoneSpecifications.LEFT_HAND.label_no_left_right)
    label_column.separator()
    label_column.label(text=HumanBoneSpecifications.LEFT_UPPER_LEG.label_no_left_right)
    label_column.label(text=HumanBoneSpecifications.LEFT_LOWER_LEG.label_no_left_right)
    label_column.label(text=HumanBoneSpecifications.LEFT_FOOT.label_no_left_right)

    search_column = row.column(align=True)
    right_left_row = search_column.row(align=True)
    right_left_row.label(text="Right")
    right_left_row.label(text="Left")

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_UPPER_ARM)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_UPPER_ARM)

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_LOWER_ARM)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_LOWER_ARM)

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_HAND)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_HAND)

    search_column.separator()

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_UPPER_LEG)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_UPPER_LEG)

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_LOWER_LEG)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_LOWER_LEG)

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_FOOT)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_FOOT)


def draw_optional_bones_layout(armature: Object, layout: UILayout) -> None:
    armature_data = armature.data
    if not isinstance(armature_data, Armature):
        return
    settings = armature_data.beyond_motion
    split_factor = 0.2

    row = layout.row(align=True).split(factor=split_factor, align=True)
    label_column = row.column(align=True)
    label_column.label(text="")
    label_column.label(text=HumanBoneSpecifications.LEFT_EYE.label_no_left_right)
    label_column.label(text=HumanBoneSpecifications.JAW.label)
    label_column.label(text=HumanBoneSpecifications.NECK.label)
    label_column.label(text=HumanBoneSpecifications.RIGHT_SHOULDER.label_no_left_right)
    label_column.label(text=HumanBoneSpecifications.UPPER_CHEST.label)
    label_column.label(text=HumanBoneSpecifications.CHEST.label)
    label_column.label(text=HumanBoneSpecifications.RIGHT_TOES.label_no_left_right)

    search_column = row.column(align=True)

    right_left_row = search_column.row(align=True)
    right_left_row.label(text="Right")
    right_left_row.label(text="Left")

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_EYE)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_EYE)

    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.JAW)
    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.NECK)

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_SHOULDER)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_SHOULDER)

    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.UPPER_CHEST)
    draw_human_bone_search(search_column, armature_data, settings, HumanBoneSpecifications.CHEST)

    right_left_row = search_column.row(align=True)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.RIGHT_TOES)
    draw_human_bone_search(right_left_row, armature_data, settings, HumanBoneSpecifications.LEFT_TOES)

    row = layout.row(align=True).split(factor=split_factor, align=True)
    label_column = row.column(align=True)
    label_column.label(text="")
    label_column.label(text="Left Thumb:")
    label_column.label(text="Left Index:")
    label_column.label(text="Left Middle:")
    label_column.label(text="Left Ring:")
    label_column.label(text="Left Little:")
    label_column.separator()
    label_column.label(text="Right Thumb:")
    label_column.label(text="Right Index:")
    label_column.label(text="Right Middle:")
    label_column.label(text="Right Ring:")
    label_column.label(text="Right Little:")

    search_column = row.column(align=True)

    finger_row = search_column.row(align=True)
    finger_row.label(text="Root")
    finger_row.label(text="")
    finger_row.label(text="Tip")

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_THUMB_METACARPAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_THUMB_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_THUMB_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_INDEX_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_INDEX_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_MIDDLE_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_RING_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_RING_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_RING_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.LEFT_LITTLE_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_THUMB_METACARPAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_THUMB_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_INDEX_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_RING_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_RING_DISTAL)

    finger_row = search_column.row(align=True)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE)
    draw_human_bone_search(finger_row, armature_data, settings, HumanBoneSpecifications.RIGHT_LITTLE_DISTAL)


class BEYONDMOTION_PT_main(Panel):
    bl_label = "Beyond Motion"
    bl_idname = "BEYONDMOTION_PT_main"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "HEADER"
    bl_ui_units_x = 26

    @classmethod
    def poll(cls, context: Context) -> bool:
        return getattr(context.space_data, "mode", "") == "TIMELINE"

    def draw(self, context: Context) -> None:
        layout = self.layout
        prefs = get_preferences(context)
        dependency_status = get_dependency_status(prefs.torch_device if prefs else "auto")
        backend, download_estimate, disk_estimate = dependency_size_estimate(prefs.torch_device if prefs else "auto")
        armature_object = _active_armature_object(context)
        model_name = "kimodo-soma-rp"
        if armature_object is not None and isinstance(armature_object.data, Armature):
            model_name = armature_object.data.beyond_motion.model_name
        runtime_status = get_runtime_setup_status(
            model_name=model_name,
            text_encoder_mode=prefs.text_encoder_mode if prefs else "auto",
            text_encoder_url=prefs.text_encoder_url if prefs else "",
            checkpoint_dir_override=prefs.checkpoint_dir if prefs else "",
            hf_token=prefs.hf_token if prefs else "",
            offline_only=bool(prefs.offline_only) if prefs else False,
        )
        if armature_object is None:
            _draw_setup_box(
                layout,
                context,
                dependency_status=dependency_status,
                runtime_status=runtime_status,
                backend=backend,
                download_estimate=download_estimate,
                disk_estimate=disk_estimate,
            )
            select_box = layout.box()
            select_box.label(text="Select an Armature", icon="ARMATURE_DATA")
            select_text = (
                "Choose a humanoid FK armature in the 3D View to map bones, auto-detect humanoid assignments, "
                "and generate Kimodo inbetweens from the keyframes you have selected."
            )
            _draw_wrapped_lines(select_box, context, select_text)
            return
        armature_data = armature_object.data
        if not isinstance(armature_data, Armature):
            return
        settings = armature_data.beyond_motion
        expected_human_bone_count = len(HumanBoneSpecifications.ALL)
        if len(settings.human_bones) < expected_human_bone_count:
            _initialize_human_bones_later(armature_data.name)
            init_box = layout.box()
            init_box.label(text="Preparing Humanoid Mapping", icon="ARMATURE_DATA")
            init_text = (
                "Beyond Motion is setting up the humanoid bone mapping slots for this armature. "
                "The panel will refresh automatically in a moment."
            )
            _draw_wrapped_lines(init_box, context, init_text)
            return

        if not dependency_status.ready or not runtime_status.ready:
            _draw_setup_box(
                layout,
                context,
                dependency_status=dependency_status,
                runtime_status=runtime_status,
                backend=backend,
                download_estimate=download_estimate,
                disk_estimate=disk_estimate,
            )
            return

        selected_frames = _draw_generation_status(layout, context, armature_object, armature_data, settings)
        _draw_wrapped_lines(
            layout.box(),
            context,
            "Click AI to enter the movement prompt and generation settings for the selected keyframes.",
            icon="INFO",
        )
        _draw_missing_required_bones(layout, settings)

        humanoid_box = layout.box()
        humanoid_header = humanoid_box.row(align=True)
        humanoid_header.label(text="Humanoid", icon="OUTLINER_OB_ARMATURE")
        humanoid_header.operator("beyond_motion.clear_human_bones", text="", icon="TRASH")
        _draw_tall_auto_assign_button(humanoid_box)
        draw_required_bones_layout(armature_object, humanoid_box.box())
        optional_header, optional_body = humanoid_box.panel(
            "beyond_motion_optional_human_bones",
            default_closed=True,
        )
        optional_header.label(text="Optional Human Bones", icon="BONE_DATA")
        if optional_body:
            draw_optional_bones_layout(armature_object, optional_body.box())
        _draw_tall_auto_assign_button(humanoid_box)

        runtime_header, runtime_body = layout.panel("beyond_motion_runtime", default_closed=True)
        runtime_header.label(text="Runtime", icon="PREFERENCES")
        if not runtime_body:
            return
        if prefs is None:
            runtime_body.label(text="Open add-on preferences to install generation dependencies.", icon="INFO")
            return
        runtime_body.label(text=f"Python: {prefs.resolved_python_executable()}")
        runtime_body.label(text=f"Device: {prefs.torch_device}")
        runtime_body.label(text=f"MPS Fallback: {'On' if prefs.enable_mps_fallback else 'Off'}")
        runtime_body.label(text=f"Text Encoder Mode: {prefs.text_encoder_mode}")
        runtime_body.label(text=f"Text Encoder: {prefs.text_encoder_url or 'auto'}")
        runtime_body.label(text=f"Offline Only: {'On' if prefs.offline_only else 'Off'}")
        if prefs.checkpoint_dir:
            runtime_body.label(text=f"Checkpoints: {prefs.checkpoint_dir}")
        runtime_body.label(
            text=(
                "Local text encoder service reachable"
                if runtime_status.text_encoder_service_reachable
                else "Using bundled local text encoder assets"
            ),
            icon="URL",
        )


class BEYONDMOTION_PT_generate(Panel):
    bl_label = "AI Interpolation"
    bl_idname = "BEYONDMOTION_PT_generate"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "HEADER"
    bl_ui_units_x = 13

    @classmethod
    def poll(cls, context: Context) -> bool:
        return getattr(context.space_data, "mode", "") == "TIMELINE"

    def draw(self, context: Context) -> None:
        layout = self.layout
        prefs = get_preferences(context)
        armature_object = _active_armature_object(context)
        if armature_object is None:
            info_box = layout.box()
            info_box.alert = True
            info_box.label(text="Select an Armature", icon="ERROR")
            _draw_wrapped_lines(
                info_box,
                context,
                "Choose a humanoid FK armature first to enter a movement prompt and generate AI in-betweens.",
                alert=True,
                icon="ERROR",
            )
            return
        armature_data = armature_object.data
        if not isinstance(armature_data, Armature):
            return
        settings = armature_data.beyond_motion
        dependency_status = get_dependency_status(prefs.torch_device if prefs else "auto")
        runtime_status = get_runtime_setup_status(
            model_name=settings.model_name,
            text_encoder_mode=prefs.text_encoder_mode if prefs else "auto",
            text_encoder_url=prefs.text_encoder_url if prefs else "",
            checkpoint_dir_override=prefs.checkpoint_dir if prefs else "",
            hf_token=prefs.hf_token if prefs else "",
            offline_only=bool(prefs.offline_only) if prefs else False,
        )
        if not dependency_status.ready or not runtime_status.ready:
            backend, download_estimate, disk_estimate = dependency_size_estimate(prefs.torch_device if prefs else "auto")
            _draw_setup_box(
                layout,
                context,
                dependency_status=dependency_status,
                runtime_status=runtime_status,
                backend=backend,
                download_estimate=download_estimate,
                disk_estimate=disk_estimate,
            )
            return
        selected_frames = _draw_generation_status(layout, context, armature_object, armature_data, settings)
        layout.separator()

        prompt_box = layout.box()
        prompt_header = prompt_box.row()
        prompt_header.alert = True
        prompt_header.label(text="Movement Prompt", icon="TEXT")
        prompt_box.prop(settings, "prompt", text="")
        _draw_prompt_preview(prompt_box, context, settings.prompt)

        layout.separator()
        _draw_generation_settings(layout, settings)

        layout.separator()
        _draw_missing_required_bones(layout, settings)

        layout.separator()
        _draw_generation_button(
            layout,
            enabled=len(selected_frames) >= 2 and bool(settings.prompt.strip()),
        )


def _draw_timeline_header(self, context: Context) -> None:
    space_data = context.space_data
    if getattr(space_data, "type", "") != "DOPESHEET_EDITOR" or getattr(space_data, "mode", "") != "TIMELINE":
        return

    layout = self.layout
    layout.separator()
    layout.popover(panel="BEYONDMOTION_PT_main", text="Beyond", icon="ARMATURE_DATA")

    row = layout.row(align=True)
    row.enabled = _generation_popover_available(context)
    row.popover(panel="BEYONDMOTION_PT_generate", text="AI", icon="IPO_EASE_IN_OUT")


def register_header_draw() -> None:
    bpy.types.DOPESHEET_HT_header.append(_draw_timeline_header)
    bpy.types.DOPESHEET_MT_context_menu.append(_draw_keyframe_context_menu)
    bpy.types.GRAPH_MT_context_menu.append(_draw_keyframe_context_menu)


def unregister_header_draw() -> None:
    try:
        bpy.types.DOPESHEET_HT_header.remove(_draw_timeline_header)
    except Exception:
        pass
    for menu in (
        bpy.types.DOPESHEET_MT_context_menu,
        bpy.types.GRAPH_MT_context_menu,
    ):
        try:
            menu.remove(_draw_keyframe_context_menu)
        except Exception:
            pass
