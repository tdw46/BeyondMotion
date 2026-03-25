from __future__ import annotations

import traceback

from bpy.props import StringProperty
from bpy.types import Context, Object, Operator

from .dependency_manager import get_dependency_status
from .preferences import get_preferences
from .retarget import apply_generated_motion, apply_static_source_motion, build_constraint_request
from .runtime import run_generation_job
from .runtime_setup import get_runtime_setup_status
from .utils import selected_keyframes_for_object, wrap_text_to_panel


def _active_armature_object(context: Context) -> Object | None:
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    return None


def _runtime_ready_issue(context: Context, armature_object: Object):
    prefs = get_preferences(context)
    dependency_status = get_dependency_status(prefs.torch_device if prefs else "auto")
    if not dependency_status.ready:
        return "Install generation dependencies in Beyond Motion preferences first."
    settings = armature_object.data.beyond_motion
    runtime_status = get_runtime_setup_status(
        model_name=settings.model_name,
        text_encoder_mode=prefs.text_encoder_mode if prefs else "auto",
        text_encoder_url=prefs.text_encoder_url if prefs else "",
        checkpoint_dir_override=prefs.checkpoint_dir if prefs else "",
        hf_token=prefs.hf_token if prefs else "",
        offline_only=bool(prefs.offline_only) if prefs else False,
    )
    if not runtime_status.ready:
        return runtime_status.issues[0] if runtime_status.issues else "Finish Generation Setup first."
    return None


def _draw_prompt_preview(layout, context: Context, prompt: str) -> None:
    preview_box = layout.box()
    preview_box.label(text="Prompt Preview", icon="TEXT")
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        preview_box.label(text="Enter a movement prompt to preview it here.", icon="INFO")
        return
    wrapped = wrap_text_to_panel(prompt_text, context, full_width=True, preferred_chars=42)
    for line in (wrapped.splitlines() or [""]):
        preview_box.label(text=line)


class BEYONDMOTION_OT_generate_inbetweens(Operator):
    bl_idname = "beyond_motion.generate_inbetweens"
    bl_label = "AI Interpolation"
    bl_description = "Generate a constrained local motion segment with Kimodo and apply it back to the mapped rig"
    bl_options = {"REGISTER", "UNDO"}

    prompt_text: StringProperty(  # type: ignore[valid-type]
        name="Prompt",
        description="Text prompt for the full generated segment",
        options={"TEXTEDIT_UPDATE"},
    )

    def _sync_prompt_to_settings(self, context: Context) -> None:
        properties = getattr(self, "properties", None)
        if properties is None or not properties.is_property_set("prompt_text"):
            return
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return
        settings = armature_object.data.beyond_motion
        if settings.prompt != self.prompt_text:
            settings.prompt = self.prompt_text

    def invoke(self, context: Context, event) -> set[str]:
        del event
        armature_object = _active_armature_object(context)
        if armature_object is None:
            self.report({"ERROR"}, "Select an armature first.")
            return {"CANCELLED"}
        runtime_issue = _runtime_ready_issue(context, armature_object)
        if runtime_issue:
            self.report({"ERROR"}, runtime_issue)
            return {"CANCELLED"}
        source_frames = selected_keyframes_for_object(armature_object)
        if len(source_frames) < 2:
            self.report({"ERROR"}, "Select at least two keyframes in the Timeline, Dope Sheet, or Graph Editor.")
            return {"CANCELLED"}
        self.prompt_text = armature_object.data.beyond_motion.prompt
        return context.window_manager.invoke_props_dialog(self, width=280)

    def check(self, context: Context) -> bool:
        self._sync_prompt_to_settings(context)
        return True

    def draw(self, context: Context) -> None:
        layout = self.layout
        armature_object = _active_armature_object(context)
        if armature_object is None:
            layout.label(text="Select an armature first.", icon="ERROR")
            return
        settings = armature_object.data.beyond_motion
        selected_frames = selected_keyframes_for_object(armature_object)
        summary_box = layout.box()
        summary_box.label(text="Generation", icon="IPO_EASE_IN_OUT")
        if selected_frames:
            summary_box.label(
                text=f"Selected Keyframes: {len(selected_frames)} ({selected_frames[0]} to {selected_frames[-1]})",
                icon="ACTION",
            )
        else:
            summary_box.alert = True
            summary_box.label(text="Select at least two keyframes before generating.", icon="ERROR")

        summary_box.separator()
        summary_box.prop(settings, "root_target_mode")
        if settings.root_target_mode == "MOTION_ROOT":
            summary_box.prop_search(settings, "motion_root_bone", armature_object.data, "bones", text="Motion Root")
        summary_box.prop(settings, "blender_forward_axis")

        layout.separator()
        prompt_box = layout.box()
        prompt_header = prompt_box.row()
        prompt_header.alert = True
        prompt_header.label(text="Movement Prompt", icon="TEXT")
        prompt_box.prop(self, "prompt_text", text="")
        _draw_prompt_preview(prompt_box, context, self.prompt_text)

        layout.separator()
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
        settings_box.prop(settings, "hold_frame_bias")
        settings_box.prop(settings, "keypose_match_frames")

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return {"CANCELLED"}

        settings = armature_object.data.beyond_motion
        settings.ensure_human_bones()
        self._sync_prompt_to_settings(context)
        runtime_issue = _runtime_ready_issue(context, armature_object)
        if runtime_issue:
            self.report({"ERROR"}, runtime_issue)
            return {"CANCELLED"}

        if not settings.prompt.strip():
            self.report({"ERROR"}, "Enter a motion prompt before generating.")
            return {"CANCELLED"}

        source_frames = selected_keyframes_for_object(armature_object)
        if len(source_frames) < 2:
            self.report({"ERROR"}, "Select at least two keyframes in the Timeline, Dope Sheet, or Graph Editor.")
            return {"CANCELLED"}

        context.window.cursor_set("WAIT")
        try:
            request, source_data = build_constraint_request(context, armature_object, settings, source_frames)
            if request is None or not source_data.generation_required:
                apply_static_source_motion(context, armature_object, settings, source_data)
                response = {"device": "none"}
                worker_log = ""
            else:
                output, response, worker_log = run_generation_job(context, request)
                apply_generated_motion(context, armature_object, settings, source_data, output)
        except Exception as error:
            traceback.print_exc()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        finally:
            context.window.cursor_set("DEFAULT")

        if worker_log:
            print(worker_log)
        device = response.get("device", "unknown")
        if request is None or not source_data.generation_required:
            frame_count = source_frames[-1] - source_frames[0] + 1
            self.report({"INFO"}, f"Applied a static hold across {frame_count} frames.")
        else:
            self.report({"INFO"}, f"Generated {request['num_frames']} frames with {settings.model_name} on {device}.")
        return {"FINISHED"}
