from __future__ import annotations

import traceback

from bpy.types import Context, Object, Operator

from .dependency_manager import get_dependency_status
from .retarget import apply_generated_motion, build_constraint_request
from .runtime import run_generation_job
from .preferences import get_preferences
from .runtime_setup import get_runtime_setup_status
from .utils import selected_keyframes_for_object


def _active_armature_object(context: Context) -> Object | None:
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    return None


class BEYONDMOTION_OT_generate_inbetweens(Operator):
    bl_idname = "beyond_motion.generate_inbetweens"
    bl_label = "Generate Inbetweens"
    bl_description = "Generate a constrained local motion segment with Kimodo and apply it back to the mapped rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return {"CANCELLED"}

        settings = armature_object.data.beyond_motion
        settings.ensure_human_bones()
        prefs = get_preferences(context)
        dependency_status = get_dependency_status(prefs.torch_device if prefs else "auto")
        if not dependency_status.ready:
            self.report({"ERROR"}, "Install generation dependencies in Beyond Motion preferences first.")
            return {"CANCELLED"}
        runtime_status = get_runtime_setup_status(
            model_name=settings.model_name,
            text_encoder_mode=prefs.text_encoder_mode if prefs else "auto",
            text_encoder_url=prefs.text_encoder_url if prefs else "",
            checkpoint_dir_override=prefs.checkpoint_dir if prefs else "",
            hf_token=prefs.hf_token if prefs else "",
            offline_only=bool(prefs.offline_only) if prefs else False,
        )
        if not runtime_status.ready:
            self.report({"ERROR"}, runtime_status.issues[0] if runtime_status.issues else "Finish Generation Setup first.")
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
        self.report({"INFO"}, f"Generated {request['num_frames']} frames with {settings.model_name} on {device}.")
        return {"FINISHED"}
