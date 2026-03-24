from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Context, Object, Operator

from .properties import BeyondMotionArmatureSettings, update_source_frames_from_iterable
from .vrm_bridge import auto_detect_human_bones


def _active_armature_object(context: Context) -> Object | None:
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    return None


class BEYONDMOTION_OT_auto_assign_human_bones(Operator):
    bl_idname = "beyond_motion.auto_assign_human_bones"
    bl_label = "Automatic Bone Assignment"
    bl_description = "Assign humanoid bones using Beyond Motion's built-in humanoid auto-detection"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return {"CANCELLED"}

        settings = armature_object.data.beyond_motion
        settings.ensure_human_bones()
        detected = auto_detect_human_bones(armature_object)
        for human_bone_name, bone_name in detected.items():
            settings.assignment_for(human_bone_name).bone_name = bone_name
        self.report({"INFO"}, f"Assigned {len(detected)} humanoid bones.")
        return {"FINISHED"}


class BEYONDMOTION_OT_clear_human_bones(Operator):
    bl_idname = "beyond_motion.clear_human_bones"
    bl_label = "Clear Assignments"
    bl_description = "Clear all humanoid bone assignments"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return {"CANCELLED"}
        settings = armature_object.data.beyond_motion
        settings.ensure_human_bones()
        for item in settings.human_bones:
            item.bone_name = ""
        return {"FINISHED"}


class BEYONDMOTION_OT_use_selected_keyframes(Operator):
    bl_idname = "beyond_motion.use_selected_keyframes"
    bl_label = "Use Selected Keyframes"
    bl_description = "Fill the input frame list from selected keyframes on the active action"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return {"CANCELLED"}
        animation_data = armature_object.animation_data
        action = animation_data.action if animation_data else None
        if action is None:
            self.report({"ERROR"}, "The active armature has no action.")
            return {"CANCELLED"}

        frames: set[int] = set()
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                if keyframe.select_control_point:
                    frames.add(int(round(keyframe.co.x)))

        if len(frames) < 2:
            self.report({"ERROR"}, "Select at least two keyframes in the Dope Sheet or Graph Editor.")
            return {"CANCELLED"}

        settings = armature_object.data.beyond_motion
        update_source_frames_from_iterable(settings, frames)
        self.report({"INFO"}, f"Using {len(frames)} selected keyframes.")
        return {"FINISHED"}
