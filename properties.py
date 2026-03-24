from __future__ import annotations

from collections.abc import Iterable

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Armature, PropertyGroup

from .human_bones import HumanBoneSpecifications


MODEL_ITEMS = (
    ("kimodo-soma-rp", "Kimodo SOMA Rigplay", "Recommended human model trained on Rigplay 1", "ARMATURE_DATA", 0),
    ("kimodo-soma-seed", "Kimodo SOMA SEED", "Public-data SOMA model trained on BONES-SEED", "OUTLINER_OB_ARMATURE", 1),
)

CFG_TYPE_ITEMS = (
    ("separated", "Separated", "Independent text and constraint guidance", "SETTINGS", 0),
    ("regular", "Regular", "Single CFG weight", "SETTINGS", 1),
    ("nocfg", "Off", "Disable classifier-free guidance", "SETTINGS", 2),
)

ROOT_TARGET_ITEMS = (
    ("HIPS", "Hips Bone", "Apply generated translation to the mapped hips bone", "BONE_DATA", 0),
    ("MOTION_ROOT", "Motion Root Bone", "Apply generated translation to a separate motion root/controller bone", "BONE_DATA", 1),
    ("OBJECT", "Armature Object", "Apply generated translation to the armature object itself", "OBJECT_DATA", 2),
)

FORWARD_AXIS_ITEMS = (
    ("NEGATIVE_Y", "-Y Forward", "Typical Blender character facing", "AXIS_FRONT", 0),
    ("POSITIVE_Y", "+Y Forward", "Character faces Blender +Y", "AXIS_FRONT", 1),
    ("POSITIVE_X", "+X Forward", "Character faces Blender +X", "AXIS_SIDE", 2),
    ("NEGATIVE_X", "-X Forward", "Character faces Blender -X", "AXIS_SIDE", 3),
)


class BeyondMotionHumanBoneAssignment(PropertyGroup):
    human_bone_name: StringProperty()  # type: ignore[valid-type]
    bone_name: StringProperty(name="Bone")  # type: ignore[valid-type]


class BeyondMotionArmatureSettings(PropertyGroup):
    prompt: StringProperty(  # type: ignore[valid-type]
        name="Prompt",
        description="Text prompt for the full generated segment",
    )
    source_frames: StringProperty(  # type: ignore[valid-type]
        name="Input Frames",
        description="Comma-separated frame numbers used as Kimodo full-body keyframes",
    )
    model_name: EnumProperty(  # type: ignore[valid-type]
        name="Model",
        items=MODEL_ITEMS,
        default="kimodo-soma-rp",
    )
    diffusion_steps: IntProperty(  # type: ignore[valid-type]
        name="Diffusion Steps",
        default=100,
        min=10,
        max=400,
    )
    cfg_type: EnumProperty(  # type: ignore[valid-type]
        name="CFG",
        items=CFG_TYPE_ITEMS,
        default="separated",
    )
    cfg_text_weight: FloatProperty(  # type: ignore[valid-type]
        name="Text Weight",
        default=2.0,
        min=0.0,
    )
    cfg_constraint_weight: FloatProperty(  # type: ignore[valid-type]
        name="Constraint Weight",
        default=2.0,
        min=0.0,
    )
    seed: IntProperty(  # type: ignore[valid-type]
        name="Seed",
        default=-1,
        min=-1,
        description="Use -1 for a random seed",
    )
    apply_postprocess: BoolProperty(  # type: ignore[valid-type]
        name="Post-Process",
        default=False,
        description="Enable Kimodo post-processing. This may require MotionCorrection to be installed in the external Python environment",
    )
    root_target_mode: EnumProperty(  # type: ignore[valid-type]
        name="Root Target",
        items=ROOT_TARGET_ITEMS,
        default="HIPS",
    )
    motion_root_bone: StringProperty(  # type: ignore[valid-type]
        name="Motion Root Bone",
        description="Optional FK/root controller bone used when Root Target is Motion Root Bone",
    )
    blender_forward_axis: EnumProperty(  # type: ignore[valid-type]
        name="Forward Axis",
        items=FORWARD_AXIS_ITEMS,
        default="NEGATIVE_Y",
    )
    human_bones: CollectionProperty(type=BeyondMotionHumanBoneAssignment)  # type: ignore[valid-type]

    def ensure_human_bones(self) -> None:
        existing = {item.human_bone_name for item in self.human_bones}
        for spec in HumanBoneSpecifications.ALL:
            if spec.name in existing:
                continue
            item = self.human_bones.add()
            item.human_bone_name = spec.name
            item.bone_name = ""

    def assignment_for(self, human_bone_name: str) -> BeyondMotionHumanBoneAssignment:
        self.ensure_human_bones()
        for item in self.human_bones:
            if item.human_bone_name == human_bone_name:
                return item
        message = f"Missing human bone entry: {human_bone_name}"
        raise KeyError(message)

    def assignment_map(self) -> dict[str, str]:
        self.ensure_human_bones()
        return {
            item.human_bone_name: item.bone_name
            for item in self.human_bones
            if item.bone_name
        }

    def required_bones_missing(self) -> list[str]:
        mapping = self.assignment_map()
        return [spec.title for spec in HumanBoneSpecifications.ALL if spec.requirement and not mapping.get(spec.name)]

    def duplicate_bone_names(self) -> set[str]:
        counts: dict[str, int] = {}
        for bone_name in self.assignment_map().values():
            counts[bone_name] = counts.get(bone_name, 0) + 1
        return {bone_name for bone_name, count in counts.items() if count > 1}


def parse_source_frames(text: str) -> list[int]:
    frames: list[int] = []
    for chunk in (part.strip() for part in text.split(",")):
        if not chunk:
            continue
        frames.append(int(float(chunk)))
    return sorted(set(frames))


def update_source_frames_from_iterable(settings: BeyondMotionArmatureSettings, frames: Iterable[int]) -> None:
    unique_frames = sorted({int(frame) for frame in frames})
    settings.source_frames = ", ".join(str(frame) for frame in unique_frames)


def register_properties() -> None:
    bpy.types.Armature.beyond_motion = PointerProperty(type=BeyondMotionArmatureSettings)


def unregister_properties() -> None:
    if hasattr(bpy.types.Armature, "beyond_motion"):
        del bpy.types.Armature.beyond_motion
