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


def update_generation_preview(self, context) -> None:
    del self
    try:
        window_manager = getattr(context, "window_manager", None)
        windows = getattr(window_manager, "windows", []) if window_manager else []
        for window in windows:
            screen = getattr(window, "screen", None)
            if screen is None:
                continue
            for area in screen.areas:
                area.tag_redraw()
    except Exception:
        pass


class BeyondMotionHumanBoneAssignment(PropertyGroup):
    human_bone_name: StringProperty()  # type: ignore[valid-type]
    bone_name: StringProperty(name="Bone")  # type: ignore[valid-type]


class BeyondMotionArmatureSettings(PropertyGroup):
    prompt: StringProperty(  # type: ignore[valid-type]
        name="Prompt",
        description=(
            "Describe the motion you want in plain English. "
            "This steers the overall feel of the in-between animation while still respecting your keyed poses."
        ),
        options={"TEXTEDIT_UPDATE"},
        update=update_generation_preview,
    )
    source_frames: StringProperty(  # type: ignore[valid-type]
        name="Input Frames",
        description="Comma-separated frame numbers used as Kimodo full-body keyframes",
    )
    model_name: EnumProperty(  # type: ignore[valid-type]
        name="Model",
        items=MODEL_ITEMS,
        default="kimodo-soma-rp",
        description=(
            "Choose which Kimodo motion model to use. "
            "Rigplay is the general recommended option for human animation, while SEED can behave a little differently because it was trained on public data."
        ),
    )
    diffusion_steps: IntProperty(  # type: ignore[valid-type]
        name="Diffusion Steps",
        default=100,
        min=10,
        max=400,
        description=(
            "How long the model spends refining the motion. "
            "Higher values can improve stability and detail, but they also take longer and do not always give noticeably better results."
        ),
    )
    cfg_type: EnumProperty(  # type: ignore[valid-type]
        name="CFG",
        items=CFG_TYPE_ITEMS,
        default="separated",
        description=(
            "Controls how strongly the model follows your text prompt versus your keyed pose constraints. "
            "Separated gives the most control, Regular is simpler, and Off is the loosest and least guided."
        ),
    )
    cfg_text_weight: FloatProperty(  # type: ignore[valid-type]
        name="Text Weight",
        default=2.0,
        min=0.0,
        description=(
            "How strongly the animation should follow the motion prompt. "
            "Higher values push harder toward the described action, but can make the result less faithful to the exact feel of your source poses."
        ),
    )
    cfg_constraint_weight: FloatProperty(  # type: ignore[valid-type]
        name="Constraint Weight",
        default=2.0,
        min=0.0,
        description=(
            "How strongly the generated motion should stay locked to your selected key poses. "
            "Higher values make the animation more reliably match your input poses at the cost of flexibility."
        ),
    )
    seed: IntProperty(  # type: ignore[valid-type]
        name="Seed",
        default=-1,
        min=-1,
        description=(
            "Controls random variation in the generation. "
            "Use the same seed to try to reproduce a result, or -1 to get a fresh variation each time."
        ),
    )
    apply_postprocess: BoolProperty(  # type: ignore[valid-type]
        name="Post-Process",
        default=False,
        description=(
            "Run Kimodo's cleanup pass after generation. "
            "This can smooth or correct some motion issues, but it may take longer and depends on the extra postprocess runtime being available."
        ),
    )
    keypose_match_frames: IntProperty(  # type: ignore[valid-type]
        name="Keypose Match",
        default=0,
        min=0,
        max=12,
        description=(
            "How many frames on each side of your keyed poses should blend back toward the original animation after generation. "
            "Set this to 0 to keep the current adjacent-override behavior only. Higher values add an Animation Layers-style pose matching pass that hits your original key poses more exactly, while surrounding frames ease into and out of them."
        ),
    )
    root_target_mode: EnumProperty(  # type: ignore[valid-type]
        name="Root Target",
        items=ROOT_TARGET_ITEMS,
        default="HIPS",
        description=(
            "Choose where generated root translation should be applied in Blender. "
            "Use Hips for typical FK rigs, Motion Root for a dedicated controller bone, or Armature Object if the whole rig should move as one object."
        ),
    )
    motion_root_bone: StringProperty(  # type: ignore[valid-type]
        name="Motion Root Bone",
        description=(
            "Pick the controller or root bone that should receive generated movement when Root Target is set to Motion Root Bone. "
            "This is useful for rigs that separate body motion from the hips bone."
        ),
    )
    blender_forward_axis: EnumProperty(  # type: ignore[valid-type]
        name="Forward Axis",
        items=FORWARD_AXIS_ITEMS,
        default="NEGATIVE_Y",
        description=(
            "Tell Beyond Motion which direction your character faces in its Blender rest orientation. "
            "If this is wrong, the generated motion can look twisted or rotated the wrong way."
        ),
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
