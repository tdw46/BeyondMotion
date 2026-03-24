from __future__ import annotations

import re
from collections.abc import Mapping
from functools import cache

from bpy.types import Armature, Bone, Object

from .human_bones import HumanBoneSpecification, HumanBoneSpecifications


# Internalized humanoid auto-detection adapted from the VRM add-on mappings so
# Beyond Motion can remain self-contained.
FULLWIDTH_ASCII_TO_ASCII_MAP = {
    chr(ascii_char + 0x0000_FEE0): chr(ascii_char)
    for ascii_char in range(0x21, 0x7E)
}


@cache
def canonicalize_bone_name(bone_name: str) -> str:
    bone_name = "".join(FULLWIDTH_ASCII_TO_ASCII_MAP.get(char, char) for char in bone_name)
    bone_name = re.sub(r"([a-z])([A-Z])", r"\1.\2", bone_name)
    bone_name = bone_name.lower()
    bone_name = "".join(" " if char.isspace() else char for char in bone_name)
    bone_name = re.sub(r"(\d+)", r".\1.", bone_name).strip(".")
    components = re.split(r"[-._: (){}[\]<>]+", bone_name)
    for patterns, replacement in {
        ("l", "左"): "left",
        ("r", "右"): "right",
    }.items():
        components = [replacement if component in patterns else component for component in components]
    return ".".join(components)


def match_bone_name(bone_name1: str, bone_name2: str) -> bool:
    return canonicalize_bone_name(bone_name1) == canonicalize_bone_name(bone_name2)


def _match_count(armature: Armature, mapping: Mapping[str, HumanBoneSpecification]) -> int:
    count = 0
    filtered_mapping = {
        bpy_name: specification
        for bpy_name, specification in mapping.items()
        if any(match_bone_name(bpy_name, bone.name) for bone in armature.bones)
    }

    for bpy_name, specification in filtered_mapping.items():
        bone = next((bone for bone in armature.bones if match_bone_name(bpy_name, bone.name)), None)
        if bone is None:
            continue

        parent_specification: HumanBoneSpecification | None = None
        search_parent = specification.parent
        while search_parent:
            if search_parent in filtered_mapping.values():
                parent_specification = search_parent
                break
            search_parent = search_parent.parent

        found = False
        search_bone: Bone | None = bone.parent
        while search_bone:
            search_specification = next(
                (
                    candidate_specification
                    for candidate_bpy_name, candidate_specification in filtered_mapping.items()
                    if match_bone_name(candidate_bpy_name, search_bone.name)
                ),
                None,
            )
            if search_specification:
                found = search_specification == parent_specification
                break
            search_bone = search_bone.parent

        if found or parent_specification is None:
            count += 1

    return count


def _match_counts(
    armature: Armature,
    mapping: Mapping[str, HumanBoneSpecification],
) -> tuple[int, int]:
    required_mapping = {
        bpy_name: specification
        for bpy_name, specification in mapping.items()
        if specification.requirement
    }
    return (_match_count(armature, required_mapping), _match_count(armature, mapping))


def _sorted_required_first(
    armature: Armature,
    mapping: Mapping[str, HumanBoneSpecification],
) -> dict[str, HumanBoneSpecification]:
    bpy_bone_name_mapping: dict[str, HumanBoneSpecification] = {
        bpy_bone_name: specification
        for original_bone_name, specification in mapping.items()
        if (
            bpy_bone_name := next(
                (
                    bone.name
                    for bone in armature.bones
                    if match_bone_name(original_bone_name, bone.name)
                ),
                None,
            )
        )
    }
    sorted_mapping: dict[str, HumanBoneSpecification] = {}
    sorted_mapping.update(
        {
            bpy_name: specification
            for bpy_name, specification in bpy_bone_name_mapping.items()
            if specification.requirement
        }
    )
    sorted_mapping.update(
        {
            bpy_name: specification
            for bpy_name, specification in bpy_bone_name_mapping.items()
            if not specification.requirement
        }
    )
    return sorted_mapping


def _required_human_bone_count() -> int:
    return sum(1 for specification in HumanBoneSpecifications.ALL if specification.requirement)


def _prefixed_mapping(key_prefix: str, mapping: Mapping[str, HumanBoneSpecification]) -> dict[str, HumanBoneSpecification]:
    return {key_prefix + key: value for key, value in mapping.items()}


def _create_biped_config(armature: Object) -> tuple[str, Mapping[str, HumanBoneSpecification]]:
    biped_mapping = {
        (None, "Pelvis"): HumanBoneSpecifications.HIPS,
        (None, "Spine"): HumanBoneSpecifications.SPINE,
        (None, "Spine2"): HumanBoneSpecifications.CHEST,
        (None, "Neck"): HumanBoneSpecifications.NECK,
        (None, "Head"): HumanBoneSpecifications.HEAD,
        ("R", "Clavicle"): HumanBoneSpecifications.RIGHT_SHOULDER,
        ("R", "UpperArm"): HumanBoneSpecifications.RIGHT_UPPER_ARM,
        ("R", "Forearm"): HumanBoneSpecifications.RIGHT_LOWER_ARM,
        ("R", "Hand"): HumanBoneSpecifications.RIGHT_HAND,
        ("L", "Clavicle"): HumanBoneSpecifications.LEFT_SHOULDER,
        ("L", "UpperArm"): HumanBoneSpecifications.LEFT_UPPER_ARM,
        ("L", "Forearm"): HumanBoneSpecifications.LEFT_LOWER_ARM,
        ("L", "Hand"): HumanBoneSpecifications.LEFT_HAND,
        ("R", "Thigh"): HumanBoneSpecifications.RIGHT_UPPER_LEG,
        ("R", "Calf"): HumanBoneSpecifications.RIGHT_LOWER_LEG,
        ("R", "Foot"): HumanBoneSpecifications.RIGHT_FOOT,
        ("R", "Toe0"): HumanBoneSpecifications.RIGHT_TOES,
        ("L", "Thigh"): HumanBoneSpecifications.LEFT_UPPER_LEG,
        ("L", "Calf"): HumanBoneSpecifications.LEFT_LOWER_LEG,
        ("L", "Foot"): HumanBoneSpecifications.LEFT_FOOT,
        ("L", "Toe0"): HumanBoneSpecifications.LEFT_TOES,
        ("R", "Finger0"): HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
        ("R", "Finger01"): HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
        ("R", "Finger02"): HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
        ("L", "Finger0"): HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
        ("L", "Finger01"): HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
        ("L", "Finger02"): HumanBoneSpecifications.LEFT_THUMB_DISTAL,
        ("R", "Finger1"): HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
        ("R", "Finger11"): HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
        ("R", "Finger12"): HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
        ("L", "Finger1"): HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
        ("L", "Finger11"): HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
        ("L", "Finger12"): HumanBoneSpecifications.LEFT_INDEX_DISTAL,
        ("R", "Finger2"): HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
        ("R", "Finger21"): HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
        ("R", "Finger22"): HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
        ("L", "Finger2"): HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
        ("L", "Finger21"): HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
        ("L", "Finger22"): HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
        ("R", "Finger3"): HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
        ("R", "Finger31"): HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
        ("R", "Finger32"): HumanBoneSpecifications.RIGHT_RING_DISTAL,
        ("L", "Finger3"): HumanBoneSpecifications.LEFT_RING_PROXIMAL,
        ("L", "Finger31"): HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
        ("L", "Finger32"): HumanBoneSpecifications.LEFT_RING_DISTAL,
        ("R", "Finger4"): HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
        ("R", "Finger41"): HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
        ("R", "Finger42"): HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
        ("L", "Finger4"): HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
        ("L", "Finger41"): HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
        ("L", "Finger42"): HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
    }
    mapping: dict[str, HumanBoneSpecification] = {}
    for bone in armature.pose.bones:
        for (bone_lr, bone_suffix), specification in biped_mapping.items():
            if (
                bone.name.startswith("Bip001")
                and bone.name.endswith(bone_suffix)
                and (bone_lr is None or bone_lr in bone.name)
            ):
                mapping[bone.name] = specification
    if any(spec.requirement and spec not in mapping.values() for spec in HumanBoneSpecifications.ALL):
        return ("Bip001", {})
    return ("Bip001", mapping)


def _create_mmd_config(armature: Object) -> tuple[str, Mapping[str, HumanBoneSpecification]]:
    pairs = [
        ("頭", HumanBoneSpecifications.HEAD),
        ("右目", HumanBoneSpecifications.RIGHT_EYE),
        ("左目", HumanBoneSpecifications.LEFT_EYE),
        ("首", HumanBoneSpecifications.NECK),
        ("上半身2", HumanBoneSpecifications.CHEST),
        ("上半身", HumanBoneSpecifications.SPINE),
        ("センター", HumanBoneSpecifications.HIPS),
        ("右肩", HumanBoneSpecifications.RIGHT_SHOULDER),
        ("右腕", HumanBoneSpecifications.RIGHT_UPPER_ARM),
        ("右ひじ", HumanBoneSpecifications.RIGHT_LOWER_ARM),
        ("右手", HumanBoneSpecifications.RIGHT_HAND),
        ("右手首", HumanBoneSpecifications.RIGHT_HAND),
        ("右足", HumanBoneSpecifications.RIGHT_UPPER_LEG),
        ("右ひざ", HumanBoneSpecifications.RIGHT_LOWER_LEG),
        ("右足首", HumanBoneSpecifications.RIGHT_FOOT),
        ("右つま先", HumanBoneSpecifications.RIGHT_TOES),
        ("右足先EX", HumanBoneSpecifications.RIGHT_TOES),
        ("左肩", HumanBoneSpecifications.LEFT_SHOULDER),
        ("左腕", HumanBoneSpecifications.LEFT_UPPER_ARM),
        ("左ひじ", HumanBoneSpecifications.LEFT_LOWER_ARM),
        ("左手", HumanBoneSpecifications.LEFT_HAND),
        ("左手首", HumanBoneSpecifications.LEFT_HAND),
        ("左足", HumanBoneSpecifications.LEFT_UPPER_LEG),
        ("左ひざ", HumanBoneSpecifications.LEFT_LOWER_LEG),
        ("左足首", HumanBoneSpecifications.LEFT_FOOT),
        ("左つま先", HumanBoneSpecifications.LEFT_TOES),
        ("左足先EX", HumanBoneSpecifications.LEFT_TOES),
        ("右親指０", HumanBoneSpecifications.RIGHT_THUMB_METACARPAL),
        ("右親指１", HumanBoneSpecifications.RIGHT_THUMB_METACARPAL),
        ("右親指２", HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL),
        ("右人指１", HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL),
        ("右人指２", HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE),
        ("右人指３", HumanBoneSpecifications.RIGHT_INDEX_DISTAL),
        ("右中指１", HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL),
        ("右中指２", HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE),
        ("右中指３", HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL),
        ("右薬指１", HumanBoneSpecifications.RIGHT_RING_PROXIMAL),
        ("右薬指２", HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE),
        ("右薬指３", HumanBoneSpecifications.RIGHT_RING_DISTAL),
        ("右小指１", HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL),
        ("右小指２", HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE),
        ("右小指３", HumanBoneSpecifications.RIGHT_LITTLE_DISTAL),
        ("左親指０", HumanBoneSpecifications.LEFT_THUMB_METACARPAL),
        ("左親指１", HumanBoneSpecifications.LEFT_THUMB_METACARPAL),
        ("左親指２", HumanBoneSpecifications.LEFT_THUMB_PROXIMAL),
        ("左人指１", HumanBoneSpecifications.LEFT_INDEX_PROXIMAL),
        ("左人指２", HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE),
        ("左人指３", HumanBoneSpecifications.LEFT_INDEX_DISTAL),
        ("左中指１", HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL),
        ("左中指２", HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE),
        ("左中指３", HumanBoneSpecifications.LEFT_MIDDLE_DISTAL),
        ("左薬指１", HumanBoneSpecifications.LEFT_RING_PROXIMAL),
        ("左薬指２", HumanBoneSpecifications.LEFT_RING_INTERMEDIATE),
        ("左薬指３", HumanBoneSpecifications.LEFT_RING_DISTAL),
        ("左小指１", HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL),
        ("左小指２", HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE),
        ("左小指３", HumanBoneSpecifications.LEFT_LITTLE_DISTAL),
    ]
    mmd_name_to_bpy_name: dict[str, str] = {}
    for bone in armature.pose.bones:
        mmd_bone = getattr(bone, "mmd_bone", None)
        if not mmd_bone:
            continue
        name_j = getattr(mmd_bone, "name_j", None)
        if isinstance(name_j, str):
            mmd_name_to_bpy_name[name_j] = bone.name

    mapping: dict[str, HumanBoneSpecification] = {}
    for mmd_name, specification in pairs:
        bpy_bone_name = mmd_name_to_bpy_name.get(mmd_name)
        if not bpy_bone_name:
            continue
        if bpy_bone_name in mapping or specification in mapping.values():
            continue
        mapping[bpy_bone_name] = specification

    if any(spec.requirement and spec not in mapping.values() for spec in HumanBoneSpecifications.ALL):
        return ("MikuMikuDance", {})
    return ("MikuMikuDance", mapping)


LEFT_PATTERN = re.compile(r"^J_(Adj|Bip|Opt|Sec)_L_")
RIGHT_PATTERN = re.compile(r"^J_(Adj|Bip|Opt|Sec)_R_")
FULL_PATTERN = re.compile(r"^J_(Adj|Bip|Opt|Sec)_([CLR]_)?")


def _symmetrise_vroid_bone_name(bone_name: str) -> str:
    left = LEFT_PATTERN.sub("", bone_name)
    if left != bone_name:
        return left + "_L"
    right = RIGHT_PATTERN.sub("", bone_name)
    if right != bone_name:
        return right + "_R"
    return FULL_PATTERN.sub("", bone_name)


MIXAMO_MAPPING = {
    "mixamorig:Head": HumanBoneSpecifications.HEAD,
    "mixamorig:Neck": HumanBoneSpecifications.NECK,
    "mixamorig:Spine2": HumanBoneSpecifications.UPPER_CHEST,
    "mixamorig:Spine1": HumanBoneSpecifications.CHEST,
    "mixamorig:Spine": HumanBoneSpecifications.SPINE,
    "mixamorig:Hips": HumanBoneSpecifications.HIPS,
    "mixamorig:RightShoulder": HumanBoneSpecifications.RIGHT_SHOULDER,
    "mixamorig:RightArm": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "mixamorig:RightForeArm": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "mixamorig:RightHand": HumanBoneSpecifications.RIGHT_HAND,
    "mixamorig:LeftShoulder": HumanBoneSpecifications.LEFT_SHOULDER,
    "mixamorig:LeftArm": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "mixamorig:LeftForeArm": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "mixamorig:LeftHand": HumanBoneSpecifications.LEFT_HAND,
    "mixamorig:RightUpLeg": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "mixamorig:RightLeg": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "mixamorig:RightFoot": HumanBoneSpecifications.RIGHT_FOOT,
    "mixamorig:RightToeBase": HumanBoneSpecifications.RIGHT_TOES,
    "mixamorig:LeftUpLeg": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "mixamorig:LeftLeg": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "mixamorig:LeftFoot": HumanBoneSpecifications.LEFT_FOOT,
    "mixamorig:LeftToeBase": HumanBoneSpecifications.LEFT_TOES,
    "mixamorig:RightHandThumb1": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "mixamorig:RightHandThumb2": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "mixamorig:RightHandThumb3": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "mixamorig:RightHandIndex1": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "mixamorig:RightHandIndex2": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "mixamorig:RightHandIndex3": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "mixamorig:RightHandMiddle1": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "mixamorig:RightHandMiddle2": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "mixamorig:RightHandMiddle3": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "mixamorig:RightHandRing1": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "mixamorig:RightHandRing2": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "mixamorig:RightHandRing3": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "mixamorig:RightHandPinky1": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "mixamorig:RightHandPinky2": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "mixamorig:RightHandPinky3": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
    "mixamorig:LeftHandThumb1": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "mixamorig:LeftHandThumb2": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "mixamorig:LeftHandThumb3": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "mixamorig:LeftHandIndex1": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "mixamorig:LeftHandIndex2": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "mixamorig:LeftHandIndex3": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "mixamorig:LeftHandMiddle1": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "mixamorig:LeftHandMiddle2": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "mixamorig:LeftHandMiddle3": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "mixamorig:LeftHandRing1": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "mixamorig:LeftHandRing2": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "mixamorig:LeftHandRing3": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "mixamorig:LeftHandPinky1": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "mixamorig:LeftHandPinky2": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "mixamorig:LeftHandPinky3": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
}

UNREAL_MAPPING = {
    "pelvis": HumanBoneSpecifications.HIPS,
    "spine_01": HumanBoneSpecifications.SPINE,
    "spine_02": HumanBoneSpecifications.CHEST,
    "spine_03": HumanBoneSpecifications.UPPER_CHEST,
    "neck_01": HumanBoneSpecifications.NECK,
    "head": HumanBoneSpecifications.HEAD,
    "clavicle_r": HumanBoneSpecifications.RIGHT_SHOULDER,
    "upperarm_r": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "lowerarm_r": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "hand_r": HumanBoneSpecifications.RIGHT_HAND,
    "clavicle_l": HumanBoneSpecifications.LEFT_SHOULDER,
    "upperarm_l": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "lowerarm_l": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "hand_l": HumanBoneSpecifications.LEFT_HAND,
    "thigh_r": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "calf_r": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "foot_r": HumanBoneSpecifications.RIGHT_FOOT,
    "ball_r": HumanBoneSpecifications.RIGHT_TOES,
    "thigh_l": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "calf_l": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "foot_l": HumanBoneSpecifications.LEFT_FOOT,
    "ball_l": HumanBoneSpecifications.LEFT_TOES,
    "thumb_01_r": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "thumb_02_r": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "thumb_03_r": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "thumb_01_l": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "thumb_02_l": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "thumb_03_l": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "index_01_r": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "index_02_r": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "index_03_r": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "index_01_l": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "index_02_l": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "index_03_l": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "middle_01_r": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "middle_02_r": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "middle_03_r": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "middle_01_l": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "middle_02_l": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "middle_03_l": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "ring_01_r": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "ring_02_r": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "ring_03_r": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "ring_01_l": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "ring_02_l": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "ring_03_l": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "pinky_01_r": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "pinky_02_r": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "pinky_03_r": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
    "pinky_01_l": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "pinky_02_l": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "pinky_03_l": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
}

READY_PLAYER_ME_MAPPING = {
    "Head": HumanBoneSpecifications.HEAD,
    "RightEye": HumanBoneSpecifications.RIGHT_EYE,
    "LeftEye": HumanBoneSpecifications.LEFT_EYE,
    "Neck": HumanBoneSpecifications.NECK,
    "Spine2": HumanBoneSpecifications.UPPER_CHEST,
    "Spine1": HumanBoneSpecifications.CHEST,
    "Spine": HumanBoneSpecifications.SPINE,
    "Hips": HumanBoneSpecifications.HIPS,
    "RightShoulder": HumanBoneSpecifications.RIGHT_SHOULDER,
    "RightArm": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "RightForeArm": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "RightHand": HumanBoneSpecifications.RIGHT_HAND,
    "LeftShoulder": HumanBoneSpecifications.LEFT_SHOULDER,
    "LeftArm": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "LeftForeArm": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "LeftHand": HumanBoneSpecifications.LEFT_HAND,
    "RightUpLeg": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "RightLeg": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "RightFoot": HumanBoneSpecifications.RIGHT_FOOT,
    "RightToeBase": HumanBoneSpecifications.RIGHT_TOES,
    "LeftUpLeg": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "LeftLeg": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "LeftFoot": HumanBoneSpecifications.LEFT_FOOT,
    "LeftToeBase": HumanBoneSpecifications.LEFT_TOES,
    "RightHandThumb1": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "RightHandThumb2": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "RightHandThumb3": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "RightHandIndex1": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "RightHandIndex2": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "RightHandIndex3": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "RightHandMiddle1": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "RightHandMiddle2": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "RightHandMiddle3": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "RightHandRing1": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "RightHandRing2": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "RightHandRing3": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "RightHandPinky1": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "RightHandPinky2": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "RightHandPinky3": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
    "LeftHandThumb1": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "LeftHandThumb2": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "LeftHandThumb3": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "LeftHandIndex1": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "LeftHandIndex2": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "LeftHandIndex3": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "LeftHandMiddle1": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "LeftHandMiddle2": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "LeftHandMiddle3": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "LeftHandRing1": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "LeftHandRing2": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "LeftHandRing3": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "LeftHandPinky1": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "LeftHandPinky2": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "LeftHandPinky3": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
}

CATS_FIXED_MODEL_MAPPING = {
    "Hips": HumanBoneSpecifications.HIPS,
    "Spine": HumanBoneSpecifications.SPINE,
    "Chest": HumanBoneSpecifications.CHEST,
    "Neck": HumanBoneSpecifications.NECK,
    "Head": HumanBoneSpecifications.HEAD,
    "Eye_R": HumanBoneSpecifications.RIGHT_EYE,
    "Eye_L": HumanBoneSpecifications.LEFT_EYE,
    "Right shoulder": HumanBoneSpecifications.RIGHT_SHOULDER,
    "Right arm": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "Right elbow": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "Right wrist": HumanBoneSpecifications.RIGHT_HAND,
    "Left shoulder": HumanBoneSpecifications.LEFT_SHOULDER,
    "Left arm": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "Left elbow": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "Left wrist": HumanBoneSpecifications.LEFT_HAND,
    "Right leg": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "Right knee": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "Right ankle": HumanBoneSpecifications.RIGHT_FOOT,
    "Right toe": HumanBoneSpecifications.RIGHT_TOES,
    "Left leg": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "Left knee": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "Left ankle": HumanBoneSpecifications.LEFT_FOOT,
    "Left toe": HumanBoneSpecifications.LEFT_TOES,
    "Thumb0_R": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "Thumb1_R": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "Thumb2_R": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "Thumb0_L": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "Thumb1_L": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "Thumb2_L": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "IndexFinger1_R": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "IndexFinger2_R": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "IndexFinger3_R": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "IndexFinger1_L": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "IndexFinger2_L": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "IndexFinger3_L": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "MiddleFinger1_R": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "MiddleFinger2_R": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "MiddleFinger3_R": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "MiddleFinger1_L": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "MiddleFinger2_L": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "MiddleFinger3_L": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "RingFinger1_R": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "RingFinger2_R": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "RingFinger3_R": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "RingFinger1_L": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "RingFinger2_L": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "RingFinger3_L": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "LittleFinger1_R": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "LittleFinger2_R": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "LittleFinger3_R": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
    "LittleFinger1_L": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "LittleFinger2_L": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "LittleFinger3_L": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
}

ROCKETBOX_TEMPLATE_MAPPING = {
    "Head": HumanBoneSpecifications.HEAD,
    "REye": HumanBoneSpecifications.RIGHT_EYE,
    "LEye": HumanBoneSpecifications.LEFT_EYE,
    "MJaw": HumanBoneSpecifications.JAW,
    "Spine2": HumanBoneSpecifications.CHEST,
    "Spine1": HumanBoneSpecifications.SPINE,
    "Pelvis": HumanBoneSpecifications.HIPS,
    "R Clavicle": HumanBoneSpecifications.RIGHT_SHOULDER,
    "R UpperArm": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "R Forearm": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "R Hand": HumanBoneSpecifications.RIGHT_HAND,
    "L Clavicle": HumanBoneSpecifications.LEFT_SHOULDER,
    "L UpperArm": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "L Forearm": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "L Hand": HumanBoneSpecifications.LEFT_HAND,
    "R Thigh": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "R Calf": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "R Foot": HumanBoneSpecifications.RIGHT_FOOT,
    "R Toe0": HumanBoneSpecifications.RIGHT_TOES,
    "L Thigh": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "L Calf": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "L Foot": HumanBoneSpecifications.LEFT_FOOT,
    "L Toe0": HumanBoneSpecifications.LEFT_TOES,
    "R Finger0": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "R Finger01": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "R Finger02": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "R Finger1": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "R Finger11": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "R Finger12": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "R Finger2": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "R Finger21": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "R Finger22": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "R Finger3": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "R Finger31": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "R Finger32": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "R Finger4": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "R Finger41": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "R Finger42": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
    "L Finger0": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "L Finger01": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "L Finger02": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "L Finger1": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "L Finger11": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "L Finger12": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "L Finger2": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "L Finger21": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "L Finger22": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "L Finger3": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "L Finger31": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "L Finger32": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "L Finger4": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "L Finger41": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "L Finger42": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
}

RIGIFY_META_RIG_MAPPING = {
    "spine.006": HumanBoneSpecifications.HEAD,
    "spine.001": HumanBoneSpecifications.SPINE,
    "spine": HumanBoneSpecifications.HIPS,
    "upper_arm.R": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "forearm.R": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "hand.R": HumanBoneSpecifications.RIGHT_HAND,
    "upper_arm.L": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "forearm.L": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "hand.L": HumanBoneSpecifications.LEFT_HAND,
    "thigh.R": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "shin.R": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "foot.R": HumanBoneSpecifications.RIGHT_FOOT,
    "thigh.L": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "shin.L": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "foot.L": HumanBoneSpecifications.LEFT_FOOT,
    "eye.R": HumanBoneSpecifications.RIGHT_EYE,
    "eye.L": HumanBoneSpecifications.LEFT_EYE,
    "jaw": HumanBoneSpecifications.JAW,
    "spine.004": HumanBoneSpecifications.NECK,
    "shoulder.L": HumanBoneSpecifications.LEFT_SHOULDER,
    "shoulder.R": HumanBoneSpecifications.RIGHT_SHOULDER,
    "spine.003": HumanBoneSpecifications.UPPER_CHEST,
    "spine.002": HumanBoneSpecifications.CHEST,
    "toe.R": HumanBoneSpecifications.RIGHT_TOES,
    "toe.L": HumanBoneSpecifications.LEFT_TOES,
    "thumb.01.L": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "thumb.02.L": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "thumb.03.L": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "f_index.01.L": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "f_index.02.L": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "f_index.03.L": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "f_middle.01.L": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "f_middle.02.L": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "f_middle.03.L": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "f_ring.01.L": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "f_ring.02.L": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "f_ring.03.L": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "f_pinky.01.L": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "f_pinky.02.L": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "f_pinky.03.L": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
    "thumb.01.R": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "thumb.02.R": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "thumb.03.R": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "f_index.01.R": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "f_index.02.R": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "f_index.03.R": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "f_middle.01.R": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "f_middle.02.R": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "f_middle.03.R": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "f_ring.01.R": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "f_ring.02.R": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "f_ring.03.R": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "f_pinky.01.R": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "f_pinky.02.R": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "f_pinky.03.R": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
}

VROID_MAPPING = {
    "J_Bip_C_Hips": HumanBoneSpecifications.HIPS,
    "J_Bip_C_Spine": HumanBoneSpecifications.SPINE,
    "J_Bip_C_Chest": HumanBoneSpecifications.CHEST,
    "J_Bip_C_UpperChest": HumanBoneSpecifications.UPPER_CHEST,
    "J_Bip_C_Neck": HumanBoneSpecifications.NECK,
    "J_Bip_C_Head": HumanBoneSpecifications.HEAD,
    "J_Adj_L_FaceEye": HumanBoneSpecifications.LEFT_EYE,
    "J_Adj_R_FaceEye": HumanBoneSpecifications.RIGHT_EYE,
    "J_Bip_L_UpperLeg": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "J_Bip_L_LowerLeg": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "J_Bip_L_Foot": HumanBoneSpecifications.LEFT_FOOT,
    "J_Bip_L_ToeBase": HumanBoneSpecifications.LEFT_TOES,
    "J_Bip_R_UpperLeg": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "J_Bip_R_LowerLeg": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "J_Bip_R_Foot": HumanBoneSpecifications.RIGHT_FOOT,
    "J_Bip_R_ToeBase": HumanBoneSpecifications.RIGHT_TOES,
    "J_Bip_L_Shoulder": HumanBoneSpecifications.LEFT_SHOULDER,
    "J_Bip_L_UpperArm": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "J_Bip_L_LowerArm": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "J_Bip_L_Hand": HumanBoneSpecifications.LEFT_HAND,
    "J_Bip_R_Shoulder": HumanBoneSpecifications.RIGHT_SHOULDER,
    "J_Bip_R_UpperArm": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "J_Bip_R_LowerArm": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "J_Bip_R_Hand": HumanBoneSpecifications.RIGHT_HAND,
    "J_Bip_L_Thumb1": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "J_Bip_L_Thumb2": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "J_Bip_L_Thumb3": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "J_Bip_L_Index1": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "J_Bip_L_Index2": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "J_Bip_L_Index3": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "J_Bip_L_Middle1": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "J_Bip_L_Middle2": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "J_Bip_L_Middle3": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "J_Bip_L_Ring1": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "J_Bip_L_Ring2": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "J_Bip_L_Ring3": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "J_Bip_L_Little1": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "J_Bip_L_Little2": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "J_Bip_L_Little3": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
    "J_Bip_R_Thumb1": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "J_Bip_R_Thumb2": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "J_Bip_R_Thumb3": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "J_Bip_R_Index1": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "J_Bip_R_Index2": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "J_Bip_R_Index3": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "J_Bip_R_Middle1": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "J_Bip_R_Middle2": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "J_Bip_R_Middle3": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "J_Bip_R_Ring1": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "J_Bip_R_Ring2": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "J_Bip_R_Ring3": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "J_Bip_R_Little1": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "J_Bip_R_Little2": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "J_Bip_R_Little3": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
}

VRM_ADDON_MAPPING = {
    "head": HumanBoneSpecifications.HEAD,
    "spine": HumanBoneSpecifications.SPINE,
    "hips": HumanBoneSpecifications.HIPS,
    "upper_arm.R": HumanBoneSpecifications.RIGHT_UPPER_ARM,
    "lower_arm.R": HumanBoneSpecifications.RIGHT_LOWER_ARM,
    "hand.R": HumanBoneSpecifications.RIGHT_HAND,
    "upper_arm.L": HumanBoneSpecifications.LEFT_UPPER_ARM,
    "lower_arm.L": HumanBoneSpecifications.LEFT_LOWER_ARM,
    "hand.L": HumanBoneSpecifications.LEFT_HAND,
    "upper_leg.R": HumanBoneSpecifications.RIGHT_UPPER_LEG,
    "lower_leg.R": HumanBoneSpecifications.RIGHT_LOWER_LEG,
    "foot.R": HumanBoneSpecifications.RIGHT_FOOT,
    "upper_leg.L": HumanBoneSpecifications.LEFT_UPPER_LEG,
    "lower_leg.L": HumanBoneSpecifications.LEFT_LOWER_LEG,
    "foot.L": HumanBoneSpecifications.LEFT_FOOT,
    "eye.R": HumanBoneSpecifications.RIGHT_EYE,
    "eye.L": HumanBoneSpecifications.LEFT_EYE,
    "neck": HumanBoneSpecifications.NECK,
    "shoulder.L": HumanBoneSpecifications.LEFT_SHOULDER,
    "shoulder.R": HumanBoneSpecifications.RIGHT_SHOULDER,
    "upper_chest": HumanBoneSpecifications.UPPER_CHEST,
    "chest": HumanBoneSpecifications.CHEST,
    "toes.R": HumanBoneSpecifications.RIGHT_TOES,
    "toes.L": HumanBoneSpecifications.LEFT_TOES,
    "thumb_metacarpal.L": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "thumb.metacarpal.L": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
    "thumb_proximal.L": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "thumb.proximal.L": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
    "thumb_distal.L": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "thumb.distal.L": HumanBoneSpecifications.LEFT_THUMB_DISTAL,
    "index_proximal.L": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "index.proximal.L": HumanBoneSpecifications.LEFT_INDEX_PROXIMAL,
    "index_intermediate.L": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "index.intermediate.L": HumanBoneSpecifications.LEFT_INDEX_INTERMEDIATE,
    "index_distal.L": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "index.distal.L": HumanBoneSpecifications.LEFT_INDEX_DISTAL,
    "middle_proximal.L": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "middle.proximal.L": HumanBoneSpecifications.LEFT_MIDDLE_PROXIMAL,
    "middle_intermediate.L": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "middle.intermediate.L": HumanBoneSpecifications.LEFT_MIDDLE_INTERMEDIATE,
    "middle_distal.L": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "middle.distal.L": HumanBoneSpecifications.LEFT_MIDDLE_DISTAL,
    "ring_proximal.L": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "ring.proximal.L": HumanBoneSpecifications.LEFT_RING_PROXIMAL,
    "ring_intermediate.L": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "ring.intermediate.L": HumanBoneSpecifications.LEFT_RING_INTERMEDIATE,
    "ring_distal.L": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "ring.distal.L": HumanBoneSpecifications.LEFT_RING_DISTAL,
    "little_proximal.L": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "little.proximal.L": HumanBoneSpecifications.LEFT_LITTLE_PROXIMAL,
    "little_intermediate.L": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "little.intermediate.L": HumanBoneSpecifications.LEFT_LITTLE_INTERMEDIATE,
    "little_distal.L": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
    "little.distal.L": HumanBoneSpecifications.LEFT_LITTLE_DISTAL,
    "thumb_metacarpal.R": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "thumb.metacarpal.R": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
    "thumb_proximal.R": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "thumb.proximal.R": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
    "thumb_distal.R": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "thumb.distal.R": HumanBoneSpecifications.RIGHT_THUMB_DISTAL,
    "index_proximal.R": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "index.proximal.R": HumanBoneSpecifications.RIGHT_INDEX_PROXIMAL,
    "index_intermediate.R": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "index.intermediate.R": HumanBoneSpecifications.RIGHT_INDEX_INTERMEDIATE,
    "index_distal.R": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "index.distal.R": HumanBoneSpecifications.RIGHT_INDEX_DISTAL,
    "middle_proximal.R": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "middle.proximal.R": HumanBoneSpecifications.RIGHT_MIDDLE_PROXIMAL,
    "middle_intermediate.R": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "middle.intermediate.R": HumanBoneSpecifications.RIGHT_MIDDLE_INTERMEDIATE,
    "middle_distal.R": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "middle.distal.R": HumanBoneSpecifications.RIGHT_MIDDLE_DISTAL,
    "ring_proximal.R": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "ring.proximal.R": HumanBoneSpecifications.RIGHT_RING_PROXIMAL,
    "ring_intermediate.R": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "ring.intermediate.R": HumanBoneSpecifications.RIGHT_RING_INTERMEDIATE,
    "ring_distal.R": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "ring.distal.R": HumanBoneSpecifications.RIGHT_RING_DISTAL,
    "little_proximal.R": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "little.proximal.R": HumanBoneSpecifications.RIGHT_LITTLE_PROXIMAL,
    "little_intermediate.R": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "little.intermediate.R": HumanBoneSpecifications.RIGHT_LITTLE_INTERMEDIATE,
    "little_distal.R": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
    "little.distal.R": HumanBoneSpecifications.RIGHT_LITTLE_DISTAL,
}


def _mapping_configs(armature: Object) -> list[tuple[str, Mapping[str, HumanBoneSpecification]]]:
    return [
        _create_mmd_config(armature),
        _create_biped_config(armature),
        ("Mixamo", MIXAMO_MAPPING),
        ("Unreal", UNREAL_MAPPING),
        ("Ready Player Me", READY_PLAYER_ME_MAPPING),
        ("Cats Blender Plugin Fixed Model", CATS_FIXED_MODEL_MAPPING),
        ("Microsoft Rocketbox (Bip01)", _prefixed_mapping("Bip01 ", ROCKETBOX_TEMPLATE_MAPPING)),
        ("Microsoft Rocketbox (Bip02)", _prefixed_mapping("Bip02 ", ROCKETBOX_TEMPLATE_MAPPING)),
        ("Rigify Meta-Rig", RIGIFY_META_RIG_MAPPING),
        ("VRoid", VROID_MAPPING),
        ("VRoid (Symmetrised)", {_symmetrise_vroid_bone_name(key): value for key, value in VROID_MAPPING.items()}),
        ("VRM Add-on (VRM1)", VRM_ADDON_MAPPING),
        (
            "VRM Add-on (VRM0)",
            {
                **VRM_ADDON_MAPPING,
                "thumb_proximal.L": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
                "thumb.proximal.L": HumanBoneSpecifications.LEFT_THUMB_METACARPAL,
                "thumb_intermediate.L": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
                "thumb.intermediate.L": HumanBoneSpecifications.LEFT_THUMB_PROXIMAL,
                "thumb_proximal.R": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
                "thumb.proximal.R": HumanBoneSpecifications.RIGHT_THUMB_METACARPAL,
                "thumb_intermediate.R": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
                "thumb.intermediate.R": HumanBoneSpecifications.RIGHT_THUMB_PROXIMAL,
            },
        ),
    ]


def auto_detect_human_bones(armature_object: Object) -> dict[str, str]:
    if armature_object.type != "ARMATURE":
        return {}
    armature_data = armature_object.data
    if not isinstance(armature_data, Armature):
        return {}

    _best_name, best_mapping = max(
        _mapping_configs(armature_object),
        key=lambda item: _match_counts(armature_data, item[1]),
        default=("", {}),
    )
    required_count, _all_count = _match_counts(armature_data, best_mapping)
    if required_count < _required_human_bone_count():
        canonicalize_bone_name.cache_clear()
        return {}

    result = {
        specification.name: blender_bone_name
        for blender_bone_name, specification in _sorted_required_first(armature_data, best_mapping).items()
    }
    canonicalize_bone_name.cache_clear()
    return result
