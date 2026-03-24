from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanBoneSpecification:
    name: str
    title: str
    icon: str
    requirement: bool
    parent_name: str | None = None

    @property
    def label(self) -> str:
        return f"{self.title}:"

    @property
    def label_no_left_right(self) -> str:
        title = self.title
        if title.startswith("Left "):
            title = title[5:]
        elif title.startswith("Right "):
            title = title[6:]
        return f"{title}:"

    @property
    def parent(self) -> "HumanBoneSpecification | None":
        if self.parent_name is None:
            return None
        return HumanBoneSpecifications.get(self.parent_name)


def _spec(
    name: str,
    title: str,
    icon: str,
    requirement: bool,
    *,
    parent_name: str | None = None,
) -> HumanBoneSpecification:
    return HumanBoneSpecification(
        name=name,
        title=title,
        icon=icon,
        requirement=requirement,
        parent_name=parent_name,
    )


class HumanBoneSpecifications:
    HIPS = _spec("hips", "Hips", "USER", True)
    SPINE = _spec("spine", "Spine", "USER", True, parent_name="hips")
    CHEST = _spec("chest", "Chest", "USER", False, parent_name="spine")
    UPPER_CHEST = _spec("upperChest", "Upper Chest", "USER", False, parent_name="chest")
    NECK = _spec("neck", "Neck", "USER", False, parent_name="upperChest")

    HEAD = _spec("head", "Head", "USER", True, parent_name="neck")
    LEFT_EYE = _spec("leftEye", "Left Eye", "HIDE_OFF", False, parent_name="head")
    RIGHT_EYE = _spec("rightEye", "Right Eye", "HIDE_OFF", False, parent_name="head")
    JAW = _spec("jaw", "Jaw", "USER", False, parent_name="head")

    LEFT_UPPER_LEG = _spec("leftUpperLeg", "Left Upper Leg", "MOD_DYNAMICPAINT", True, parent_name="hips")
    LEFT_LOWER_LEG = _spec("leftLowerLeg", "Left Lower Leg", "MOD_DYNAMICPAINT", True, parent_name="leftUpperLeg")
    LEFT_FOOT = _spec("leftFoot", "Left Foot", "MOD_DYNAMICPAINT", True, parent_name="leftLowerLeg")
    LEFT_TOES = _spec("leftToes", "Left Toes", "MOD_DYNAMICPAINT", False, parent_name="leftFoot")
    RIGHT_UPPER_LEG = _spec("rightUpperLeg", "Right Upper Leg", "MOD_DYNAMICPAINT", True, parent_name="hips")
    RIGHT_LOWER_LEG = _spec("rightLowerLeg", "Right Lower Leg", "MOD_DYNAMICPAINT", True, parent_name="rightUpperLeg")
    RIGHT_FOOT = _spec("rightFoot", "Right Foot", "MOD_DYNAMICPAINT", True, parent_name="rightLowerLeg")
    RIGHT_TOES = _spec("rightToes", "Right Toes", "MOD_DYNAMICPAINT", False, parent_name="rightFoot")

    LEFT_SHOULDER = _spec("leftShoulder", "Left Shoulder", "VIEW_PAN", False, parent_name="upperChest")
    LEFT_UPPER_ARM = _spec("leftUpperArm", "Left Upper Arm", "VIEW_PAN", True, parent_name="leftShoulder")
    LEFT_LOWER_ARM = _spec("leftLowerArm", "Left Lower Arm", "VIEW_PAN", True, parent_name="leftUpperArm")
    LEFT_HAND = _spec("leftHand", "Left Hand", "VIEW_PAN", True, parent_name="leftLowerArm")
    RIGHT_SHOULDER = _spec("rightShoulder", "Right Shoulder", "VIEW_PAN", False, parent_name="upperChest")
    RIGHT_UPPER_ARM = _spec("rightUpperArm", "Right Upper Arm", "VIEW_PAN", True, parent_name="rightShoulder")
    RIGHT_LOWER_ARM = _spec("rightLowerArm", "Right Lower Arm", "VIEW_PAN", True, parent_name="rightUpperArm")
    RIGHT_HAND = _spec("rightHand", "Right Hand", "VIEW_PAN", True, parent_name="rightLowerArm")

    LEFT_THUMB_METACARPAL = _spec("leftThumbMetacarpal", "Left Thumb", "VIEW_PAN", False, parent_name="leftHand")
    LEFT_THUMB_PROXIMAL = _spec("leftThumbProximal", "Left Thumb Proximal", "VIEW_PAN", False, parent_name="leftThumbMetacarpal")
    LEFT_THUMB_DISTAL = _spec("leftThumbDistal", "Left Thumb Distal", "VIEW_PAN", False, parent_name="leftThumbProximal")
    LEFT_INDEX_PROXIMAL = _spec("leftIndexProximal", "Left Index Proximal", "VIEW_PAN", False, parent_name="leftHand")
    LEFT_INDEX_INTERMEDIATE = _spec("leftIndexIntermediate", "Left Index Intermediate", "VIEW_PAN", False, parent_name="leftIndexProximal")
    LEFT_INDEX_DISTAL = _spec("leftIndexDistal", "Left Index Distal", "VIEW_PAN", False, parent_name="leftIndexIntermediate")
    LEFT_MIDDLE_PROXIMAL = _spec("leftMiddleProximal", "Left Middle Proximal", "VIEW_PAN", False, parent_name="leftHand")
    LEFT_MIDDLE_INTERMEDIATE = _spec("leftMiddleIntermediate", "Left Middle Intermediate", "VIEW_PAN", False, parent_name="leftMiddleProximal")
    LEFT_MIDDLE_DISTAL = _spec("leftMiddleDistal", "Left Middle Distal", "VIEW_PAN", False, parent_name="leftMiddleIntermediate")
    LEFT_RING_PROXIMAL = _spec("leftRingProximal", "Left Ring Proximal", "VIEW_PAN", False, parent_name="leftHand")
    LEFT_RING_INTERMEDIATE = _spec("leftRingIntermediate", "Left Ring Intermediate", "VIEW_PAN", False, parent_name="leftRingProximal")
    LEFT_RING_DISTAL = _spec("leftRingDistal", "Left Ring Distal", "VIEW_PAN", False, parent_name="leftRingIntermediate")
    LEFT_LITTLE_PROXIMAL = _spec("leftLittleProximal", "Left Little Proximal", "VIEW_PAN", False, parent_name="leftHand")
    LEFT_LITTLE_INTERMEDIATE = _spec("leftLittleIntermediate", "Left Little Intermediate", "VIEW_PAN", False, parent_name="leftLittleProximal")
    LEFT_LITTLE_DISTAL = _spec("leftLittleDistal", "Left Little Distal", "VIEW_PAN", False, parent_name="leftLittleIntermediate")
    RIGHT_THUMB_METACARPAL = _spec("rightThumbMetacarpal", "Right Thumb", "VIEW_PAN", False, parent_name="rightHand")
    RIGHT_THUMB_PROXIMAL = _spec("rightThumbProximal", "Right Thumb Proximal", "VIEW_PAN", False, parent_name="rightThumbMetacarpal")
    RIGHT_THUMB_DISTAL = _spec("rightThumbDistal", "Right Thumb Distal", "VIEW_PAN", False, parent_name="rightThumbProximal")
    RIGHT_INDEX_PROXIMAL = _spec("rightIndexProximal", "Right Index Proximal", "VIEW_PAN", False, parent_name="rightHand")
    RIGHT_INDEX_INTERMEDIATE = _spec("rightIndexIntermediate", "Right Index Intermediate", "VIEW_PAN", False, parent_name="rightIndexProximal")
    RIGHT_INDEX_DISTAL = _spec("rightIndexDistal", "Right Index Distal", "VIEW_PAN", False, parent_name="rightIndexIntermediate")
    RIGHT_MIDDLE_PROXIMAL = _spec("rightMiddleProximal", "Right Middle Proximal", "VIEW_PAN", False, parent_name="rightHand")
    RIGHT_MIDDLE_INTERMEDIATE = _spec("rightMiddleIntermediate", "Right Middle Intermediate", "VIEW_PAN", False, parent_name="rightMiddleProximal")
    RIGHT_MIDDLE_DISTAL = _spec("rightMiddleDistal", "Right Middle Distal", "VIEW_PAN", False, parent_name="rightMiddleIntermediate")
    RIGHT_RING_PROXIMAL = _spec("rightRingProximal", "Right Ring Proximal", "VIEW_PAN", False, parent_name="rightHand")
    RIGHT_RING_INTERMEDIATE = _spec("rightRingIntermediate", "Right Ring Intermediate", "VIEW_PAN", False, parent_name="rightRingProximal")
    RIGHT_RING_DISTAL = _spec("rightRingDistal", "Right Ring Distal", "VIEW_PAN", False, parent_name="rightRingIntermediate")
    RIGHT_LITTLE_PROXIMAL = _spec("rightLittleProximal", "Right Little Proximal", "VIEW_PAN", False, parent_name="rightHand")
    RIGHT_LITTLE_INTERMEDIATE = _spec("rightLittleIntermediate", "Right Little Intermediate", "VIEW_PAN", False, parent_name="rightLittleProximal")
    RIGHT_LITTLE_DISTAL = _spec("rightLittleDistal", "Right Little Distal", "VIEW_PAN", False, parent_name="rightLittleIntermediate")

    ALL = [
        HIPS,
        SPINE,
        CHEST,
        UPPER_CHEST,
        NECK,
        HEAD,
        LEFT_EYE,
        RIGHT_EYE,
        JAW,
        LEFT_UPPER_LEG,
        LEFT_LOWER_LEG,
        LEFT_FOOT,
        LEFT_TOES,
        RIGHT_UPPER_LEG,
        RIGHT_LOWER_LEG,
        RIGHT_FOOT,
        RIGHT_TOES,
        LEFT_SHOULDER,
        LEFT_UPPER_ARM,
        LEFT_LOWER_ARM,
        LEFT_HAND,
        RIGHT_SHOULDER,
        RIGHT_UPPER_ARM,
        RIGHT_LOWER_ARM,
        RIGHT_HAND,
        LEFT_THUMB_METACARPAL,
        LEFT_THUMB_PROXIMAL,
        LEFT_THUMB_DISTAL,
        LEFT_INDEX_PROXIMAL,
        LEFT_INDEX_INTERMEDIATE,
        LEFT_INDEX_DISTAL,
        LEFT_MIDDLE_PROXIMAL,
        LEFT_MIDDLE_INTERMEDIATE,
        LEFT_MIDDLE_DISTAL,
        LEFT_RING_PROXIMAL,
        LEFT_RING_INTERMEDIATE,
        LEFT_RING_DISTAL,
        LEFT_LITTLE_PROXIMAL,
        LEFT_LITTLE_INTERMEDIATE,
        LEFT_LITTLE_DISTAL,
        RIGHT_THUMB_METACARPAL,
        RIGHT_THUMB_PROXIMAL,
        RIGHT_THUMB_DISTAL,
        RIGHT_INDEX_PROXIMAL,
        RIGHT_INDEX_INTERMEDIATE,
        RIGHT_INDEX_DISTAL,
        RIGHT_MIDDLE_PROXIMAL,
        RIGHT_MIDDLE_INTERMEDIATE,
        RIGHT_MIDDLE_DISTAL,
        RIGHT_RING_PROXIMAL,
        RIGHT_RING_INTERMEDIATE,
        RIGHT_RING_DISTAL,
        RIGHT_LITTLE_PROXIMAL,
        RIGHT_LITTLE_INTERMEDIATE,
        RIGHT_LITTLE_DISTAL,
    ]
    BY_NAME = {spec.name: spec for spec in ALL}

    @classmethod
    def get(cls, name: str) -> HumanBoneSpecification:
        return cls.BY_NAME[name]
