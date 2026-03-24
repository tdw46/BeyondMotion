from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bpy
import numpy as np
from bpy.types import Context, Object, PoseBone
from mathutils import Matrix, Quaternion, Vector

from .human_bones import HumanBoneSpecifications
from .properties import BeyondMotionArmatureSettings

SOMA77_BONE_ORDER = [
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumbEnd",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumbEnd",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinkyEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
]
SOMA77_INDEX = {name: index for index, name in enumerate(SOMA77_BONE_ORDER)}

CANONICAL_TO_SOMA = {
    "hips": "Hips",
    "spine": "Spine1",
    "chest": "Spine2",
    "upperChest": "Chest",
    "neck": "Neck1",
    "head": "Head",
    "jaw": "Jaw",
    "leftEye": "LeftEye",
    "rightEye": "RightEye",
    "leftShoulder": "LeftShoulder",
    "leftUpperArm": "LeftArm",
    "leftLowerArm": "LeftForeArm",
    "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder",
    "rightUpperArm": "RightArm",
    "rightLowerArm": "RightForeArm",
    "rightHand": "RightHand",
    "leftUpperLeg": "LeftLeg",
    "leftLowerLeg": "LeftShin",
    "leftFoot": "LeftFoot",
    "leftToes": "LeftToeBase",
    "rightUpperLeg": "RightLeg",
    "rightLowerLeg": "RightShin",
    "rightFoot": "RightFoot",
    "rightToes": "RightToeBase",
    "leftThumbMetacarpal": "LeftHandThumb1",
    "leftThumbProximal": "LeftHandThumb2",
    "leftThumbDistal": "LeftHandThumb3",
    "leftIndexProximal": "LeftHandIndex1",
    "leftIndexIntermediate": "LeftHandIndex2",
    "leftIndexDistal": "LeftHandIndex3",
    "leftMiddleProximal": "LeftHandMiddle1",
    "leftMiddleIntermediate": "LeftHandMiddle2",
    "leftMiddleDistal": "LeftHandMiddle3",
    "leftRingProximal": "LeftHandRing1",
    "leftRingIntermediate": "LeftHandRing2",
    "leftRingDistal": "LeftHandRing3",
    "leftLittleProximal": "LeftHandPinky1",
    "leftLittleIntermediate": "LeftHandPinky2",
    "leftLittleDistal": "LeftHandPinky3",
    "rightThumbMetacarpal": "RightHandThumb1",
    "rightThumbProximal": "RightHandThumb2",
    "rightThumbDistal": "RightHandThumb3",
    "rightIndexProximal": "RightHandIndex1",
    "rightIndexIntermediate": "RightHandIndex2",
    "rightIndexDistal": "RightHandIndex3",
    "rightMiddleProximal": "RightHandMiddle1",
    "rightMiddleIntermediate": "RightHandMiddle2",
    "rightMiddleDistal": "RightHandMiddle3",
    "rightRingProximal": "RightHandRing1",
    "rightRingIntermediate": "RightHandRing2",
    "rightRingDistal": "RightHandRing3",
    "rightLittleProximal": "RightHandPinky1",
    "rightLittleIntermediate": "RightHandPinky2",
    "rightLittleDistal": "RightHandPinky3",
}


@dataclass
class CapturedSourceData:
    source_frames: list[int]
    relative_frame_indices: list[int]
    source_rotations: dict[int, dict[str, Matrix]]
    root_control_locations: dict[int, Any]
    first_root_control_location: Any


def axis_angle_vector_from_matrix(matrix: Matrix) -> list[float]:
    quaternion = matrix.to_quaternion()
    axis, angle = quaternion.to_axis_angle()
    return [axis.x * angle, axis.y * angle, axis.z * angle]


def matrix_from_numpy(matrix_values: np.ndarray) -> Matrix:
    return Matrix((
        (float(matrix_values[0][0]), float(matrix_values[0][1]), float(matrix_values[0][2])),
        (float(matrix_values[1][0]), float(matrix_values[1][1]), float(matrix_values[1][2])),
        (float(matrix_values[2][0]), float(matrix_values[2][1]), float(matrix_values[2][2])),
    ))


def blender_position_to_kimodo(vector: Vector, forward_axis: str) -> list[float]:
    if forward_axis == "NEGATIVE_Y":
        return [vector.x, vector.z, -vector.y]
    if forward_axis == "POSITIVE_Y":
        return [vector.x, vector.z, vector.y]
    if forward_axis == "POSITIVE_X":
        return [-vector.y, vector.z, vector.x]
    return [vector.y, vector.z, -vector.x]


def kimodo_position_to_blender(values: np.ndarray, forward_axis: str) -> Vector:
    x, y, z = (float(values[0]), float(values[1]), float(values[2]))
    if forward_axis == "NEGATIVE_Y":
        return Vector((x, -z, y))
    if forward_axis == "POSITIVE_Y":
        return Vector((x, z, y))
    if forward_axis == "POSITIVE_X":
        return Vector((z, -x, y))
    return Vector((-z, x, y))


def _get_pose_bone(armature_object: Object, bone_name: str) -> PoseBone | None:
    if not bone_name:
        return None
    return armature_object.pose.bones.get(bone_name)


def _matrix_basis_rotation(pose_bone: PoseBone) -> Matrix:
    return pose_bone.matrix_basis.to_3x3().normalized()


def _target_root_identifier(settings: BeyondMotionArmatureSettings) -> str:
    if settings.root_target_mode == "OBJECT":
        return "OBJECT"
    if settings.root_target_mode == "MOTION_ROOT" and settings.motion_root_bone:
        return settings.motion_root_bone
    return settings.assignment_for("hips").bone_name


def _capture_root_control_location(
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
) -> Vector:
    identifier = _target_root_identifier(settings)
    if identifier == "OBJECT":
        return armature_object.location.copy()
    pose_bone = _get_pose_bone(armature_object, identifier)
    if pose_bone is None:
        return Vector((0.0, 0.0, 0.0))
    return pose_bone.location.copy()


def _set_root_control_location(
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    location: Vector,
    frame: int,
) -> None:
    identifier = _target_root_identifier(settings)
    if identifier == "OBJECT":
        armature_object.location = location
        armature_object.keyframe_insert(data_path="location", frame=frame)
        return
    pose_bone = _get_pose_bone(armature_object, identifier)
    if pose_bone is None:
        return
    pose_bone.location = location
    pose_bone.keyframe_insert(data_path="location", frame=frame)


def build_constraint_request(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_frames: list[int],
) -> tuple[dict[str, Any], CapturedSourceData]:
    if len(source_frames) < 2:
        raise ValueError("At least two source frames are required.")

    settings.ensure_human_bones()
    assignment_map = settings.assignment_map()
    missing = settings.required_bones_missing()
    if missing:
        raise ValueError("Missing required humanoid bones: " + ", ".join(missing))

    hips_bone_name = assignment_map.get("hips")
    hips_bone = _get_pose_bone(armature_object, hips_bone_name or "")
    if hips_bone is None:
        raise ValueError("A hips bone assignment is required.")

    scene = context.scene
    original_frame = scene.frame_current

    relative_frame_indices = [frame - source_frames[0] for frame in source_frames]
    local_joints_rot: list[list[list[list[float]]]] = []
    root_positions: list[list[float]] = []
    smooth_root_2d: list[list[float]] = []
    source_rotations: dict[int, dict[str, Matrix]] = {}
    root_control_locations: dict[int, Vector] = {}

    try:
        for frame in source_frames:
            scene.frame_set(frame)
            context.view_layer.update()

            soma_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(SOMA77_BONE_ORDER), axis=0)
            frame_rotations: dict[str, Matrix] = {}
            for human_bone_name, blender_bone_name in assignment_map.items():
                pose_bone = _get_pose_bone(armature_object, blender_bone_name)
                if pose_bone is None:
                    continue
                delta_matrix = _matrix_basis_rotation(pose_bone)
                frame_rotations[human_bone_name] = delta_matrix.copy()

                soma_bone_name = CANONICAL_TO_SOMA.get(human_bone_name)
                if not soma_bone_name:
                    continue
                soma_rotations[SOMA77_INDEX[soma_bone_name]] = np.array(delta_matrix, dtype=np.float32)

            source_rotations[frame] = frame_rotations

            local_joints_rot.append(
                [axis_angle_vector_from_matrix(matrix_from_numpy(rotation_matrix)) for rotation_matrix in soma_rotations]
            )

            hips_position = hips_bone.head.copy()
            kimodo_position = blender_position_to_kimodo(hips_position, settings.blender_forward_axis)
            root_positions.append(kimodo_position)
            smooth_root_2d.append([kimodo_position[0], kimodo_position[2]])

            root_control_locations[frame] = _capture_root_control_location(armature_object, settings)
    finally:
        scene.frame_set(original_frame)
        context.view_layer.update()

    constraints = [
        {
            "type": "fullbody",
            "frame_indices": relative_frame_indices,
            "local_joints_rot": local_joints_rot,
            "root_positions": root_positions,
            "smooth_root_2d": smooth_root_2d,
        }
    ]
    request = {
        "prompt": settings.prompt.strip(),
        "model_name": settings.model_name,
        "num_frames": int(source_frames[-1] - source_frames[0] + 1),
        "diffusion_steps": int(settings.diffusion_steps),
        "constraints": constraints,
        "cfg_type": settings.cfg_type,
        "cfg_text_weight": float(settings.cfg_text_weight),
        "cfg_constraint_weight": float(settings.cfg_constraint_weight),
        "seed": None if settings.seed < 0 else int(settings.seed),
        "post_processing": bool(settings.apply_postprocess),
    }
    captured = CapturedSourceData(
        source_frames=source_frames,
        relative_frame_indices=relative_frame_indices,
        source_rotations=source_rotations,
        root_control_locations=root_control_locations,
        first_root_control_location=root_control_locations[source_frames[0]].copy(),
    )
    return request, captured


def _apply_rotation_to_pose_bone(pose_bone: PoseBone, local_rotation: Matrix, frame: int) -> None:
    quaternion = local_rotation.to_quaternion()
    if pose_bone.rotation_mode == "QUATERNION":
        pose_bone.rotation_quaternion = quaternion
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        return
    if pose_bone.rotation_mode == "AXIS_ANGLE":
        axis, angle = quaternion.to_axis_angle()
        pose_bone.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        pose_bone.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
        return
    pose_bone.rotation_euler = quaternion.to_euler(pose_bone.rotation_mode)
    pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)


def ensure_action(armature_object: Object) -> None:
    armature_object.animation_data_create()
    if armature_object.animation_data and armature_object.animation_data.action is None:
        armature_object.animation_data.action = bpy.data.actions.new(name=f"{armature_object.name}_BeyondMotion")


def apply_generated_motion(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_data: CapturedSourceData,
    output: dict[str, np.ndarray],
) -> None:
    settings.ensure_human_bones()
    assignment_map = settings.assignment_map()
    if not assignment_map:
        raise ValueError("No humanoid bones are assigned.")

    ensure_action(armature_object)

    local_rot_mats = output["local_rot_mats"]
    root_positions = output["root_positions"]
    frame_start = source_data.source_frames[0]
    source_frame_set = set(source_data.source_frames)

    first_generated_root = kimodo_position_to_blender(root_positions[0], settings.blender_forward_axis)
    original_frame = context.scene.frame_current
    try:
        for relative_index in range(local_rot_mats.shape[0]):
            scene_frame = frame_start + relative_index
            context.scene.frame_set(scene_frame)
            context.view_layer.update()

            if scene_frame in source_frame_set:
                rotations = source_data.source_rotations[scene_frame]
                root_location = source_data.root_control_locations[scene_frame]
            else:
                rotations = {}
                for human_bone_name, blender_bone_name in assignment_map.items():
                    soma_bone_name = CANONICAL_TO_SOMA.get(human_bone_name)
                    if not soma_bone_name:
                        continue
                    soma_matrix = matrix_from_numpy(local_rot_mats[relative_index, SOMA77_INDEX[soma_bone_name]])
                    rotations[human_bone_name] = soma_matrix

                generated_root = kimodo_position_to_blender(root_positions[relative_index], settings.blender_forward_axis)
                root_location = source_data.first_root_control_location + (generated_root - first_generated_root)

            for human_bone_name, blender_bone_name in assignment_map.items():
                pose_bone = _get_pose_bone(armature_object, blender_bone_name)
                rotation_matrix = rotations.get(human_bone_name)
                if pose_bone is None or rotation_matrix is None:
                    continue
                _apply_rotation_to_pose_bone(pose_bone, rotation_matrix, scene_frame)

            _set_root_control_location(armature_object, settings, root_location, scene_frame)
    finally:
        context.scene.frame_set(original_frame)
        context.view_layer.update()
