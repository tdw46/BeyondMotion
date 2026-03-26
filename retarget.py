from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
import re
from typing import Any, Iterator

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
SOMA77_PARENT = {
    "Hips": None,
    "Spine1": "Hips",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "Neck1": "Chest",
    "Neck2": "Neck1",
    "Head": "Neck2",
    "HeadEnd": "Head",
    "Jaw": "Head",
    "LeftEye": "Head",
    "RightEye": "Head",
    "LeftShoulder": "Chest",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand",
    "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2",
    "LeftHandThumbEnd": "LeftHandThumb3",
    "LeftHandIndex1": "LeftHand",
    "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2",
    "LeftHandIndex4": "LeftHandIndex3",
    "LeftHandIndexEnd": "LeftHandIndex4",
    "LeftHandMiddle1": "LeftHand",
    "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftHandMiddle4": "LeftHandMiddle3",
    "LeftHandMiddleEnd": "LeftHandMiddle4",
    "LeftHandRing1": "LeftHand",
    "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2",
    "LeftHandRing4": "LeftHandRing3",
    "LeftHandRingEnd": "LeftHandRing4",
    "LeftHandPinky1": "LeftHand",
    "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2",
    "LeftHandPinky4": "LeftHandPinky3",
    "LeftHandPinkyEnd": "LeftHandPinky4",
    "RightShoulder": "Chest",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand",
    "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2",
    "RightHandThumbEnd": "RightHandThumb3",
    "RightHandIndex1": "RightHand",
    "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2",
    "RightHandIndex4": "RightHandIndex3",
    "RightHandIndexEnd": "RightHandIndex4",
    "RightHandMiddle1": "RightHand",
    "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2",
    "RightHandMiddle4": "RightHandMiddle3",
    "RightHandMiddleEnd": "RightHandMiddle4",
    "RightHandRing1": "RightHand",
    "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2",
    "RightHandRing4": "RightHandRing3",
    "RightHandRingEnd": "RightHandRing4",
    "RightHandPinky1": "RightHand",
    "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2",
    "RightHandPinky4": "RightHandPinky3",
    "RightHandPinkyEnd": "RightHandPinky4",
    "LeftLeg": "Hips",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftShin",
    "LeftToeBase": "LeftFoot",
    "LeftToeEnd": "LeftToeBase",
    "RightLeg": "Hips",
    "RightShin": "RightLeg",
    "RightFoot": "RightShin",
    "RightToeBase": "RightFoot",
    "RightToeEnd": "RightToeBase",
}
HUMAN_BONE_PROCESS_ORDER = {spec.name: index for index, spec in enumerate(HumanBoneSpecifications.ALL)}
WEIGHT_SHIFT_SPEED_MPS = 0.30
WALKING_SPEED_MAX_MPS = 2.10
TURN_PROMPT_THRESHOLD_DEGREES = 5.0
MIN_HEURISTIC_SEGMENT_FRAMES = 3

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
    constraint_frames: list[int]
    relative_frame_indices: list[int]
    source_rotations: dict[int, dict[str, Matrix]]
    root_control_locations: dict[int, Any]
    source_root_positions: dict[int, Vector]
    first_root_control_location: Any
    generation_required: bool
    source_loop_matches: bool
    injected_turn_frames: tuple[int, ...] = ()


@dataclass
class PromptSegmentAnalysis:
    start_frame: int
    end_frame: int
    duration_frames: int
    segment_kind: str
    prompt: str
    displacement: float
    average_speed: float
    turn_degrees: float


@dataclass
class InternalPromptSegment:
    start_frame: int
    end_frame: int
    num_frames: int
    prompt: str
    segment_kind: str
    path_frames: tuple[int, ...] = ()
    turn_degrees: float = 0.0


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


def _orthonormalize_rotation(matrix: Matrix) -> Matrix:
    return matrix.to_quaternion().to_matrix()


def _rotation_distance(a: Matrix, b: Matrix) -> float:
    return float(a.to_quaternion().rotation_difference(b.to_quaternion()).angle)


def _nlerp_quaternion(a: Quaternion, b: Quaternion, factor: float) -> Quaternion:
    dot = (a.w * b.w) + (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
    bw, bx, by, bz = b.w, b.x, b.y, b.z
    if dot < 0.0:
        bw, bx, by, bz = -bw, -bx, -by, -bz
    blended = Quaternion((
        (1.0 - factor) * a.w + factor * bw,
        (1.0 - factor) * a.x + factor * bx,
        (1.0 - factor) * a.y + factor * by,
        (1.0 - factor) * a.z + factor * bz,
    ))
    blended.normalize()
    return blended


def _blend_rotation_matrices(a: Matrix, b: Matrix, factor: float) -> Matrix:
    if factor <= 0.0:
        return a.copy()
    if factor >= 1.0:
        return b.copy()
    return _nlerp_quaternion(a.to_quaternion(), b.to_quaternion(), factor).to_matrix()


def _blend_vectors(a: Vector, b: Vector, factor: float) -> Vector:
    if factor <= 0.0:
        return a.copy()
    if factor >= 1.0:
        return b.copy()
    return a + ((b - a) * factor)


def _scaled_rotation_delta(delta: Matrix, factor: float) -> Matrix:
    if factor <= 0.0:
        return Matrix.Identity(3)
    if factor >= 1.0:
        return delta.copy()
    identity = Quaternion((1.0, 0.0, 0.0, 0.0))
    return _nlerp_quaternion(identity, delta.to_quaternion(), factor).to_matrix()


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


def _rotation_basis_matrix(forward_axis: str) -> Matrix:
    if forward_axis == "NEGATIVE_Y":
        return Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
    if forward_axis == "POSITIVE_Y":
        return Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    if forward_axis == "POSITIVE_X":
        return Matrix(((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    return Matrix(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)))


def blender_rotation_to_kimodo(matrix: Matrix, forward_axis: str) -> Matrix:
    basis = _rotation_basis_matrix(forward_axis)
    return _orthonormalize_rotation(basis @ matrix @ basis.transposed())


def kimodo_rotation_to_blender(matrix: Matrix, forward_axis: str) -> Matrix:
    basis = _rotation_basis_matrix(forward_axis)
    return _orthonormalize_rotation(basis.transposed() @ matrix @ basis)


def _get_pose_bone(armature_object: Object, bone_name: str) -> PoseBone | None:
    if not bone_name:
        return None
    return armature_object.pose.bones.get(bone_name)


def _matrix_basis_rotation(pose_bone: PoseBone) -> Matrix:
    return _orthonormalize_rotation(pose_bone.matrix_basis.to_3x3())


def _bone_rest_global_rotation(pose_bone: PoseBone) -> Matrix:
    return _orthonormalize_rotation(pose_bone.bone.matrix_local.to_3x3())


def _bone_pose_global_rotation(pose_bone: PoseBone) -> Matrix:
    return _orthonormalize_rotation(pose_bone.matrix.to_3x3())


def _bone_pose_global_delta_rotation(pose_bone: PoseBone) -> Matrix:
    rest_global = _bone_rest_global_rotation(pose_bone)
    pose_global = _bone_pose_global_rotation(pose_bone)
    return _orthonormalize_rotation(pose_global @ rest_global.inverted_safe())


def _pose_bone_depth(pose_bone: PoseBone) -> int:
    depth = 0
    current = pose_bone.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth


def _mapped_pose_bones_in_application_order(
    armature_object: Object,
    assignment_map: dict[str, str],
) -> list[tuple[str, PoseBone]]:
    mapped: list[tuple[str, PoseBone]] = []
    for human_bone_name, blender_bone_name in assignment_map.items():
        pose_bone = _get_pose_bone(armature_object, blender_bone_name)
        if pose_bone is None:
            continue
        mapped.append((human_bone_name, pose_bone))
    mapped.sort(
        key=lambda item: (
            _pose_bone_depth(item[1]),
            HUMAN_BONE_PROCESS_ORDER.get(item[0], 1_000),
            item[1].name,
        )
    )
    return mapped


def _build_soma_local_rotations(
    armature_object: Object,
    assignment_map: dict[str, str],
    forward_axis: str,
) -> np.ndarray:
    soma_global_by_name: dict[str, Matrix] = {}
    for human_bone_name, blender_bone_name in assignment_map.items():
        soma_bone_name = CANONICAL_TO_SOMA.get(human_bone_name)
        if soma_bone_name is None:
            continue
        pose_bone = _get_pose_bone(armature_object, blender_bone_name)
        if pose_bone is None:
            continue
        blender_delta = _bone_pose_global_delta_rotation(pose_bone)
        soma_global_by_name[soma_bone_name] = blender_rotation_to_kimodo(blender_delta, forward_axis)

    resolved_soma_globals: dict[str, Matrix] = {}
    soma_local_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(SOMA77_BONE_ORDER), axis=0)
    for soma_bone_name in SOMA77_BONE_ORDER:
        parent_name = SOMA77_PARENT[soma_bone_name]
        global_rotation = soma_global_by_name.get(soma_bone_name)
        if global_rotation is None:
            if parent_name is None:
                global_rotation = Matrix.Identity(3)
            else:
                global_rotation = resolved_soma_globals[parent_name].copy()
        global_rotation = _orthonormalize_rotation(global_rotation)
        resolved_soma_globals[soma_bone_name] = global_rotation

        if parent_name is None:
            local_rotation = global_rotation
        else:
            local_rotation = resolved_soma_globals[parent_name].inverted_safe() @ global_rotation
        soma_local_rotations[SOMA77_INDEX[soma_bone_name]] = np.array(
            _orthonormalize_rotation(local_rotation),
            dtype=np.float32,
        )
    return soma_local_rotations


def _basis_rotation_from_target_global_rotation(
    pose_bone: PoseBone,
    target_pose_global_rotation: Matrix,
    parent_pose_global_rotation: Matrix | None,
) -> Matrix:
    rest_global_rotation = _bone_rest_global_rotation(pose_bone)
    if pose_bone.parent is None:
        return _orthonormalize_rotation(rest_global_rotation.inverted_safe() @ target_pose_global_rotation)

    parent_rest_global_rotation = _bone_rest_global_rotation(pose_bone.parent)
    if parent_pose_global_rotation is None:
        parent_pose_global_rotation = _bone_pose_global_rotation(pose_bone.parent)
    return _orthonormalize_rotation(
        rest_global_rotation.inverted_safe()
        @ parent_rest_global_rotation
        @ parent_pose_global_rotation.inverted_safe()
        @ target_pose_global_rotation
    )


def _basis_matrix_from_target_pose_matrix(
    pose_bone: PoseBone,
    target_pose_matrix: Matrix,
    parent_pose_matrix: Matrix | None,
) -> Matrix:
    bone_rest_matrix = pose_bone.bone.matrix_local.copy()
    if pose_bone.parent is None:
        return bone_rest_matrix.inverted_safe() @ target_pose_matrix

    parent_rest_matrix = pose_bone.parent.bone.matrix_local.copy()
    if parent_pose_matrix is None:
        parent_pose_matrix = pose_bone.parent.matrix.copy()
    return bone_rest_matrix.inverted_safe() @ parent_rest_matrix @ parent_pose_matrix.inverted_safe() @ target_pose_matrix


def _expand_soma_output_to_77(output: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    local_rot_mats = output["local_rot_mats"]
    global_rot_mats = output.get("global_rot_mats")
    joint_count = int(local_rot_mats.shape[1])
    if joint_count == len(SOMA77_BONE_ORDER):
        if global_rot_mats is None:
            raise ValueError("Kimodo output is missing global rotations for the SOMA 77 skeleton.")
        return local_rot_mats, global_rot_mats

    if joint_count != 30:
        raise ValueError(f"Unsupported SOMA joint count: {joint_count}. Expected 30 or 77 joints.")

    try:
        import torch
        from kimodo.skeleton import SOMASkeleton30
    except ImportError as exc:
        raise RuntimeError("Kimodo SOMA conversion helpers are unavailable in the extension runtime.") from exc

    soma30 = SOMASkeleton30()
    local_rot_tensor = torch.as_tensor(local_rot_mats, dtype=torch.float32)
    root_positions_tensor = torch.as_tensor(output["root_positions"], dtype=torch.float32)
    local_rot_77 = soma30.to_SOMASkeleton77(local_rot_tensor)
    global_rot_77, _, _ = soma30.somaskel77.fk(local_rot_77, root_positions_tensor)
    return (
        local_rot_77.detach().cpu().numpy(),
        global_rot_77.detach().cpu().numpy(),
    )


def _frames_match_pose(
    frame_a: int,
    frame_b: int,
    source_rotations: dict[int, dict[str, Matrix]],
    root_control_locations: dict[int, Vector],
    *,
    rotation_tolerance: float = 1.0e-3,
    location_tolerance: float = 1.0e-4,
) -> bool:
    rotations_a = source_rotations.get(frame_a, {})
    rotations_b = source_rotations.get(frame_b, {})
    if rotations_a.keys() != rotations_b.keys():
        return False

    for human_bone_name, rotation_a in rotations_a.items():
        if _rotation_distance(rotation_a, rotations_b[human_bone_name]) > rotation_tolerance:
            return False

    root_a = root_control_locations.get(frame_a)
    root_b = root_control_locations.get(frame_b)
    if root_a is None or root_b is None:
        return False
    return (root_a - root_b).length <= location_tolerance


def _sequence_is_static_hold(
    source_frames: list[int],
    source_rotations: dict[int, dict[str, Matrix]],
    root_control_locations: dict[int, Vector],
) -> bool:
    if len(source_frames) < 2:
        return False
    return all(
        _frames_match_pose(frame_a, frame_b, source_rotations, root_control_locations)
        for frame_a, frame_b in zip(source_frames, source_frames[1:])
    )


def _apply_hold_frame_bias(
    source_frames: list[int],
    source_rotations: dict[int, dict[str, Matrix]],
    root_control_locations: dict[int, Vector],
    bias_mode: str,
) -> list[int]:
    if len(source_frames) <= 1 or bias_mode == "NONE":
        return source_frames.copy()

    filtered_frames: list[int] = []
    run_start = 0
    while run_start < len(source_frames):
        run_end = run_start
        while (
            run_end + 1 < len(source_frames)
            and _frames_match_pose(
                source_frames[run_end],
                source_frames[run_end + 1],
                source_rotations,
                root_control_locations,
            )
        ):
            run_end += 1

        run_frames = source_frames[run_start:run_end + 1]
        if len(run_frames) == 1:
            filtered_frames.extend(run_frames)
        elif bias_mode == "FIRST":
            filtered_frames.extend(run_frames[1:])
        elif bias_mode == "LAST":
            filtered_frames.extend(run_frames[:-1])
        else:
            filtered_frames.extend(run_frames)
        run_start = run_end + 1

    return filtered_frames or source_frames.copy()


def _preferred_generated_override_frame(
    constraint_frames: list[int],
    scene_frame: int,
    frame_start: int,
    total_frame_count: int,
) -> int | None:
    generated_frames = [
        frame_start + relative_index
        for relative_index in range(total_frame_count)
        if (frame_start + relative_index) not in constraint_frames
    ]
    if not generated_frames:
        return None

    previous_generated = [frame for frame in generated_frames if frame < scene_frame]
    if previous_generated:
        return previous_generated[-1]

    next_generated = [frame for frame in generated_frames if frame > scene_frame]
    if next_generated:
        return next_generated[0]

    return min(generated_frames, key=lambda frame: abs(frame - scene_frame))


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
    if settings.root_target_mode == "MOTION_ROOT":
        return pose_bone.matrix.translation.copy()
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
    if settings.root_target_mode == "MOTION_ROOT":
        parent_pose_matrix = pose_bone.parent.matrix.copy() if pose_bone.parent is not None else None
        target_pose_matrix = pose_bone.matrix.copy()
        target_pose_matrix.translation = location
        basis_matrix = _basis_matrix_from_target_pose_matrix(
            pose_bone,
            target_pose_matrix,
            parent_pose_matrix,
        )
        _apply_location_to_pose_bone(pose_bone, basis_matrix.to_translation(), frame)
        return
    pose_bone.location = location
    pose_bone.keyframe_insert(data_path="location", frame=frame)


def _capture_local_pose_sample(
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    assignment_map: dict[str, str],
) -> tuple[dict[str, Matrix], Vector]:
    rotations: dict[str, Matrix] = {}
    for human_bone_name, blender_bone_name in assignment_map.items():
        pose_bone = _get_pose_bone(armature_object, blender_bone_name)
        if pose_bone is None:
            continue
        rotations[human_bone_name] = _matrix_basis_rotation(pose_bone).copy()
    return rotations, _capture_root_control_location(armature_object, settings).copy()


def _capture_source_keyframe_data(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_frames: list[int],
    assignment_map: dict[str, str],
    hips_bone: PoseBone,
) -> tuple[
    dict[int, dict[str, Matrix]],
    dict[int, Vector],
    dict[int, Vector],
    dict[int, float | None],
    dict[int, float | None],
]:
    scene = context.scene
    original_frame = scene.frame_current
    source_rotations: dict[int, dict[str, Matrix]] = {}
    root_control_locations: dict[int, Vector] = {}
    source_root_positions: dict[int, Vector] = {}
    hips_heading_degrees: dict[int, float | None] = {}
    root_heading_degrees: dict[int, float | None] = {}

    try:
        for frame in source_frames:
            scene.frame_set(frame)
            context.view_layer.update()

            frame_rotations: dict[str, Matrix] = {}
            for human_bone_name, blender_bone_name in assignment_map.items():
                pose_bone = _get_pose_bone(armature_object, blender_bone_name)
                if pose_bone is None:
                    continue
                frame_rotations[human_bone_name] = _matrix_basis_rotation(pose_bone).copy()

            source_rotations[frame] = frame_rotations
            source_root_positions[frame] = hips_bone.head.copy()
            root_control_locations[frame] = _capture_root_control_location(armature_object, settings)
            hips_heading_degrees[frame] = _rotation_heading_degrees(
                armature_object,
                _bone_pose_global_delta_rotation(hips_bone),
                settings.blender_forward_axis,
            )
            root_identifier = _target_root_identifier(settings)
            if root_identifier == "OBJECT":
                root_heading_degrees[frame] = _heading_degrees_from_direction(
                    armature_object.matrix_world.to_3x3() @ _forward_axis_vector(settings.blender_forward_axis)
                )
            else:
                root_pose_bone = _get_pose_bone(armature_object, root_identifier)
                root_heading_degrees[frame] = (
                    _rotation_heading_degrees(
                        armature_object,
                        _bone_pose_global_delta_rotation(root_pose_bone),
                        settings.blender_forward_axis,
                    )
                    if root_pose_bone is not None
                    else None
                )
    finally:
        scene.frame_set(original_frame)
        context.view_layer.update()

    return (
        source_rotations,
        root_control_locations,
        source_root_positions,
        hips_heading_degrees,
        root_heading_degrees,
    )


def _scene_fps(context: Context) -> float:
    render = context.scene.render
    fps = float(getattr(render, "fps", 24.0) or 24.0)
    fps_base = float(getattr(render, "fps_base", 1.0) or 1.0)
    return fps / fps_base if fps_base else fps


def _segment_prompt_text(segment_kind: str, *, large_distance: bool = False) -> str:
    if segment_kind == "HOLD":
        return "A person naturally holds the same pose with subtle breathing and balance adjustments."
    if segment_kind == "SHIFT":
        if large_distance:
            return "A person turns in place and naturally shifts weight into the next pose without stepping far away."
        return "A person stays mostly in place and naturally shifts weight into the next pose."
    if segment_kind == "RUN":
        return "A person runs naturally through the keyed poses with clear momentum and grounded footfalls."
    return "A person walks at a natural relaxed pace through the keyed poses."


def _forward_axis_vector(forward_axis: str) -> Vector:
    if forward_axis == "NEGATIVE_Y":
        return Vector((0.0, -1.0, 0.0))
    if forward_axis == "POSITIVE_Y":
        return Vector((0.0, 1.0, 0.0))
    if forward_axis == "POSITIVE_X":
        return Vector((1.0, 0.0, 0.0))
    return Vector((-1.0, 0.0, 0.0))


def _signed_heading_delta_degrees(start_heading: float, end_heading: float) -> float:
    delta = end_heading - start_heading
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def _heading_degrees_from_direction(direction: Vector) -> float | None:
    planar = Vector((direction.x, direction.y, 0.0))
    if planar.length <= 1.0e-6:
        return None
    planar.normalize()
    return degrees(atan2(planar.y, planar.x))


def _rotation_heading_degrees(
    armature_object: Object,
    rotation_matrix: Matrix,
    forward_axis: str,
) -> float | None:
    world_rotation = armature_object.matrix_world.to_3x3()
    forward_direction = world_rotation @ (rotation_matrix @ _forward_axis_vector(forward_axis))
    return _heading_degrees_from_direction(forward_direction)


def _average_turn_delta_degrees(
    hips_heading_degrees: dict[int, float | None],
    root_heading_degrees: dict[int, float | None],
    frame_a: int,
    frame_b: int,
) -> float:
    contributors: list[float] = []
    hips_start = hips_heading_degrees.get(frame_a)
    hips_end = hips_heading_degrees.get(frame_b)
    if hips_start is not None and hips_end is not None:
        contributors.append(_signed_heading_delta_degrees(hips_start, hips_end))

    root_start = root_heading_degrees.get(frame_a)
    root_end = root_heading_degrees.get(frame_b)
    if root_start is not None and root_end is not None:
        contributors.append(_signed_heading_delta_degrees(root_start, root_end))

    if not contributors:
        return 0.0
    return sum(contributors) / float(len(contributors))


def _average_heading_degrees(
    hips_heading_degrees: dict[int, float | None],
    root_heading_degrees: dict[int, float | None],
    frame: int,
) -> float | None:
    contributors = [
        value
        for value in (
            hips_heading_degrees.get(frame),
            root_heading_degrees.get(frame),
        )
        if value is not None
    ]
    if not contributors:
        return None
    return sum(contributors) / float(len(contributors))


def _direction_from_heading_degrees(heading_degrees: float) -> Vector:
    radians = np.deg2rad(heading_degrees)
    return Vector((float(np.cos(radians)), float(np.sin(radians)), 0.0))


def _target_is_in_front(
    start_position: Vector,
    end_position: Vector,
    start_heading_degrees: float | None,
) -> bool:
    if start_heading_degrees is None:
        return False
    displacement = Vector((end_position.x - start_position.x, end_position.y - start_position.y, 0.0))
    if displacement.length <= 1.0e-6:
        return False
    forward = _direction_from_heading_degrees(start_heading_degrees)
    return displacement.dot(forward) > 0.0


def _turn_phrase(turn_degrees: float, *, gradual_threshold: float = 45.0) -> str:
    magnitude = abs(turn_degrees)
    rounded = int(round(magnitude))
    if rounded <= 0:
        return ""
    direction = "left" if turn_degrees >= 0.0 else "right"
    if magnitude < gradual_threshold:
        return f"gradually turns {direction} by {rounded} degrees"
    return f"turns {direction} by {rounded} degrees"


def _prompt_for_segment(segment_kind: str, turn_degrees: float, *, large_distance: bool = False) -> str:
    turn_phrase = _turn_phrase(turn_degrees)
    if segment_kind in {"WALK", "RUN"} and turn_phrase:
        if segment_kind == "RUN":
            if abs(turn_degrees) < 45.0:
                return (
                    f"A person {turn_phrase} while running naturally through the keyed poses with clear momentum "
                    "and grounded footfalls."
                )
            return (
                f"A person {turn_phrase}, then runs naturally through the keyed poses with clear momentum "
                "and grounded footfalls."
            )
        if abs(turn_degrees) < 45.0:
            return f"A person {turn_phrase} while walking at a natural relaxed pace through the keyed poses."
        return f"A person {turn_phrase}, then walks at a natural relaxed pace through the keyed poses."

    if segment_kind in {"HOLD", "SHIFT"} and abs(turn_degrees) >= TURN_PROMPT_THRESHOLD_DEGREES:
        return f"A person {_turn_phrase(turn_degrees)} then holds the new pose."

    return _segment_prompt_text(segment_kind, large_distance=large_distance)


def _short_span_prompt(segment_kind: str, poses_match: bool) -> str:
    if poses_match or segment_kind == "HOLD":
        return "A person naturally holds the same pose with subtle breathing and balance adjustments."
    return "A person smoothly transitions into the next pose."


def analyze_prompt_segments(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_frames: list[int],
) -> list[PromptSegmentAnalysis]:
    if len(source_frames) < 2:
        return []

    settings.ensure_human_bones()
    assignment_map = settings.assignment_map()
    hips_bone_name = assignment_map.get("hips")
    hips_bone = _get_pose_bone(armature_object, hips_bone_name or "")
    if hips_bone is None:
        raise ValueError("A hips bone assignment is required.")

    source_rotations, root_control_locations, source_root_positions, hips_heading_degrees, root_heading_degrees = _capture_source_keyframe_data(
        context,
        armature_object,
        settings,
        source_frames,
        assignment_map,
        hips_bone,
    )
    fps = max(_scene_fps(context), 1.0)
    analyses: list[PromptSegmentAnalysis] = []
    for frame_a, frame_b in zip(source_frames, source_frames[1:]):
        duration_frames = max(int(frame_b - frame_a), 1)
        if duration_frames < MIN_HEURISTIC_SEGMENT_FRAMES:
            continue
        duration_seconds = max(duration_frames / fps, 1.0 / fps)
        displacement = float((source_root_positions[frame_b] - source_root_positions[frame_a]).length)
        average_speed = displacement / duration_seconds
        poses_match = _frames_match_pose(frame_a, frame_b, source_rotations, root_control_locations)
        turn_degrees = _average_turn_delta_degrees(hips_heading_degrees, root_heading_degrees, frame_a, frame_b)

        hips_a = source_rotations.get(frame_a, {}).get("hips")
        hips_b = source_rotations.get(frame_b, {}).get("hips")
        hips_turn = _rotation_distance(hips_a, hips_b) if hips_a is not None and hips_b is not None else 0.0

        if poses_match:
            segment_kind = "HOLD"
        elif displacement < 0.06 or average_speed < WEIGHT_SHIFT_SPEED_MPS:
            segment_kind = "SHIFT"
        elif average_speed <= WALKING_SPEED_MAX_MPS:
            segment_kind = "WALK"
        else:
            segment_kind = "RUN"

        prompt = _prompt_for_segment(
            segment_kind,
            turn_degrees,
            large_distance=(segment_kind == "SHIFT" and hips_turn >= 0.45),
        )
        analyses.append(
            PromptSegmentAnalysis(
                start_frame=int(frame_a),
                end_frame=int(frame_b),
                duration_frames=duration_frames,
                segment_kind=segment_kind,
                prompt=prompt,
                displacement=displacement,
                average_speed=average_speed,
                turn_degrees=turn_degrees,
            )
        )
    return analyses


def prompt_segments_from_settings(
    settings: BeyondMotionArmatureSettings,
    source_frames: list[int],
) -> list[dict[str, Any]]:
    if not settings.prompt_segments_match_frames(source_frames):
        return []
    segments: list[dict[str, Any]] = []
    for item in settings.prompt_segments:
        prompt = str(item.prompt or "").strip()
        segments.append(
            {
                "start_frame": int(item.start_frame),
                "end_frame": int(item.end_frame),
                "duration_frames": max(int(item.duration_frames), 1),
                "segment_kind": str(item.segment_kind or "SHIFT"),
                "prompt": prompt,
                "turn_degrees": float(item.turn_degrees or 0.0),
            }
        )
    return segments


def _request_num_frames_by_segment(source_frames: list[int]) -> list[int]:
    if len(source_frames) < 2:
        return []
    num_frames: list[int] = []
    for index, (frame_a, frame_b) in enumerate(zip(source_frames, source_frames[1:])):
        segment_frames = int(frame_b - frame_a)
        if index == len(source_frames) - 2:
            segment_frames += 1
        num_frames.append(max(segment_frames, 1))
    return num_frames


def _build_root2d_constraint(
    frame_start: int,
    root_positions_by_frame: dict[int, Vector],
    prompt_segments: list[InternalPromptSegment],
    forward_axis: str,
) -> dict[str, Any] | None:
    root2d_by_frame: dict[int, list[float]] = {}
    for segment in prompt_segments:
        if segment.segment_kind not in {"WALK", "RUN"}:
            continue
        anchor_frames = tuple(
            frame
            for frame in (segment.path_frames or (segment.start_frame, segment.end_frame))
            if frame in root_positions_by_frame
        )
        if len(anchor_frames) < 2:
            continue
        for start_frame, end_frame in zip(anchor_frames, anchor_frames[1:]):
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            start_root = root_positions_by_frame.get(start_frame)
            end_root = root_positions_by_frame.get(end_frame)
            if start_root is None or end_root is None or end_frame <= start_frame:
                continue
            frame_span = max(end_frame - start_frame, 1)
            for scene_frame in range(start_frame, end_frame + 1):
                factor = (scene_frame - start_frame) / float(frame_span)
                blended_root = _blend_vectors(start_root, end_root, factor)
                kimodo_position = blender_position_to_kimodo(blended_root, forward_axis)
                root2d_by_frame[scene_frame - frame_start] = [kimodo_position[0], kimodo_position[2]]

    if not root2d_by_frame:
        return None

    ordered_frames = sorted(root2d_by_frame)
    return {
        "type": "root2d",
        "frame_indices": ordered_frames,
        "smooth_root_2d": [root2d_by_frame[frame] for frame in ordered_frames],
    }


def _turn_only_prompt(turn_degrees: float) -> str:
    turn_phrase = _turn_phrase(turn_degrees, gradual_threshold=45.0)
    if not turn_phrase:
        return "A person quickly reorients into the new facing direction."
    return f"A person {turn_phrase}."


def _straight_locomotion_prompt(segment_kind: str) -> str:
    if segment_kind == "RUN":
        return "A person runs naturally through the keyed poses with clear momentum and grounded footfalls."
    return "A person walks at a natural relaxed pace through the keyed poses."


def _merged_locomotion_prompt(segment_kind: str, total_turn_degrees: float) -> str:
    base_prompt = _straight_locomotion_prompt(segment_kind)
    if abs(total_turn_degrees) < 10.0 or abs(total_turn_degrees) >= 45.0:
        return base_prompt

    turn_phrase = _turn_phrase(total_turn_degrees, gradual_threshold=45.0)
    if not turn_phrase:
        return base_prompt
    if segment_kind == "RUN":
        return f"A person {turn_phrase} while running naturally through the keyed poses with clear momentum and grounded footfalls."
    return f"A person {turn_phrase} while walking at a natural relaxed pace through the keyed poses."


def _merge_path_frames(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    merged: list[int] = []
    for frame in (*a, *b):
        if not merged or merged[-1] != int(frame):
            merged.append(int(frame))
    return tuple(merged)


def _merge_internal_locomotion_segments(
    internal_segments: list[InternalPromptSegment],
) -> list[InternalPromptSegment]:
    if not internal_segments:
        return []

    merged_segments: list[InternalPromptSegment] = []
    for segment in internal_segments:
        if (
            merged_segments
            and segment.segment_kind in {"WALK", "RUN"}
            and merged_segments[-1].segment_kind == segment.segment_kind
            and segment.start_frame <= (merged_segments[-1].end_frame + 1)
        ):
            previous = merged_segments[-1]
            combined_turn = previous.turn_degrees + segment.turn_degrees
            merged_segments[-1] = InternalPromptSegment(
                start_frame=previous.start_frame,
                end_frame=max(previous.end_frame, segment.end_frame),
                num_frames=1,
                prompt=_merged_locomotion_prompt(segment.segment_kind, combined_turn),
                segment_kind=segment.segment_kind,
                path_frames=_merge_path_frames(previous.path_frames, segment.path_frames),
                turn_degrees=combined_turn,
            )
            continue
        merged_segments.append(segment)
    return merged_segments


def _prompt_without_turn_instruction(prompt: str, segment_kind: str) -> str:
    cleaned = str(prompt or "").strip()
    if not cleaned:
        return _straight_locomotion_prompt(segment_kind)

    patterns = (
        r"^A person gradually turns (?:left|right) by \d+ degrees and (.+)$",
        r"^A person turns (?:left|right) by \d+ degrees, then (.+)$",
        r"^A person turns (?:left|right) by \d+ degrees then (.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        remainder = match.group(1).strip()
        if not remainder:
            break
        if remainder.lower().startswith("a person "):
            return remainder
        return f"A person {remainder[0].lower() + remainder[1:]}" if remainder else _straight_locomotion_prompt(segment_kind)
    return cleaned


def _build_internal_prompt_plan(
    context: Context,
    visible_prompt_segments: list[dict[str, Any]],
    source_root_positions: dict[int, Vector],
    hips_heading_degrees: dict[int, float | None],
    root_heading_degrees: dict[int, float | None],
) -> tuple[list[InternalPromptSegment], dict[int, dict[str, Any]]]:
    fps = max(_scene_fps(context), 1.0)
    internal_segments: list[InternalPromptSegment] = []
    injected_turn_frames: dict[int, dict[str, Any]] = {}

    for segment in visible_prompt_segments:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        segment_kind = str(segment["segment_kind"])
        prompt = str(segment["prompt"])
        turn_degrees = float(segment.get("turn_degrees", 0.0) or 0.0)
        duration_frames = max(end_frame - start_frame, 1)

        if (
            segment_kind in {"WALK", "RUN"}
            and duration_frames >= MIN_HEURISTIC_SEGMENT_FRAMES
            and abs(turn_degrees) >= 45.0
            and end_frame > start_frame + 2
        ):
            seconds = 0.5 if segment_kind == "RUN" else 0.8
            offset_frames = max(1, int(round(seconds * fps)))
            end_is_in_front = _target_is_in_front(
                source_root_positions[start_frame],
                source_root_positions[end_frame],
                _average_heading_degrees(hips_heading_degrees, root_heading_degrees, start_frame),
            )
            if end_is_in_front:
                locomotion_end_frame = max(start_frame + 1, end_frame - offset_frames)
                turn_start_frame = locomotion_end_frame + 1
                if locomotion_end_frame < end_frame and turn_start_frame < end_frame:
                    injected_info = {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "segment_kind": segment_kind,
                        "inject_mode": "END",
                    }
                    injected_turn_frames[locomotion_end_frame] = dict(injected_info)
                    injected_turn_frames[turn_start_frame] = dict(injected_info)
                    source_root_positions[locomotion_end_frame] = source_root_positions[end_frame].copy()
                    source_root_positions[turn_start_frame] = source_root_positions[end_frame].copy()
                    internal_segments.append(
                        InternalPromptSegment(
                            start_frame=start_frame,
                            end_frame=locomotion_end_frame,
                            num_frames=1,
                            prompt=_prompt_without_turn_instruction(prompt, segment_kind),
                            segment_kind=segment_kind,
                            path_frames=(start_frame, locomotion_end_frame),
                        )
                    )
                    internal_segments.append(
                        InternalPromptSegment(
                            start_frame=turn_start_frame,
                            end_frame=end_frame,
                            num_frames=1,
                            prompt=_turn_only_prompt(turn_degrees),
                            segment_kind="TURN",
                            turn_degrees=turn_degrees,
                        )
                    )
                    continue

            turn_end_frame = min(start_frame + offset_frames, end_frame - 2)
            turn_start_frame = turn_end_frame + 1
            if turn_end_frame > start_frame and turn_start_frame < end_frame:
                injected_info = {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "segment_kind": segment_kind,
                    "inject_mode": "START",
                }
                injected_turn_frames[turn_end_frame] = dict(injected_info)
                injected_turn_frames[turn_start_frame] = dict(injected_info)
                source_root_positions[turn_end_frame] = source_root_positions[start_frame].copy()
                source_root_positions[turn_start_frame] = source_root_positions[start_frame].copy()
                internal_segments.append(
                    InternalPromptSegment(
                        start_frame=start_frame,
                        end_frame=turn_end_frame,
                        num_frames=1,
                        prompt=_turn_only_prompt(turn_degrees),
                        segment_kind="TURN",
                        turn_degrees=turn_degrees,
                    )
                )
                internal_segments.append(
                    InternalPromptSegment(
                        start_frame=turn_start_frame,
                        end_frame=end_frame,
                        num_frames=1,
                        prompt=_prompt_without_turn_instruction(prompt, segment_kind),
                        segment_kind=segment_kind,
                        path_frames=(turn_start_frame, end_frame),
                    )
                )
                continue

        internal_segments.append(
            InternalPromptSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=1,
                prompt=prompt,
                segment_kind=segment_kind,
                path_frames=(start_frame, end_frame) if segment_kind in {"WALK", "RUN"} else (),
                turn_degrees=turn_degrees,
            )
        )

    internal_segments = _merge_internal_locomotion_segments(internal_segments)

    for index, segment in enumerate(internal_segments):
        if index < len(internal_segments) - 1:
            next_segment = internal_segments[index + 1]
            segment.num_frames = max(next_segment.start_frame - segment.start_frame, 1)
        else:
            segment.num_frames = max((segment.end_frame - segment.start_frame) + 1, 1)
    return internal_segments, injected_turn_frames


def build_constraint_request(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_frames: list[int],
) -> tuple[dict[str, Any] | None, CapturedSourceData]:
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

    visible_prompt_segments = prompt_segments_from_settings(settings, source_frames)
    expected_segment_count = sum(
        1 for frame_a, frame_b in zip(source_frames, source_frames[1:]) if int(frame_b) - int(frame_a) >= MIN_HEURISTIC_SEGMENT_FRAMES
    )
    if len(visible_prompt_segments) != expected_segment_count:
        raise ValueError("Process Keyframes first so Beyond Motion can build prompts for the keyframe spans with at least 3 frames between them.")
    if any(not str(segment.get("prompt", "")).strip() for segment in visible_prompt_segments):
        raise ValueError("Fill in every visible segment prompt before generating.")

    scene = context.scene
    original_frame = scene.frame_current

    source_rotations, root_control_locations, source_root_positions, _hips_heading_degrees, _root_heading_degrees = _capture_source_keyframe_data(
        context,
        armature_object,
        settings,
        source_frames,
        assignment_map,
        hips_bone,
    )

    generation_required = not _sequence_is_static_hold(source_frames, source_rotations, root_control_locations)
    constraint_frames = (
        _apply_hold_frame_bias(
            source_frames,
            source_rotations,
            root_control_locations,
            str(getattr(settings, "hold_frame_bias", "NONE") or "NONE"),
        )
        if generation_required
        else source_frames.copy()
    )
    relative_frame_indices = [frame - source_frames[0] for frame in constraint_frames]
    source_frame_local_joints_rot: dict[int, list[list[float]]] = {}
    source_frame_root_positions: dict[int, list[float]] = {}
    source_frame_root2d: dict[int, list[float]] = {}
    try:
        for frame in source_frames:
            scene.frame_set(frame)
            context.view_layer.update()
            soma_rotations = _build_soma_local_rotations(
                armature_object,
                assignment_map,
                settings.blender_forward_axis,
            )
            source_frame_local_joints_rot[frame] = [
                axis_angle_vector_from_matrix(matrix_from_numpy(rotation_matrix)) for rotation_matrix in soma_rotations
            ]

            kimodo_position = blender_position_to_kimodo(hips_bone.head.copy(), settings.blender_forward_axis)
            source_frame_root_positions[frame] = kimodo_position
            source_frame_root2d[frame] = [kimodo_position[0], kimodo_position[2]]
    finally:
        scene.frame_set(original_frame)
        context.view_layer.update()

    request = None
    if generation_required:
        if not visible_prompt_segments:
            total_duration_frames = max(source_frames[-1] - source_frames[0], 1)
            total_duration_seconds = max(total_duration_frames / max(_scene_fps(context), 1.0), 1.0 / max(_scene_fps(context), 1.0))
            total_displacement = float((source_root_positions[source_frames[-1]] - source_root_positions[source_frames[0]]).length)
            total_speed = total_displacement / total_duration_seconds
            total_turn = _average_turn_delta_degrees(_hips_heading_degrees, _root_heading_degrees, source_frames[0], source_frames[-1])
            if total_displacement < 0.06 or total_speed < WEIGHT_SHIFT_SPEED_MPS:
                segment_kind = "SHIFT"
            elif total_speed <= WALKING_SPEED_MAX_MPS:
                segment_kind = "WALK"
            else:
                segment_kind = "RUN"
            visible_prompt_segments = [{
                "start_frame": int(source_frames[0]),
                "end_frame": int(source_frames[-1]),
                "duration_frames": int(total_duration_frames),
                "segment_kind": segment_kind,
                "prompt": _prompt_for_segment(segment_kind, total_turn),
                "turn_degrees": float(total_turn),
            }]
        internal_prompt_segments, injected_turn_frames = _build_internal_prompt_plan(
            context,
            visible_prompt_segments,
            source_root_positions,
            _hips_heading_degrees,
            _root_heading_degrees,
        )
        internal_constraint_frames = sorted(set(constraint_frames) | set(injected_turn_frames.keys()))
        internal_relative_frame_indices = [frame - source_frames[0] for frame in internal_constraint_frames]
        local_joints_rot: list[list[list[float]]] = []
        root_positions: list[list[float]] = []
        smooth_root_2d: list[list[float]] = []
        hips_index = SOMA77_INDEX["Hips"]
        for frame in internal_constraint_frames:
            if frame in injected_turn_frames:
                start_frame = int(injected_turn_frames[frame]["start_frame"])
                end_frame = int(injected_turn_frames[frame]["end_frame"])
                inject_mode = str(injected_turn_frames[frame].get("inject_mode", "START"))
                if inject_mode == "END":
                    injected_rotations = [rotation.copy() for rotation in source_frame_local_joints_rot[end_frame]]
                    injected_rotations[hips_index] = source_frame_local_joints_rot[start_frame][hips_index].copy()
                    root_positions.append(source_frame_root_positions[end_frame])
                    smooth_root_2d.append(source_frame_root2d[end_frame])
                else:
                    injected_rotations = [rotation.copy() for rotation in source_frame_local_joints_rot[start_frame]]
                    injected_rotations[hips_index] = source_frame_local_joints_rot[end_frame][hips_index].copy()
                    root_positions.append(source_frame_root_positions[start_frame])
                    smooth_root_2d.append(source_frame_root2d[start_frame])
                local_joints_rot.append(injected_rotations)
            else:
                local_joints_rot.append(source_frame_local_joints_rot[frame])
                root_positions.append(source_frame_root_positions[frame])
                smooth_root_2d.append(source_frame_root2d[frame])

        constraints = [
            {
                "type": "fullbody",
                "frame_indices": internal_relative_frame_indices,
                "local_joints_rot": local_joints_rot,
                "root_positions": root_positions,
                "smooth_root_2d": smooth_root_2d,
            }
        ]
        if bool(getattr(settings, "use_locomotion_root_path", True)):
            root2d_constraint = _build_root2d_constraint(
                source_frames[0],
                source_root_positions,
                internal_prompt_segments,
                settings.blender_forward_axis,
            )
            if root2d_constraint is not None:
                constraints.append(root2d_constraint)

        prompts = [segment.prompt for segment in internal_prompt_segments]
        num_frames = [max(int(segment.num_frames), 1) for segment in internal_prompt_segments]
        request = {
            "prompt": prompts if len(prompts) > 1 else prompts[0],
            "model_name": settings.model_name,
            "num_frames": num_frames if len(num_frames) > 1 else num_frames[0],
            "diffusion_steps": int(settings.diffusion_steps),
            "constraints": constraints,
            "cfg_type": settings.cfg_type,
            "cfg_text_weight": float(settings.cfg_text_weight),
            "cfg_constraint_weight": float(settings.cfg_constraint_weight),
            "seed": None if settings.seed < 0 else int(settings.seed),
            "post_processing": bool(settings.apply_postprocess),
            "multi_prompt": len(prompts) > 1,
            "num_transition_frames": 4,
        }
    captured = CapturedSourceData(
        source_frames=source_frames,
        constraint_frames=constraint_frames,
        relative_frame_indices=relative_frame_indices,
        source_rotations=source_rotations,
        root_control_locations=root_control_locations,
        source_root_positions=source_root_positions,
        first_root_control_location=root_control_locations[source_frames[0]].copy(),
        generation_required=generation_required,
        source_loop_matches=_frames_match_pose(
            source_frames[0],
            source_frames[-1],
            source_rotations,
            root_control_locations,
        ),
        injected_turn_frames=tuple(sorted(injected_turn_frames.keys())) if generation_required else (),
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


def _apply_location_to_pose_bone(pose_bone: PoseBone, location: Vector, frame: int) -> None:
    pose_bone.location = location
    pose_bone.keyframe_insert(data_path="location", frame=frame)


def _delete_rotation_keyframes(pose_bone: PoseBone, frame: int) -> None:
    pose_bone.keyframe_delete(data_path="rotation_quaternion", frame=frame)
    pose_bone.keyframe_delete(data_path="rotation_axis_angle", frame=frame)
    pose_bone.keyframe_delete(data_path="rotation_euler", frame=frame)


def _delete_generated_frame_keys(
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    assignment_map: dict[str, str],
    frame: int,
) -> None:
    for blender_bone_name in assignment_map.values():
        pose_bone = _get_pose_bone(armature_object, blender_bone_name)
        if pose_bone is None:
            continue
        _delete_rotation_keyframes(pose_bone, frame)

    if settings.root_target_mode == "OBJECT":
        armature_object.keyframe_delete(data_path="location", frame=frame)
        return

    root_bone_name = (
        assignment_map.get("hips", "")
        if settings.root_target_mode == "HIPS"
        else settings.motion_root_bone
    )
    root_pose_bone = _get_pose_bone(armature_object, root_bone_name or "")
    if root_pose_bone is not None:
        root_pose_bone.keyframe_delete(data_path="location", frame=frame)


def ensure_action(armature_object: Object) -> None:
    armature_object.animation_data_create()
    if armature_object.animation_data and armature_object.animation_data.action is None:
        armature_object.animation_data.action = bpy.data.actions.new(name=f"{armature_object.name}_BeyondMotion")


def apply_static_source_motion(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_data: CapturedSourceData,
) -> None:
    settings.ensure_human_bones()
    assignment_map = settings.assignment_map()
    if not assignment_map:
        raise ValueError("No humanoid bones are assigned.")

    ensure_action(armature_object)

    hold_frame = source_data.source_frames[0]
    rotations = source_data.source_rotations[hold_frame]
    root_location = source_data.root_control_locations[hold_frame]
    original_frame = context.scene.frame_current
    try:
        for scene_frame in range(source_data.source_frames[0], source_data.source_frames[-1] + 1):
            context.scene.frame_set(scene_frame)
            context.view_layer.update()
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


def _generated_motion_sample(
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    assignment_map: dict[str, str],
    mapped_pose_bones: list[tuple[str, PoseBone]],
    global_rot_mats: np.ndarray,
    root_positions: np.ndarray,
    source_data: CapturedSourceData,
    relative_index: int,
) -> tuple[dict[str, Matrix], Vector, Vector, dict[str, Matrix]]:
    rotations: dict[str, Matrix] = {}
    target_pose_global_rotations: dict[str, Matrix] = {}
    for human_bone_name, blender_bone_name in assignment_map.items():
        soma_bone_name = CANONICAL_TO_SOMA.get(human_bone_name)
        if not soma_bone_name:
            continue
        pose_bone = _get_pose_bone(armature_object, blender_bone_name)
        if pose_bone is None:
            continue
        soma_global_rotation = matrix_from_numpy(global_rot_mats[relative_index, SOMA77_INDEX[soma_bone_name]])
        blender_delta_global = kimodo_rotation_to_blender(soma_global_rotation, settings.blender_forward_axis)
        target_pose_global_rotations[blender_bone_name] = _orthonormalize_rotation(
            blender_delta_global @ _bone_rest_global_rotation(pose_bone)
        )

    for human_bone_name, pose_bone in mapped_pose_bones:
        target_pose_global_rotation = target_pose_global_rotations.get(pose_bone.name)
        if target_pose_global_rotation is None:
            continue
        parent_target_pose_rotation = None
        if pose_bone.parent is not None:
            parent_target_pose_rotation = target_pose_global_rotations.get(pose_bone.parent.name)
        rotations[human_bone_name] = _basis_rotation_from_target_global_rotation(
            pose_bone,
            target_pose_global_rotation,
            parent_target_pose_rotation,
        )

    generated_root = kimodo_position_to_blender(root_positions[relative_index], settings.blender_forward_axis)
    root_location = source_data.first_root_control_location + (generated_root - source_data.source_root_positions[source_data.source_frames[0]])
    return rotations, root_location, generated_root, target_pose_global_rotations


def _apply_sampled_frame(
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    assignment_map: dict[str, str],
    scene_frame: int,
    rotations: dict[str, Matrix],
    root_location: Vector,
    generated_root: Vector,
    target_pose_global_rotations: dict[str, Matrix],
) -> None:
    hips_location_applied = False
    if settings.root_target_mode == "HIPS":
        hips_bone_name = assignment_map.get("hips")
        hips_pose_bone = _get_pose_bone(armature_object, hips_bone_name or "")
        hips_target_pose_rotation = (
            target_pose_global_rotations.get(hips_pose_bone.name) if hips_pose_bone is not None else None
        )
        if hips_pose_bone is not None and hips_target_pose_rotation is not None:
            parent_target_pose_matrix = None
            if hips_pose_bone.parent is not None:
                parent_target_pose_rotation = target_pose_global_rotations.get(hips_pose_bone.parent.name)
                if parent_target_pose_rotation is not None:
                    parent_target_pose_matrix = parent_target_pose_rotation.to_4x4()
                    parent_target_pose_matrix.translation = hips_pose_bone.parent.matrix.translation.copy()
            target_pose_matrix = hips_target_pose_rotation.to_4x4()
            target_pose_matrix.translation = generated_root
            hips_basis_matrix = _basis_matrix_from_target_pose_matrix(
                hips_pose_bone,
                target_pose_matrix,
                parent_target_pose_matrix,
            )
            _apply_location_to_pose_bone(hips_pose_bone, hips_basis_matrix.to_translation(), scene_frame)
            hips_location_applied = True

    for human_bone_name, blender_bone_name in assignment_map.items():
        pose_bone = _get_pose_bone(armature_object, blender_bone_name)
        rotation_matrix = rotations.get(human_bone_name)
        if pose_bone is None or rotation_matrix is None:
            continue
        _apply_rotation_to_pose_bone(pose_bone, rotation_matrix, scene_frame)

    if not hips_location_applied:
        _set_root_control_location(armature_object, settings, root_location, scene_frame)


def _override_smoothing_factor(scene_frame: int, overridden_keyframe: int, window: int = 2) -> float:
    frame_delta = abs(scene_frame - overridden_keyframe)
    if frame_delta == 0 or frame_delta > window:
        return 0.0
    return (window + 1 - frame_delta) / float(window + 1)


def _keypose_match_factor(scene_frame: int, source_frame: int, window: int) -> float:
    frame_delta = abs(scene_frame - source_frame)
    if frame_delta > window:
        return 0.0
    return (window + 1 - frame_delta) / float(window + 1)


def _keypose_match_quarter_window(span: int) -> int:
    return max(1, int(np.ceil(max(span, 1) * 0.25)))


def _keypose_match_source_influences(
    scene_frame: int,
    source_frames: list[int],
    break_frames: tuple[int, ...] = (),
) -> list[tuple[int, float]]:
    if not source_frames:
        return []
    ordered_frames = sorted(int(frame) for frame in source_frames)
    ordered_breaks = sorted(int(frame) for frame in break_frames)
    if scene_frame in ordered_frames:
        return [(int(scene_frame), 1.0)]
    if scene_frame <= ordered_frames[0]:
        return [(ordered_frames[0], 1.0)]
    if scene_frame >= ordered_frames[-1]:
        return [(ordered_frames[-1], 1.0)]

    for previous_frame, next_frame in zip(ordered_frames, ordered_frames[1:]):
        if previous_frame < scene_frame < next_frame:
            segment_breaks = [
                frame
                for frame in ordered_breaks
                if previous_frame < frame < next_frame
            ]
            span = max(next_frame - previous_frame, 1)
            quarter_window = _keypose_match_quarter_window(span)
            start_blend_end = min(previous_frame + quarter_window, next_frame - 1)
            end_blend_start = max(next_frame - quarter_window, previous_frame + 1)

            if segment_breaks:
                first_break = segment_breaks[0]
                last_break = segment_breaks[-1]
                start_blend_end = min(start_blend_end, first_break - 1)
                end_blend_start = max(end_blend_start, last_break + 1)
                if first_break <= scene_frame <= last_break:
                    return []

            if previous_frame < scene_frame <= start_blend_end:
                denominator = max(start_blend_end - previous_frame, 1)
                factor = 1.0 - ((scene_frame - previous_frame) / float(denominator))
                return [(previous_frame, max(0.0, factor))]

            if end_blend_start <= scene_frame < next_frame:
                denominator = max(next_frame - end_blend_start, 1)
                factor = (scene_frame - end_blend_start) / float(denominator)
                return [(next_frame, max(0.0, factor))]

            return []
    return []


def _keypose_match_affected_frames(
    source_frames: list[int],
    break_frames: tuple[int, ...] = (),
) -> list[int]:
    if not source_frames:
        return []
    frame_start = int(min(source_frames))
    frame_end = int(max(source_frames))
    affected_frames: list[int] = []
    for scene_frame in range(frame_start, frame_end + 1):
        if _keypose_match_source_influences(scene_frame, source_frames, break_frames):
            affected_frames.append(scene_frame)
    return affected_frames


def _keypose_match_step_count(source_data: CapturedSourceData, window: int) -> int:
    if window <= 0 or not source_data.constraint_frames:
        return 0
    affected_frames = _keypose_match_affected_frames(
        source_data.constraint_frames,
        source_data.injected_turn_frames,
    )
    return (len(affected_frames) * 2) + len(source_data.constraint_frames)


def _iter_keypose_match_pass(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_data: CapturedSourceData,
    assignment_map: dict[str, str],
    window: int,
) -> Iterator[dict[str, Any]]:
    if window <= 0 or not source_data.constraint_frames:
        return

    affected_frames = _keypose_match_affected_frames(
        source_data.constraint_frames,
        source_data.injected_turn_frames,
    )
    if not affected_frames:
        return
    total_steps = _keypose_match_step_count(source_data, window)
    if total_steps <= 0:
        return

    # Animation Layers / multikey-style pass:
    # capture the generated result first, compute additive deltas from the original keyed poses,
    # then feather those deltas across neighboring frames before baking them back into one action.
    base_rotations_by_frame: dict[int, dict[str, Matrix]] = {}
    base_roots_by_frame: dict[int, Vector] = {}
    rotation_deltas_by_source_frame: dict[int, dict[str, Matrix]] = {}
    root_deltas_by_source_frame: dict[int, Vector] = {}
    original_frame = context.scene.frame_current
    completed_steps = 0

    try:
        for scene_frame in affected_frames:
            context.scene.frame_set(scene_frame)
            context.view_layer.update()
            rotations, root_location = _capture_local_pose_sample(armature_object, settings, assignment_map)
            base_rotations_by_frame[scene_frame] = rotations
            base_roots_by_frame[scene_frame] = root_location
            completed_steps += 1
            yield {
                "progress": completed_steps / float(total_steps),
                "status_text": "Applying keypose matching...",
                "detail_text": f"Capturing base motion at frame {scene_frame}.",
            }

        for source_frame in source_data.constraint_frames:
            base_rotations = base_rotations_by_frame.get(source_frame)
            base_root = base_roots_by_frame.get(source_frame)
            if base_rotations is None or base_root is None:
                continue

            source_rotations = source_data.source_rotations.get(source_frame, {})
            delta_rotations: dict[str, Matrix] = {}
            for human_bone_name, base_rotation in base_rotations.items():
                source_rotation = source_rotations.get(human_bone_name)
                if source_rotation is None:
                    continue
                delta_rotations[human_bone_name] = _orthonormalize_rotation(
                    base_rotation.inverted_safe() @ source_rotation
                )
            rotation_deltas_by_source_frame[source_frame] = delta_rotations
            root_deltas_by_source_frame[source_frame] = source_data.root_control_locations[source_frame] - base_root
            completed_steps += 1
            yield {
                "progress": completed_steps / float(total_steps),
                "status_text": "Applying keypose matching...",
                "detail_text": f"Preparing additive correction for frame {source_frame}.",
            }

        for scene_frame in affected_frames:
            base_rotations = base_rotations_by_frame.get(scene_frame, {})
            corrected_rotations = {name: matrix.copy() for name, matrix in base_rotations.items()}
            corrected_root = base_roots_by_frame.get(scene_frame, Vector((0.0, 0.0, 0.0))).copy()

            active_source_frames = _keypose_match_source_influences(
                scene_frame,
                source_data.constraint_frames,
                source_data.injected_turn_frames,
            )

            if not active_source_frames:
                completed_steps += 1
                yield {
                    "progress": completed_steps / float(total_steps),
                    "status_text": "Applying keypose matching...",
                    "detail_text": f"Preserving injected turn motion at frame {scene_frame}.",
                }
                continue

            for source_frame, factor in active_source_frames:
                for human_bone_name, delta_rotation in rotation_deltas_by_source_frame.get(source_frame, {}).items():
                    current_rotation = corrected_rotations.get(human_bone_name)
                    if current_rotation is None:
                        continue
                    corrected_rotations[human_bone_name] = _orthonormalize_rotation(
                        current_rotation @ _scaled_rotation_delta(delta_rotation, factor)
                    )
                corrected_root += root_deltas_by_source_frame.get(source_frame, Vector((0.0, 0.0, 0.0))) * factor

            context.scene.frame_set(scene_frame)
            context.view_layer.update()
            for human_bone_name, blender_bone_name in assignment_map.items():
                pose_bone = _get_pose_bone(armature_object, blender_bone_name)
                rotation_matrix = corrected_rotations.get(human_bone_name)
                if pose_bone is None or rotation_matrix is None:
                    continue
                _apply_rotation_to_pose_bone(pose_bone, rotation_matrix, scene_frame)
            _set_root_control_location(armature_object, settings, corrected_root, scene_frame)
            completed_steps += 1
            yield {
                "progress": completed_steps / float(total_steps),
                "status_text": "Applying keypose matching...",
                "detail_text": f"Blending corrected keypose influence at frame {scene_frame}.",
            }
    finally:
        context.scene.frame_set(original_frame)
        context.view_layer.update()


def _apply_keypose_match_pass(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_data: CapturedSourceData,
    assignment_map: dict[str, str],
    window: int,
) -> None:
    for _ in _iter_keypose_match_pass(
        context,
        armature_object,
        settings,
        source_data,
        assignment_map,
        window,
    ):
        pass


def iter_apply_generated_motion(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_data: CapturedSourceData,
    output: dict[str, np.ndarray],
) -> Iterator[dict[str, Any]]:
    settings.ensure_human_bones()
    assignment_map = settings.assignment_map()
    if not assignment_map:
        raise ValueError("No humanoid bones are assigned.")

    ensure_action(armature_object)

    local_rot_mats, global_rot_mats = _expand_soma_output_to_77(output)
    root_positions = output["root_positions"]
    frame_start = source_data.source_frames[0]
    source_frame_set = set(source_data.constraint_frames)
    mapped_pose_bones = _mapped_pose_bones_in_application_order(armature_object, assignment_map)
    override_source_frames_to_delete: set[int] = set()
    overridden_constraint_frames: dict[int, int] = {}
    sample_cache: dict[int, tuple[dict[str, Matrix], Vector, Vector, dict[str, Matrix]]] = {}
    keypose_window = int(getattr(settings, "keypose_match_frames", 0) or 0)
    keypose_steps = _keypose_match_step_count(source_data, keypose_window)
    use_adjacent_override_pass = keypose_window <= 0

    def cached_generated_motion_sample(relative_index: int) -> tuple[dict[str, Matrix], Vector, Vector, dict[str, Matrix]]:
        cached = sample_cache.get(relative_index)
        if cached is not None:
            rotations, root_location, generated_root, target_pose_global_rotations = cached
            return (
                {key: value.copy() for key, value in rotations.items()},
                root_location.copy(),
                generated_root.copy(),
                {key: value.copy() for key, value in target_pose_global_rotations.items()},
            )

        rotations, root_location, generated_root, target_pose_global_rotations = _generated_motion_sample(
            armature_object,
            settings,
            assignment_map,
            mapped_pose_bones,
            global_rot_mats,
            root_positions,
            source_data,
            relative_index,
        )
        sample_cache[relative_index] = (
            {key: value.copy() for key, value in rotations.items()},
            root_location.copy(),
            generated_root.copy(),
            {key: value.copy() for key, value in target_pose_global_rotations.items()},
        )
        return (
            rotations,
            root_location,
            generated_root,
            target_pose_global_rotations,
        )

    if use_adjacent_override_pass:
        for constraint_frame in source_data.constraint_frames:
            override_frame = _preferred_generated_override_frame(
                source_data.constraint_frames,
                constraint_frame,
                frame_start,
                local_rot_mats.shape[0],
            )
            if override_frame is None:
                continue
            overridden_constraint_frames[constraint_frame] = override_frame - frame_start
            if abs(override_frame - constraint_frame) == 1:
                override_source_frames_to_delete.add(override_frame)

    total_steps = local_rot_mats.shape[0]
    if source_data.source_loop_matches and use_adjacent_override_pass:
        total_steps += 1
    if use_adjacent_override_pass:
        total_steps += len(override_source_frames_to_delete)
    total_steps += keypose_steps
    total_steps = max(total_steps, 1)
    completed_steps = 0
    original_frame = context.scene.frame_current
    try:
        for relative_index in range(local_rot_mats.shape[0]):
            scene_frame = frame_start + relative_index
            context.scene.frame_set(scene_frame)
            context.view_layer.update()

            if scene_frame in source_frame_set:
                override_relative_index = overridden_constraint_frames.get(scene_frame)
                if override_relative_index is None:
                    rotations = source_data.source_rotations[scene_frame]
                    root_location = source_data.root_control_locations[scene_frame]
                    generated_root = source_data.source_root_positions[scene_frame]
                    target_pose_global_rotations = {}
                else:
                    rotations, root_location, generated_root, target_pose_global_rotations = cached_generated_motion_sample(
                        override_relative_index,
                    )
                hips_location_applied = False
            else:
                rotations, root_location, generated_root, target_pose_global_rotations = cached_generated_motion_sample(
                    relative_index,
                )
                if use_adjacent_override_pass:
                    for overridden_keyframe, overridden_relative_index in overridden_constraint_frames.items():
                        smoothing_factor = _override_smoothing_factor(scene_frame, overridden_keyframe, window=2)
                        if smoothing_factor <= 0.0:
                            continue
                        (
                            overridden_rotations,
                            overridden_root_location,
                            overridden_generated_root,
                            overridden_target_pose_global_rotations,
                        ) = cached_generated_motion_sample(overridden_relative_index)
                        for human_bone_name, rotation_matrix in list(rotations.items()):
                            overridden_rotation = overridden_rotations.get(human_bone_name)
                            if overridden_rotation is None:
                                continue
                            rotations[human_bone_name] = _blend_rotation_matrices(
                                rotation_matrix,
                                overridden_rotation,
                                smoothing_factor,
                            )
                        for bone_name, target_matrix in list(target_pose_global_rotations.items()):
                            overridden_target = overridden_target_pose_global_rotations.get(bone_name)
                            if overridden_target is None:
                                continue
                            target_pose_global_rotations[bone_name] = _blend_rotation_matrices(
                                target_matrix,
                                overridden_target,
                                smoothing_factor,
                            )
                        root_location = _blend_vectors(root_location, overridden_root_location, smoothing_factor)
                        generated_root = _blend_vectors(generated_root, overridden_generated_root, smoothing_factor)
            _apply_sampled_frame(
                armature_object,
                settings,
                assignment_map,
                scene_frame,
                rotations,
                root_location,
                generated_root,
                target_pose_global_rotations,
            )
            completed_steps += 1
            yield {
                "progress": completed_steps / float(total_steps),
                "status_text": "Applying generated motion in Blender...",
                "detail_text": f"Writing generated keys at frame {scene_frame}.",
            }

        if source_data.source_loop_matches and use_adjacent_override_pass:
            first_frame = source_data.source_frames[0]
            last_frame = source_data.source_frames[-1]
            first_override_relative_index = overridden_constraint_frames.get(first_frame)
            if first_override_relative_index is not None and last_frame > first_frame:
                (
                    loop_rotations,
                    loop_root_location,
                    loop_generated_root,
                    loop_target_pose_global_rotations,
                ) = cached_generated_motion_sample(first_override_relative_index)
                context.scene.frame_set(last_frame)
                context.view_layer.update()
                _apply_sampled_frame(
                    armature_object,
                    settings,
                    assignment_map,
                    last_frame,
                    loop_rotations,
                    loop_root_location,
                    loop_generated_root,
                    loop_target_pose_global_rotations,
                )
                for gap_frame in (last_frame - 1, last_frame - 2):
                    if gap_frame <= first_frame:
                        continue
                    _delete_generated_frame_keys(armature_object, settings, assignment_map, gap_frame)
                completed_steps += 1
                yield {
                    "progress": completed_steps / float(total_steps),
                    "status_text": "Applying generated motion in Blender...",
                    "detail_text": f"Preserving loop continuity at frame {last_frame}.",
                }

        if use_adjacent_override_pass:
            for frame in sorted(override_source_frames_to_delete):
                _delete_generated_frame_keys(armature_object, settings, assignment_map, frame)
                completed_steps += 1
                yield {
                    "progress": completed_steps / float(total_steps),
                    "status_text": "Cleaning adjacent override keys...",
                    "detail_text": f"Removing temporary override keys at frame {frame}.",
                }

        if keypose_window > 0:
            keypose_base_step = completed_steps
            for step in _iter_keypose_match_pass(
                context,
                armature_object,
                settings,
                source_data,
                assignment_map,
                keypose_window,
            ):
                keypose_progress = float(step.get("progress", 0.0) or 0.0)
                progress = (keypose_base_step + (keypose_progress * keypose_steps)) / float(total_steps)
                yield {
                    "progress": progress,
                    "status_text": str(step.get("status_text", "Applying keypose matching...")),
                    "detail_text": str(step.get("detail_text", "")),
                }
            completed_steps = keypose_base_step + keypose_steps
    finally:
        context.scene.frame_set(original_frame)
        context.view_layer.update()


def apply_generated_motion(
    context: Context,
    armature_object: Object,
    settings: BeyondMotionArmatureSettings,
    source_data: CapturedSourceData,
    output: dict[str, np.ndarray],
) -> None:
    for _ in iter_apply_generated_motion(
        context,
        armature_object,
        settings,
        source_data,
        output,
    ):
        pass
