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


def _collapse_constraint_frames(
    source_frames: list[int],
    source_rotations: dict[int, dict[str, Matrix]],
    root_control_locations: dict[int, Vector],
) -> list[int]:
    if len(source_frames) <= 2:
        return source_frames.copy()

    constraint_frames = [source_frames[0]]
    for index in range(1, len(source_frames) - 1):
        frame = source_frames[index]
        next_frame = source_frames[index + 1]
        if _frames_match_pose(frame, next_frame, source_rotations, root_control_locations):
            continue
        constraint_frames.append(frame)
    constraint_frames.append(source_frames[-1])
    return constraint_frames


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

    scene = context.scene
    original_frame = scene.frame_current

    source_rotations: dict[int, dict[str, Matrix]] = {}
    root_control_locations: dict[int, Vector] = {}
    source_root_positions: dict[int, Vector] = {}

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
    finally:
        scene.frame_set(original_frame)
        context.view_layer.update()

    generation_required = not _sequence_is_static_hold(source_frames, source_rotations, root_control_locations)
    constraint_frames = (
        _collapse_constraint_frames(source_frames, source_rotations, root_control_locations)
        if generation_required
        else source_frames.copy()
    )
    relative_frame_indices = [frame - source_frames[0] for frame in constraint_frames]
    local_joints_rot: list[list[list[list[float]]]] = []
    root_positions: list[list[float]] = []
    smooth_root_2d: list[list[float]] = []
    try:
        for frame in constraint_frames:
            scene.frame_set(frame)
            context.view_layer.update()
            soma_rotations = _build_soma_local_rotations(
                armature_object,
                assignment_map,
                settings.blender_forward_axis,
            )
            local_joints_rot.append(
                [axis_angle_vector_from_matrix(matrix_from_numpy(rotation_matrix)) for rotation_matrix in soma_rotations]
            )

            kimodo_position = blender_position_to_kimodo(hips_bone.head.copy(), settings.blender_forward_axis)
            root_positions.append(kimodo_position)
            smooth_root_2d.append([kimodo_position[0], kimodo_position[2]])
    finally:
        scene.frame_set(original_frame)
        context.view_layer.update()

    request = None
    if generation_required:
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

    local_rot_mats, global_rot_mats = _expand_soma_output_to_77(output)
    root_positions = output["root_positions"]
    frame_start = source_data.source_frames[0]
    source_frame_set = set(source_data.constraint_frames)
    mapped_pose_bones = _mapped_pose_bones_in_application_order(armature_object, assignment_map)
    override_source_frames_to_delete: set[int] = set()
    overridden_constraint_frames: dict[int, int] = {}
    sample_cache: dict[int, tuple[dict[str, Matrix], Vector, Vector, dict[str, Matrix]]] = {}

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

        if source_data.source_loop_matches:
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

        for frame in sorted(override_source_frames_to_delete):
            _delete_generated_frame_keys(armature_object, settings, assignment_map, frame)
    finally:
        context.scene.frame_set(original_frame)
        context.view_layer.update()
