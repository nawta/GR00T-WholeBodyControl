"""Unit tests for mirror_smpl_motion (--mirror-smpl) in pico_manager_thread_server.py.

The script lives in gear_sonic/scripts/ which is not a package, so it is loaded
by file path. Run explicitly (root pyproject testpaths only covers decoupled_wbc):

    pytest gear_sonic/tests/test_mirror_smpl.py -v
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pico_manager_thread_server.py"
_spec = importlib.util.spec_from_file_location("pico_manager_thread_server", _SCRIPT)
pico = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(pico)
except ImportError as e:  # e.g. zmq/torch only ship with the teleop/sim extras
    pytest.skip(f"pico_manager_thread_server deps unavailable: {e}", allow_module_level=True)

# Reflection across the XZ plane (X-forward, Y-left, Z-up)
S = np.diag([1.0, -1.0, 1.0])

_MIRROR_Y = np.array([1.0, -1.0, 1.0])

# Independently derived expected permutations (NOT read from the module under
# test) from the compute_human_joints ordering: SMPL-X body joints 0-21 plus
# thumb tips (gear_sonic/trl/utils/torch_transform.py, SMPLH_JOINT_NAMES in
# gear_sonic/trl/utils/smplx/body_model/utils.py).
# fmt: off
_EXPECTED_PERM_24 = [
    0,       # pelvis
    2, 1,    # left_hip <-> right_hip
    3,       # spine1
    5, 4,    # left_knee <-> right_knee
    6,       # spine2
    8, 7,    # left_ankle <-> right_ankle
    9,       # spine3
    11, 10,  # left_foot <-> right_foot
    12,      # neck
    14, 13,  # left_collar <-> right_collar
    15,      # head
    17, 16,  # left_shoulder <-> right_shoulder
    19, 18,  # left_elbow <-> right_elbow
    21, 20,  # left_wrist <-> right_wrist
    23, 22,  # left_thumb3 <-> right_thumb3
]
# fmt: on


def test_perm_literals():
    """Anchor the permutation contents against independently derived literals.

    The functional tests below compare against pico.SMPL_MIRROR_PERM_* and would
    be tautological w.r.t. the perm contents without this anchor.
    """
    np.testing.assert_array_equal(pico.SMPL_MIRROR_PERM_24, _EXPECTED_PERM_24)
    # smpl_pose covers SMPL joints 1..21 (array index i = SMPL joint i+1), so the
    # 21-joint perm is structurally determined by the 24-joint one.
    np.testing.assert_array_equal(pico.SMPL_MIRROR_PERM_21, np.array(_EXPECTED_PERM_24[1:22]) - 1)


def test_perms_are_involutions_and_cover_all_indices():
    perm24 = pico.SMPL_MIRROR_PERM_24
    perm21 = pico.SMPL_MIRROR_PERM_21
    np.testing.assert_array_equal(np.sort(perm24), np.arange(24))
    np.testing.assert_array_equal(np.sort(perm21), np.arange(21))
    np.testing.assert_array_equal(perm24[perm24], np.arange(24))
    np.testing.assert_array_equal(perm21[perm21], np.arange(21))
    # Midline joints keep their index: pelvis(0), spine1(3), spine2(6), spine3(9),
    # neck(12), head(15); in the 21-joint body pose those are indices 2,5,8,11,14.
    for idx in (0, 3, 6, 9, 12, 15):
        assert perm24[idx] == idx
    for idx in (2, 5, 8, 11, 14):
        assert perm21[idx] == idx


def test_joints_y_negation_and_lr_swap():
    joints = np.zeros((24, 3), dtype=np.float32)
    for i in range(24):
        joints[i] = [i, 100.0 + i, 200.0 + i]
    pose = np.zeros((21, 3), dtype=np.float32)

    mirrored, _, _ = pico.mirror_smpl_motion(joints, pose)

    assert mirrored.dtype == np.float32
    # All 9 left/right pairs from the compute_human_joints ordering, anchored by
    # explicit indices (independent of SMPL_MIRROR_PERM_24): hips, knees, ankles,
    # feet, collars, shoulders, elbows, wrists, thumb tips.
    lr_pairs = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23)]
    for left, right in lr_pairs:
        np.testing.assert_array_equal(mirrored[left], joints[right] * _MIRROR_Y)
        np.testing.assert_array_equal(mirrored[right], joints[left] * _MIRROR_Y)
    # Midline joints: Y-negated in place, no swap
    for mid in (0, 3, 6, 9, 12, 15):
        np.testing.assert_array_equal(mirrored[mid], joints[mid] * _MIRROR_Y)


def test_axis_angle_mirror_matches_matrix_conjugation():
    """The closed-form (ax,ay,az) -> (-ax,ay,-az) must equal the literal S @ R @ S."""
    rng = np.random.RandomState(1)
    pose = rng.uniform(-np.pi * 0.9, np.pi * 0.9, size=(21, 3)).astype(np.float32)
    joints = rng.uniform(-1.0, 1.0, size=(24, 3)).astype(np.float32)

    _, mirrored_pose, _ = pico.mirror_smpl_motion(joints, pose)

    assert mirrored_pose.dtype == np.float32
    for k in range(21):
        src_aa = pose[pico.SMPL_MIRROR_PERM_21[k]]
        expected = S @ sRot.from_rotvec(src_aa).as_matrix() @ S
        actual = sRot.from_rotvec(mirrored_pose[k]).as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_quat_mirror_matches_matrix_conjugation():
    """(w,x,y,z) -> (w,-x,y,-z) must equal S @ R @ S, up to global quaternion sign."""
    rng = np.random.RandomState(2)
    joints = np.zeros((24, 3), dtype=np.float32)
    pose = np.zeros((21, 3), dtype=np.float32)
    for _ in range(50):
        quat_wxyz = rng.normal(size=4)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        quat_wxyz = quat_wxyz.astype(np.float32)

        _, _, mirrored = pico.mirror_smpl_motion(joints, pose, quat_wxyz)

        rot = sRot.from_quat(quat_wxyz, scalar_first=True)
        expected_xyzw = sRot.from_matrix(S @ rot.as_matrix() @ S).as_quat()
        actual_xyzw = np.array([mirrored[1], mirrored[2], mirrored[3], mirrored[0]])
        if np.dot(actual_xyzw, expected_xyzw) < 0.0:
            expected_xyzw = -expected_xyzw
        np.testing.assert_allclose(actual_xyzw, expected_xyzw, atol=1e-6)


def test_mirror_is_involution():
    rng = np.random.RandomState(3)
    joints = rng.uniform(-1.0, 1.0, size=(24, 3)).astype(np.float32)
    pose = rng.uniform(-np.pi, np.pi, size=(21, 3)).astype(np.float32)
    quat = rng.normal(size=4).astype(np.float32)
    quat /= np.linalg.norm(quat)

    j1, p1, q1 = pico.mirror_smpl_motion(joints, pose, quat)
    j2, p2, q2 = pico.mirror_smpl_motion(j1, p1, q1)

    np.testing.assert_array_equal(j2, joints)
    np.testing.assert_array_equal(p2, pose)
    np.testing.assert_array_equal(q2, quat)
    # Inputs must not be modified in place
    assert not np.array_equal(j1, joints)


def test_quat_is_optional():
    joints = np.zeros((24, 3), dtype=np.float32)
    pose = np.zeros((21, 3), dtype=np.float32)
    j, p, q = pico.mirror_smpl_motion(joints, pose)
    assert q is None
    assert j.shape == (24, 3) and p.shape == (21, 3)


def test_shape_asserts():
    with pytest.raises(AssertionError):
        pico.mirror_smpl_motion(np.zeros((23, 3)), np.zeros((21, 3)))
    with pytest.raises(AssertionError):
        pico.mirror_smpl_motion(np.zeros((24, 3)), np.zeros((24, 3)))
    with pytest.raises(AssertionError):
        pico.mirror_smpl_motion(np.zeros((24, 3)), np.zeros((21, 3)), np.zeros(3))
