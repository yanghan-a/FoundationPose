"""
立方体颜色-轴校正模块 (Cube Color Corrector)

FoundationPose 对立方体做 6DoF 姿态估计时，由于 24 重旋转对称性，register 会随机落入
24 个等价解之一。本模块通过枚举 24 个对称旋转候选，利用相机图像像素颜色匹配选出正确朝向。

用法:
    from cube_color_corrector import CubeColorCorrector

    corrector = CubeColorCorrector(half_size=0.0275, debug=True)

    # 在 register 后调用 (仅需一次，track 阶段不需要):
    tf_centered = est.get_tf_to_centered_mesh().data.cpu().numpy()
    pose, pose_centered = corrector.correct(pose, image_rgb, K, tf_centered)
    est.pose_last = torch.as_tensor(pose_centered, device='cuda', dtype=torch.float).reshape(1, 4, 4)

颜色-轴映射 (与 mesh 贴图一致):
    +X = Red,    -X = Orange
    +Y = Blue,   -Y = Green
    +Z = White,  -Z = Yellow

参数:
    half_size     方块半边长 (米), 55mm 方块 = 0.0275
    patch_radius  采样 patch 半径 (像素), 默认 3 → 7x7 patch
    debug         是否打印调试信息
"""

import numpy as np
import cv2
import itertools


def generate_cube_rotations():
    """生成立方体 24 个旋转对称矩阵 (signed permutation matrices, det=+1).

    Returns:
        list of 24 个 3x3 numpy 旋转矩阵
    """
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = signs[i]
            if np.linalg.det(R) > 0.5:  # det = +1 (avoid float issues)
                rotations.append(R)
    assert len(rotations) == 24, f"Expected 24 rotations, got {len(rotations)}"
    return rotations


def sample_face_color(image, u, v, patch_radius):
    """在图像投影点 (u, v) 周围采样小 patch, 取中位数颜色.

    Args:
        image: HxWx3 RGB uint8 图像
        u, v: 投影点坐标 (float)
        patch_radius: patch 半径 (像素)

    Returns:
        (3,) RGB 中位数颜色 (float)
    """
    H, W = image.shape[:2]
    ui, vi = int(round(u)), int(round(v))

    y0 = max(0, vi - patch_radius)
    y1 = min(H, vi + patch_radius + 1)
    x0 = max(0, ui - patch_radius)
    x1 = min(W, ui + patch_radius + 1)

    if y1 <= y0 or x1 <= x0:
        return np.array([128.0, 128.0, 128.0])

    patch = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    return np.median(patch, axis=0)


def color_match_score(sampled_rgb, expected_rgb):
    """HSV 色相距离打分, 白色特殊处理.

    Args:
        sampled_rgb: (3,) 采样到的 RGB 颜色 (0-255 float)
        expected_rgb: (3,) 期望的 RGB 颜色 (0-255 uint8-like)

    Returns:
        float 匹配分数 [0, 1]
    """
    # Convert to HSV (OpenCV expects uint8 BGR)
    s_bgr = np.uint8([[sampled_rgb[::-1]]])
    e_bgr = np.uint8([[expected_rgb[::-1]]])
    s_hsv = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    e_hsv = cv2.cvtColor(e_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)

    s_h, s_s, s_v = s_hsv  # H: 0-180, S: 0-255, V: 0-255
    e_h, e_s, e_v = e_hsv

    # White detection: expected has very low S and high V
    is_white_expected = (e_s < 30 and e_v > 200)

    if is_white_expected:
        # White face: sampled should have low S and high V
        # S < 60 (out of 255) and V > 150
        if s_s < 60 and s_v > 150:
            # Better score for lower saturation and higher value
            return 1.0 - s_s / 255.0
        else:
            return 0.0
    else:
        # Chromatic face: hue Gaussian distance
        # OpenCV H is 0-180 (half-degree), convert to full degrees for sigma
        hue_diff = abs(s_h - e_h)
        hue_diff = min(hue_diff, 180 - hue_diff)  # circular distance in OpenCV units
        hue_diff_deg = hue_diff * 2.0  # convert to full degrees

        sigma = 15.0
        score = np.exp(-(hue_diff_deg ** 2) / (2 * sigma ** 2))

        # Penalize low saturation samples (gray/white shouldn't match chromatic)
        if s_s < 40:
            score *= 0.2

        return score


class CubeColorCorrector:
    """立方体颜色-轴校正器.

    在 FoundationPose register 后, 枚举 24 个立方体对称旋转候选,
    通过图像像素颜色匹配选出与物理方块颜色朝向一致的 pose.

    Attributes:
        half_size: 方块半边长 (米)
        patch_radius: 采样 patch 半径 (像素)
        debug: 是否打印调试信息
    """

    # 面颜色定义: (法线方向, 期望 RGB 颜色)
    # 与 mesh 贴图一致: Red=+X, Blue=+Y, White=+Z
    FACE_COLORS = [
        (np.array([1, 0, 0], dtype=np.float64),  np.array([255, 0, 0])),      # +X = Red
        (np.array([-1, 0, 0], dtype=np.float64), np.array([255, 140, 0])),     # -X = Orange
        (np.array([0, 1, 0], dtype=np.float64),  np.array([0, 0, 255])),       # +Y = Blue
        (np.array([0, -1, 0], dtype=np.float64), np.array([0, 255, 0])),       # -Y = Green
        (np.array([0, 0, 1], dtype=np.float64),  np.array([255, 255, 255])),   # +Z = White
        (np.array([0, 0, -1], dtype=np.float64), np.array([255, 255, 0])),     # -Z = Yellow
    ]

    def __init__(self, half_size=0.0275, patch_radius=3, debug=False):
        self.half_size = half_size
        self.patch_radius = patch_radius
        self.debug = debug

        # 预计算 24 个 4x4 旋转矩阵
        rots_3x3 = generate_cube_rotations()
        self.rotations_3x3 = rots_3x3
        self.rotations_4x4 = []
        for R in rots_3x3:
            M = np.eye(4)
            M[:3, :3] = R
            self.rotations_4x4.append(M)

        # 预计算面信息: (法线, 面中心, 期望RGB)
        self.faces = []
        for normal, color in self.FACE_COLORS:
            center = self.half_size * normal
            self.faces.append((normal, center, color))

        if self.debug:
            print(f"[CubeColorCorrector] half_size={half_size}m, patch_radius={patch_radius}px, "
                  f"24 rotations, 6 faces")

    def correct(self, pose, image_rgb, K, tf_to_centered_mesh):
        """校正 register 后的 pose, 选出颜色匹配最佳的对称旋转.

        Args:
            pose: 4x4 numpy, register 返回的 original-mesh pose (相机坐标系)
            image_rgb: HxWx3 RGB uint8 图像
            K: 3x3 相机内参矩阵
            tf_to_centered_mesh: 4x4 numpy, est.get_tf_to_centered_mesh() 结果

        Returns:
            corrected_pose: 4x4 numpy, 校正后的 original-mesh pose
            corrected_centered: 4x4 numpy, 校正后的 centered-mesh pose (用于更新 est.pose_last)
        """
        H, W = image_rgb.shape[:2]

        # centered_pose: pose of the centered mesh
        tf_inv = np.linalg.inv(tf_to_centered_mesh)
        centered_pose = pose @ tf_inv

        R_cam = centered_pose[:3, :3]
        t_cam = centered_pose[:3, 3]

        best_score = -np.inf
        best_idx = 0

        scores_debug = []

        for idx, R_sym in enumerate(self.rotations_3x3):
            score = 0.0
            total_weight = 0.0

            for face_normal, face_center, expected_rgb in self.faces:
                # Transform face normal and center to camera frame
                rotated_normal = R_sym @ face_normal
                rotated_center = R_sym @ face_center

                n_cam = R_cam @ rotated_normal
                c_cam = R_cam @ rotated_center + t_cam

                # Visibility: face must point toward camera (z < -0.1)
                if n_cam[2] >= -0.1:
                    continue

                # Must be in front of camera
                if c_cam[2] <= 0.01:
                    continue

                # Project face center to image
                proj = K @ c_cam
                u = proj[0] / proj[2]
                v = proj[1] / proj[2]

                # Bounds check
                if u < 0 or u >= W or v < 0 or v >= H:
                    continue

                # Sample color
                sampled_rgb = sample_face_color(image_rgb, u, v, self.patch_radius)

                # Score
                s = color_match_score(sampled_rgb, expected_rgb)
                weight = -n_cam[2]  # more facing camera = higher weight
                score += s * weight
                total_weight += weight

            if total_weight > 0:
                score /= total_weight

            scores_debug.append(score)

            if score > best_score:
                best_score = score
                best_idx = idx

        # Apply best rotation
        R_sym_4x4 = self.rotations_4x4[best_idx]
        corrected_centered = centered_pose @ R_sym_4x4
        corrected_pose = corrected_centered @ tf_to_centered_mesh

        if self.debug:
            # Sort scores for debugging
            sorted_scores = sorted(enumerate(scores_debug), key=lambda x: -x[1])
            print(f"[CubeColorCorrector] Best rotation idx={best_idx}, "
                  f"score={best_score:.3f}")
            print(f"  Top-3 scores: "
                  f"#{sorted_scores[0][0]}={sorted_scores[0][1]:.3f}, "
                  f"#{sorted_scores[1][0]}={sorted_scores[1][1]:.3f}, "
                  f"#{sorted_scores[2][0]}={sorted_scores[2][1]:.3f}")
            is_identity = np.allclose(self.rotations_3x3[best_idx], np.eye(3))
            print(f"  Identity rotation: {is_identity}")

        return corrected_pose, corrected_centered
