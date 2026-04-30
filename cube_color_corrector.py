"""
立方体颜色-轴校正模块 (Cube Color Corrector)

FoundationPose 对立方体做 6DoF 姿态估计时，由于 24 重旋转对称性，register 会随机落入
24 个等价解之一。本模块通过枚举 24 个对称旋转候选，利用相机图像像素颜色匹配选出正确朝向。

颜色匹配使用 Lab 色彩空间欧氏距离 (sigma=25)，相比 HSV 色相距离对 Red/Orange/Yellow
等相近颜色辨别力提升 3x+（Lab 距离 ~40-50 vs HSV 仅 ~15° hue）。
白色判定同时检查 L* 和 chroma，避免 Yellow(L*=248) 被误判为 White(L*=255)。
debug 模式下打印 best 候选和 runner-up 的逐面详情 (采样RGB、期望RGB、Lab距离、score)。

用法:
    from cube_color_corrector import CubeColorCorrector

    corrector = CubeColorCorrector(half_size=0.0275, debug=True)

    # 在 register 后调用 (仅需一次，track 阶段不需要):
    tf_centered = est.get_tf_to_centered_mesh().data.cpu().numpy()
    pose, pose_centered = corrector.correct(pose, image_rgb, K, tf_centered, mask=mask)
    est.pose_last = torch.as_tensor(pose_centered, device='cuda', dtype=torch.float).reshape(1, 4, 4)

颜色-轴映射 (与 mesh 贴图一致):
    +X = Red,    -X = Orange
    +Y = Blue,   -Y = Green
    +Z = White,  -Z = Yellow

参数:
    half_size     方块半边长 (米), 55mm 方块 = 0.0275
    patch_radius  每个采样点的 patch 半径 (像素), 默认 3 → 7x7 patch
                  每个面采 3x3=9 个点, 各点取 7x7 patch 中位数, 再对 9 个点取中位数
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
    """Lab 色彩空间欧氏距离打分, 白色特殊处理.

    相比 HSV 色相距离, Lab 空间对 Red/Orange/Yellow 等相近颜色辨别力更强:
    Red-Orange Lab 距离 ~40-50 (vs HSV 仅 ~15° hue).

    Args:
        sampled_rgb: (3,) 采样到的 RGB 颜色 (0-255 float)
        expected_rgb: (3,) 期望的 RGB 颜色 (0-255 uint8-like)

    Returns:
        float 匹配分数 [0, 1]
    """
    # RGB → Lab (OpenCV: L 0-255, a/b 0-255, neutral=128)
    s_bgr = np.uint8([[np.clip(sampled_rgb[::-1], 0, 255)]])
    e_bgr = np.uint8([[np.clip(expected_rgb[::-1], 0, 255)]])
    s_lab = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(float)
    e_lab = cv2.cvtColor(e_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(float)

    # White detection: expected has very high L* AND low chroma
    # (Yellow also has L*=248, but chroma~97; White has chroma~0)
    e_chroma = np.sqrt((e_lab[1] - 128)**2 + (e_lab[2] - 128)**2)
    is_white_expected = (e_lab[0] > 230) and (e_chroma < 20)
    if is_white_expected:
        # Sampled should have high L* and low chroma
        s_chroma = np.sqrt((s_lab[1] - 128)**2 + (s_lab[2] - 128)**2)
        if s_lab[0] > 180 and s_chroma < 40:
            return 1.0 - s_chroma / 100.0
        return 0.0

    # Chromatic: Lab Euclidean distance
    dist = np.linalg.norm(s_lab - e_lab)
    sigma = 25.0
    score = np.exp(-(dist**2) / (2 * sigma**2))

    # Penalize low-chroma samples (gray/white shouldn't match chromatic)
    chroma = np.sqrt((s_lab[1] - 128)**2 + (s_lab[2] - 128)**2)
    if chroma < 15:
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

    # 面颜色定义: (法线方向, 期望 RGB 颜色, 名称)
    # 与 mesh 贴图一致: Red=+X, Blue=+Y, White=+Z
    FACE_COLORS = [
        (np.array([1, 0, 0], dtype=np.float64),  np.array([255, 0, 0]),       "+X Red"),
        (np.array([-1, 0, 0], dtype=np.float64), np.array([230, 90, 0]),      "-X Orange"),
        (np.array([0, 1, 0], dtype=np.float64),  np.array([0, 0, 255]),       "+Y Blue"),
        (np.array([0, -1, 0], dtype=np.float64), np.array([0, 255, 0]),       "-Y Green"),
        (np.array([0, 0, 1], dtype=np.float64),  np.array([255, 255, 255]),   "+Z White"),
        (np.array([0, 0, -1], dtype=np.float64), np.array([255, 255, 0]),     "-Z Yellow"),
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

        # 预计算面信息: (法线, 采样点列表, 期望RGB, 名称)
        # 每个面在面上取 3x3 = 9 个采样点, 避免单点采到边缘或反光
        self.faces = []
        sample_offsets = [-0.5, 0.0, 0.5]  # 相对 half_size 的偏移比例
        for normal, color, name in self.FACE_COLORS:
            # 找到面的两个切线方向
            axis = np.argmax(np.abs(normal))
            tangent_axes = [i for i in range(3) if i != axis]
            sample_points = []
            for s in sample_offsets:
                for t in sample_offsets:
                    pt = self.half_size * normal.copy()
                    pt[tangent_axes[0]] = s * self.half_size
                    pt[tangent_axes[1]] = t * self.half_size
                    sample_points.append(pt)
            self.faces.append((normal, sample_points, color, name))

        if self.debug:
            print(f"[CubeColorCorrector] half_size={half_size}m, patch_radius={patch_radius}px, "
                  f"24 rotations, 6 faces, 9 samples/face")

    def correct(self, pose, image_rgb, K, tf_to_centered_mesh, mask=None):
        """校正 register 后的 pose, 选出颜色匹配最佳的对称旋转.

        Args:
            pose: 4x4 numpy, register 返回的 original-mesh pose (相机坐标系)
            image_rgb: HxWx3 RGB uint8 图像
            K: 3x3 相机内参矩阵
            tf_to_centered_mesh: 4x4 numpy, est.get_tf_to_centered_mesh() 结果
            mask: HxW bool numpy, register 时使用的 mask 区域 (None=不约束)

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

        # 存储每个候选的逐面详情 (用于 debug)
        face_details_per_candidate = []

        for idx, R_sym in enumerate(self.rotations_3x3):
            score = 0.0
            total_weight = 0.0
            face_details = []

            for face_normal, sample_points, expected_rgb, face_name in self.faces:
                # Transform face normal to camera frame (用第一个采样点算可见性)
                rotated_normal = R_sym @ face_normal

                n_cam = R_cam @ rotated_normal

                # Visibility: face must point toward camera (z < -0.1)
                if n_cam[2] >= -0.1:
                    face_details.append((face_name, 'hidden', None, expected_rgb, 0, 0))
                    continue

                # 对面上 9 个采样点投影并采样颜色
                valid_samples = []
                for sp in sample_points:
                    rotated_sp = R_sym @ sp
                    sp_cam = R_cam @ rotated_sp + t_cam

                    if sp_cam[2] <= 0.01:
                        continue

                    proj = K @ sp_cam
                    u = proj[0] / proj[2]
                    v = proj[1] / proj[2]

                    if u < 0 or u >= W or v < 0 or v >= H:
                        continue

                    c = sample_face_color(image_rgb, u, v, self.patch_radius)
                    valid_samples.append(c)

                if len(valid_samples) == 0:
                    face_details.append((face_name, 'OOB', None, expected_rgb, 0, 0))
                    continue

                # 取所有有效采样点的中位数颜色
                sampled_rgb = np.median(np.array(valid_samples), axis=0)

                # Score
                s = color_match_score(sampled_rgb, expected_rgb)
                weight = -n_cam[2]  # more facing camera = higher weight
                score += s * weight
                total_weight += weight
                face_details.append((face_name, 'visible', sampled_rgb, expected_rgb, s, weight,
                                     len(valid_samples)))

            if total_weight > 0:
                score /= total_weight

            scores_debug.append(score)
            face_details_per_candidate.append(face_details)

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

            # 打印 best 候选的逐面详情
            best_details = face_details_per_candidate[best_idx]
            print(f"  --- Best candidate #{best_idx} per-face details ---")
            for detail in best_details:
                face_name, status = detail[0], detail[1]
                if status != 'visible':
                    print(f"    {face_name:12s}: [{status}]")
                else:
                    sampled, expected, s, w = detail[2], detail[3], detail[4], detail[5]
                    n_samples = detail[6] if len(detail) > 6 else 1
                    # 计算 Lab 距离
                    s_bgr = np.uint8([[np.clip(sampled[::-1], 0, 255)]])
                    e_bgr = np.uint8([[np.clip(expected[::-1], 0, 255)]])
                    s_lab = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(float)
                    e_lab = cv2.cvtColor(e_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(float)
                    lab_dist = np.linalg.norm(s_lab - e_lab)
                    print(f"    {face_name:12s}: sampled=({sampled[0]:.0f},{sampled[1]:.0f},{sampled[2]:.0f}) "
                          f"expected=({expected[0]},{expected[1]},{expected[2]}) "
                          f"Lab_dist={lab_dist:.1f} score={s:.3f} weight={w:.2f} pts={n_samples}")

            # 如果 top-2 分数很接近, 也打印第 2 名的详情
            if len(sorted_scores) >= 2:
                second_idx = sorted_scores[1][0]
                score_gap = sorted_scores[0][1] - sorted_scores[1][1]
                if score_gap < 0.15:
                    print(f"  --- Runner-up #{second_idx} (gap={score_gap:.3f}) per-face details ---")
                    for detail in face_details_per_candidate[second_idx]:
                        face_name, status = detail[0], detail[1]
                        if status != 'visible':
                            print(f"    {face_name:12s}: [{status}]")
                        else:
                            sampled, expected, s, w = detail[2], detail[3], detail[4], detail[5]
                            n_samples = detail[6] if len(detail) > 6 else 1
                            s_bgr = np.uint8([[np.clip(sampled[::-1], 0, 255)]])
                            e_bgr = np.uint8([[np.clip(expected[::-1], 0, 255)]])
                            s_lab = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(float)
                            e_lab = cv2.cvtColor(e_bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(float)
                            lab_dist = np.linalg.norm(s_lab - e_lab)
                            print(f"    {face_name:12s}: sampled=({sampled[0]:.0f},{sampled[1]:.0f},{sampled[2]:.0f}) "
                                  f"expected=({expected[0]},{expected[1]},{expected[2]}) "
                                  f"Lab_dist={lab_dist:.1f} score={s:.3f} weight={w:.2f} pts={n_samples}")

        return corrected_pose, corrected_centered
