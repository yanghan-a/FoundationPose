"""
FoundationPose 实时 6DoF 跟踪脚本 — 固定区域自动框选模式 (Intel RealSense D435i)

与 run_live.py 功能相同，但使用固定位置矩形框自动生成 mask，无需手动鼠标框选。
适用于相机固定、方块在视野中位置大致固定的场景。
通过 --box_cx/--box_cy/--box_w/--box_h 参数调节框选区域的中心和大小。

用法:
  python run_live_auto_nocolor.py --mesh_file <path_to_mesh.obj> [选项]

参数:
  --mesh_file           物体 mesh 文件路径 (.obj/.ply)
  --est_refine_iter     初始位姿估计的 refine 迭代次数 (默认 5)
  --track_refine_iter   跟踪时的 refine 迭代次数 (默认 2)
  --cam_width           相机分辨率宽 (默认 640)
  --cam_height          相机分辨率高 (默认 480)
  --cam_fps             相机帧率 (默认 60)
  --motion_predict      启用运动模型预测 (默认开启)
  --no_motion_predict   关闭运动模型预测
  --exposure            手动曝光值 (微秒, 如 100/200/500, 不传则自动曝光)
  --recovery_thresh     丢失检测阈值, 帧间平移跳变超过此值(米)判定异常 (默认 auto=直径*0.5)
  --debug               调试等级, 0=仅终端FPS, 1=实时显示, 2=保存图片 (默认 1)
  --debug_dir           调试图片保存目录 (默认 ./debug_live)
  --zmq_port            ZMQ PUB端口, 发布cube pose供其他进程订阅 (默认 5555, 0=禁用)
  --world_tag_id        世界坐标系 AprilTag ID (默认 0)
  --world_tag_size      AprilTag 物理尺寸 (米, 默认 0.048)
  --world_sample_frames 世界坐标系采样帧数 (默认 100)
  --no_world_frame      跳过世界坐标系标定, 直接发布相机坐标系下的 pose
  --box_cx              固定框中心 x 比例 (0~1, 0.5=居中, 默认 0.5)
  --box_cy              固定框中心 y 比例 (0~1, 0.5=居中, 默认 0.5)
  --box_w               固定框宽度占图像宽度比例 (0~1, 默认 0.3)
  --box_h               固定框高度占图像高度比例 (0~1, 默认 0.3)

固定区域框选:
  启动后在图像指定位置生成固定大小的矩形 mask, 无需任何图像分割。
  遮挡恢复后优先用 last_good_pose 投影区域 auto re-register,
  若失败则回退到固定区域框选。
  按 'r' 键可触发重新 register (使用固定区域)。

ZMQ发布 (与 cube_world_observer.py 兼容):
  默认端口 5555, JSON 格式:
    {"timestamp", "frame", "world_fixed": bool,
     "cube1": {"position": {"x","y","z"}, "orientation": {"x","y","z","w"}}}
  orientation 为 scipy 四元数 (x,y,z,w), CubeReceiver 自动转 MuJoCo (w,x,y,z)
  默认通过 AprilTag 标定世界坐标系, 与 cube_world_observer.py 行为一致
  按 'w' 键可重新标定世界坐标系

注意:
  此版本不含 24 对称姿态颜色校正 (无 CubeColorCorrector)。
  Register 后直接使用 FoundationPose 返回的 pose, 不做颜色-坐标轴绑定。

示例:
  python run_live_auto_nocolor.py --mesh_file demo_data/cube55/mesh/textured_simple.obj
  python run_live_auto_nocolor.py --mesh_file demo_data/cube55/mesh/textured_simple.obj \\
      --box_cx 0.4 --box_cy 0.4 --box_w 0.5 --box_h 0.5
"""

from estimater import *
from datareader import *
import argparse
import json
import time
import threading
from collections import defaultdict
from typing import Optional
import zmq
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

try:
    from pupil_apriltags import Detector as AprilTagDetector
except ImportError:
    AprilTagDetector = None


class FixedBoxSelector:
    """固定区域框选: 在图像指定位置生成固定大小的矩形 mask.

    相机固定后方块在视野中的位置也固定, 无需任何图像分割处理,
    直接在预设位置生成矩形 mask 用于 FoundationPose register.

    通过命令行参数 --box_cx, --box_cy, --box_w, --box_h 调节位置和大小.
    默认值为图像中央, 宽高各占图像的 30%.
    """

    def __init__(self, cx_ratio=0.5, cy_ratio=0.5, w_ratio=0.3, h_ratio=0.3, debug=False):
        """
        Args:
            cx_ratio: 框中心 x 占图像宽度的比例 (0~1, 0.5=居中)
            cy_ratio: 框中心 y 占图像高度的比例 (0~1, 0.5=居中)
            w_ratio:  框宽度占图像宽度的比例 (0~1)
            h_ratio:  框高度占图像高度的比例 (0~1)
            debug:    是否在画面上显示框选区域
        """
        self.cx_ratio = cx_ratio
        self.cy_ratio = cy_ratio
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.debug = debug

    def segment(self, image_rgb, depth=None) -> np.ndarray:
        """生成固定位置的矩形 mask.

        Args:
            image_rgb: HxWx3 RGB uint8 图像 (仅用于获取尺寸)
            depth: 未使用, 保留接口兼容性

        Returns:
            bool mask (H,W), 始终返回有效 mask (不会返回 None)
        """
        H, W = image_rgb.shape[:2]

        cx = int(W * self.cx_ratio)
        cy = int(H * self.cy_ratio)
        bw = int(W * self.w_ratio)
        bh = int(H * self.h_ratio)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(W, cx + bw // 2)
        y2 = min(H, cy + bh // 2)

        mask = np.zeros((H, W), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask


class FrameProfiler:
    """每帧各阶段耗时统计 (使用 CUDA sync 获取真实 GPU 耗时)"""

    def __init__(self, window=60, use_cuda_sync=True):
        self.window = window
        self.use_cuda_sync = use_cuda_sync
        self.history = defaultdict(list)      # name -> list of durations
        self._pending_name = None
        self._pending_t0 = None

    def tick(self, name):
        """开始计时一个新阶段 (自动结束上一个阶段)"""
        now_t = self._sync_time()
        if self._pending_name is not None:
            dur = now_t - self._pending_t0
            self._record(self._pending_name, dur)
        self._pending_name = name
        self._pending_t0 = now_t

    def tock(self, name=None):
        """显式结束当前阶段"""
        now_t = self._sync_time()
        if self._pending_name is not None:
            dur = now_t - self._pending_t0
            self._record(self._pending_name, dur)
            self._pending_name = None

    def _sync_time(self):
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _record(self, name, dur):
        h = self.history[name]
        h.append(dur)
        if len(h) > self.window:
            h.pop(0)

    def summary(self):
        """返回各阶段平均耗时 (ms) 的排序列表"""
        result = []
        for name, vals in self.history.items():
            if len(vals) == 0:
                continue
            avg_ms = sum(vals) / len(vals) * 1000
            result.append((name, avg_ms))
        result.sort(key=lambda x: -x[1])
        return result

    def summary_str(self):
        lines = []
        total = 0
        for name, avg_ms in self.summary():
            lines.append(f"  {name:30s} {avg_ms:7.2f} ms")
            total += avg_ms
        lines.insert(0, f"  {'TOTAL':30s} {total:7.2f} ms")
        return "\n".join(lines)


class MotionPredictor:
    """恒速运动模型 (纯 GPU torch)"""

    def __init__(self):
        self.pose_prev = None
        self.pose_curr = None
        self.time_prev = 0.0
        self.time_curr = 0.0

    def update(self, pose_tensor, timestamp):
        self.pose_prev = self.pose_curr
        self.time_prev = self.time_curr
        self.pose_curr = pose_tensor.detach().clone().reshape(4, 4)
        self.time_curr = timestamp

    def predict(self):
        if self.pose_prev is None or self.pose_curr is None:
            return None
        dt = self.time_curr - self.time_prev
        if dt < 1e-6:
            return None
        delta = self.pose_curr @ torch.inverse(self.pose_prev)
        return delta @ self.pose_curr

    def reset(self):
        self.pose_prev = None
        self.pose_curr = None
        self.time_prev = 0.0
        self.time_curr = 0.0


class RealSenseCamera:
    """Intel RealSense D435i RGB-D 相机封装，后台线程持续抓帧"""

    def __init__(self, width=424, height=240, fps=60, exposure=None):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(config)

        color_sensor = profile.get_device().first_color_sensor()
        if exposure is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, exposure)
            logging.info(f"Manual exposure set to {exposure} us")
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            logging.info("Auto exposure enabled")

        self.align = rs.align(rs.stream.color)

        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.W = intrinsics.width
        self.H = intrinsics.height
        logging.info(f"RealSense started: {self.W}x{self.H} @ {fps}fps")
        logging.info(f"Camera K:\n{self.K}")

        for _ in range(30):
            self.pipeline.wait_for_frames()

        self._latest_color = None
        self._latest_depth = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
                depth[depth < 0.001] = 0
                depth[depth > 3.0] = 0

                with self._lock:
                    self._latest_color = color
                    self._latest_depth = depth
            except Exception:
                pass

    def get_frame(self):
        with self._lock:
            return self._latest_color, self._latest_depth

    def get_frame_blocking(self):
        while True:
            c, d = self.get_frame()
            if c is not None:
                return c, d
            time.sleep(0.001)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.pipeline.stop()


class FPSCounter:
    def __init__(self, window=30):
        self.window = window
        self.timestamps = []

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt == 0:
            return 0.0
        return (len(self.timestamps) - 1) / dt


class WorldFrameCalibrator:
    """通过 AprilTag 建立世界坐标系，与 cube_world_observer.py 逻辑一致。

    采样若干帧 AprilTag 检测结果，取平均后固定世界坐标系。
    之后可将相机坐标系下的物体 pose 转换到世界坐标系。
    """

    def __init__(self, cam_K, tag_id=0, tag_size=0.048, sample_frames=100,
                 world_frame_correction=None):
        """
        Args:
            cam_K: 3x3 相机内参矩阵
            tag_id: 世界坐标系 AprilTag ID (默认 0)
            tag_size: AprilTag 物理尺寸 (米)
            sample_frames: 采样帧数
            world_frame_correction: 世界坐标系修正矩阵 (3x3) 或 None
        """
        if AprilTagDetector is None:
            raise ImportError("pupil_apriltags not installed. Run: pip install pupil-apriltags")

        self.cam_K = cam_K
        self.tag_id = tag_id
        self.tag_size = tag_size
        self.sample_target = sample_frames
        self.world_frame_correction = world_frame_correction

        self.detector = AprilTagDetector(
            families="tag36h11", nthreads=4, quad_decimate=1.0,
            quad_sigma=0.0, decode_sharpening=0.25,
        )

        self.world_pose = None
        self.is_fixed = False
        self._samples_R = []
        self._samples_t = []

    def feed_frame(self, gray):
        """喂入一帧灰度图进行 AprilTag 检测。

        Returns:
            True if 世界坐标系刚刚标定完成, False otherwise.
        """
        if self.is_fixed:
            return False

        fx, fy, cx, cy = self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, 2], self.cam_K[1, 2]
        results = self.detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.tag_size,
        )

        for r in results:
            if r.tag_id == self.tag_id:
                self._samples_R.append(r.pose_R)
                self._samples_t.append(r.pose_t.flatten())
                break

        n = len(self._samples_R)
        if n > 0 and n % 20 == 0:
            print(f"[World Sampling] {n}/{self.sample_target} frames...")

        if n >= self.sample_target:
            return self._finalize()
        return False

    def _finalize(self):
        """从采样数据中计算并固定世界坐标系。"""
        if len(self._samples_R) < 10:
            print(f"[World Sampling] Failed: only {len(self._samples_R)} samples")
            return False

        quats = []
        for R in self._samples_R:
            q = Rotation.from_matrix(R).as_quat()
            quats.append(q)
        quats = np.array(quats)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]
        avg_quat = quats.mean(axis=0)
        avg_quat /= np.linalg.norm(avg_quat)
        avg_R = Rotation.from_quat(avg_quat).as_matrix()

        avg_t = np.mean(self._samples_t, axis=0)

        if self.world_frame_correction is not None:
            correction_R = np.array(self.world_frame_correction)
            avg_R = avg_R @ correction_R.T
            print(f"[World Sampling] Applied world frame correction")

        self.world_pose = (avg_R, avg_t)
        self.is_fixed = True
        print(f"[World Sampling] Complete! Averaged {len(self._samples_R)} samples. World frame FIXED.")
        return True

    def reset(self):
        """重新开始采样。"""
        self._samples_R = []
        self._samples_t = []
        self.is_fixed = False
        self.world_pose = None
        print(f"[World Sampling] Reset. Collecting {self.sample_target} frames...")

    def cam_to_world(self, pose_4x4):
        """将相机坐标系下的 4x4 pose 转换到世界坐标系。"""
        if self.world_pose is None:
            return None, None

        R_world_cam, t_world_cam = self.world_pose
        R_cam_world = R_world_cam.T
        t_cam_world = -R_cam_world @ t_world_cam

        R_obj_cam = pose_4x4[:3, :3]
        t_obj_cam = pose_4x4[:3, 3]

        R_obj_world = R_cam_world @ R_obj_cam
        t_obj_world = R_cam_world @ t_obj_cam + t_cam_world

        return R_obj_world, t_obj_world


def make_mask_from_pose(pose_4x4, mesh_pts, K, H, W, pad_ratio=0.5):
    """把 mesh 顶点按 pose 投影到图像，取 2D bbox 扩大 pad_ratio 生成 mask"""
    pts_cam = (pose_4x4[:3, :3] @ mesh_pts.T + pose_4x4[:3, 3:4]).T
    valid = pts_cam[:, 2] > 0.01
    if valid.sum() < 10:
        return None
    pts_cam = pts_cam[valid]
    uvs = (K @ pts_cam.T).T
    uvs = uvs[:, :2] / uvs[:, 2:3]
    u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
    v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()
    du = (u_max - u_min) * pad_ratio
    dv = (v_max - v_min) * pad_ratio
    x1 = max(0, int(u_min - du))
    y1 = max(0, int(v_min - dv))
    x2 = min(W, int(u_max + du))
    y2 = min(H, int(v_max + dv))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def check_pose_size_consistency(pose_4x4, mesh_diameter, K, depth_map, mesh_pts,
                                 size_tol=0.5, depth_tol=0.4):
    """检查 pose 投影尺寸是否与目标物体一致。"""
    z = pose_4x4[2, 3]
    if z < 0.01:
        return False, f"z={z:.3f} too small"

    pts_cam = (pose_4x4[:3, :3] @ mesh_pts.T + pose_4x4[:3, 3:4]).T
    valid = pts_cam[:, 2] > 0.01
    if valid.sum() < 10:
        return False, "too few valid projected points"
    pts_cam = pts_cam[valid]
    uvs = (K @ pts_cam.T).T
    uvs = uvs[:, :2] / uvs[:, 2:3]

    bbox_w = uvs[:, 0].max() - uvs[:, 0].min()
    bbox_h = uvs[:, 1].max() - uvs[:, 1].min()
    actual_size = max(bbox_w, bbox_h)

    f = (K[0, 0] + K[1, 1]) / 2.0
    expected_size = mesh_diameter * f / z

    ratio = actual_size / max(expected_size, 1e-6)
    if ratio < (1 - size_tol) or ratio > (1 + size_tol):
        return False, f"size mismatch: actual={actual_size:.0f}px, expected={expected_size:.0f}px, ratio={ratio:.2f}"

    H, W = depth_map.shape[:2]
    u_min = max(0, int(uvs[:, 0].min()))
    u_max = min(W, int(uvs[:, 0].max()))
    v_min = max(0, int(uvs[:, 1].min()))
    v_max = min(H, int(uvs[:, 1].max()))
    if u_max > u_min and v_max > v_min:
        roi_depth = depth_map[v_min:v_max, u_min:u_max]
        valid_depth = roi_depth[(roi_depth > 0.01) & (roi_depth < 3.0)]
        if len(valid_depth) > 10:
            median_depth = np.median(valid_depth)
            depth_ratio = abs(median_depth - z) / max(z, 0.01)
            if depth_ratio > depth_tol:
                return False, f"depth mismatch: pose_z={z:.3f}m, roi_median={median_depth:.3f}m, ratio={depth_ratio:.2f}"

    return True, "ok"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, required=True)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--cam_width', type=int, default=640)
    parser.add_argument('--cam_height', type=int, default=480)
    parser.add_argument('--cam_fps', type=int, default=60)
    parser.add_argument('--exposure', type=int, default=None,
                        help='手动曝光值(微秒), 不传则自动曝光')
    parser.add_argument('--motion_predict', action='store_true', default=True)
    parser.add_argument('--no_motion_predict', dest='motion_predict', action='store_false')
    parser.add_argument('--recovery_thresh', type=float, default=0,
                        help='丢失检测阈值(米), 0=auto(直径*0.5)')
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_live')
    parser.add_argument('--zmq_port', type=int, default=5555,
                        help='ZMQ PUB port for cube pose (default 5555, 0=disable)')
    parser.add_argument('--world_tag_id', type=int, default=0,
                        help='AprilTag ID for world frame (default 0)')
    parser.add_argument('--world_tag_size', type=float, default=0.048,
                        help='AprilTag physical size in meters (default 0.048)')
    parser.add_argument('--world_sample_frames', type=int, default=100,
                        help='Frames to sample for world frame averaging (default 100)')
    parser.add_argument('--no_world_frame', action='store_true',
                        help='Skip world frame calibration, publish in camera frame')
    parser.add_argument('--box_cx', type=float, default=0.4,
                        help='固定框中心 x 比例 (0~1, 0.5=居中)')
    parser.add_argument('--box_cy', type=float, default=0.5,
                        help='固定框中心 y 比例 (0~1, 0.5=居中)')
    parser.add_argument('--box_w', type=float, default=0.35,
                        help='固定框宽度占图像宽度比例 (0~1)')
    parser.add_argument('--box_h', type=float, default=0.35,
                        help='固定框高度占图像高度比例 (0~1)')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    inv_to_origin = np.linalg.inv(to_origin)  # 预计算，避免每帧重复求逆


    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    cam = RealSenseCamera(width=args.cam_width, height=args.cam_height, fps=args.cam_fps,
                          exposure=args.exposure)
    auto_seg = FixedBoxSelector(cx_ratio=args.box_cx, cy_ratio=args.box_cy,
                                w_ratio=args.box_w, h_ratio=args.box_h,
                                debug=(debug >= 1))
    fps_counter = FPSCounter()
    motion = MotionPredictor()

    # 丢失恢复
    mesh_diameter = est.diameter
    recovery_thresh = args.recovery_thresh if args.recovery_thresh > 0 else mesh_diameter * 0.5
    mesh_pts_np = np.asarray(mesh.vertices)
    REVERT_LIMIT = 5        # 回退阶段最多持续帧数, 连续几帧确认丢失后立即 re-register

    # ZMQ publisher for cube pose
    zmq_context = None
    zmq_socket = None
    if args.zmq_port > 0:
        zmq_context = zmq.Context()
        zmq_socket = zmq_context.socket(zmq.PUB)
        zmq_socket.bind(f"tcp://*:{args.zmq_port}")
        logging.info(f"ZMQ cube pose publisher on port {args.zmq_port}")

    # World frame calibrator (AprilTag)
    WORLD_FRAME_CORRECTION = np.array([
        [ 0.9848,  0.0000,  0.1736],
        [ 0.0000, -1.0000,  0.0000],
        [ 0.1736,  0.0000, -0.9848],
    ])
    world_calibrator = None
    if not args.no_world_frame:
        world_calibrator = WorldFrameCalibrator(
            cam_K=cam.K,
            tag_id=args.world_tag_id,
            tag_size=args.world_tag_size,
            sample_frames=args.world_sample_frames,
            world_frame_correction=WORLD_FRAME_CORRECTION,
        )
        logging.info(f"World frame calibration enabled: tag_id={args.world_tag_id}, "
                     f"tag_size={args.world_tag_size}m, samples={args.world_sample_frames}")

    pose = None
    last_good_pose = None
    last_good_pose_centered = None
    lost_count = 0
    frame_idx = 0
    need_register = True

    logging.info(f"Motion prediction: {'ON' if args.motion_predict else 'OFF'}")
    logging.info(f"Recovery: thresh={recovery_thresh:.4f}m, diameter={mesh_diameter:.4f}m, "
                 f"revert={REVERT_LIMIT}f (then immediate re-register)")
    logging.info("Controls: 'r' = auto re-register, 'w' = re-calibrate world, 'q' = quit, 'p' = profiling")
    logging.info("Auto mode: parallelogram detection for cube detection (no manual selection)")

    # 初始化完成后压制日志 (每帧 ~15 次 logging.info 严重影响性能)
    logging.getLogger().setLevel(logging.WARNING)

    profiler = FrameProfiler(window=60, use_cuda_sync=True)
    PROFILE_INTERVAL = 120  # 每 N 帧自动打印一次 profiling

    try:
        while True:
            profiler.tick('get_frame')
            color, depth = cam.get_frame_blocking()
            t_now = time.time()

            if need_register:
                logging.getLogger().setLevel(logging.INFO)

                # 等待清场: 固定等待 1 秒, 让遮挡物 (如手) 移开后再抓帧
                if not hasattr(est, '_reregister_wait_start'):
                    est._reregister_wait_start = time.time()
                elapsed = time.time() - est._reregister_wait_start
                if elapsed < 1.0:
                    if debug >= 1:
                        vis_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                        wait_text = f"Waiting for clear view... ({elapsed:.1f}s / 1.0s)"
                        cv2.putText(vis_bgr, wait_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        if last_good_pose is not None:
                            center_pose = last_good_pose @ inv_to_origin
                            draw_posed_3d_box(cam.K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox)
                        cv2.imshow('FoundationPose Live Tracking (Auto)', vis_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                del est._reregister_wait_start

                # 等待结束, 重新抓取最新帧
                color, depth = cam.get_frame_blocking()

                # 尝试用上次好的 pose 投影区域自动 re-register
                auto_mask = None
                if last_good_pose is not None:
                    auto_mask = make_mask_from_pose(
                        last_good_pose, mesh_pts_np, cam.K, cam.H, cam.W, pad_ratio=0.5)

                if auto_mask is not None and auto_mask.sum() > 100:
                    logging.info("Auto re-register from last good pose...")
                    mask = auto_mask
                else:
                    # Fallback: 固定区域框选
                    logging.info("Auto segmentation: using fixed-position box...")
                    mask = auto_seg.segment(color)

                # register 比 track 显存开销大很多，先释放缓存防 OOM
                torch.cuda.empty_cache()

                t0 = time.time()
                pose = est.register(K=cam.K, rgb=color, depth=depth,
                                    ob_mask=mask, iteration=args.est_refine_iter)
                logging.info(f"Pose estimated in {time.time()-t0:.2f}s")

                # Register 结果校验: 投影尺寸 + 深度是否与 mesh 一致
                reg_valid, reg_reason = check_pose_size_consistency(
                    pose, mesh_diameter, cam.K, depth, mesh_pts_np,
                    size_tol=0.5, depth_tol=0.4)
                if not reg_valid:
                    logging.warning(f"[REGISTER] Pose rejected: {reg_reason}")
                    print(f"[REGISTER] Result doesn't match target size: {reg_reason}")
                    print(f"[REGISTER] Auto retrying...")
                    last_good_pose = None  # 强制颜色分割重试
                    continue

                # 直接使用 register 返回的 pose (不做 24 对称姿态颜色校正)

                need_register = False
                frame_idx = 0
                lost_count = 0
                fps_counter = FPSCounter()
                motion.reset()
                motion.update(est.pose_last, t_now)
                last_good_pose = pose.copy()
                last_good_pose_centered = est.pose_last.detach().clone()

                logging.getLogger().setLevel(logging.WARNING)
            else:
                # 运动模型外推
                profiler.tick('motion_predict')
                if args.motion_predict:
                    predicted = motion.predict()
                    if predicted is not None:
                        est.pose_last = predicted.reshape(1, 4, 4)

                profiler.tick('track_one_total')
                pose = est.track_one(rgb=color, depth=depth, K=cam.K,
                                     iteration=args.track_refine_iter,
                                     profiler=profiler)

                profiler.tick('motion_update')
                motion.update(est.pose_last, time.time())

                # === 丢失检测 ===
                profiler.tick('loss_detection')
                z = pose[2, 3]
                trans_jump = np.linalg.norm(pose[:3, 3] - last_good_pose[:3, 3]) if last_good_pose is not None else 0.0
                if last_good_pose is not None:
                    R_delta = pose[:3, :3] @ last_good_pose[:3, :3].T
                    rot_jump = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))
                else:
                    rot_jump = 0.0
                jump_bad = (trans_jump > recovery_thresh) or (rot_jump > np.deg2rad(45)) or (z < 0.02) or (z > 3.0)

                # 投影尺寸 + 深度一致性校验
                size_valid, size_reason = check_pose_size_consistency(
                    pose, mesh_diameter, cam.K, depth, mesh_pts_np)

                if not size_valid and not jump_bad:
                    # 尺寸不对但跳变检测没触发 = tracker 平滑漂移到了错误目标
                    print(f"[SIZE CHECK] Target mismatch: {size_reason}")
                    print(f"[SIZE CHECK] Tracker locked on wrong object, auto re-register...")
                    need_register = True
                    last_good_pose = None  # 清除, 强制颜色分割
                    lost_count = 0
                    frame_idx += 1
                    fps_counter.tick()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                is_bad = jump_bad or (not size_valid)

                if is_bad:
                    lost_count += 1
                    if lost_count <= REVERT_LIMIT:
                        # 短暂回退确认: 连续几帧都丢失才算真丢
                        if last_good_pose_centered is not None:
                            est.pose_last = last_good_pose_centered.clone().reshape(1, 4, 4)
                        motion.reset()
                    else:
                        # 确认丢失 → 立即 re-register
                        print(f"[RECOVERY] Lost for {lost_count} frames, auto re-register...")
                        need_register = True
                        lost_count = 0

                    frame_idx += 1
                    fps_counter.tick()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        need_register = True
                        last_good_pose = None
                    continue
                else:
                    if lost_count > 0:
                        print(f"[RECOVERY] Tracking recovered after {lost_count} lost frames!")
                    lost_count = 0
                    last_good_pose = pose.copy()
                    last_good_pose_centered = est.pose_last.detach().clone()

            # World frame calibration (feed every frame until fixed)
            if world_calibrator is not None and not world_calibrator.is_fixed:
                gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                just_fixed = world_calibrator.feed_frame(gray)
                if just_fixed:
                    logging.getLogger().setLevel(logging.INFO)
                    logging.info("World frame calibrated!")
                    logging.getLogger().setLevel(logging.WARNING)

            corrected_pose = pose

            # Publish cube pose via ZMQ
            if zmq_socket is not None:
                profiler.tick('zmq_publish')

                if world_calibrator is not None and world_calibrator.is_fixed:
                    R_world, t_world = world_calibrator.cam_to_world(corrected_pose)
                    world_fixed = True
                elif world_calibrator is None:
                    R_world = corrected_pose[:3, :3]
                    t_world = corrected_pose[:3, 3]
                    world_fixed = True
                else:
                    R_world = corrected_pose[:3, :3]
                    t_world = corrected_pose[:3, 3]
                    world_fixed = False

                quat_xyzw = Rotation.from_matrix(R_world).as_quat()
                msg = {
                    'timestamp': time.time(),
                    'frame': frame_idx,
                    'world_detected': world_calibrator is None or world_calibrator.is_fixed,
                    'world_fixed': world_fixed,
                    'cube1': {
                        'position': {
                            'x': float(t_world[0]),
                            'y': float(t_world[1]),
                            'z': float(t_world[2]),
                        },
                        'orientation': {
                            'x': float(quat_xyzw[0]),
                            'y': float(quat_xyzw[1]),
                            'z': float(quat_xyzw[2]),
                            'w': float(quat_xyzw[3]),
                        },
                        'timestamp': time.time(),
                    }
                }
                zmq_socket.send_string(json.dumps(msg))

            frame_idx += 1
            fps_counter.tick()

            profiler.tick('visualization')
            if debug >= 1:
                center_pose = corrected_pose @ inv_to_origin
                vis_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                draw_posed_3d_box(cam.K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox)
                axis_pose = corrected_pose.copy()
                axis_pose[:3, 3] = center_pose[:3, 3]
                vis_bgr = draw_xyz_axis(vis_bgr, ob_in_cam=axis_pose, scale=0.1,
                              K=cam.K, thickness=3, transparency=0, is_input_rgb=False,
                              axis_colors=((0,0,255), (255,0,0), (255,255,255)))
                _axis_pts = np.array([[0.1,0,0,1],[0,0.1,0,1],[0,0,0.1,1]], dtype=np.float64)
                _pts_cam = (axis_pose @ _axis_pts.T)[:3, :]
                _proj = cam.K @ _pts_cam
                _uvs = (_proj[:2, :] / _proj[2:3, :]).T.astype(int)
                for _label, _uv, _clr in [("X", _uvs[0], (0,0,255)), ("Y", _uvs[1], (255,0,0)), ("Z", _uvs[2], (255,255,255))]:
                    cv2.putText(vis_bgr, _label, tuple(_uv + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _clr, 2)

                if world_calibrator is not None and world_calibrator.is_fixed:
                    R_world, t_world = world_calibrator.cam_to_world(corrected_pose)
                    pos = t_world
                    euler_deg = Rotation.from_matrix(R_world).as_euler('xyz', degrees=True)
                    frame_label = "world"
                else:
                    pos = corrected_pose[:3, 3]
                    euler_deg = Rotation.from_matrix(corrected_pose[:3, :3]).as_euler('xyz', degrees=True)
                    frame_label = "cam"

                label = f"FPS:{fps_counter.fps:.1f}"
                if args.motion_predict:
                    label += " +motion"
                label += " [AUTO]"
                if world_calibrator is not None:
                    if world_calibrator.is_fixed:
                        label += " W:OK"
                    else:
                        n = len(world_calibrator._samples_R)
                        label += f" W:{n}/{world_calibrator.sample_target}"
                cv2.putText(vis_bgr, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                pos_text = f"Pos({frame_label}): x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} m"
                cv2.putText(vis_bgr, pos_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                rot_text = f"Rot({frame_label}): rx={euler_deg[0]:.1f} ry={euler_deg[1]:.1f} rz={euler_deg[2]:.1f} deg"
                cv2.putText(vis_bgr, rot_text, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # AprilTag 检测并绘制坐标轴
                if AprilTagDetector is not None:
                    gray_vis = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                    fx, fy = cam.K[0, 0], cam.K[1, 1]
                    cx, cy = cam.K[0, 2], cam.K[1, 2]
                    tag_results = world_calibrator.detector.detect(
                        gray_vis, estimate_tag_pose=True,
                        camera_params=(fx, fy, cx, cy),
                        tag_size=args.world_tag_size,
                    ) if world_calibrator is not None else []
                    for det in tag_results:
                        corners = det.corners.astype(int)
                        for i in range(4):
                            cv2.line(vis_bgr, tuple(corners[i]), tuple(corners[(i+1)%4]),
                                     (0, 255, 0), 2)
                        ctr = det.center.astype(int)
                        cv2.putText(vis_bgr, f"id:{det.tag_id}", (ctr[0]-20, ctr[1]-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if det.pose_R is not None and det.pose_t is not None:
                            tag_R = det.pose_R @ WORLD_FRAME_CORRECTION.T
                            tag_t = det.pose_t.flatten()
                            axis_len = args.world_tag_size * 1.0
                            origin = tag_t
                            x_end = tag_t + tag_R[:, 0] * axis_len
                            y_end = tag_t + tag_R[:, 1] * axis_len
                            z_end = tag_t + tag_R[:, 2] * axis_len
                            def proj_pt(pt3d):
                                u = fx * pt3d[0] / pt3d[2] + cx
                                v = fy * pt3d[1] / pt3d[2] + cy
                                return (int(u), int(v))
                            o2d = proj_pt(origin)
                            x2d = proj_pt(x_end)
                            y2d = proj_pt(y_end)
                            z2d = proj_pt(z_end)
                            cv2.arrowedLine(vis_bgr, o2d, x2d, (0, 0, 255), 2, tipLength=0.15)
                            cv2.arrowedLine(vis_bgr, o2d, y2d, (0, 255, 0), 2, tipLength=0.15)
                            cv2.arrowedLine(vis_bgr, o2d, z2d, (255, 0, 0), 2, tipLength=0.15)
                            cv2.putText(vis_bgr, "X", (x2d[0]+3, x2d[1]-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            cv2.putText(vis_bgr, "Y", (y2d[0]+3, y2d[1]-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            cv2.putText(vis_bgr, "Z", (z2d[0]+3, z2d[1]-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # 绘制固定框选区域 (cyan 虚线矩形)
            if debug >= 1:
                H_img, W_img = vis_bgr.shape[:2]
                box_cx = int(W_img * auto_seg.cx_ratio)
                box_cy = int(H_img * auto_seg.cy_ratio)
                box_bw = int(W_img * auto_seg.w_ratio)
                box_bh = int(H_img * auto_seg.h_ratio)
                box_x1 = max(0, box_cx - box_bw // 2)
                box_y1 = max(0, box_cy - box_bh // 2)
                box_x2 = min(W_img, box_cx + box_bw // 2)
                box_y2 = min(H_img, box_cy + box_bh // 2)
                cv2.rectangle(vis_bgr, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 0), 1)
                cv2.putText(vis_bgr, "BOX", (box_x1 + 2, box_y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            profiler.tick('imshow')
            if debug >= 1:
                cv2.imshow('FoundationPose Live Tracking (Auto)', vis_bgr)

            if debug >= 2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{debug_dir}/track_vis/{frame_idx:06d}.png',
                                cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))

            profiler.tick('waitkey')
            key = cv2.waitKey(1) & 0xFF
            profiler.tock()

            if key == ord('q'):
                break
            elif key == ord('r'):
                # 'r' = 触发自动重新分割
                need_register = True
                last_good_pose = None  # 强制颜色分割
            elif key == ord('w'):
                if world_calibrator is not None:
                    world_calibrator.reset()
            elif key == ord('p'):
                print(f"\n=== Frame Profiling (avg over last {profiler.window} frames) ===")
                print(profiler.summary_str())
                print(f"  Tracking FPS: {fps_counter.fps:.1f}")
                print()

            # 定期自动打印
            if frame_idx > 0 and frame_idx % PROFILE_INTERVAL == 0:
                print(f"\n=== Frame {frame_idx} Profiling (avg over last {profiler.window} frames) ===")
                print(profiler.summary_str())
                print(f"  Tracking FPS: {fps_counter.fps:.1f}")
                print()

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if zmq_socket is not None:
            zmq_socket.close()
        if zmq_context is not None:
            zmq_context.term()
        # 退出前打印最终 profiling
        if profiler.history:
            print(f"\n=== Final Profiling Summary ===")
            print(profiler.summary_str())
            print()
