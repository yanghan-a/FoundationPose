"""
FoundationPose 实时 6DoF 跟踪脚本 (Intel RealSense D435i)

用法:
  python run_live.py --mesh_file <path_to_mesh.obj> [选项]

参数:
  --mesh_file           物体 mesh 文件路径 (.obj/.ply)
  --est_refine_iter     初始位姿估计的 refine 迭代次数 (默认 5)
  --track_refine_iter   跟踪时的 refine 迭代次数 (默认 1, 越小越快)
  --cam_width           相机分辨率宽 (默认 424, 低分辨率=高帧率)
  --cam_height          相机分辨率高 (默认 240)
  --cam_fps             相机帧率 (默认 60, D435i 支持 60@424x240)
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

ZMQ发布 (与 cube_world_observer.py 兼容):
  默认端口 5555, JSON 格式:
    {"timestamp", "frame", "world_fixed": bool,
     "cube1": {"position": {"x","y","z"}, "orientation": {"x","y","z","w"}}}
  orientation 为 scipy 四元数 (x,y,z,w), CubeReceiver 自动转 MuJoCo (w,x,y,z)
  默认通过 AprilTag 标定世界坐标系, 与 cube_world_observer.py 行为一致
  按 'w' 键可重新标定世界坐标系

颜色-坐标轴绑定 (自动校正 + 右手系):
  立方体有 24 重旋转对称性, FoundationPose register 会随机落入任一等价解。
  register 后自动枚举 24 个对称旋转候选, 通过图像像素颜色匹配 (HSV 色相距离)
  选出与物理方块颜色朝向一致的 pose (由 CubeColorCorrector 实现)。
  校正后坐标轴绑定: +X=红色面, +Y=蓝色面(-Y=绿色面), +Z=白色面
  仅在 register 时做一次校正, track 阶段保持 register 确定的朝向。
  可视化箭头颜色: X=红色, Y=蓝色, Z=白色
  箭头端点附带 X/Y/Z 文字标注
  屏幕实时显示位置 (x,y,z 米) 和欧拉角 (rx,ry,rz 度)
  世界坐标系标定完成后, 显示和发布的位姿均为世界坐标系下 (标注 "world"); 未标定时为相机坐标系 (标注 "cam")

AprilTag 坐标轴可视化:
  每帧检测画面中的 AprilTag, 绘制 tag 边框 (绿色) 和 ID 标注,
  并在 tag 中心画出 XYZ 坐标轴箭头 (X=红, Y=绿, Z=蓝), 轴长等于 tag 尺寸
  坐标轴已应用 WORLD_FRAME_CORRECTION, 与 cube_world_observer.py 的世界坐标系一致

丢失恢复机制:
  每帧检测位姿跳变 (平移突变 / z<0 飞出画面):
    - 连续 <=3 帧异常: 回退到上次好的 pose，重置 motion predictor，继续跟踪
    - 连续 >3 帧异常: 判定彻底丢失，用上次好 pose 投影 bbox 区域自动 re-register
    - 自动 re-register 失败: 弹出手动框选

示例:
  # 默认配置 (424x240@60fps, ~30Hz tracking)
  python run_live.py --mesh_file demo_data/cube59/mesh/textured_simple.obj

  # 高分辨率 (640x480, ~20Hz)
  python run_live.py --mesh_file demo_data/cube59/mesh/textured_simple.obj \\
      --cam_width 640 --cam_height 480 --cam_fps 30

  # 关闭运动预测
  python run_live.py --mesh_file demo_data/cube59/mesh/textured_simple.obj \\
      --no_motion_predict
"""

from estimater import *
from datareader import *
import argparse
import json
import time
import threading
from collections import defaultdict
import zmq
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from cube_color_corrector import CubeColorCorrector

try:
    from pupil_apriltags import Detector as AprilTagDetector
except ImportError:
    AprilTagDetector = None


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


class BBoxSelector:
    """鼠标框选工具"""

    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selecting = True
            self.done = False
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.selecting = False
            self.done = True

    def select(self, image):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.done = False

        win_name = "Draw box around object, press ENTER to confirm"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        display = image[..., ::-1].copy()

        while True:
            show = display.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(show, self.start_point, self.end_point, (0, 255, 0), 2)
            cv2.imshow(win_name, show)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and self.done:
                break
            if key == 27:
                cv2.destroyWindow(win_name)
                return None

        cv2.destroyWindow(win_name)

        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])

        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask.astype(bool)


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

        # world_pose = (R, t): R 的列向量是世界坐标轴在相机坐标系下的方向,
        # t 是世界原点在相机坐标系下的位置
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

        # 平均旋转 (四元数平均)
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

        # 平均平移
        avg_t = np.mean(self._samples_t, axis=0)

        # 应用世界坐标系修正
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
        """将相机坐标系下的 4x4 pose 转换到世界坐标系。

        Args:
            pose_4x4: 4x4 物体在相机坐标系下的位姿矩阵

        Returns:
            (R_world, t_world): 物体在世界坐标系下的旋转和平移,
            如果世界坐标系未标定则返回 (None, None)
        """
        if self.world_pose is None:
            return None, None

        R_world_cam, t_world_cam = self.world_pose
        # R_world_cam 列向量是世界轴在相机系下的表示
        # R_cam_world = R_world_cam.T 将相机系下的向量转到世界系
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

    # 颜色-坐标轴修正: mesh 原始轴 → 物理方块颜色轴 (右手系)
    # mesh 原始 (贴图交换蓝绿后): +X=Blue, +Y=White, +Z=Red
    # 目标:      +X=Red,   +Y=Blue,  +Z=White (对面: -X=Orange, -Y=Green, -Z=Yellow)
    # 用法: corrected_pose = pose @ R_color_correction
    # R 的列向量 = 新坐标轴在 mesh 坐标系中的方向
    R_color_correction = np.eye(4)
    # R_color_correction[:3, :3] = np.array([
    #     [0, 1, 0],   # col0: new +X(Red)   = mesh +Z → [0,0,1] read col-wise
    #     [0, 0, 1],   # col1: new +Y(Green) = mesh +X → [1,0,0] read col-wise
    #     [1, 0, 0],   # col2: new +Z(White) = mesh +Y → [0,1,0] read col-wise
    # ])
    logging.info(f"Color-axis correction: Red=+X, Blue=+Y(-Y=Green), White=+Z")

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    color_corrector = CubeColorCorrector(half_size=0.0275, debug=(debug >= 1))

    cam = RealSenseCamera(width=args.cam_width, height=args.cam_height, fps=args.cam_fps,
                          exposure=args.exposure)
    selector = BBoxSelector()
    fps_counter = FPSCounter()
    motion = MotionPredictor()

    # 丢失恢复
    mesh_diameter = est.diameter
    recovery_thresh = args.recovery_thresh if args.recovery_thresh > 0 else mesh_diameter * 0.5
    mesh_pts_np = np.asarray(mesh.vertices)

    # ZMQ publisher for cube pose
    zmq_context = None
    zmq_socket = None
    if args.zmq_port > 0:
        zmq_context = zmq.Context()
        zmq_socket = zmq_context.socket(zmq.PUB)
        zmq_socket.bind(f"tcp://*:{args.zmq_port}")
        logging.info(f"ZMQ cube pose publisher on port {args.zmq_port}")

    # World frame calibrator (AprilTag)
    # 与 cube_world_observer.py 中的 WORLD_FRAME_CORRECTION 一致
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
    REVERT_LIMIT = 3
    frame_idx = 0
    need_register = True

    logging.info(f"Motion prediction: {'ON' if args.motion_predict else 'OFF'}")
    logging.info(f"Recovery: thresh={recovery_thresh:.4f}m, diameter={mesh_diameter:.4f}m")
    logging.info("Controls: 'r' = re-select, 'w' = re-calibrate world, 'q' = quit, 'p' = profiling")

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

                # 尝试用上次好的 pose 投影区域自动 re-register
                auto_mask = None
                if last_good_pose is not None:
                    auto_mask = make_mask_from_pose(
                        last_good_pose, mesh_pts_np, cam.K, cam.H, cam.W, pad_ratio=0.5)

                if auto_mask is not None and auto_mask.sum() > 100:
                    logging.info("Auto re-register from last good pose...")
                    mask = auto_mask
                else:
                    logging.info("Select object with mouse, press ENTER to confirm")
                    mask = selector.select(color)
                    if mask is None:
                        continue

                # register 比 track 显存开销大很多，先释放缓存防 OOM
                torch.cuda.empty_cache()

                t0 = time.time()
                pose = est.register(K=cam.K, rgb=color, depth=depth,
                                    ob_mask=mask, iteration=args.est_refine_iter)
                logging.info(f"Pose estimated in {time.time()-t0:.2f}s")

                # 颜色-轴校正: 从 24 个对称旋转候选中选出颜色匹配最佳的
                tf_centered = est.get_tf_to_centered_mesh().data.cpu().numpy()
                pose, pose_centered_np = color_corrector.correct(
                    pose, color, cam.K, tf_centered)
                est.pose_last = torch.as_tensor(
                    pose_centered_np, device='cuda', dtype=torch.float
                ).reshape(1, 4, 4)

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
                is_bad = (trans_jump > recovery_thresh) or (rot_jump > np.deg2rad(45)) or (z < 0.02) or (z > 3.0)

                if is_bad:
                    lost_count += 1
                    if lost_count <= REVERT_LIMIT:
                        # 回退到上次好的 pose 重试
                        if last_good_pose_centered is not None:
                            est.pose_last = last_good_pose_centered.clone().reshape(1, 4, 4)
                        motion.reset()
                    else:
                        # 彻底丢失 → 自动 re-register
                        print(f"[RECOVERY] Lost tracking ({lost_count} bad frames), re-registering...")
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
                    lost_count = 0
                    last_good_pose = pose.copy()
                    last_good_pose_centered = est.pose_last.detach().clone()

            # World frame calibration (feed every frame until fixed)
            if world_calibrator is not None and not world_calibrator.is_fixed:
                # RealSense color is RGB, convert to gray for AprilTag
                gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                just_fixed = world_calibrator.feed_frame(gray)
                if just_fixed:
                    logging.getLogger().setLevel(logging.INFO)
                    logging.info("World frame calibrated!")
                    logging.getLogger().setLevel(logging.WARNING)

            # 应用颜色-坐标轴修正
            corrected_pose = pose @ R_color_correction

            # Publish cube pose via ZMQ (same format as cube_world_observer.py)
            if zmq_socket is not None:
                profiler.tick('zmq_publish')

                # Transform to world frame if calibrated (使用修正后的 pose)
                if world_calibrator is not None and world_calibrator.is_fixed:
                    R_world, t_world = world_calibrator.cam_to_world(corrected_pose)
                    world_fixed = True
                elif world_calibrator is None:
                    # --no_world_frame: publish camera-frame pose directly
                    R_world = corrected_pose[:3, :3]
                    t_world = corrected_pose[:3, 3]
                    world_fixed = True
                else:
                    # World frame not yet calibrated, publish camera-frame as fallback
                    R_world = corrected_pose[:3, :3]
                    t_world = corrected_pose[:3, 3]
                    world_fixed = False

                quat_xyzw = Rotation.from_matrix(R_world).as_quat()  # (x, y, z, w)
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
                center_pose = corrected_pose @ inv_to_origin  # OBB 变换用于 bbox 绘制
                # 先转 BGR 一次，后续 draw 直接在 BGR 上 in-place 操作，避免反复转换+copy
                vis_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                draw_posed_3d_box(cam.K, img=vis_bgr, ob_in_cam=center_pose, bbox=bbox)
                # 坐标轴: 用 corrected_pose 的旋转 (颜色-轴绑定), 平移取 center_pose (视觉中心)
                # inv_to_origin 包含 OBB 旋转会破坏颜色-轴对齐, 所以坐标轴不能用 center_pose
                axis_pose = corrected_pose.copy()
                axis_pose[:3, 3] = center_pose[:3, 3]  # 平移用 OBB 中心, 旋转用颜色修正
                # 坐标轴颜色: X=红(Red), Y=蓝(Blue, 对面绿), Z=白(White)
                vis_bgr = draw_xyz_axis(vis_bgr, ob_in_cam=axis_pose, scale=0.1,
                              K=cam.K, thickness=3, transparency=0, is_input_rgb=False,
                              axis_colors=((0,0,255), (255,0,0), (255,255,255)))
                # 在箭头端点附近标注 X/Y/Z 文字
                _axis_pts = np.array([[0.1,0,0,1],[0,0.1,0,1],[0,0,0.1,1]], dtype=np.float64)
                _pts_cam = (axis_pose @ _axis_pts.T)[:3, :]
                _proj = cam.K @ _pts_cam
                _uvs = (_proj[:2, :] / _proj[2:3, :]).T.astype(int)
                for _label, _uv, _clr in [("X", _uvs[0], (0,0,255)), ("Y", _uvs[1], (255,0,0)), ("Z", _uvs[2], (255,255,255))]:
                    cv2.putText(vis_bgr, _label, tuple(_uv + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _clr, 2)

                # 提取姿态信息: 位置 + 欧拉角 (世界坐标系优先, 未标定时用相机坐标系)
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
                if world_calibrator is not None:
                    if world_calibrator.is_fixed:
                        label += " W:OK"
                    else:
                        n = len(world_calibrator._samples_R)
                        label += f" W:{n}/{world_calibrator.sample_target}"
                cv2.putText(vis_bgr, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # 位置信息
                pos_text = f"Pos({frame_label}): x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} m"
                cv2.putText(vis_bgr, pos_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # 欧拉角信息 (XYZ convention, degrees)
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
                        # 画 tag 边框
                        corners = det.corners.astype(int)
                        for i in range(4):
                            cv2.line(vis_bgr, tuple(corners[i]), tuple(corners[(i+1)%4]),
                                     (0, 255, 0), 2)
                        # 标注 tag ID
                        ctr = det.center.astype(int)
                        cv2.putText(vis_bgr, f"id:{det.tag_id}", (ctr[0]-20, ctr[1]-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # 绘制 tag 坐标轴 (XYZ), 应用 WORLD_FRAME_CORRECTION 与 cube_world_observer 对齐
                        if det.pose_R is not None and det.pose_t is not None:
                            tag_R = det.pose_R @ WORLD_FRAME_CORRECTION.T  # 修正后的世界坐标轴
                            tag_t = det.pose_t.flatten()  # 3
                            axis_len = args.world_tag_size * 1.0  # 轴长 = tag 尺寸
                            # 3D 轴端点 (tag 坐标系原点 + 各轴方向)
                            origin = tag_t
                            x_end = tag_t + tag_R[:, 0] * axis_len
                            y_end = tag_t + tag_R[:, 1] * axis_len
                            z_end = tag_t + tag_R[:, 2] * axis_len
                            # 投影到图像
                            def proj_pt(pt3d):
                                u = fx * pt3d[0] / pt3d[2] + cx
                                v = fy * pt3d[1] / pt3d[2] + cy
                                return (int(u), int(v))
                            o2d = proj_pt(origin)
                            x2d = proj_pt(x_end)
                            y2d = proj_pt(y_end)
                            z2d = proj_pt(z_end)
                            cv2.arrowedLine(vis_bgr, o2d, x2d, (0, 0, 255), 2, tipLength=0.15)  # X=红
                            cv2.arrowedLine(vis_bgr, o2d, y2d, (0, 255, 0), 2, tipLength=0.15)  # Y=绿
                            cv2.arrowedLine(vis_bgr, o2d, z2d, (255, 0, 0), 2, tipLength=0.15)  # Z=蓝
                            cv2.putText(vis_bgr, "X", (x2d[0]+3, x2d[1]-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            cv2.putText(vis_bgr, "Y", (y2d[0]+3, y2d[1]-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            cv2.putText(vis_bgr, "Z", (z2d[0]+3, z2d[1]-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            profiler.tick('imshow')
            if debug >= 1:
                cv2.imshow('FoundationPose Live Tracking', vis_bgr)

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
                need_register = True
                last_good_pose = None
            elif key == ord('w'):
                # 重新标定世界坐标系
                if world_calibrator is not None:
                    world_calibrator.reset()
            elif key == ord('p'):
                # 手动打印 profiling
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
