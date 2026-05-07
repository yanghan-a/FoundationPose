"""
FoundationPose 单物体追踪 on Intel RealSense.

通用单物体版本 (与 run_track_cube54.py 同结构, 但 mesh 必填, 不预设 cube54).

  - 通过 --mesh_file 加载 1 个 mesh
  - 首帧鼠标框选 2D bbox 作为初始 mask -> est.register(...)
  - 后续帧 est.track_one(...)

用法:


  python track_single_publish.py --mesh_file demo_data/cube54/mesh/textured_cube54.obj --track_refine_iter 1 --color_exposure 5000 --depth_exposure 5000 --color_gain 80 --depth_max 0.6 --cam_width 640 --cam_height 480 --cam_fps 90 --score_vis_topk 5 --auto_bbox --debug 1


按键:
  q  退出
  r  重新框选 (触发 re-register)

debug 等级:
  0  tracking 阶段静默 (无窗口, Ctrl+C 退出)
  1  tracking 阶段显示窗口 (q 退出 / r 重新框选)
两档下 register 阶段都会在 debug_dir 下保存 vis_score_NNN.png 用于检查最高分匹配.
tracking 阶段始终不写文件、不影响帧率.
"""

from estimater import *
from datareader import *
import argparse
import json
import re
import time
import threading
from collections import defaultdict
import pyrealsense2 as rs

# AprilTag world frame + ZMQ publisher 依赖. 失败给明确安装提示而不是默认 ImportError.
try:
  from pupil_apriltags import Detector as AprilTagDetector
except ImportError as e:
  raise ImportError(
      "pupil_apriltags not installed. Run: pip install pupil-apriltags") from e
try:
  import zmq
except ImportError as e:
  raise ImportError("pyzmq not installed. Run: pip install pyzmq") from e


# ---------------------------------------------------------------------------
# AprilTag 世界系: 启动采样 N 帧 + 固定; 复刻 cube_world_observer.py 的约定.
# ---------------------------------------------------------------------------
WORLD_TAG_ID = 0
WORLD_TAG_SIZE = 0.048   # 米
WORLD_SAMPLE_FRAMES = 100

# WORLD_FRAME_CORRECTION = R_flip @ R_y(angle), 其中:
#   R_flip = diag([1, -1, -1])  (绕 X 轴 180°, det=+1)
#   R_y(angle) = palm→tag100 的 Y 轴补偿
# 应用方式: avg_R = avg_R @ correction_R.T
# 含义: correction_R 表达 "AprilTag 系 → 目标世界系" 的旋转, 右乘 .T 把
#   world_in_cam 的旋转部分修正到目标轴系下, 之后 inv(world_in_cam) @ ob_in_cam
#   出来就是物体在矫正后世界系下的 pose.
PALM_TO_TAG100_Y_DEG = 10.0


def _build_world_frame_correction(angle_deg):
  """Build WORLD_FRAME_CORRECTION = R_flip @ R_y(angle).

  Args:
      angle_deg: Y-axis rotation in degrees (palm→tag100 compensation)

  Returns:
      3x3 proper rotation matrix (det=+1)
  """
  theta = np.radians(angle_deg)
  c, s = np.cos(theta), np.sin(theta)
  R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
  R_flip = np.diag([1.0, -1.0, -1.0])
  R = R_flip @ R_y
  det = np.linalg.det(R)
  assert abs(det - 1.0) < 1e-6, f"WORLD_FRAME_CORRECTION det={det:.6f}, expected +1"
  return R


WORLD_FRAME_CORRECTION = _build_world_frame_correction(PALM_TO_TAG100_Y_DEG)


def _mat_to_quat_xyzw(R):
  """Rotation matrix -> quaternion (x, y, z, w) via Shepperd's method.

  Numerically stable across all branches; no scipy per-frame overhead.
  """
  trace = R[0, 0] + R[1, 1] + R[2, 2]
  if trace > 0:
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (R[2, 1] - R[1, 2]) * s
    y = (R[0, 2] - R[2, 0]) * s
    z = (R[1, 0] - R[0, 1]) * s
  elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
    w = (R[2, 1] - R[1, 2]) / s
    x = 0.25 * s
    y = (R[0, 1] + R[1, 0]) / s
    z = (R[0, 2] + R[2, 0]) / s
  elif R[1, 1] > R[2, 2]:
    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
    w = (R[0, 2] - R[2, 0]) / s
    x = (R[0, 1] + R[1, 0]) / s
    y = 0.25 * s
    z = (R[1, 2] + R[2, 1]) / s
  else:
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    w = (R[1, 0] - R[0, 1]) / s
    x = (R[0, 2] + R[2, 0]) / s
    y = (R[1, 2] + R[2, 1]) / s
    z = 0.25 * s
  q = np.array([x, y, z, w])
  q /= np.linalg.norm(q)
  return q


def _quat_to_euler_xyz_deg(q_xyzw):
  """Quaternion (x,y,z,w) -> intrinsic XYZ Euler angles in degrees."""
  x, y, z, w = q_xyzw
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = np.arctan2(sinr_cosp, cosr_cosp)
  sinp = 2.0 * (w * y - z * x)
  sinp = np.clip(sinp, -1.0, 1.0)
  pitch = np.arcsin(sinp)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = np.arctan2(siny_cosp, cosy_cosp)
  return np.degrees(np.array([roll, pitch, yaw]))


def _quat_average(quats):
  """Hemisphere-aligned mean of unit quaternions, then normalize.

  Avoids quaternion sign ambiguity (q == -q) by flipping any sample whose
  dot product with the first sample is negative before averaging.
  """
  quats = np.array(quats, dtype=np.float64)
  for i in range(1, len(quats)):
    if np.dot(quats[i], quats[0]) < 0:
      quats[i] = -quats[i]
  q = quats.mean(axis=0)
  q /= np.linalg.norm(q)
  return q


class FrameProfiler:
  """每阶段耗时统计 (CUDA sync 后取真实 GPU 耗时), 按帧聚合."""

  def __init__(self, window=60, use_cuda_sync=True):
    self.window = window
    self.use_cuda_sync = use_cuda_sync
    self.history = defaultdict(list)
    self._pending_name = None
    self._pending_t0 = None
    self._cur_frame = defaultdict(float)

  def tick(self, name):
    now = self._sync_time()
    if self._pending_name is not None:
      self._cur_frame[self._pending_name] += now - self._pending_t0
    self._pending_name = name
    self._pending_t0 = now

  def tock(self, name=None):
    if self._pending_name is None:
      return
    now = self._sync_time()
    self._cur_frame[self._pending_name] += now - self._pending_t0
    self._pending_name = None

  def frame_mark(self):
    self.tock()
    all_stages = set(self.history.keys()) | set(self._cur_frame.keys())
    for s in all_stages:
      v = self._cur_frame.get(s, 0.0)
      h = self.history[s]
      h.append(v)
      if len(h) > self.window:
        h.pop(0)
    self._cur_frame = defaultdict(float)

  def _sync_time(self):
    if self.use_cuda_sync and torch.cuda.is_available():
      torch.cuda.synchronize()
    return time.perf_counter()

  def summary(self):
    out = []
    for name, vals in self.history.items():
      if vals:
        out.append((name, sum(vals) / len(vals) * 1000.0))
    out.sort(key=lambda x: -x[1])
    return out

  def summary_str(self):
    lines = []
    total = 0.0
    for name, ms in self.summary():
      lines.append(f"  {name:28s} {ms:7.2f} ms")
      total += ms
    lines.insert(0, f"  {'TOTAL (real per-frame)':28s} {total:7.2f} ms")
    return "\n".join(lines)


class RealSenseRGBD:
  """RealSense 异步抓帧, depth 对齐到 color, 单位米."""

  def __init__(self, width=640, height=480, fps=30, depth_max=3.0,
               color_exposure=None, depth_exposure=None,
               color_gain=None, white_balance=None):
    self.pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = self.pipeline.start(cfg)
    self.align = rs.align(rs.stream.color)
    self.product_name = profile.get_device().get_info(rs.camera_info.name)

    color_sensor = None
    depth_sensor = None
    for s in profile.get_device().query_sensors():
      for sp in s.get_stream_profiles():
        st = sp.stream_type()
        if st == rs.stream.color and color_sensor is None:
          color_sensor = s
        elif st == rs.stream.depth and depth_sensor is None:
          depth_sensor = s
      if color_sensor is not None and depth_sensor is not None:
        break
    if depth_sensor is None:
      raise RuntimeError("no depth sensor found on device")
    if color_sensor is None:
      raise RuntimeError("no color sensor found on device")
    if color_sensor is depth_sensor:
      print(f"[RealSense] color & depth share one sensor (D405-style)")

    try:
      self.depth_scale = depth_sensor.as_depth_sensor().get_depth_scale()
    except Exception:
      try:
        self.depth_scale = depth_sensor.get_option(rs.option.depth_units)
      except Exception:
        self.depth_scale = 0.001
    self.depth_max = float(depth_max)

    if color_exposure is None:
      color_sensor.set_option(rs.option.enable_auto_exposure, 1)
      print(f"[RealSense] color: AE on")
    else:
      color_sensor.set_option(rs.option.enable_auto_exposure, 0)
      color_sensor.set_option(rs.option.exposure, float(color_exposure))
      print(f"[RealSense] color: manual exposure {color_exposure}us")
    if color_gain is not None:
      color_sensor.set_option(rs.option.gain, float(color_gain))
      print(f"[RealSense] color: manual gain {color_gain}")
    if white_balance is None:
      try:
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
      except Exception:
        pass
    else:
      try:
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, float(white_balance))
        print(f"[RealSense] color: manual WB {white_balance}K")
      except Exception:
        pass

    if depth_exposure is None:
      try:
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        print(f"[RealSense] depth: AE on")
      except Exception:
        pass
    else:
      try:
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        depth_sensor.set_option(rs.option.exposure, float(depth_exposure))
        print(f"[RealSense] depth: manual exposure {depth_exposure}us")
      except Exception:
        pass

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    self.K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1.0],
    ])
    self.W, self.H = intr.width, intr.height
    print(f"[RealSense] {self.product_name} {self.W}x{self.H}@{fps} "
          f"depth_scale={self.depth_scale} depth_max={self.depth_max}m")
    print(f"[RealSense] K=\n{self.K}")

    # 预热 N 帧让自动曝光稳定. 30 帧偏多, 10 帧实测够用; 手动曝光时可以更少.
    for _ in range(10):
      self.pipeline.wait_for_frames()

    self._latest_color = None
    self._latest_depth = None
    self._latest_id = 0
    self._lock = threading.Lock()
    self._running = True
    self._thread = threading.Thread(target=self._grab_loop, daemon=True)
    self._thread.start()

  def _grab_loop(self):
    while self._running:
      try:
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
          continue
        rgb = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
        d = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        d[(d < 0.001) | (d > self.depth_max)] = 0
        with self._lock:
          self._latest_color = rgb
          self._latest_depth = d
          self._latest_id += 1
      except Exception as e:
        print(f"[RealSense] grab error: {e}")

  def get(self):
    with self._lock:
      if self._latest_color is None:
        return None, None, 0
      return self._latest_color, self._latest_depth, self._latest_id

  def get_blocking(self):
    while True:
      c, d, fid = self.get()
      if c is not None:
        return c, d, fid
      time.sleep(0.001)

  def stop(self):
    self._running = False
    self._thread.join(timeout=2)
    self.pipeline.stop()


class BBoxSelector:
  def __init__(self):
    self.p0 = None
    self.p1 = None
    self.dragging = False
    self.done = False

  def _cb(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.p0 = (x, y)
      self.p1 = (x, y)
      self.dragging = True
      self.done = False
    elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
      self.p1 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
      self.p1 = (x, y)
      self.dragging = False
      self.done = True

  def select(self, rgb):
    self.p0 = self.p1 = None
    self.dragging = False
    self.done = False
    win = "Drag a box around the object, then press ENTER (ESC to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, self._cb)
    bgr = rgb[..., ::-1].copy()
    while True:
      show = bgr.copy()
      if self.p0 and self.p1:
        cv2.rectangle(show, self.p0, self.p1, (0, 255, 0), 2)
      cv2.imshow(win, show)
      key = cv2.waitKey(30) & 0xFF
      if key == 13 and self.done:
        break
      if key == 27:
        cv2.destroyWindow(win)
        return None
    cv2.destroyWindow(win)

    x1, x2 = sorted([self.p0[0], self.p1[0]])
    y1, y2 = sorted([self.p0[1], self.p1[1]])
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return mask


class WorldFrameSampler:
  """启动时采样 N 帧 AprilTag pose, 四元数平均后固定一个 "world" 系.

  约定:
    world_pose = (R_world_in_cam, t_world_in_cam)
      - R_world_in_cam: 世界三轴(列)在相机系下的方向
      - t_world_in_cam: 世界原点在相机系下的位置(米)
    transform 公式: ob_in_world = inv(world_in_cam) @ ob_in_cam.

  矫正应用方式 (右乘 correction_R.T) 完全照搬 cube_world_observer.py:
    correction_R 表达 "AprilTag 系 → 目标世界系" 的旋转,
    avg_R = avg_R @ correction_R.T 就是把 world_in_cam 修正到目标轴系下.

  接口:
    start()                      重置采样状态(供 'w' 键调用)
    step(rgb)  -> (detected, corners or None)
                                 喂一帧彩图(RGB), 采样阶段累计 R/t,
                                 满帧后 _finalize 设 self.world_pose 与 _fixed=True;
                                 固定后是 no-op, 返回 (False, None).
    transform(ob_in_cam)         4x4 -> 4x4 ob_in_world, world_pose=None 返回 None.
  """

  def __init__(self, K, tag_id=WORLD_TAG_ID, tag_size=WORLD_TAG_SIZE,
               sample_target=WORLD_SAMPLE_FRAMES,
               correction=WORLD_FRAME_CORRECTION,
               quad_decimate=2.0):
    self._fxfycxcy = (float(K[0, 0]), float(K[1, 1]),
                      float(K[0, 2]), float(K[1, 2]))
    self.tag_id = int(tag_id)
    self.tag_size = float(tag_size)
    self.sample_target = int(sample_target)
    self.correction = correction
    # quad_decimate=2.0: RealSense 640x480 比工业相机分辨率低, 比 cube_world_observer
    # 用的 3.0 更稳, 又不至于慢到拖累采样阶段.
    self.detector = AprilTagDetector(
        families="tag36h11", nthreads=4, quad_decimate=quad_decimate,
        quad_sigma=0.0, decode_sharpening=0.25)

    self._samples_R = []
    self._samples_t = []
    self._fixed = False
    self.world_pose = None      # (R_world_in_cam, t_world_in_cam) — 矫正后
    self.world_pose_4x4 = None  # 同 world_pose 的 4x4 形式, 给 draw_xyz_axis 用
    self._inv_cache = None      # 缓存 (R^T, -R^T @ t) 避免每帧重算

  @property
  def fixed(self):
    return self._fixed

  @property
  def n_collected(self):
    return len(self._samples_R)

  def start(self):
    self._samples_R = []
    self._samples_t = []
    self._fixed = False
    self.world_pose = None
    self.world_pose_4x4 = None
    self._inv_cache = None

  def step(self, rgb):
    """喂一帧彩图(RGB). 固定后立即 no-op (返回 (False, None))."""
    if self._fixed:
      return False, None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    results = self.detector.detect(
        gray, estimate_tag_pose=True,
        camera_params=self._fxfycxcy, tag_size=self.tag_size)
    for r in results:
      if r.tag_id == self.tag_id:
        self._samples_R.append(np.asarray(r.pose_R, dtype=np.float64))
        self._samples_t.append(np.asarray(r.pose_t, dtype=np.float64).flatten())
        if len(self._samples_R) >= self.sample_target:
          self._finalize()
        return True, r.corners
    return False, None

  def _finalize(self):
    if len(self._samples_R) < 10:
      print(f"[world] _finalize: only {len(self._samples_R)} samples, "
            f"not enough — keeping sampling state open")
      self._samples_R = []
      self._samples_t = []
      return
    quats = [_mat_to_quat_xyzw(R) for R in self._samples_R]
    avg_q = _quat_average(quats)
    avg_R = self._quat_to_mat(avg_q)
    avg_t = np.mean(self._samples_t, axis=0)

    if self.correction is not None:
      avg_R = avg_R @ np.asarray(self.correction).T
      print(f"[world] applied frame correction "
            f"(det={np.linalg.det(self.correction):.3f})")

    self.world_pose = (avg_R, avg_t)
    # 4x4 形式预先建好, draw_xyz_axis 每帧直接复用, 避免每帧 np.eye+切片赋值.
    self.world_pose_4x4 = np.eye(4)
    self.world_pose_4x4[:3, :3] = avg_R
    self.world_pose_4x4[:3, 3] = avg_t
    R_T = avg_R.T
    self._inv_cache = (R_T, -R_T @ avg_t)
    self._fixed = True
    print(f"[world] FIXED after {len(self._samples_R)} samples. "
          f"Press 'w' to resample.")

  @staticmethod
  def _quat_to_mat(q_xyzw):
    """xyzw 四元数 -> 3x3 旋转矩阵."""
    x, y, z, w = q_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])

  def transform(self, ob_in_cam):
    """ob_in_cam (4x4) -> ob_in_world (4x4). 未固定返回 None."""
    if self.world_pose is None or self._inv_cache is None:
      return None
    R_T, neg_R_T_t = self._inv_cache
    ob_in_world = np.eye(4)
    ob_in_world[:3, :3] = R_T @ ob_in_cam[:3, :3]
    ob_in_world[:3, 3] = R_T @ ob_in_cam[:3, 3] + neg_R_T_t
    return ob_in_world

  def world_axes_in_cam(self, axis_length=0.05):
    """世界三轴端点在相机系下的 4 个 3D 点 (origin + xyz_end), 用于可视化."""
    if self.world_pose is None:
      return None
    R, t = self.world_pose
    origin = t.reshape(3)
    pts = np.stack([
        origin,
        origin + axis_length * R[:, 0],
        origin + axis_length * R[:, 1],
        origin + axis_length * R[:, 2],
    ], axis=0)
    return pts


# register 完成后会在 debug_dir 留下这些中间产物, 我们只想留 vis_score.png,
# 其余清理掉以保持目录干净.
_REGISTER_INTERMEDIATE_FILES = [
    'ob_mask.png', 'color.png', 'depth.png',
    'scene_raw.ply', 'scene_complete.ply',
    'init_center.ply', 'vis_refiner.png', 'model_tf.obj',
]

_VIS_SCORE_PATTERN = re.compile(r'^vis_score_(\d+)\.png$')


def next_register_count(debug_dir):
  """扫描 debug_dir 找已有的 vis_score_NNN.png, 返回下一个可用编号 (max+1, 没有则 0).

  这样每次新启动脚本时编号自动从历史最大值 +1 开始, 不会覆盖之前 session 的打分图;
  同一次运行内多次按 r 重新框选则继续顺延. 用户可以通过删除 debug_dir 重置编号.
  """
  if not os.path.isdir(debug_dir):
    return 0
  max_idx = -1
  for fn in os.listdir(debug_dir):
    m = _VIS_SCORE_PATTERN.match(fn)
    if m:
      idx = int(m.group(1))
      if idx > max_idx:
        max_idx = idx
  return max_idx + 1


def keep_only_score_vis(debug_dir, register_count):
  """register 后只保留 vis_score.png 并按次累计重命名为 vis_score_NNN.png.

  estimater.debug=2 时会写一堆中间产物, 这里全部清理只留打分图.
  按 register_count 累计命名, 这样多次按 r 重新框选都能各留一张, 不会互相覆盖.
  """
  src = os.path.join(debug_dir, 'vis_score.png')
  if os.path.exists(src):
    dst = os.path.join(debug_dir, f'vis_score_{register_count:03d}.png')
    os.replace(src, dst)
    print(f"[register] saved score visualization -> {dst}")
  else:
    print(f"[register] WARNING: vis_score.png not generated by estimater")
  for fn in _REGISTER_INTERMEDIATE_FILES:
    p = os.path.join(debug_dir, fn)
    if os.path.exists(p):
      try:
        os.remove(p)
      except OSError:
        pass


def make_center_bbox_mask(H, W, h_frac=0.5, w_frac=0.5):
  """画面中央占长宽各 frac 的矩形 mask. 用于 --auto_bbox 模式跳过手动框选.
  默认 0.5 表示 mask 占整图正中 H/2 x W/2 区域.
  """
  mask = np.zeros((H, W), dtype=bool)
  cy, cx = H // 2, W // 2
  hh = int(H * h_frac / 2)
  hw = int(W * w_frac / 2)
  mask[cy - hh:cy + hh, cx - hw:cx + hw] = True
  return mask


def warmup_estimator(est, K, H, W):
  """用 dummy 数据跑一次 1-iteration 的 register, 把 nvdiffrast / refiner / scorer 的
  CUDA kernel JIT 编译和 cudnn benchmark 一次性付清. 这样真正用户框选后的第一次
  register 就能直接进入稳态 (~0.5s), 不再被首次编译开销拖累 1-3 秒.

  注意: register 内部会改 est.pose_last / poses / scores / ob_mask / K / H / W,
  跑完后 reset est.pose_last = None, 真正的用户 register 会重新走完整流程覆盖所有状态.
  """
  rgb_dummy = np.zeros((H, W, 3), dtype=np.uint8)
  depth_dummy = np.full((H, W), 0.5, dtype=np.float32)   # 0.5m 平面深度, 保证有效
  mask_dummy = np.zeros((H, W), dtype=bool)
  mask_dummy[H // 4:3 * H // 4, W // 4:3 * W // 4] = True

  print('[warmup] dummy register to JIT-compile CUDA kernels...')
  prev_debug = est.debug
  est.debug = 0   # 不写任何中间产物
  t0 = time.time()
  try:
    est.register(K=K, rgb=rgb_dummy, depth=depth_dummy,
                 ob_mask=mask_dummy, iteration=1)
  except Exception as e:
    print(f'[warmup] register failed (ignored): {e}')
  finally:
    est.debug = prev_debug
  est.pose_last = None    # 防止污染真正 tracking 起点
  print(f'[warmup] done in {time.time() - t0:.2f}s')


def main():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--mesh_file', type=str, required=True,
                      help='待追踪物体的 mesh 文件 (.obj/.ply)')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1, choices=[0, 1],
                      help='0=tracking 阶段静默(无窗口, Ctrl+C 退出); '
                           '1=tracking 阶段显示窗口. 两档 register 都会保存 vis_score_NNN.png.')
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_single')
  parser.add_argument('--cam_width', type=int, default=640)
  parser.add_argument('--cam_height', type=int, default=480)
  parser.add_argument('--cam_fps', type=int, default=30)
  parser.add_argument('--depth_max', type=float, default=3.0,
                      help='超过此距离(米)的深度被清零. D435i 用 3.0, D405 建议 0.6')
  parser.add_argument('--color_exposure', type=int, default=None,
                      help='color 手动曝光(us), 不传 = 自动曝光')
  parser.add_argument('--depth_exposure', type=int, default=None,
                      help='depth 手动曝光(us), 不传 = 自动')
  parser.add_argument('--color_gain', type=int, default=None,
                      help='color 手动增益, 配合短曝光使用以保持亮度')
  parser.add_argument('--white_balance', type=int, default=None,
                      help='color 手动白平衡(K), 例如 4000~6500, 不传 = 自动')
  parser.add_argument('--show_every', type=int, default=1,
                      help='每 N 帧 imshow 一次 (1=每帧都显示), 仅 --debug 1 时生效')
  parser.add_argument('--score_vis_topk', type=int, default=10,
                      help='vis_score.png 里画 top-N 得分的 pose. 0 或负值表示画全部 252 个.')
  parser.add_argument('--auto_bbox', action='store_true',
                      help='跳过手动框选, 用画面中央 H/2 x W/2 的矩形作为初始 mask. '
                           '使用前把目标物体放在画面中央并距离相机适中.')
  # AprilTag 世界系 + ZMQ 发布相关参数
  parser.add_argument('--world_samples', type=int, default=WORLD_SAMPLE_FRAMES,
                      help=f'AprilTag 启动采样帧数. 默认 {WORLD_SAMPLE_FRAMES}.')
  parser.add_argument('--world_tag_id', type=int, default=WORLD_TAG_ID,
                      help=f'AprilTag(tag36h11) 用作世界系的 tag id. 默认 {WORLD_TAG_ID}.')
  parser.add_argument('--world_tag_size', type=float, default=WORLD_TAG_SIZE,
                      help=f'AprilTag 物理边长(米). 默认 {WORLD_TAG_SIZE}.')
  parser.add_argument('--world_correction_y_deg', type=float,
                      default=PALM_TO_TAG100_Y_DEG,
                      help=f'世界系矫正: 在 R_flip(绕X 180°)之前先绕 Y 旋转的角度(度). '
                           f'默认 {PALM_TO_TAG100_Y_DEG} 是 cube_world_observer 在 MV 工业相机下'
                           f'的标定值, RealSense 安装位姿不同的话先试 0 看 raw AprilTag 系再调.')
  parser.add_argument('--no_world_correction', action='store_true',
                      help='完全跳过世界系矫正, 直接用 AprilTag 系作为世界系. '
                           '想先看 raw 朝向再决定要不要补角度时打开.')
  parser.add_argument('--zmq_port', type=int, default=5555,
                      help='ZMQ PUB 端口, 发布物体在 AprilTag 世界系下的 pose. 默认 5555.')
  parser.add_argument('--no_zmq', action='store_true',
                      help='关闭 ZMQ 发布, 仅做屏幕显示和 stdout 打印.')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  logging.getLogger().setLevel(logging.WARNING)

  debug = args.debug
  debug_dir = args.debug_dir
  os.makedirs(debug_dir, exist_ok=True)
  # 从 debug_dir 已有的 vis_score_NNN.png 取最大编号 +1, 历次 session 的打分图全部保留.
  register_count = next_register_count(debug_dir)
  print(f"[init] vis_score numbering starts at {register_count:03d} "
        f"(debug_dir={debug_dir})")

  mesh = trimesh.load(args.mesh_file)
  # 用 mesh.bounds 画 AABB, 跳过 trimesh.bounds.oriented_bounds():
  # 后者对完美对称的物体 (例如正方体) 会返回退化 OBB,
  # 导致后续 pose @ inv(to_origin) 把显示的物体坐标系转到 OBB 系而非 obj 原生系.
  bbox = np.stack([mesh.bounds[0], mesh.bounds[1]], axis=0).reshape(2, 3)
  print(f"[init] loaded mesh: {args.mesh_file}, extents={mesh.extents}")

  scorer = ScorePredictor()
  scorer.vis_topk = args.score_vis_topk if args.score_vis_topk > 0 else None
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  # estimater 默认静默 (debug=0); register 时临时设为 2 让它生成 vis_score.png 等中间产物,
  # 之后立刻切回 0, 这样 track_one 内部不会再渲染 vis (refiner.predict 的 get_vis=False),
  # tracking 帧率不受 debug 拖累.
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                       mesh=mesh, scorer=scorer, refiner=refiner,
                       debug_dir=debug_dir, debug=0, glctx=glctx)
  print("[init] estimator ready")

  cam = RealSenseRGBD(width=args.cam_width, height=args.cam_height,
                      fps=args.cam_fps, depth_max=args.depth_max,
                      color_exposure=args.color_exposure,
                      depth_exposure=args.depth_exposure,
                      color_gain=args.color_gain,
                      white_balance=args.white_balance)
  K = cam.K

  # 把首次 GPU JIT 编译开销 (~1-3s) 在用户框选前一次性付清, 真正第一次 register 才能快
  warmup_estimator(est, K, cam.H, cam.W)

  selector = BBoxSelector()

  # AprilTag 世界系采样器: 启动时采 N 帧固定, 之后每帧 step() 是 no-op.
  if args.no_world_correction:
    correction = None
    print(f"[world] correction DISABLED (--no_world_correction), "
          f"using raw AprilTag frame as world frame")
  else:
    correction = _build_world_frame_correction(args.world_correction_y_deg)
    print(f"[world] correction = R_flip @ R_y({args.world_correction_y_deg:+.2f}°)")
  sampler = WorldFrameSampler(K, tag_id=args.world_tag_id,
                              tag_size=args.world_tag_size,
                              sample_target=args.world_samples,
                              correction=correction)
  sampler.start()
  print(f"[world] sampling enabled: tag36h11 id={args.world_tag_id} "
        f"size={args.world_tag_size}m target={args.world_samples} frames")

  # ZMQ PUB socket: 物体在世界系下的 pose 单向广播.
  zmq_ctx = None
  zmq_sock = None
  if not args.no_zmq:
    zmq_ctx = zmq.Context()
    zmq_sock = zmq_ctx.socket(zmq.PUB)
    zmq_sock.bind(f"tcp://*:{args.zmq_port}")
    print(f"[zmq] PUB on tcp://*:{args.zmq_port}")
  else:
    print(f"[zmq] disabled (--no_zmq)")

  pose = None
  frame_id = 0
  need_register = True
  last_cam_id = -1
  register_bbox_2d = None    # (x1,y1,x2,y2) register 时 mask 的外接矩形, 用于可视化检查
  last_world_xyz_log = 0.0   # 上次 stdout 打印 ob_in_world 的时间戳, 控制频率

  fps_window = []
  fps_window_size = 30
  last_print_t = time.time()
  fps_smoothed = 0.0

  profiler = FrameProfiler(window=60, use_cuda_sync=True)
  PROFILE_PRINT_INTERVAL = 2.0
  last_profile_print_t = time.time()

  try:
    while True:
      t_frame_start = time.time()

      profiler.tick('cam_get')
      if need_register:
        color, depth, cam_id = cam.get_blocking()
      else:
        color, depth, cam_id = cam.get()
        if color is None or cam_id == last_cam_id:
          profiler.tock()
          profiler._cur_frame.clear()
          time.sleep(0.001)
          continue
      last_cam_id = cam_id

      if need_register:
        profiler.tock()
        if args.auto_bbox:
          H_img, W_img = color.shape[:2]
          mask = make_center_bbox_mask(H_img, W_img)
          print(f"[register] auto bbox: center {W_img//2} x {H_img//2} rectangle")
        else:
          print("[register] Please draw a 2D bbox to initialize tracking")
          mask = selector.select(color)
          if mask is None:
            print("[register] cancelled")
            break
        if mask.sum() < 50:
          print("[register] mask too small, please re-select")
          continue

        # 记下 mask 的外接矩形, tracking 阶段会叠在画面上方便检查框选区域是否套住物体
        ys, xs = np.where(mask)
        register_bbox_2d = (int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max()))

        # register 阶段临时打开 estimater 内部 debug, 让它生成 vis_score.png
        est.debug = 2
        try:
          pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                              iteration=args.est_refine_iter)
        finally:
          est.debug = 0
        print(f"[register] done, pose=\n{pose}")
        keep_only_score_vis(debug_dir, register_count)
        register_count += 1

        need_register = False
        frame_id = 0
        profiler.history.clear()
        last_profile_print_t = time.time()
        continue
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K,
                             iteration=args.track_refine_iter,
                             profiler=profiler)

      # 喂给 AprilTag 采样器 (固定后是 no-op), 然后把 ob_in_cam 转到世界系.
      profiler.tick('world_sampler')
      world_detected, world_corners = sampler.step(color)
      ob_in_world = sampler.transform(pose)

      # 仅 debug>=1 才画轴和 bbox, 否则连这点开销都省 (debug=0 = 真静默最快)
      vis = None
      if debug >= 1:
        profiler.tick('vis_render')
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.05, K=K,
                            thickness=3, transparency=0, is_input_rgb=True)
        # 已固定的世界轴: 在物体轴旁多画一组, 方便目视验证 ob_in_world 是否合理.
        # 注意: 必须用 transparency=0 走 fast path. slow path 在 is_input_rgb=True
        # 下会先 cvtColor RGB→BGR 再用 RGB 颜色元组直接画到 BGR 图上, 等于把 X 轴
        # 画成蓝色、Z 轴画成红色 (Y 对称, 看起来正常), 视觉上像 R 没矫正过来。
        # 同时 fast path 砍掉了 3 次 full-image diff 和 2 次 cvtColor, vis_render
        # 从 ~15ms 降到 ~3ms。
        # 用浅色区分世界轴和物体轴: 浅红 / 浅绿 / 浅蓝 (axis_colors 旁路 is_input_rgb).
        if sampler.world_pose_4x4 is not None:
          vis = draw_xyz_axis(
              vis, ob_in_cam=sampler.world_pose_4x4, scale=0.05, K=K,
              thickness=2, transparency=0, is_input_rgb=True,
              axis_colors=((255, 140, 140), (140, 255, 140), (140, 140, 255)))

      t_now = time.time()
      fps_window.append(t_now)
      if len(fps_window) > fps_window_size:
        fps_window.pop(0)
      if len(fps_window) >= 2:
        fps_smoothed = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0] + 1e-9)
      frame_ms = (t_now - t_frame_start) * 1000.0
      if t_now - last_print_t >= 1.0:
        print(f"[fps] FPS={fps_smoothed:5.1f}  frame={frame_ms:5.1f}ms  id={frame_id}")
        last_print_t = t_now

      # ob_in_world 周期 stdout 打印 (与 fps 节奏分离, 仅世界系固定后才打)
      if ob_in_world is not None and t_now - last_world_xyz_log >= 1.0:
        t = ob_in_world[:3, 3]
        q = _mat_to_quat_xyzw(ob_in_world[:3, :3])
        rpy = _quat_to_euler_xyz_deg(q)
        print(f"[world] xyz=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f}) "
              f"rpy=({rpy[0]:+6.1f},{rpy[1]:+6.1f},{rpy[2]:+6.1f}) "
              f"quat_xyzw=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})")
        last_world_xyz_log = t_now

      # ZMQ 发布 ob_in_world (字段格式与 cube_world_observer.py 对齐, key 用 'cube1' 以匹配 CubeReceiver)
      if zmq_sock is not None and ob_in_world is not None:
        t = ob_in_world[:3, 3]
        q = _mat_to_quat_xyzw(ob_in_world[:3, :3])
        try:
          zmq_sock.send_string(json.dumps({
              'timestamp': time.time(),
              'frame': frame_id,
              'world_fixed': sampler.fixed,
              'cube1': {
                  'position': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
                  'orientation': {'x': float(q[0]), 'y': float(q[1]),
                                  'z': float(q[2]), 'w': float(q[3])},
              },
          }), zmq.NOBLOCK)
        except zmq.Again:
          pass

      if debug >= 1 and vis is not None and frame_id % args.show_every == 0:
        profiler.tick('imshow_waitkey')
        vis_bgr = vis[..., ::-1].copy()
        # 把 register 时的 bbox 用黄色细线画出来, 方便目视确认初始 mask 框是否覆盖物体
        if register_bbox_2d is not None:
          x1, y1, x2, y2 = register_bbox_2d
          cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)
          cv2.putText(vis_bgr, 'register bbox', (x1, max(y1 - 5, 12)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_bgr, f"FPS: {fps_smoothed:5.1f}  ({frame_ms:5.1f} ms)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 世界系状态行 + 进度条 / FIXED 文本
        if not sampler.fixed:
          n = sampler.n_collected
          tgt = args.world_samples
          progress = n / max(tgt, 1)
          bar_w, bar_h, bar_x, bar_y = 200, 18, 10, 50
          cv2.rectangle(vis_bgr, (bar_x, bar_y),
                        (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
          cv2.rectangle(vis_bgr, (bar_x, bar_y),
                        (bar_x + int(bar_w * progress), bar_y + bar_h),
                        (0, 255, 255), -1)
          cv2.rectangle(vis_bgr, (bar_x, bar_y),
                        (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
          cv2.putText(vis_bgr, f"World Sampling: {n}/{tgt}",
                      (bar_x, bar_y + bar_h + 18),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
          if world_corners is not None:
            cv2.polylines(vis_bgr,
                          [np.asarray(world_corners).astype(int)],
                          True, (255, 0, 255), 2)
        else:
          cv2.putText(vis_bgr, "WORLD FIXED", (10, 55),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          if ob_in_world is not None:
            t = ob_in_world[:3, 3]
            q = _mat_to_quat_xyzw(ob_in_world[:3, :3])
            rpy = _quat_to_euler_xyz_deg(q)
            cv2.putText(vis_bgr,
                        f"xyz=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            cv2.putText(vis_bgr,
                        f"rpy=({rpy[0]:+6.1f},{rpy[1]:+6.1f},{rpy[2]:+6.1f})",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        cv2.imshow('track_single (q=quit, r=re-register, w=resample world)',
                   vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          profiler.frame_mark()
          break
        if key == ord('r'):
          profiler.frame_mark()
          need_register = True
          continue
        if key == ord('w'):
          sampler.start()
          print("[world] resampling started — waving the AprilTag in view")
          profiler.frame_mark()
          continue

      profiler.frame_mark()

      if t_now - last_profile_print_t >= PROFILE_PRINT_INTERVAL:
        print(
            f"\n===== Stage profile (avg over last {profiler.window} frames) =====\n"
            f"{profiler.summary_str()}\n"
            f"================================================================="
        )
        last_profile_print_t = t_now

      frame_id += 1

  finally:
    cam.stop()
    cv2.destroyAllWindows()
    if zmq_sock is not None:
      try:
        zmq_sock.close(linger=0)
      except Exception:
        pass
    if zmq_ctx is not None:
      try:
        zmq_ctx.term()
      except Exception:
        pass


if __name__ == '__main__':
  main()
