"""
FoundationPose tracking on cube55 with Intel RealSense.

基于 run_demo.py 的结构改造:
  - 输入源换成 RealSense (RGB + 对齐 depth + 自动读取内参 K)
  - 首帧鼠标框选 2D bbox 作为初始 mask -> est.register(...)
  - 后续帧 est.track_one(...)
  - 默认 debug=3, 方便查看 register 阶段错误匹配的中间渲染、打分以及每帧追踪结果

用法:
  python run_track_cube55.py
  python run_track_cube55.py --debug 3 --cam_width 640 --cam_height 480 --cam_fps 60

按键:
  q  退出
  r  重新框选 (触发 re-register)

debug 等级 (沿用 run_demo.py / estimater.py 内的语义):
  0  仅终端输出
  1  实时显示 bbox+xyz 轴
  2  额外保存 track_vis/{i}.png 以及 register 阶段的 ob_mask.png/color.png/depth.png/scene_*.ply/init_center.ply/vis_refiner.png/vis_score.png
  3  额外保存 model_tf.obj 等
"""

from estimater import *
from datareader import *
import argparse
import time
import threading
from collections import defaultdict
import pyrealsense2 as rs


class FrameProfiler:
  """每阶段耗时统计 (CUDA sync 后取真实 GPU 耗时), **按帧聚合**。

  关键: 一帧内多次 tick(同一 name) 会累加, 跨帧通过 frame_mark() 提交。
  所以 summary 中每个阶段是 "本帧总耗时" 的窗口平均, TOTAL = 实际平均帧耗时,
  不会再因为 refiner 单帧迭代多次或 imshow 跨帧执行而失真。

  用法:
    p = FrameProfiler()
    while ...:
      p.tick('a'); ...; p.tick('b'); ...; p.frame_mark()
    print(p.summary_str())
  """

  def __init__(self, window=60, use_cuda_sync=True):
    self.window = window
    self.use_cuda_sync = use_cuda_sync
    self.history = defaultdict(list)        # stage -> list of per-frame total durations
    self._pending_name = None
    self._pending_t0 = None
    self._cur_frame = defaultdict(float)    # 本帧内每个 stage 累计耗时

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
    """提交本帧统计。stage 只在某些帧触发的, 缺席帧用 0 补齐, 这样平均才对。"""
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
  """RealSense 异步抓帧, depth 对齐到 color, 单位米。

  后台 daemon 线程持续 wait_for_frames + align, 主线程通过 get() 拿最新一帧。
  这样主循环不会被相机硬件帧率阻塞。

  设备无关: D435i / D405 / D455 均可。depth_max 默认按 D435i 的 3m 设置,
  使用 D405 这种短距相机时建议传 depth_max=0.6 以滤掉超量程的噪声深度。
  """

  def __init__(self, width=640, height=480, fps=30, depth_max=3.0,
               color_exposure=None, depth_exposure=None,
               color_gain=None, white_balance=None):
    """
    Args:
      color_exposure: None=自动曝光开; int=手动曝光值(微秒, 例如 100/200/500/1000)
      depth_exposure: None=自动; int=手动 (D405 这种全局快门相机, 手动曝光对快速运动有意义)
      color_gain:     None=自动; int=手动增益 (一般配合手动曝光使用, 0~128 量级)
      white_balance:  None=自动; int=手动白平衡色温 (K, 例如 4000~6500)
    """
    self.pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = self.pipeline.start(cfg)
    self.align = rs.align(rs.stream.color)

    # 设备型号, 仅用于日志
    self.product_name = profile.get_device().get_info(rs.camera_info.name)

    # ---- 反查 color / depth 各自所属的 sensor ----
    # D435i: color 和 depth 是两个独立的物理传感器, query_sensors() 返回 2 个对象。
    # D405:  双 RGB stereo, color 和 depth 是同一个物理传感器, query_sensors() 返回 1 个,
    #        first_color_sensor() 会抛 "Could not find requested sensor type"。
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
    shared_sensor = (color_sensor is depth_sensor)
    if shared_sensor:
      print(f"[RealSense] color & depth share one sensor "
            f"(D405-style); color/depth_exposure 会写到同一个 option")

    # depth_scale: query_sensors() 返回的是基类 rs.sensor, 没有 get_depth_scale。
    # 优先 cast 到 depth_sensor; 失败再退回到 rs.option.depth_units。
    try:
      self.depth_scale = depth_sensor.as_depth_sensor().get_depth_scale()
    except Exception:
      try:
        self.depth_scale = depth_sensor.get_option(rs.option.depth_units)
      except Exception:
        self.depth_scale = 0.001  # 兜底, 大多数 RealSense 是 1mm
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

    for _ in range(30):
      self.pipeline.wait_for_frames()

    self._latest_color = None
    self._latest_depth = None
    self._latest_id = 0          # 单调递增, 主线程可据此判断是否拿到了新帧
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
    """返回 (rgb, depth, frame_id)。frame_id 单调递增, 同一帧多次调用 frame_id 不变。"""
    with self._lock:
      if self._latest_color is None:
        return None, None, 0
      return self._latest_color, self._latest_depth, self._latest_id

  def get_blocking(self):
    """阻塞直到至少有一帧可用 (用于 register 前的首帧)。"""
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
    win = "Drag a box around the cube, then press ENTER (ESC to cancel)"
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


def main():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--mesh_file', type=str,
                      default=f'{code_dir}/demo_data/cube55/mesh/textured_simple.obj')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=3)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_cube55')
  parser.add_argument('--cam_width', type=int, default=640)
  parser.add_argument('--cam_height', type=int, default=480)
  parser.add_argument('--cam_fps', type=int, default=30)
  parser.add_argument('--depth_max', type=float, default=3.0,
                      help='超过此距离(米)的深度被清零。D435i 用 3.0, D405 建议 0.6')
  parser.add_argument('--color_exposure', type=int, default=None,
                      help='color 手动曝光(us), 不传 = 自动曝光。快速运动建议 100~300')
  parser.add_argument('--depth_exposure', type=int, default=None,
                      help='depth 手动曝光(us), 不传 = 自动')
  parser.add_argument('--color_gain', type=int, default=None,
                      help='color 手动增益, 配合短曝光使用以保持亮度')
  parser.add_argument('--white_balance', type=int, default=None,
                      help='color 手动白平衡(K), 例如 4000~6500, 不传 = 自动')
  parser.add_argument('--show_every', type=int, default=2,
                      help='每 N 帧 imshow 一次 (1=每帧都显示)')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  # estimater / refiner / scorer 内部有大量 logging.info("Welcome"/"forward start"/...),
  # 每帧每迭代都会打, 实测刷屏。这里把 root logger 降到 WARNING, 只保留我们自己的
  # print() 输出 (FPS / profile / register 状态)。
  logging.getLogger().setLevel(logging.WARNING)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  mesh = trimesh.load(args.mesh_file)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
  print(f"[init] loaded mesh: {args.mesh_file}, extents={extents}")

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                       mesh=mesh, scorer=scorer, refiner=refiner,
                       debug_dir=debug_dir, debug=debug, glctx=glctx)
  print("[init] estimator ready")

  cam = RealSenseRGBD(width=args.cam_width, height=args.cam_height,
                      fps=args.cam_fps, depth_max=args.depth_max,
                      color_exposure=args.color_exposure,
                      depth_exposure=args.depth_exposure,
                      color_gain=args.color_gain,
                      white_balance=args.white_balance)
  K = cam.K
  selector = BBoxSelector()

  pose = None
  frame_id = 0
  need_register = True
  last_cam_id = -1       # 上一帧使用的相机帧 id, 用于跳过重复帧

  fps_window = []        # 最近若干帧的时间戳, 用于估算 tracking FPS
  fps_window_size = 30
  last_print_t = time.time()
  fps_smoothed = 0.0

  profiler = FrameProfiler(window=60, use_cuda_sync=True)
  PROFILE_PRINT_INTERVAL = 2.0   # 秒, 每隔 N 秒打印一次阶段耗时
  last_profile_print_t = time.time()

  try:
    while True:
      t_frame_start = time.time()

      profiler.tick('cam_get')
      if need_register:
        # register 前必须有一帧
        color, depth, cam_id = cam.get_blocking()
      else:
        color, depth, cam_id = cam.get()
        if color is None or cam_id == last_cam_id:
          # 没拿到新帧, 主循环空转避免重复处理同一帧
          profiler.tock()
          # 这一轮没真正处理一帧, 不调 frame_mark, 避免污染统计
          profiler._cur_frame.clear()
          time.sleep(0.001)
          continue
      last_cam_id = cam_id

      if need_register:
        profiler.tock()
        print("[register] Please draw a 2D bbox to initialize tracking")
        mask = selector.select(color)
        if mask is None:
          print("[register] cancelled")
          break
        if mask.sum() < 50:
          print("[register] mask too small, please re-select")
          continue
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                            iteration=args.est_refine_iter)
        print(f"[register] done, pose=\n{pose}")

        if debug >= 3:
          m = mesh.copy()
          m.apply_transform(pose)
          m.export(f'{debug_dir}/model_tf.obj')
          xyz_map = depth2xyzmap(depth, K)
          valid = depth >= 0.001
          pcd = toOpen3dCloud(xyz_map[valid], color[valid])
          o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

        need_register = False
        frame_id = 0
        # register 比较慢, 不计入稳态 profile, 直接清掉历史
        profiler.history.clear()
        last_profile_print_t = time.time()
        continue
      else:
        # track_one 内部会 tick 'depth_to_gpu' / 'erode_depth' / 'bilateral_filter' /
        # 'depth2xyzmap' / 'refiner_predict' / 'pose_to_cpu', refiner.predict 内部会
        # 进一步 tick 'refiner_crop_render' / 'refiner_data_concat' / 'refiner_nn_forward' /
        # 'refiner_pose_update'。这里只要把 profiler 传进去即可。
        pose = est.track_one(rgb=color, depth=depth, K=K,
                             iteration=args.track_refine_iter,
                             profiler=profiler)

      profiler.tick('save_pose')
      np.savetxt(f'{debug_dir}/ob_in_cam/{frame_id:05d}.txt', pose.reshape(4, 4))

      profiler.tick('vis_render')
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=K,
                          thickness=3, transparency=0, is_input_rgb=True)

      # ---- FPS 统计 ----
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

      # ---- 显示 (每 SHOW_EVERY 帧一次, 节省 imshow+waitKey 开销) ----
      vis_bgr = None
      if frame_id % args.show_every == 0:
        profiler.tick('imshow_waitkey')
        vis_bgr = vis[..., ::-1].copy()
        cv2.putText(vis_bgr, f"FPS: {fps_smoothed:5.1f}  ({frame_ms:5.1f} ms)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('cube55 tracking (q=quit, r=re-register)', vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          profiler.frame_mark()
          break
        if key == ord('r'):
          profiler.frame_mark()
          need_register = True
          continue

      if debug >= 2:
        profiler.tick('save_vis')
        if vis_bgr is None:
          vis_bgr = vis[..., ::-1].copy()
          cv2.putText(vis_bgr, f"FPS: {fps_smoothed:5.1f}  ({frame_ms:5.1f} ms)",
                      (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        imageio.imwrite(f'{debug_dir}/track_vis/{frame_id:05d}.png', vis_bgr[..., ::-1])

      profiler.frame_mark()  # 提交本帧所有阶段统计

      # ---- 每隔 N 秒打印阶段耗时 ----
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


if __name__ == '__main__':
  main()
