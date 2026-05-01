"""
FoundationPose 多物体追踪 (track anything) on Intel RealSense.

基于 run_track_cube54.py 改造为多物体版本:
  - 通过 --mesh_files 或 --mesh_dir 加载 N 个 mesh
  - 首帧依次为每个物体框选 2D bbox -> 各自的 est.register(...)
  - 后续每帧对每个物体串行调 est.track_one(...)
  - 一帧内叠加所有物体的 bbox (默认绿色) + xyz 轴, 与 cube54 版样式一致

用法:
  # 显式列表
  python track_anything.py --mesh_files demo_data/cube54/mesh/textured_cube54.obj --track_refine_iter 1 --color_exposure 5000 --depth_exposure 5000 --color_gain 80 --depth_max 0.6 --cam_width 640 --cam_height 480 --cam_fps 90 --score_vis_topk 5 --debug 1

  # 目录扫描
  python track_anything.py --mesh_dir my_meshes --debug 1 --cam_fps 60

按键:
  q  退出
  r  全部重新框选 (re-register all objects)
  ESC (在框选阶段) 跳过当前物体

debug 等级:
  0  tracking 阶段静默 (无窗口, Ctrl+C 退出)
  1  tracking 阶段显示窗口 (q 退出 / r 全部重新框选)
两档下 register 阶段每个物体都会在 debug_dir/{obj_name}/ 下保存 vis_score_NNN.png.
tracking 阶段始终不写文件、不影响帧率.
"""

from estimater import *
from datareader import *
import argparse
import time
import threading
import glob
from collections import defaultdict
import pyrealsense2 as rs


class FrameProfiler:
  """每阶段耗时统计 (CUDA sync 后取真实 GPU 耗时), 按帧聚合.

  N 个物体串行 track 时, 同一 stage 在一帧内会被 tick N 次,
  history 里每帧记录的是 N 个物体加起来的总耗时. 所以阶段耗时 / N
  约等于单物体平均耗时.
  """

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
  """RealSense 异步抓帧, depth 对齐到 color, 单位米. 与 cube54 版相同."""

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

    for _ in range(30):
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
  """鼠标拖框选择矩形 mask. 标题栏会显示当前正在框选的物体名."""

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

  def select(self, rgb, prompt_text):
    """返回 mask (H,W bool) 或 None (用户按 ESC 跳过)."""
    self.p0 = self.p1 = None
    self.dragging = False
    self.done = False
    win = "Track Anything - draw bbox, ENTER=submit, ESC=skip, q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, self._cb)
    bgr = rgb[..., ::-1].copy()
    H, W = rgb.shape[:2]

    while True:
      show = bgr.copy()
      cv2.putText(show, prompt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                  0.7, (0, 255, 0), 2, cv2.LINE_AA)
      if self.p0 and self.p1:
        cv2.rectangle(show, self.p0, self.p1, (0, 255, 0), 2)
      cv2.imshow(win, show)
      key = cv2.waitKey(30) & 0xFF
      if key == 13 and self.done:   # ENTER
        break
      if key == 27:   # ESC, skip
        cv2.destroyWindow(win)
        return None, 'skip'
      if key == ord('q'):   # quit all
        cv2.destroyWindow(win)
        return None, 'quit'
    cv2.destroyWindow(win)

    x1, x2 = sorted([self.p0[0], self.p1[0]])
    y1, y2 = sorted([self.p0[1], self.p1[1]])
    mask = np.zeros((H, W), dtype=bool)
    mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    return mask, 'ok'


# register 完成后 estimater 会留下这些中间产物, 我们只想留 vis_score.png
_REGISTER_INTERMEDIATE_FILES = [
    'ob_mask.png', 'color.png', 'depth.png',
    'scene_raw.ply', 'scene_complete.ply',
    'init_center.ply', 'vis_refiner.png', 'model_tf.obj',
]


class TrackedObject:
  """单个被追踪物体. 持有独立的 FoundationPose 实例和 mesh 信息.

  scorer / refiner / glctx 由外部传入并在多个 TrackedObject 间共享.
  这样 N 个物体只占用 1 份网络权重和 GPU context.

  estimater 默认静默 (debug=0); register 时临时切到 2 让它生成 vis_score.png,
  完成后立刻切回 0 + 清理中间产物 + 把 vis_score.png 重命名为 vis_score_NNN.png.
  这样后续 track_one 内部不渲染 vis, 帧率不受拖累, 而且每次 re-register 都留一份打分图.
  """

  def __init__(self, name, mesh_file, scorer, refiner, glctx, debug_dir):
    self.name = name
    self.mesh_file = mesh_file
    self.mesh = trimesh.load(mesh_file)
    # 用 mesh.bounds 画 AABB, 跳过 oriented_bounds (规避对称物体 OBB 退化)
    self.bbox = np.stack([self.mesh.bounds[0], self.mesh.bounds[1]],
                         axis=0).reshape(2, 3)

    # 每个物体一个 debug 子目录, 用来存 vis_score_NNN.png
    self.obj_debug_dir = os.path.join(debug_dir, name)
    os.makedirs(self.obj_debug_dir, exist_ok=True)

    self.est = FoundationPose(
        model_pts=self.mesh.vertices,
        model_normals=self.mesh.vertex_normals,
        mesh=self.mesh,
        scorer=scorer, refiner=refiner, glctx=glctx,
        debug_dir=self.obj_debug_dir, debug=0,
    )
    self.pose = None              # (4,4) np array, 未 register 时为 None
    self.registered = False
    self.register_count = 0       # 累计 register 次数, 用于 vis_score 命名

  def _keep_only_score_vis(self):
    """register 完成后只保留 vis_score.png, 重命名累计存; 删掉其他中间产物."""
    src = os.path.join(self.obj_debug_dir, 'vis_score.png')
    if os.path.exists(src):
      dst = os.path.join(self.obj_debug_dir, f'vis_score_{self.register_count:03d}.png')
      os.replace(src, dst)
      print(f"[register] {self.name}: saved score visualization -> {dst}")
    else:
      print(f"[register] WARNING: {self.name}: vis_score.png not generated")
    for fn in _REGISTER_INTERMEDIATE_FILES:
      p = os.path.join(self.obj_debug_dir, fn)
      if os.path.exists(p):
        try:
          os.remove(p)
        except OSError:
          pass

  def register(self, K, rgb, depth, mask, iteration):
    # 临时打开 estimater 内部 debug 让它生成 vis_score.png 等
    self.est.debug = 2
    try:
      self.pose = self.est.register(K=K, rgb=rgb, depth=depth,
                                    ob_mask=mask, iteration=iteration)
    finally:
      self.est.debug = 0
    self.registered = True
    print(f"[register] {self.name} done, pose=\n{self.pose}")
    self._keep_only_score_vis()
    self.register_count += 1

  def track(self, K, rgb, depth, iteration, profiler=None):
    if not self.registered:
      return
    self.pose = self.est.track_one(rgb=rgb, depth=depth, K=K,
                                   iteration=iteration, profiler=profiler)


def collect_mesh_files(args):
  """从 --mesh_files 或 --mesh_dir 拼出 mesh 文件列表."""
  files = []
  if args.mesh_files:
    files.extend(args.mesh_files)
  if args.mesh_dir:
    for ext in ('*.obj', '*.OBJ', '*.ply', '*.PLY'):
      files.extend(sorted(glob.glob(os.path.join(args.mesh_dir, ext))))
  # 去重保序
  seen = set()
  uniq = []
  for f in files:
    f = os.path.abspath(f)
    if f not in seen:
      seen.add(f)
      uniq.append(f)
  if not uniq:
    raise SystemExit("no mesh provided. Use --mesh_files or --mesh_dir.")
  return uniq


def make_object_name(mesh_path, used_names):
  """从 mesh 文件名生成不重复的物体名."""
  base = os.path.splitext(os.path.basename(mesh_path))[0]
  name = base
  i = 1
  while name in used_names:
    i += 1
    name = f"{base}_{i}"
  used_names.add(name)
  return name


def draw_object_overlay(vis, obj, K):
  """在 vis (RGB) 上画一个物体的 bbox + xyz 轴, 样式与 cube54 版一致:
  bbox 走 draw_posed_3d_box 的默认绿色 (0,255,0);
  xyz 轴默认 R/G/B (is_input_rgb=True 路径).
  """
  if obj.pose is None:
    return vis
  vis = draw_posed_3d_box(K, img=vis, ob_in_cam=obj.pose, bbox=obj.bbox)
  vis = draw_xyz_axis(vis, ob_in_cam=obj.pose, scale=0.05, K=K,
                      thickness=3, transparency=0, is_input_rgb=True)
  return vis


def main():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--mesh_files', type=str, nargs='*', default=None,
                      help='显式指定一个或多个 mesh 文件 (.obj/.ply)')
  parser.add_argument('--mesh_dir', type=str, default=None,
                      help='指定一个目录, 自动扫描 *.obj/*.ply')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1, choices=[0, 1],
                      help='0=tracking 静默(无窗口, Ctrl+C 退出); 1=tracking 显示窗口. '
                           '两档下 register 都会保存 {obj}/vis_score_NNN.png.')
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_anything')
  parser.add_argument('--cam_width', type=int, default=640)
  parser.add_argument('--cam_height', type=int, default=480)
  parser.add_argument('--cam_fps', type=int, default=30)
  parser.add_argument('--depth_max', type=float, default=3.0)
  parser.add_argument('--color_exposure', type=int, default=None)
  parser.add_argument('--depth_exposure', type=int, default=None)
  parser.add_argument('--color_gain', type=int, default=None)
  parser.add_argument('--white_balance', type=int, default=None)
  parser.add_argument('--show_every', type=int, default=1,
                      help='每 N 帧 imshow 一次 (1=每帧都显示), 仅 --debug 1 时生效')
  parser.add_argument('--score_vis_topk', type=int, default=10,
                      help='vis_score.png 里画 top-N 得分的 pose. 0 或负值表示画全部 252 个.')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  logging.getLogger().setLevel(logging.WARNING)

  debug = args.debug
  debug_dir = args.debug_dir
  os.makedirs(debug_dir, exist_ok=True)

  mesh_files = collect_mesh_files(args)
  print(f"[init] {len(mesh_files)} mesh(es):")
  for f in mesh_files:
    print(f"       {f}")

  # ---- 共享网络权重 / GPU context ----
  scorer = ScorePredictor()
  scorer.vis_topk = args.score_vis_topk if args.score_vis_topk > 0 else None
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()

  # ---- 实例化 N 个 TrackedObject ----
  used_names = set()
  objects = []
  for idx, mesh_file in enumerate(mesh_files):
    name = make_object_name(mesh_file, used_names)
    obj = TrackedObject(name=name, mesh_file=mesh_file,
                        scorer=scorer, refiner=refiner, glctx=glctx,
                        debug_dir=debug_dir)
    objects.append(obj)
    print(f"[init] obj[{idx}] name={name} extents={obj.mesh.extents}")

  cam = RealSenseRGBD(width=args.cam_width, height=args.cam_height,
                      fps=args.cam_fps, depth_max=args.depth_max,
                      color_exposure=args.color_exposure,
                      depth_exposure=args.depth_exposure,
                      color_gain=args.color_gain,
                      white_balance=args.white_balance)
  K = cam.K
  selector = BBoxSelector()

  need_register_all = True
  last_cam_id = -1
  frame_id = 0

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
      if need_register_all:
        color, depth, cam_id = cam.get_blocking()
      else:
        color, depth, cam_id = cam.get()
        if color is None or cam_id == last_cam_id:
          profiler.tock()
          profiler._cur_frame.clear()
          time.sleep(0.001)
          continue
      last_cam_id = cam_id

      # ---------- REGISTER 阶段 ----------
      if need_register_all:
        profiler.tock()
        for obj in objects:
          obj.registered = False
          obj.pose = None

        do_quit = False
        for idx, obj in enumerate(objects):
          # 每个物体框选前重新抓一帧, 避免用户思考期间画面陈旧
          color, depth, cam_id = cam.get_blocking()
          last_cam_id = cam_id
          prompt = f"[{idx+1}/{len(objects)}] draw bbox for: {obj.name}"
          mask, action = selector.select(color, prompt)
          if action == 'quit':
            do_quit = True
            break
          if action == 'skip' or mask is None:
            print(f"[register] skipped {obj.name}")
            continue
          if mask.sum() < 50:
            print(f"[register] mask too small for {obj.name}, skipping")
            continue
          obj.register(K=K, rgb=color, depth=depth, mask=mask,
                       iteration=args.est_refine_iter)

        if do_quit:
          break

        # 至少有一个 register 成功才进入 tracking
        if not any(o.registered for o in objects):
          print("[register] no object registered, retry")
          continue

        need_register_all = False
        frame_id = 0
        profiler.history.clear()
        last_profile_print_t = time.time()
        continue

      # ---------- TRACKING 阶段 ----------
      for obj in objects:
        if obj.registered:
          obj.track(K=K, rgb=color, depth=depth,
                    iteration=args.track_refine_iter, profiler=profiler)

      # ---------- 可视化 (仅 debug>=1) ----------
      vis = None
      if debug >= 1:
        profiler.tick('vis_render')
        vis = color.copy()
        for obj in objects:
          if obj.registered:
            vis = draw_object_overlay(vis, obj, K)

      # FPS 统计
      t_now = time.time()
      fps_window.append(t_now)
      if len(fps_window) > fps_window_size:
        fps_window.pop(0)
      if len(fps_window) >= 2:
        fps_smoothed = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0] + 1e-9)
      frame_ms = (t_now - t_frame_start) * 1000.0
      if t_now - last_print_t >= 1.0:
        n_active = sum(1 for o in objects if o.registered)
        print(f"[fps] FPS={fps_smoothed:5.1f}  frame={frame_ms:5.1f}ms  "
              f"objs={n_active}/{len(objects)}  id={frame_id}")
        last_print_t = t_now

      if debug >= 1 and vis is not None and frame_id % args.show_every == 0:
        profiler.tick('imshow_waitkey')
        vis_bgr = vis[..., ::-1].copy()
        cv2.putText(vis_bgr, f"FPS: {fps_smoothed:5.1f}  ({frame_ms:5.1f} ms)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Track Anything (q=quit, r=re-register all)', vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          profiler.frame_mark()
          break
        if key == ord('r'):
          profiler.frame_mark()
          need_register_all = True
          continue

      profiler.frame_mark()

      if t_now - last_profile_print_t >= PROFILE_PRINT_INTERVAL:
        n_active = sum(1 for o in objects if o.registered)
        print(
            f"\n===== Stage profile (avg over last {profiler.window} frames, "
            f"sum across {n_active} objects) =====\n"
            f"{profiler.summary_str()}\n"
            f"==================================================="
        )
        last_profile_print_t = t_now

      frame_id += 1

  finally:
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
