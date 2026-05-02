"""harvest.py - 用仿真粗 mesh 跑 model-based 跟踪, 在 overlay 看着对的帧按 SPACE
落盘 (rgb/depth/ob_in_cam/masks/cam_K), 给 reconstruct.py 重建真相机纹理 mesh 用.

累加模式: 启动时扫 --out 目录, 找四个 modality 已存在的最大 idx, 新帧从
max+1 开始. 多次运行同一个 --out 会逐次追加, 不覆盖旧帧. 想全清重来就先
rm -rf real_data/your_obj/{rgb,depth,ob_in_cam,masks}.

照 track_single.py 的结构改造: 同样 RealSenseRGBD + BBoxSelector + register/track_one
循环, 只是把 tracking 阶段的按键加一个 SPACE 当前帧落盘.

落盘的 rgb 是相机原图 - 屏幕上的 AABB/坐标轴 overlay 只用于交互, 不会写到
保存的 PNG 里 (draw_posed_3d_box 是 in-place mutator, 必须传 color.copy()).

落盘的 mask 是用 mesh + pose 通过 nvdiffrast 渲出的轮廓 (silhouette), 紧贴物体
而非粗球, 避免 NOF 把背景吃进 mesh.

可视化: 同 track_single, 在物体上绘制 AABB 边框 + xyz 坐标轴.
画面左上角显示 'saved this session: N (total: M)' - session 是本次运行新增的,
total 是 out_dir 累计已存的.

按键:
  SPACE 保存当前帧 (rgb / depth / ob_in_cam / mask) 到 --out
  R     重新框选 + register (跟丢了就按这个)
  Q     退出

debug 等级 (与 track_single 对齐):
  0  tracking 阶段静默 (无窗口, Ctrl+C 退出, 无法 SPACE 挑帧 - 一般别用)
  1  tracking 阶段显示窗口 (默认; SPACE/R/Q 都生效)
  两档 register 都会在 debug_dir/ 下保存 vis_score_NNN.png 用于检查首帧匹配.
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
import cv2
import torch
import trimesh
import nvdiffrast.torch as dr

_FP_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _FP_ROOT not in sys.path:
  sys.path.insert(0, _FP_ROOT)

from estimater import (FoundationPose, ScorePredictor, PoseRefinePredictor,
                       set_logging_format, set_seed)            # noqa: E402
from Utils import (draw_posed_3d_box, draw_xyz_axis,
                   make_mesh_tensors, nvdiffrast_render)        # noqa: E402
from track_single import (RealSenseRGBD, BBoxSelector,
                          warmup_estimator, make_center_bbox_mask,
                          keep_only_score_vis)                  # noqa: E402
from model_free_pipeline.capture import save_frame              # noqa: E402


def save_pose(out_dir, idx, ob_in_cam):
  d = os.path.join(out_dir, 'ob_in_cam')
  os.makedirs(d, exist_ok=True)
  np.savetxt(os.path.join(d, f'{idx:06d}.txt'), ob_in_cam, fmt='%.8f')


def save_mask(out_dir, idx, mask_uint8):
  d = os.path.join(out_dir, 'masks')
  os.makedirs(d, exist_ok=True)
  cv2.imwrite(os.path.join(d, f'{idx:06d}.png'), mask_uint8)


def _find_next_idx(out_dir):
  """扫所有 modality 目录, 返回下一个可用 idx (= max(已有) + 1, 或 0).
  四个目录取并集, 防止某帧某 modality 漏写导致后续撞号."""
  import glob as _glob
  max_idx = -1
  for sub, ext in (('rgb', 'png'), ('depth', 'png'),
                   ('ob_in_cam', 'txt'), ('masks', 'png')):
    for f in _glob.glob(os.path.join(out_dir, sub, f'*.{ext}')):
      stem = os.path.splitext(os.path.basename(f))[0]
      try:
        max_idx = max(max_idx, int(stem))
      except ValueError:
        continue
  return max_idx + 1


def render_silhouette(mesh_tensors, ob_in_cam, K, H, W, glctx):
  """用 mesh + pose 通过 nvdiffrast 渲 mesh 在像素上的覆盖, 返回 0/255 uint8 mask.
  这是仿真粗 mesh 的轮廓投影 -> 比球形 ob_radius 紧得多, 直接作为 NOF 训练 mask."""
  ob = torch.as_tensor(ob_in_cam.reshape(1, 4, 4),
                       dtype=torch.float, device='cuda')
  _, depth, _ = nvdiffrast_render(
      K=K, H=H, W=W, ob_in_cams=ob, glctx=glctx,
      mesh_tensors=mesh_tensors, use_light=False)
  # depth 是 (1, H, W), 没渲到的像素值 ~= 0; > 1cm 才算物体, 避开 z=0 边缘噪声
  sil = (depth[0] > 0.01).cpu().numpy().astype(np.uint8) * 255
  return sil


def main():
  parser = argparse.ArgumentParser(
      description='harvest rgb/depth/pose from model-based tracking, '
                  'used to feed model-free reconstruction.')
  parser.add_argument('--mesh_file', required=True,
                      help='仿真粗 mesh, 用作 model-based 跟踪初值')
  parser.add_argument('--out', required=True, help='harvest 输出目录')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--cam_width', type=int, default=640)
  parser.add_argument('--cam_height', type=int, default=480)
  parser.add_argument('--cam_fps', type=int, default=30)
  parser.add_argument('--depth_max', type=float, default=1.5)
  parser.add_argument('--color_exposure', type=int, default=None)
  parser.add_argument('--depth_exposure', type=int, default=None)
  parser.add_argument('--color_gain', type=int, default=None)
  parser.add_argument('--white_balance', type=int, default=None)
  parser.add_argument('--score_vis_topk', type=int, default=10)
  parser.add_argument('--auto_bbox', action='store_true',
                      help='跳过手动框选, 用画面中心矩形 mask 初始化')
  parser.add_argument('--debug', type=int, default=1, choices=[0, 1],
                      help='0=tracking 阶段静默(无窗口, Ctrl+C 退出); '
                           '1=tracking 阶段显示窗口. 两档 register 都会保存 '
                           'vis_score_NNN.png.')
  parser.add_argument('--debug_dir', default=None,
                      help='register 阶段 vis_score.png 的存放位置, 默认 out/_register_debug')
  parser.add_argument('--axis_scale', type=float, default=0.05,
                      help='画在物体上的坐标轴长度 (米), 默认 5cm')
  args = parser.parse_args()

  os.makedirs(args.out, exist_ok=True)
  for sub in ('rgb', 'depth', 'ob_in_cam', 'masks'):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)
  if args.debug_dir is None:
    args.debug_dir = os.path.join(args.out, '_register_debug')
  os.makedirs(args.debug_dir, exist_ok=True)

  # 累加模式: 如果 out_dir 已经有数据, 新帧编号从 max(已有 idx)+1 开始,
  # 不覆盖旧帧. 四个 modality 取并集然后取最大值, 防止某个目录漏一帧导致
  # 后续编号撞车.
  start_idx = _find_next_idx(args.out)
  if start_idx > 0:
    print(f'[harvest] 累加模式: 已有 {start_idx} 帧, 新帧从 idx={start_idx} 开始')
  else:
    print(f'[harvest] 全新会话: 从 idx=0 开始')

  set_logging_format()
  set_seed(0)
  logging.getLogger().setLevel(logging.WARNING)

  mesh = trimesh.load(args.mesh_file)
  # 用 mesh.bounds 画 AABB; 不用 trimesh.bounds.oriented_bounds() 因为那个对完美
  # 对称的物体 (例如正方体) 会退化, 导致显示的坐标系被转到 OBB 系而非 mesh 原生系.
  bbox = np.stack([mesh.bounds[0], mesh.bounds[1]], axis=0).reshape(2, 3)
  print(f'[harvest] sim mesh: {args.mesh_file}  extents={mesh.extents}')

  scorer = ScorePredictor()
  scorer.vis_topk = args.score_vis_topk if args.score_vis_topk > 0 else None
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices,
                       model_normals=mesh.vertex_normals,
                       mesh=mesh, scorer=scorer, refiner=refiner,
                       debug_dir=args.debug_dir, debug=0, glctx=glctx)

  # 给 silhouette 渲染用; 复用 estimater 的 glctx 节省 GPU 上下文
  mesh_tensors_for_sil = make_mesh_tensors(mesh)

  cam = RealSenseRGBD(width=args.cam_width, height=args.cam_height,
                      fps=args.cam_fps, depth_max=args.depth_max,
                      color_exposure=args.color_exposure,
                      depth_exposure=args.depth_exposure,
                      color_gain=args.color_gain,
                      white_balance=args.white_balance)
  K = cam.K
  np.savetxt(os.path.join(args.out, 'cam_K.txt'), K, fmt='%.6f')
  print(f'[harvest] saved K -> {args.out}/cam_K.txt')

  warmup_estimator(est, K, cam.H, cam.W)

  selector = BBoxSelector()
  pose = None
  need_register = True
  saved = start_idx        # 累加: 从已有最大 idx + 1 开始, 旧文件不被覆盖
  saved_in_session = 0     # 本次会话内增量计数, 用于屏幕显示
  register_count = 0
  last_cam_id = -1

  fps_window, fps_smoothed = [], 0.0
  win = 'harvest (SPACE save | R re-register | Q quit)'

  try:
    while True:
      t_frame_start = time.time()
      if need_register:
        color, depth, cam_id = cam.get_blocking()
      else:
        color, depth, cam_id = cam.get()
        if color is None or cam_id == last_cam_id:
          time.sleep(0.001)
          continue
      last_cam_id = cam_id

      if need_register:
        if args.auto_bbox:
          mask = make_center_bbox_mask(*color.shape[:2])
          print('[register] auto bbox')
        else:
          print('[register] draw bbox to init')
          mask = selector.select(color)
          if mask is None:
            print('[register] cancelled')
            break
        if mask.sum() < 50:
          print('[register] mask too small, re-select')
          continue

        est.debug = 2
        try:
          pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                              iteration=args.est_refine_iter)
        finally:
          est.debug = 0
        keep_only_score_vis(args.debug_dir, register_count)
        register_count += 1
        need_register = False
        print(f'[register] done, pose=\n{pose}')
        continue

      pose = est.track_one(rgb=color, depth=depth, K=K,
                           iteration=args.track_refine_iter)

      now = time.time()
      fps_window.append(now)
      if len(fps_window) > 30:
        fps_window.pop(0)
      if len(fps_window) >= 2:
        fps_smoothed = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0] + 1e-9)
      frame_ms = (now - t_frame_start) * 1000.0

      if args.debug >= 1:
        # 注意: draw_posed_3d_box 会 in-place 改第二个参数; 必须传 copy 否则 color
        # 本身被画上 AABB, 后续 save_frame 落盘的就是污染图. draw_xyz_axis 内部
        # 已经自己 copy, 安全.
        vis = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=args.axis_scale,
                            K=K, thickness=3, transparency=0,
                            is_input_rgb=True)
        vis_bgr = vis[..., ::-1].copy()
        cv2.putText(vis_bgr, f'saved this session: {saved_in_session}  '
                             f'(total: {saved})',
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_bgr, f'FPS {fps_smoothed:5.1f}  ({frame_ms:5.1f} ms)',
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, vis_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
          # rgb 落盘前必须用未被任何 overlay 污染的 color (draw_posed_3d_box 已经
          # 改成传 copy, 这里 color 是干净的相机原图)
          sil = render_silhouette(mesh_tensors_for_sil, pose, K,
                                  cam.H, cam.W, glctx)
          save_frame(args.out, saved, color, depth)
          save_pose(args.out, saved, pose)
          save_mask(args.out, saved, sil)
          print(f'[harvest] SAVED idx={saved:06d} '
                f'(session #{saved_in_session + 1}, '
                f'mask coverage {(sil > 0).mean()*100:.2f}%)')
          saved += 1
          saved_in_session += 1
        elif key == ord('r') or key == ord('R'):
          print('[harvest] re-register')
          need_register = True
        elif key == ord('q') or key == ord('Q') or key == 27:
          break
  finally:
    cam.stop()
    cv2.destroyAllWindows()
    print(f'[harvest] done; this session +{saved_in_session} frames; '
          f'total now {saved} frames in {args.out}')


if __name__ == '__main__':
  main()
