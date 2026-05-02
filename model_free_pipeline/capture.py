"""RealSense 采集 + 实时落盘.

复用 track_single.py 中的 RealSenseRGBD 抓帧类, 不重新封装. 这里只负责:
- 按用户按键 (SPACE) 保存当前帧到 out_dir
- 同步把内参写到 cam_K.txt
- 实时画面叠加已保存帧数
"""
import os
import sys
import time
import cv2
import numpy as np

# 让本模块单独跑也能 import 到根目录的 track_single
_FP_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _FP_ROOT not in sys.path:
  sys.path.insert(0, _FP_ROOT)

from track_single import RealSenseRGBD  # noqa: E402


def save_frame(out_dir, idx, rgb, depth_m):
  """rgb: HxWx3 uint8 RGB; depth_m: HxW float32 米. 落盘成 datareader 期望的格式."""
  rgb_path = os.path.join(out_dir, 'rgb', f'{idx:06d}.png')
  depth_path = os.path.join(out_dir, 'depth', f'{idx:06d}.png')
  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  cv2.imwrite(rgb_path, bgr)
  d_mm = (depth_m * 1000.0).astype(np.uint16)
  cv2.imwrite(depth_path, d_mm)


def record_session(out_dir, cam_cfg, n_frames, min_save_interval_ms=80):
  """交互式采集. 返回实际保存的帧数.

  按键:
    SPACE 保存当前帧
    A     连续模式开关 (按住每隔 min_save_interval_ms 自动保存)
    Q     提前结束
  """
  for sub in ('rgb', 'depth'):
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

  cam = RealSenseRGBD(
      width=cam_cfg.get('width', 640),
      height=cam_cfg.get('height', 480),
      fps=cam_cfg.get('fps', 30),
      depth_max=cam_cfg.get('depth_max', 1.5),
      color_exposure=cam_cfg.get('color_exposure'),
      depth_exposure=cam_cfg.get('depth_exposure'),
      color_gain=cam_cfg.get('color_gain'),
      white_balance=cam_cfg.get('white_balance'),
  )
  K_path = os.path.join(out_dir, 'cam_K.txt')
  np.savetxt(K_path, cam.K, fmt='%.6f')
  print(f'[capture] saved K -> {K_path}')

  win = 'capture (SPACE save, A continuous, Q quit)'
  cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
  saved = 0
  continuous = False
  last_save = 0.0
  try:
    while saved < n_frames:
      rgb, depth, fid = cam.get()
      if rgb is None:
        time.sleep(0.005)
        continue

      now = time.time() * 1000.0
      do_save = False
      key = cv2.waitKey(1) & 0xFF
      if key == ord(' '):
        do_save = True
      elif key == ord('a') or key == ord('A'):
        continuous = not continuous
        print(f'[capture] continuous={continuous}')
      elif key == ord('q') or key == ord('Q') or key == 27:
        break
      if continuous and (now - last_save) >= min_save_interval_ms:
        do_save = True

      if do_save:
        save_frame(out_dir, saved, rgb, depth)
        saved += 1
        last_save = now
        print(f'[capture] saved frame {saved}/{n_frames}')

      vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
      txt = f'{saved}/{n_frames}  cont={continuous}'
      cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                  0.8, (0, 255, 0), 2)
      cv2.imshow(win, vis)
  finally:
    cv2.destroyWindow(win)
    cam.stop()

  return saved
