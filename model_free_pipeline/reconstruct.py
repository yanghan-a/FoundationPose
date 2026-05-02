"""调 BundleSDF 的 NeuralObjectField 重建带纹理 mesh.

不重新实现 — 直接 wrap `bundlesdf/run_nerf.py:run_neural_object_field()`.
那个函数内部已经做了:
- compute_scene_bounds (DBSCAN 估场景缩放/平移)
- preprocess_data (归一化坐标)
- NerfRunner.train (含 pose 优化, optimize_poses=1)
- extract_mesh + mesh_texture_from_train_images
- mesh_to_real_world (反归一化回真实尺度)

只需用户提供 RGB + depth + 每帧物体位姿. masks 可选, 缺了就用 depth+pose+
ob_radius 自动派一个粗 mask (球或 AABB), NOF 对粗 mask 鲁棒.

支持的目录布局 (自动嗅探):
  cam_K.txt       或  K.txt
  rgb/{stem}.png
  depth/{stem}.png        (uint16 mm)
  masks/{stem}.png        (optional, 0/255)
  ob_in_cam/{stem}.txt    或  cam_in_ob/{stem}.txt    (4x4)
"""
import os
import sys
import glob
import shutil
import yaml
import cv2
import numpy as np
import imageio


_FP_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_BSDF_DIR = os.path.join(_FP_ROOT, 'bundlesdf')
for p in (_FP_ROOT, _BSDF_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)


def _find_K(out_dir):
  for name in ('cam_K.txt', 'K.txt'):
    p = os.path.join(out_dir, name)
    if os.path.exists(p):
      return np.loadtxt(p).reshape(3, 3)
  raise FileNotFoundError(f'no cam_K.txt / K.txt under {out_dir}')


def _find_dir(out_dir, names):
  for n in names:
    p = os.path.join(out_dir, n)
    if os.path.isdir(p):
      return p, n
  return None, None


def _list_stems(d, ext):
  return sorted(os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(d, f'*.{ext}')))


def _auto_mask(depth, K, cam_in_ob, ob_radius, ob_box=None):
  """从 depth + pose 反算粗 mask. 默认球 (半径 ob_radius), 也可以传 ob_box=(hx,hy,hz)
  用 AABB. 单位米."""
  H, W = depth.shape
  fx, fy = K[0, 0], K[1, 1]
  cx, cy = K[0, 2], K[1, 2]
  ys, xs = np.mgrid[:H, :W]
  Z = depth.astype(np.float32)
  with np.errstate(invalid='ignore'):
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
  pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)   # HxWx4
  pts_ob = pts_cam @ cam_in_ob.T
  if ob_box is None:
    d2 = pts_ob[..., 0]**2 + pts_ob[..., 1]**2 + pts_ob[..., 2]**2
    inside = d2 < ob_radius**2
  else:
    hx, hy, hz = ob_box
    inside = ((np.abs(pts_ob[..., 0]) < hx) &
              (np.abs(pts_ob[..., 1]) < hy) &
              (np.abs(pts_ob[..., 2]) < hz))
  valid = (Z > 0.05) & (Z < 5.0)
  m = (valid & inside).astype(np.uint8) * 255
  # 形态学闭运算填一些小洞
  k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
  return m


def _subsample_indices(n_total, max_frames, poses=None, mode='uniform'):
  """从 n_total 帧选 max_frames 帧的下标. 支持:
  - uniform: 等步长 (简单, 默认)
  - farthest: 在 ob_in_cam 平移上做 farthest point sampling, 视角更分散
  """
  if n_total <= max_frames:
    return np.arange(n_total)
  if mode == 'uniform' or poses is None:
    return np.linspace(0, n_total - 1, max_frames).astype(int)
  if mode == 'farthest':
    # 用每帧 cam 在 ob 系下的位置作为代表特征点
    cam_pos = poses[:, :3, 3]   # (N, 3) translations
    chosen = [0]
    dists = np.linalg.norm(cam_pos - cam_pos[0], axis=1)
    for _ in range(max_frames - 1):
      next_i = int(np.argmax(dists))
      chosen.append(next_i)
      d_new = np.linalg.norm(cam_pos - cam_pos[next_i], axis=1)
      dists = np.minimum(dists, d_new)
    return np.array(sorted(set(chosen)))
  raise ValueError(f'unknown subsample mode: {mode}')


def _load_session(out_dir, ob_radius=0.15, ob_box=None,
                  save_auto_masks=True, max_frames=None,
                  subsample_mode='uniform'):
  K = _find_K(out_dir)

  rgb_dir, _ = _find_dir(out_dir, ('rgb',))
  if rgb_dir is None:
    raise FileNotFoundError(f'no rgb/ under {out_dir}')
  depth_dir, _ = _find_dir(out_dir, ('depth', 'depth_enhanced'))
  if depth_dir is None:
    raise FileNotFoundError(f'no depth/ or depth_enhanced/ under {out_dir}')

  pose_dir, pose_name = _find_dir(out_dir, ('ob_in_cam', 'cam_in_ob'))
  if pose_dir is None:
    raise FileNotFoundError(f'no ob_in_cam/ or cam_in_ob/ under {out_dir}')
  invert_pose = (pose_name == 'ob_in_cam')

  mask_dir, _ = _find_dir(out_dir, ('masks', 'mask'))
  auto_mask = mask_dir is None
  if auto_mask:
    print(f'[reconstruct] no masks/ found, auto-derive from depth+pose '
          f'(ob_radius={ob_radius}m, ob_box={ob_box})')
    if save_auto_masks:
      mask_dir = os.path.join(out_dir, 'masks')
      os.makedirs(mask_dir, exist_ok=True)

  rgb_stems = set(_list_stems(rgb_dir, 'png'))
  depth_stems = set(_list_stems(depth_dir, 'png'))
  pose_stems = set(_list_stems(pose_dir, 'txt'))
  common = sorted(rgb_stems & depth_stems & pose_stems)
  if not common:
    raise RuntimeError(
        f'no stems present in rgb/depth/pose all 3. counts: '
        f'rgb={len(rgb_stems)} depth={len(depth_stems)} pose={len(pose_stems)}')

  # 太多帧会爆 8GB 显存 (NerfRunner 内部 rays 是 N×H×W 量级).
  # 先把 pose 全读出来用于 farthest sampling, 然后再决定加载哪些帧.
  if max_frames is not None and len(common) > max_frames:
    all_poses = []
    for stem in common:
      T = np.loadtxt(os.path.join(pose_dir, f'{stem}.txt')).reshape(4, 4)
      cam_in_ob_T = np.linalg.inv(T) if invert_pose else T
      all_poses.append(cam_in_ob_T)
    all_poses = np.stack(all_poses, axis=0)
    keep = _subsample_indices(len(common), max_frames,
                              poses=all_poses, mode=subsample_mode)
    selected = [common[i] for i in keep]
    print(f'[reconstruct] {len(common)} frames > max_frames={max_frames}, '
          f'subsample to {len(selected)} via "{subsample_mode}"')
    common = selected

  rgbs, depths, masks, poses = [], [], [], []
  for stem in common:
    rgb = imageio.imread(os.path.join(rgb_dir, f'{stem}.png'))
    if rgb.ndim == 3 and rgb.shape[2] == 4:
      rgb = rgb[..., :3]
    depth = cv2.imread(os.path.join(depth_dir, f'{stem}.png'),
                       -1).astype(np.float32) / 1e3
    T = np.loadtxt(os.path.join(pose_dir, f'{stem}.txt')).reshape(4, 4)
    cam_in_ob = np.linalg.inv(T) if invert_pose else T

    if auto_mask:
      mask = _auto_mask(depth, K, cam_in_ob,
                        ob_radius=ob_radius, ob_box=ob_box)
      if save_auto_masks:
        cv2.imwrite(os.path.join(mask_dir, f'{stem}.png'), mask)
    else:
      mask = cv2.imread(os.path.join(mask_dir, f'{stem}.png'), -1)
      if mask is None:
        # 这一帧手动 mask 缺失, 跳过
        continue
      if mask.ndim == 3:
        mask = mask[..., 0]

    rgbs.append(rgb)
    depths.append(depth)
    masks.append(mask)
    poses.append(cam_in_ob)

  if not rgbs:
    raise RuntimeError(f'no usable frames after loading from {out_dir}')

  return (K, np.asarray(rgbs), np.asarray(depths),
          np.asarray(masks), np.asarray(poses), common, pose_name)


def reconstruct(out_dir, cfg_path=None, recon_cfg=None,
                ob_radius=0.15, ob_box=None, tex_res=None,
                max_frames=150, subsample_mode='farthest'):
  """主入口. 读 out_dir 下的 rgb/depth/(masks/)/(ob_in_cam|cam_in_ob)
  -> 训 NOF -> 写 mesh.
  tex_res: 贴图分辨率. 默认 None = 用 upstream 的 1028. 设 2048 可显著提升细节."""
  # 延迟 import: 让 sys.path 修改生效再触发 nerf_runner / torch
  from run_nerf import run_neural_object_field

  if tex_res is not None:
    # upstream run_neural_object_field 调 mesh_texture_from_train_images(tex_res=1028)
    # 是显式 kwarg, 所以包装函数里的 tex_res 参数会被这个 1028 覆盖.
    # 必须通过 closure 用不同名字捕获, 然后在 wrapper 里强行覆盖 kwargs[tex_res].
    import nerf_runner as _nr
    _orig = _nr.NerfRunner.mesh_texture_from_train_images
    _forced_res = int(tex_res)
    def _hires_bake(self, mesh, rgbs_raw, **kwargs):
      kwargs['tex_res'] = _forced_res            # 无视调用方传的 1028
      return _orig(self, mesh, rgbs_raw, **kwargs)
    _nr.NerfRunner.mesh_texture_from_train_images = _hires_bake
    print(f'[reconstruct] tex_res override -> {_forced_res} (default 1028)')

  # 压掉 nerf_runner.py:1227 的 nan->uint8 cast warning.
  # 验证过那些 NaN texel 数量为 0, 是 NOF 烤贴图未覆盖区域的边缘噪音, 无害.
  import warnings
  warnings.filterwarnings(
      'ignore',
      message='invalid value encountered in cast',
      category=RuntimeWarning)

  # save_dir 必须在挂 FileHandler 之前清空+建好, 否则下面的 rmtree 会把刚打开的
  # train.log unlink 掉, 后续 logging.info 写到幽灵文件, 最终磁盘上找不到 log.
  save_dir = os.path.join(out_dir, 'nerf')
  if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
  os.makedirs(save_dir, exist_ok=True)

  # 给 root logger 加 FileHandler, 把 NerfRunner 的 "Iter: ... loss: ..."
  # 行落盘到 nerf/train.log, 便于事后用 inspect_loss.py 画收敛曲线.
  import logging as _logging
  _log_path = os.path.join(save_dir, 'train.log')
  _root = _logging.getLogger()
  if _root.level > _logging.INFO or _root.level == 0:
    _root.setLevel(_logging.INFO)
  # 移除残留的同路径 FileHandler, 避免追加多次
  for h in list(_root.handlers):
    if isinstance(h, _logging.FileHandler) and \
        os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(_log_path):
      _root.removeHandler(h)
  _fh = _logging.FileHandler(_log_path, mode='w')
  _fh.setFormatter(_logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
  _fh.setLevel(_logging.INFO)
  _root.addHandler(_fh)
  print(f'[reconstruct] training log -> {_log_path}')

  K, rgbs, depths, masks, cam_in_obs, stems, pose_name = _load_session(
      out_dir, ob_radius=ob_radius, ob_box=ob_box,
      max_frames=max_frames, subsample_mode=subsample_mode)
  print(f'[reconstruct] loaded {len(rgbs)} frames; pose source = {pose_name}')
  print(f'[reconstruct] K=\n{K}')

  if cfg_path is None:
    cfg_path = os.path.join(_BSDF_DIR, 'config_ycbv.yml')
  with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

  recon_cfg = recon_cfg or {}
  if 'n_step' in recon_cfg:
    cfg['n_step'] = int(recon_cfg['n_step'])
  if 'mesh_resolution' in recon_cfg:
    cfg['mesh_resolution'] = float(recon_cfg['mesh_resolution'])
  if 'i_print' in recon_cfg:
    cfg['i_print'] = int(recon_cfg['i_print'])

  mesh = run_neural_object_field(
      cfg=cfg, K=K, rgbs=rgbs, depths=depths, masks=masks,
      cam_in_obs=cam_in_obs, save_dir=save_dir, debug=0,
  )

  out_mesh_dir = os.path.join(out_dir, 'mesh')
  os.makedirs(out_mesh_dir, exist_ok=True)
  mesh_path = os.path.join(out_mesh_dir, 'model.obj')
  mesh.export(mesh_path)
  print(f'[reconstruct] mesh -> {mesh_path}')

  # 自动打印产物尺寸 + 贴图分辨率, 免得用户手动验
  ext_mm = mesh.extents * 1000.0
  print(f'[reconstruct] extents = [{ext_mm[0]:.2f}, {ext_mm[1]:.2f}, '
        f'{ext_mm[2]:.2f}] mm   mean = {ext_mm.mean():.2f} mm   '
        f'span = {ext_mm.max() - ext_mm.min():.2f} mm')
  tex_path = os.path.join(out_mesh_dir, 'material_0.png')
  if os.path.exists(tex_path):
    tex_im = cv2.imread(tex_path, -1)
    print(f'[reconstruct] texture = {tex_im.shape[0]}x{tex_im.shape[1]} '
          f'({os.path.getsize(tex_path) / 1024 / 1024:.2f} MB)')
  return mesh_path
