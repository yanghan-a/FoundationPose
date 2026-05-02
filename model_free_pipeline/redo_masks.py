"""redo_masks.py - 在已有 rgb/depth/ob_in_cam 数据上批量补/覆盖 silhouette mask.

适用: 之前 harvest 落的是粗球 mask, 现在想换成 mesh 轮廓 mask 但不想重采.
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import trimesh
import nvdiffrast.torch as dr

_FP = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _FP not in sys.path:
  sys.path.insert(0, _FP)

from Utils import make_mesh_tensors                          # noqa: E402
from model_free_pipeline.harvest import render_silhouette    # noqa: E402


def main():
  p = argparse.ArgumentParser(
      description='Re-render silhouette masks for an existing harvest dir.')
  p.add_argument('--out', required=True, help='harvest 工作目录')
  p.add_argument('--mesh_file', required=True, help='仿真粗 mesh, 用于轮廓渲染')
  args = p.parse_args()

  K = np.loadtxt(os.path.join(args.out, 'cam_K.txt')).reshape(3, 3)
  rgb_files = sorted(glob.glob(os.path.join(args.out, 'rgb', '*.png')))
  if not rgb_files:
    raise RuntimeError(f'no rgb/ frames under {args.out}')
  first = cv2.imread(rgb_files[0])
  H, W = first.shape[:2]

  mesh = trimesh.load(args.mesh_file)
  mt = make_mesh_tensors(mesh)
  glctx = dr.RasterizeCudaContext()

  out_masks = os.path.join(args.out, 'masks')
  os.makedirs(out_masks, exist_ok=True)

  n_ok, n_skip = 0, 0
  for f in rgb_files:
    stem = os.path.splitext(os.path.basename(f))[0]
    pose_f = os.path.join(args.out, 'ob_in_cam', f'{stem}.txt')
    if not os.path.exists(pose_f):
      n_skip += 1
      continue
    pose = np.loadtxt(pose_f).reshape(4, 4)
    sil = render_silhouette(mt, pose, K, H, W, glctx)
    cv2.imwrite(os.path.join(out_masks, f'{stem}.png'), sil)
    n_ok += 1

  print(f'[redo_masks] wrote {n_ok} masks to {out_masks} '
        f'(skipped {n_skip} for missing pose)')


if __name__ == '__main__':
  main()
