"""inspect_loss.py - 解析 nerf/train.log, 输出收敛诊断 + 可选画 loss 曲线.

用法:
  python -m model_free_pipeline.inspect_loss --out real_data/cube54
  python -m model_free_pipeline.inspect_loss --out real_data/cube54 --plot

诊断指标:
  - 末段 (最后 20%) loss 相对中段 (40%-60%) 的下降幅度. > 5% 还在下降 -> 没收敛.
  - rgb_loss / sdf_loss / fs_loss 各自的趋势 (整体 vs 单项异常).
  - valid_rays 是否一直 = N_rand (=2048), 不是的话 mask/depth 可能太松或太紧.
"""
import os
import sys
import re
import argparse
import numpy as np


_LINE_RE = re.compile(
    r'Iter:\s*(\d+).*?'
    r'valid_samples:\s*(\d+)/(\d+).*?'
    r'valid_rays:\s*(\d+)/(\d+).*?'
    r'loss:\s*([\d.eE+-]+)')

_KV_RE = re.compile(r'(\w+):\s*([\d.eE+-]+)')


def parse_log(log_path):
  """每行: 'Iter: N, valid_samples: A/B, valid_rays: C/D, loss: X, rgb_loss: Y, ...'.
  返回 dict[str -> np.ndarray] 以 iter 为 x 轴."""
  records = []
  with open(log_path, 'r') as f:
    for line in f:
      if 'Iter:' not in line or 'valid_samples' not in line:
        continue
      kvs = dict(_KV_RE.findall(line))
      m = _LINE_RE.search(line)
      if not m:
        continue
      try:
        rec = {
            'iter': int(m.group(1)),
            'valid_samples': int(m.group(2)),
            'valid_samples_total': int(m.group(3)),
            'valid_rays': int(m.group(4)),
            'valid_rays_total': int(m.group(5)),
        }
      except ValueError:
        continue
      for k, v in kvs.items():
        if k in ('Iter',):
          continue
        try:
          rec[k] = float(v)
        except ValueError:
          pass
      records.append(rec)
  if not records:
    raise RuntimeError(f'no Iter lines parsed from {log_path}')

  out = {}
  keys = set()
  for r in records:
    keys.update(r.keys())
  for k in keys:
    out[k] = np.array([r.get(k, np.nan) for r in records], dtype=np.float64)
  return out


def diagnose(d):
  it = d['iter']
  N = len(it)
  if N < 5:
    print(f'[warn] only {N} samples, increase i_print to get a meaningful curve')

  loss = d['loss']

  # 取分位段
  q1 = int(N * 0.40); q2 = int(N * 0.60)
  q3 = int(N * 0.80)
  mid_loss = float(np.mean(loss[q1:q2])) if q2 > q1 else float(loss[N // 2])
  end_loss = float(np.mean(loss[q3:])) if N - q3 > 0 else float(loss[-1])
  drop = (mid_loss - end_loss) / max(abs(mid_loss), 1e-9) * 100

  print(f'\n========== convergence diagnostic ==========')
  print(f'samples: {N} loss prints from iter {int(it[0])} to {int(it[-1])}')
  print(f'  mid (40-60%) loss = {mid_loss:.4f}')
  print(f'  end (80-100%) loss = {end_loss:.4f}')
  print(f'  drop mid -> end   = {drop:+.2f}%   '
        f'({"converging" if drop > 5 else "PLATEAUED" if abs(drop) < 2 else "OK"})')
  if drop > 5:
    print('  → 末段还在掉, n_step 不够, 加大 (建议 +50-100%) 再跑')
  elif drop < -2:
    print('  → 末段反而升了, 不稳, 可能 lr 偏大或 SDF/depth 数值不稳')
  else:
    print('  → 收敛比较平稳; 如果仍质量差, 改的就不是步数问题')

  # 各分项末段值
  for k in ('rgb_loss', 'sdf_loss', 'fs_loss', 'depth_loss',
           'eikonal_loss', 'pose_reg', 'reg_features'):
    if k in d:
      v = d[k]
      print(f'  {k:18s} start={v[0]:.4f}  mid={np.mean(v[q1:q2]):.4f}  '
            f'end={np.mean(v[q3:]) if N - q3 > 0 else v[-1]:.4f}')

  # valid_rays sanity
  vr = d['valid_rays']
  vr_total = d['valid_rays_total']
  vr_frac = vr / np.maximum(vr_total, 1)
  print(f'  valid_rays fraction: min={vr_frac.min():.3f} mean={vr_frac.mean():.3f}')
  if vr_frac.mean() < 0.95:
    print('  → 平均有效光线 < 95%, mask/depth 太松或物体出框过多')

  return mid_loss, end_loss, drop


def plot_curve(d, out_png):
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print('[warn] matplotlib not installed, skipping plot')
    return
  fig, ax = plt.subplots(2, 2, figsize=(12, 8))
  it = d['iter']
  ax[0, 0].plot(it, d['loss']); ax[0, 0].set_title('total loss')
  ax[0, 0].set_yscale('log'); ax[0, 0].grid(True, alpha=0.3)
  for k, axx in zip(('rgb_loss', 'sdf_loss', 'fs_loss'),
                    (ax[0, 1], ax[1, 0], ax[1, 1])):
    if k in d:
      axx.plot(it, d[k]); axx.set_title(k)
      axx.set_yscale('log'); axx.grid(True, alpha=0.3)
  for a in ax.ravel():
    a.set_xlabel('iter')
  plt.tight_layout()
  plt.savefig(out_png, dpi=120)
  plt.close(fig)
  print(f'[plot] saved {out_png}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', required=True, help='reconstruct 工作目录')
  p.add_argument('--plot', action='store_true', help='画 loss 曲线 PNG')
  args = p.parse_args()
  log_path = os.path.join(args.out, 'nerf', 'train.log')
  if not os.path.exists(log_path):
    print(f'[error] {log_path} 不存在. 你需要重跑一次 reconstruct '
          f'(本工具依赖新加的 train.log 落盘逻辑)')
    sys.exit(1)

  d = parse_log(log_path)
  diagnose(d)
  if args.plot:
    plot_curve(d, os.path.join(args.out, 'nerf', 'loss_curve.png'))


if __name__ == '__main__':
  main()
