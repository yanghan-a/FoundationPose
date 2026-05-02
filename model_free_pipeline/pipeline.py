"""Model-free pipeline 主入口.

子命令:
  harvest      用仿真粗 mesh 跑 model-based 跟踪, 在 overlay 看着对的帧按 SPACE
               落盘 rgb/depth/ob_in_cam. (推荐入口, 直接产生喂给 reconstruct 的数据)
  capture      仅采 rgb+depth (用户已经有别的方式提供 pose 时用)
  reconstruct  读已有 rgb/depth/(masks)/(ob_in_cam|cam_in_ob) -> NOF -> mesh/model.obj
  all_with_sim_mesh  harvest + reconstruct 串起来跑

用法:
  cd /home/l/FoundationPose

  # 推荐: 全流程 - 用仿真 mesh harvest 数据然后重建
  python -m model_free_pipeline.pipeline harvest \\
      --mesh_file demo_data/my_object/sim_mesh.obj \\
      --out demo_data/my_object
  python -m model_free_pipeline.pipeline reconstruct \\
      --out demo_data/my_object --ob_radius 0.10

  # 然后用新 mesh 替代仿真 mesh 做跟踪 (无色差)
  python track_single.py --mesh_file demo_data/my_object/mesh/model.obj
"""
import os
import sys
import argparse
import yaml

_THIS = os.path.dirname(os.path.realpath(__file__))
_FP_ROOT = os.path.dirname(_THIS)
if _FP_ROOT not in sys.path:
  sys.path.insert(0, _FP_ROOT)


def load_cfg(path=None):
  if path is None:
    path = os.path.join(_THIS, 'config.yaml')
  with open(path, 'r') as f:
    return yaml.safe_load(f)


def cmd_capture(args, cfg):
  from model_free_pipeline.capture import record_session
  os.makedirs(args.out, exist_ok=True)
  cam_cfg = cfg['capture'].copy()
  if args.n_frames is not None:
    cam_cfg['n_frames'] = args.n_frames
  saved = record_session(
      out_dir=args.out,
      cam_cfg=cam_cfg,
      n_frames=cam_cfg['n_frames'],
      min_save_interval_ms=cam_cfg.get('min_save_interval_ms', 80),
  )
  print(f'[capture] done, {saved} frames in {args.out}')


def cmd_reconstruct(args, cfg):
  """
  优先级 (高 -> 低):
    1. CLI 参数 (--n_step / --i_print / --mesh_resolution)
    2. --recon_cfg 指定的 yaml (用户给的就是基准, 全权)
    3. 当 --recon_cfg 缺省时 fallback: config.yaml 里 reconstruct: 段的默认
       叠在 bundlesdf/config_ycbv.yml 之上
  """
  from model_free_pipeline.reconstruct import reconstruct
  pipe_recon = cfg.get('reconstruct', {}) or {}
  mask_cfg = cfg.get('mask', {}) or {}
  ob_radius = args.ob_radius if args.ob_radius is not None \
      else mask_cfg.get('ob_radius', 0.15)
  ob_box = mask_cfg.get('ob_box', None)

  # 只装 CLI 显式给的 override; config.yaml 的默认值不算 CLI override.
  cli_overrides = {}
  if args.n_step is not None:
    cli_overrides['n_step'] = args.n_step
  if args.i_print is not None:
    cli_overrides['i_print'] = args.i_print
  if args.mesh_resolution is not None:
    cli_overrides['mesh_resolution'] = args.mesh_resolution

  if args.recon_cfg is None:
    # 没给 yaml: 用 config.yaml.reconstruct 的默认填进 overrides 当弱补丁
    overrides = dict(pipe_recon)
    overrides.update(cli_overrides)
  else:
    # 给了 yaml: yaml 内容全权, 只让 CLI 覆盖
    overrides = dict(cli_overrides)

  reconstruct(args.out,
              cfg_path=args.recon_cfg,
              recon_cfg=overrides,
              ob_radius=ob_radius,
              ob_box=ob_box,
              tex_res=args.tex_res,
              max_frames=args.max_frames,
              subsample_mode=args.subsample_mode)
  if args.plot:
    # reconstruct 跑完顺便给一份收敛诊断 + loss 曲线
    from model_free_pipeline.inspect_loss import parse_log, diagnose, plot_curve
    log_path = os.path.join(args.out, 'nerf', 'train.log')
    if os.path.exists(log_path):
      d = parse_log(log_path)
      diagnose(d)
      plot_curve(d, os.path.join(args.out, 'nerf', 'loss_curve.png'))
    else:
      print(f'[plot] {log_path} 不在, 跳过')


def cmd_harvest(args, cfg):
  """直接转给 harvest.py 的 main(), 但 argparse 解析过的参数透传.
  这里通过把 sys.argv 改装成 harvest.py 期望的形式实现.
  """
  if not args.mesh_file:
    raise SystemExit('harvest requires --mesh_file (sim mesh used for tracking)')
  argv = ['harvest']
  argv += ['--mesh_file', args.mesh_file, '--out', args.out]
  if args.auto_bbox:
    argv += ['--auto_bbox']
  for k in ('cam_width', 'cam_height', 'cam_fps', 'depth_max',
            'color_exposure', 'depth_exposure', 'color_gain', 'white_balance',
            'est_refine_iter', 'track_refine_iter', 'score_vis_topk',
            'debug', 'axis_scale'):
    v = getattr(args, k, None)
    if v is not None:
      argv += [f'--{k}', str(v)]
  old_argv = sys.argv
  sys.argv = argv
  try:
    from model_free_pipeline.harvest import main as harvest_main
    harvest_main()
  finally:
    sys.argv = old_argv


def cmd_all_with_sim_mesh(args, cfg):
  cmd_harvest(args, cfg)
  cmd_reconstruct(args, cfg)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('cmd', choices=[
      'harvest', 'capture', 'reconstruct', 'all_with_sim_mesh'])
  p.add_argument('--out', required=True,
                 help='工作目录. harvest/capture 写入到这里; reconstruct 从这里读 '
                      'rgb/depth/ob_in_cam/cam_K.txt, 同时把 mesh/ nerf/ masks/ '
                      '写回这里. 同一个对象走完整流程时, 两步用同一个 --out.')
  p.add_argument('--cfg', default=None, help='覆盖 config.yaml 路径')
  p.add_argument('--mesh_file', default=None,
                 help='harvest 需要: 仿真粗 mesh 路径')
  p.add_argument('--n_frames', type=int, default=None,
                 help='覆盖 capture.n_frames')
  p.add_argument('--recon_cfg', default=None,
                 help='覆盖 NerfRunner config (默认 bundlesdf/config_ycbv.yml)')
  p.add_argument('--ob_radius', type=float, default=None,
                 help='reconstruct 时 auto-mask 的物体球半径 (米); 默认 0.15')
  p.add_argument('--n_step', type=int, default=None,
                 help='reconstruct 训练步数, 覆盖 cfg 默认 1000')
  p.add_argument('--i_print', type=int, default=None,
                 help='reconstruct 打印 loss 的间隔, 默认 cfg 里的 500. '
                      '调到 50 看更密的收敛曲线')
  p.add_argument('--mesh_resolution', type=float, default=None,
                 help='reconstruct marching cubes 体素 (米), 默认 0.003')
  p.add_argument('--tex_res', type=int, default=None,
                 help='reconstruct 贴图分辨率, 默认 1028. 设 2048 可显著提升纹理细节')
  p.add_argument('--max_frames', type=int, default=150,
                 help='reconstruct 用的最大帧数. 超过会做下采样 (8GB 显存安全位 = 150). '
                      '设 0 或负值表示全用 (慎用, 大数据可能 OOM)')
  p.add_argument('--subsample_mode', choices=['uniform', 'farthest'],
                 default='farthest',
                 help='帧数超 max_frames 时下采样策略. farthest = 在 cam 位置上做 '
                      '远点采样, 视角更分散 (推荐); uniform = 等步长')
  p.add_argument('--plot', action='store_true',
                 help='reconstruct 跑完后画 loss 曲线 + 输出收敛诊断')
  # 透传给 harvest 的相机/估姿参数
  p.add_argument('--auto_bbox', action='store_true')
  p.add_argument('--cam_width', type=int, default=None)
  p.add_argument('--cam_height', type=int, default=None)
  p.add_argument('--cam_fps', type=int, default=None)
  p.add_argument('--depth_max', type=float, default=None)
  p.add_argument('--color_exposure', type=int, default=None)
  p.add_argument('--depth_exposure', type=int, default=None)
  p.add_argument('--color_gain', type=int, default=None)
  p.add_argument('--white_balance', type=int, default=None)
  p.add_argument('--est_refine_iter', type=int, default=None)
  p.add_argument('--track_refine_iter', type=int, default=None)
  p.add_argument('--score_vis_topk', type=int, default=None)
  p.add_argument('--debug', type=int, default=None, choices=[0, 1],
                 help='harvest 0=tracking 阶段静默; 1=显示窗口 (默认)')
  p.add_argument('--axis_scale', type=float, default=None,
                 help='harvest overlay 上坐标轴长度 (米), 默认 0.05')
  args = p.parse_args()

  cfg = load_cfg(args.cfg)
  os.makedirs(args.out, exist_ok=True)

  {
      'harvest': cmd_harvest,
      'capture': cmd_capture,
      'reconstruct': cmd_reconstruct,
      'all_with_sim_mesh': cmd_all_with_sim_mesh,
  }[args.cmd](args, cfg)

  # 跳过 Python 正常 shutdown.
  # 原因: nvdiffrast / OpenGL / CUDA 的 C 库析构互相踩内存, atexit 链上抛
  # `free(): invalid pointer / 已中止 (核心已转储)`. mesh.export 在 reconstruct
  # 内部已经 close+flush, NerfRunner checkpoint 在训练循环里也已经 flush,
  # 此时进程的所有文件 IO 已经落盘, os._exit 跳过 atexit 不会丢任何东西.
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)


if __name__ == '__main__':
  main()
