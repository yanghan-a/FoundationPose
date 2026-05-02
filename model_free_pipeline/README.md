# FoundationPose Model-Free Pipeline

> **场景**: 你已经有仿真 mesh 在 `track_single.py` 里跑 model-based 跟踪, 但
> 仿真和真相机色差大, 只有部分帧能稳定匹配. 想用稳定那些帧的 pose 当标签,
> 重建一个**纹理来自真相机**的新 mesh, 之后所有帧都能稳定.

## 工作流

```
仿真粗 mesh ──┐
              ▼
 ┌──────────────────────────┐
 │ 1. harvest.py            │  ── 跑 model-based 跟踪, 你眼睛看 overlay,
 │   照搬 track_single.py   │     对得好的帧按 SPACE 落盘 (rgb/depth/ob_in_cam)
 │   只在按键里加 SPACE save│
 └────────────┬─────────────┘
              ▼
   out/rgb/  out/depth/  out/ob_in_cam/  out/cam_K.txt
              ▼
 ┌──────────────────────────┐
 │ 2. reconstruct.py        │  ── 调 BundleSDF NerfRunner
 │   读上面的标签数据       │     mask 没给的话用 depth+pose 自动派粗 mask
 │   自动派 mask            │
 └────────────┬─────────────┘
              ▼
       out/mesh/model.obj      (纹理来自真相机)
              ▼
 ┌──────────────────────────┐
 │ 3. track_single.py       │  ── 用新 mesh 替代仿真 mesh 跟踪, 不再色差
 └──────────────────────────┘
```

## 用法

### 第一步: harvest 数据
```bash
cd /home/l/FoundationPose
python -m model_free_pipeline.pipeline harvest --mesh_file demo_data/cube54/mesh/textured_cube54.obj --out real_data/cube54 --color_exposure 5000 --depth_exposure 5000 --color_gain 80 --cam_fps 90 --debug 1
```
窗口里会看到仿真 mesh 在画面上的轴线 + AABB. 移动相机/物体, 看到 overlay 贴
得好就按 **SPACE**. 跟丢按 **R** 重新框选, 退出按 **Q**.

控制要点:
- 至少凑 30-60 帧, 各种视角(俯/侧/旋转)都来一些
- overlay 明显错位的帧不要存, 那种帧 pose 是错的, 标签会污染重建
- 每按一次 SPACE 终端会打 `[harvest] SAVED frame N`, 数着来

输出:
```
demo_data/my_object/
  cam_K.txt
  rgb/000000.png ...
  depth/000000.png ...
  ob_in_cam/000000.txt ...
```

### 第二步: 重建带真纹理的 mesh
```bash
python -m model_free_pipeline.pipeline reconstruct --out real_data/cube54 --ob_radius 0.05        # 物体大致包围球半径 (米)

python -m model_free_pipeline.pipeline reconstruct --out real_data/cube54 --recon_cfg model_free_pipeline/config_best.yml --plot  

python -m model_free_pipeline.pipeline reconstruct --out real_data/cube54 --recon_cfg model_free_pipeline/config_best.yml --tex_res 2048 --plot
```
NOF 训练 ~1k 步, 在 4060 Ti 上几分钟. 输出:
```
demo_data/my_object/
  nerf/                  # 训练中间产物 (可删)
  masks/                 # auto 派的粗 mask (供你检查)
  mesh/
    model.obj
    model.mtl
    material_0.png       # 真相机纹理
```

### 第三步: 用新 mesh 跟踪 (色差问题消失)
```bash
python track_single.py \
    --mesh_file demo_data/my_object/mesh/model.obj \
    --color_exposure 5000 --depth_exposure 5000 --color_gain 80
```

### 一条命令串起 1+2 步
```bash
python -m model_free_pipeline.pipeline all_with_sim_mesh \
    --mesh_file demo_data/my_object/sim_mesh.obj \
    --out demo_data/my_object \
    --ob_radius 0.10
```

## 文件清单

| 文件 | 作用 |
|---|---|
| `harvest.py`     | 跟踪 + SPACE 挑帧落盘. 完全照搬 `track_single.py` 结构, 只多一个 SPACE save. |
| `capture.py`     | 仅 RGB+depth 采集 (你不用 model-based 跟踪、有别的姿态源时用) |
| `reconstruct.py` | 读盘 → BundleSDF NerfRunner → 写带贴图 mesh. 兼容 ob_in_cam/cam_in_ob 命名. |
| `pipeline.py`    | argparse 主入口, 调上面三个 |
| `config.yaml`    | 默认参数 |

## auto-mask 工作原理

如果 `masks/` 不存在, `reconstruct` 会按下列规则给每帧派 mask:
1. depth 反投到 cam 系 → 用每帧 `cam_in_ob` 转到 ob 系
2. ob 系下点距原点 < `ob_radius` (球) 视为物体
3. 形态学闭运算填洞

NOF 对粗 mask 鲁棒(内部 DBSCAN + SDF 收紧表面). `ob_radius` 选大不选小,
物体 ≤ 10cm 用 0.10, ≤ 20cm 用 0.15-0.20.

派出的 mask 会写 `masks/`, 你可以打开看. 不合理就改 `--ob_radius` 重跑.

## 常见问题

**Q: 我的标签数据本来就齐全了 (rgb/depth/pose), 不想跑 harvest.**
A: 跳过第一步, 直接放数据然后跑 reconstruct. ob_in_cam/ 或 cam_in_ob/ 都认.

**Q: harvest 时 overlay 一直对不上怎么办?**
A: 仿真 mesh 和真物体的形状/比例对不上, 这是 model-based 本身的失败, 与色差无关.
先在 `track_single.py` 里调通 register, 再来 harvest.

**Q: 能不能自动挑帧不要按 SPACE?**
A: 现在不能 - FoundationPose 的 track_one 不返回单帧 score. 想自动需要在
harvest 里再调一次 scorer 重新打分, N 倍代价. 手动 SPACE 又快又精确.
