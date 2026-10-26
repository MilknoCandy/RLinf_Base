# RoboTwin 2.0 π₀.₅ 全任务评测（RLinf）

在 RLinf 里评测 OpenPI π₀.₅ 在 **RoboTwin 2.0 全部 50 个任务** 上的成功率（Easy / Hard），用来筛 RLT Stage-2 任务。

协议：Aloha-AgileX、14 维关节、头部 + 左右腕三相机、action chunk 50、每任务 100 条轨迹。指标：`eval/success_once`。

**评测不需要演示数据集。** 只要仿真资产 + 已转换成 RLinf OpenPI 布局的 π₀.₅ 权重。Stage-1 混训用官方 `RoboTwin_lerobot_v21.zip`；Stage-2 才按筛出来的任务做 online RL。

---

## 0. 准备清单

按顺序做完再扫表：

1. 安装 RLinf OpenPI + RoboTwin 依赖
2. 克隆 `RLinf_support` 分支的 RoboTwin，下载 **仿真资产**（meshes / URDF，不是演示）
3. 下载 **50 任务共训** π₀.₅，转成 RLinf 能加载的目录（见第 3 节）
4. 确认 `MODEL_PATH` 下有 `model.safetensors` 或 `full_weights.pt`，并备好 `norm_stats.json`
5. `export ROBOTWIN_PATH` / `ROBOT_PLATFORM=ALOHA`，跑 `eval_pi_sft_robotwin_exp.sh`

仓库里 `eval_seeds.json` 已带 22 个任务的官方评测种子，不用另下。其余 28 个任务评测时随机采样 reset seed。

---

## 1. 安装

```bash
bash requirements/install.sh embodied --model openpi --env robotwin
source .venv/bin/activate          # 已有 conda 环境则 conda activate 即可
```

本机 explorer 环境示例：`conda activate sdw_rlinf-pi312`。

---

## 2. 仿真数据：RoboTwin 仓库 + 资产

评测时 `RoboTwinEnv` 通过 `robotwin.envs.vector_env.VectorEnv` 起 50 个任务的仿真。需要的是 **代码 + assets**，不是 HuggingFace 上那 1.38TB 演示。

```bash
# 必须用 RLinf_support，不要用官方 main 直接塞进 RLinf
git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
cd RoboTwin
bash script/_download_assets.sh
```

脚本把资源放到 `RoboTwin/assets/`。YAML 里的 `env.eval.assets_path` / 环境变量 `ASSETS_PATH` 填 **`assets/` 的上一级**：

```bash
export ROBOTWIN_PATH=/abs/path/to/RoboTwin
export ROBOT_PLATFORM=ALOHA
# assets_path == $ROBOTWIN_PATH
ls "${ROBOTWIN_PATH}/assets/objects/objaverse/list.json"
```

`eval_pi_sft_robotwin_exp.sh` / `_mtp.sh` 启动时都会无条件执行（与 `install_robotwin2.sh` 相同）：

```bash
ln -sfn "${RLinf_DATAROOT}/RoboTwin/assets" "${ROBOTWIN_PATH}/assets"
```

MTP 不必手链；共享资产在 `$RLinf_DATAROOT/RoboTwin/assets`（intern 数据集）。

`evaluations/run_eval.sh`、`eval_pi_sft_robotwin_exp.sh` 和 `eval_pi_sft_robotwin_mtp.sh` 都会把 `$ROBOTWIN_PATH` 加进 `PYTHONPATH`。

### 评测不需要下的东西

| 资源 | 评测 | Stage-1 混训 | Stage-2 单任务 RL |
|---|---|---|---|
| `script/_download_assets.sh` 仿真资产 | 要 | — | 要 |
| `eval_seeds.json`（仓库已有） | 要 | — | 要 |
| `lerobot_dataset/RoboTwin_lerobot_v21.zip` | **不要** | **要**（50 任务 mix） | — |
| `lerobot_dataset/RoboTwin_lerobot_v30.zip` | **不要** | **不要**（本环境是 LeRobot v2.1） | — |
| `dataset/*/demo_clean.zip` | **不要** | 仅当 v21 不可用 | — |
| `demo_randomized` 每任务 500 条 | **不要** | **不要** | 一般不要 |

---

## 3. 模型：下哪份、怎么转成 RLinf 能加载的格式

RLinf OpenPI 评测 **只认** 下面两种目录（见 `rlinf/models/embodiment/openpi/__init__.py`）：

```
MODEL_PATH/
  model.safetensors          # OpenPI 裸 Pi0 键名：img.* / llm.* / action_in_proj.*
  config.json                # 可选；形状以 YAML 为准
  .../norm_stats.json
```

或 RLinf SFT：

```
MODEL_PATH/                  # 或 .../actor 或 .../global_step_N
  actor/model_state_dict/full_weights.pt
```

LeRobot 的 `PI05Policy`、`pretrained_model/`、`config.json` 里 `type: pi05` **不能**直接当 `MODEL_PATH`。官方 `cotrain_ckpt/Pi_05/*.tar.gz` 文件名带 `lerobot_v30`，解压后必须先看格式再转。

### 3.1 推荐：JAX OpenPI 的 50 任务共训（最容易转进 RLinf）

50 任务 × 50 条 `demo_clean`、chunk 50、Aloha-AgileX 的 JAX/Orbax 权重：

[Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k](https://huggingface.co/Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k)

```bash
hf download Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k \
  --exclude "checkpoints/*" \
  --local-dir /path/to/pi05_jax_robotwin
# 目录里应有 params/（Orbax）以及 assets/.../norm_stats.json
```

转成 RLinf OpenPI（在 `RLinf_Base` 根目录、`sdw_rlinf-pi312` 下跑）。这份 JAX dump 的每个数组是 Flax `{value: ndarray}`，`jax_to_openpi` 会拆掉；权重本身是完整的 `PaliGemma` + 四个 proj，不是坏包。

```bash
cd /path/to/RLinf_Base
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
  --input-model /path/to/pi05_jax_robotwin \
  --input-norm-stats /path/to/pi05_jax_robotwin/assets/aloha_agilex_sim/norm_stats.json \
  --output-model /path/to/pi05_robotwin_openpi \
  --output-norm-stats /path/to/pi05_robotwin_openpi/physical-intelligence/robotwin/norm_stats.json \
  --action-horizon 50 \
  --action-dim 32
```

`--input-norm-stats` 换成目录里实际的 `assets/aloha_agilex_sim/norm_stats.json`。需要 `jax` / `orbax`。评测 YAML 里 env 动作维是 14，`config.json` 的 `--action-dim 32` 是 OpenPI 内部 padded dim。

### 3.2 官方 leaderboard 共训（解压后按格式选转换）

[TianxingChen/RoboTwin2.0 `cotrain_ckpt/Pi_05`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/cotrain_ckpt/Pi_05)
（`RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz`，约 12.4 GB）

```bash
huggingface-cli download TianxingChen/RoboTwin2.0 \
  --repo-type dataset \
  --include "cotrain_ckpt/Pi_05/*" \
  --local-dir /path/to/robotwin_ckpts

tar -xzf /path/to/robotwin_ckpts/cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz \
  -C /path/to/extracted_pi05
```

解压后用目录结构判断，**不要猜**：

```bash
find /path/to/extracted_pi05 -maxdepth 3 \( \
  -name 'params' -o -name 'model.safetensors' -o -name 'norm_stats.json' \
  -o -name 'config.json' -o -name 'pretrained_model' \)
```

| 你看到的 | 含义 | 怎么转 |
|---|---|---|
| `params/`（Orbax 多文件） | JAX OpenPI | `--mode jax_to_openpi`（第 3.1 节命令） |
| `model.safetensors`，键名 `paligemma_with_expert.*` | OpenPI PyTorch | `--mode openpi_pytorch_to_openpi` |
| `model.safetensors`，键名 `img.*` / `llm.*` | 已是 RLinf OpenPI | 直接当 `MODEL_PATH`，只复制 `norm_stats.json` |
| `config.json` 含 `PI05Policy` / `lerobot`，或只有 `pretrained_model/` | LeRobot π₀.₅ | **RLinf OpenPI 加载不了**，改用 3.1 的 JAX 共训，或在 RLinf 里自己 SFT（3.4） |

查 safetensors 键名前缀：

```bash
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
p = next(Path("/path/to/extracted_pi05").rglob("model.safetensors"))
with safe_open(p, framework="pt") as f:
    keys = list(f.keys())[:8]
print("file", p)
print("keys", keys)
PY
```

OpenPI PyTorch → OpenPI：

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode openpi_pytorch_to_openpi \
  --input-model /path/to/extracted_pi05 \
  --input-norm-stats /path/to/extracted_pi05/norm_stats.json \
  --output-model /path/to/pi05_robotwin_openpi \
  --output-norm-stats /path/to/pi05_robotwin_openpi/physical-intelligence/robotwin/norm_stats.json
```

### 3.3 转换完成后必须检查

```bash
ls /path/to/pi05_robotwin_openpi/model.safetensors
ls /path/to/pi05_robotwin_openpi/physical-intelligence/robotwin/norm_stats.json
```

`norm_stats.json` 必须是 **这份共训权重自己的统计**，不能拿单任务 `adjust_bottle` 的统计去解共训模型的动作。通用 YAML 已写死：

```
rollout.model.openpi_data.norm_stats_path: ${model_path}/physical-intelligence/robotwin/norm_stats.json
rollout.model.openpi_data.adapt_to_pi: false
rollout.model.openpi_data.extra_delta_transform: true
```

JAX AgileX 共训（`pi05jax_robotwin`）**不能**走 Trossen 的关节翻转 / gripper 角度映射。`pi05_aloha_robotwin` 默认 `adapt_to_pi=True`（给 RLinf 自己的 π₀.₅ SFT 用）。评测这份转换权重必须 `false`，否则 `success_once` 会整表打成 0。若评测的是 RLinf 自己训的 SFT ckpt，再加：

```bash
EXTRA_OVERRIDES="rollout.model.openpi_data.adapt_to_pi=true"
```

### 3.4 没有现成共训时：自己用 50 任务 `demo_clean` 在 RLinf SFT

1. 下 JAX 基座 `gs://openpi-assets/checkpoints/pi05_base`（或 `lerobot/pi05_base` 再转）
2. `jax_to_openpi` 得到 `/path/to/pi05_base_openpi`
3. 下 50 任务 × 50 条 clean（`download_xpolicylab_data.sh` 不带参数，或 `RoboTwin_lerobot_v30.zip`）
4. 转成 OpenPI / LeRobot，算 `norm_stats.json`
5. 用 `config_name: pi05_aloha_robotwin`、`action_dim: 14`、`num_action_chunks: 50`、`num_images_in_input: 3` 做 SFT
6. `MODEL_PATH` 指向 `checkpoints/global_step_<N>`（可读 `full_weights.pt`）

### 3.5 不要用来做 50 任务扫表的权重

| 权重 | 原因 |
|---|---|
| `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` | 单任务 SFT，换任务会塌 |
| `RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle` | π₀ + 单任务 |
| `gs://openpi-assets/checkpoints/pi05_base` 未 SFT | 基座，没在 RoboTwin 上 finetune |
| `niejunnan25/pi05-robotwin` | 50 个**单任务** JAX（约 622 GB），不是一个共训模型 |

---

## 4. 为什么是 50 个任务，不是 46

RoboTwin 2.0 官方套件就是 **50** 个双臂任务（`RoboTwin/envs/*.py`）。本扫表默认 `TASKS=all`，**50 个都会跑**。

仓库训练文档里写过 4 个任务「尚未支持」：`open_laptop`、`place_fan`、`place_object_scale`、`put_object_cabinet`。那是因为 **RL 训练用的 dense reward 还没接**。本评测 `use_custom_reward: False`，只读仿真 `success`，这 4 个照样进 VectorEnv。

RLinf 在 `rlinf/envs/sim/robotwin/eval_compat.py` 里打了评测补丁：`open_laptop` / `put_object_cabinet` / `place_object_scale` 在 `check_success` 里补上 `arm_tag`（policy eval 不走 `play_once`）；`place_fan` 的 `Reward()` 构造函数不匹配会被接住；`SubEnv.setup_task` 最多试 25 个 seed，不再无限重试把扫表挂死。`place_bread_basket` 这类 `vk::Device::createFenceUnique: ErrorOutOfHostMemory` 是 SAPIEN 主机内存，不是逻辑 bug，扫表脚本会记 rc=255 后继续，可用 `START_FROM=` 单任务重跑。

| `TASKS` | 数量 | 用途 |
|---|---|---|
| `all`（默认） | **50** | 全任务 π₀.₅ 评测 |
| `supported` | 46 | 仅当上述 4 个 VectorEnv 崩溃时再跳过 |
| `seeded` | 22 | 只用 `eval_seeds.json` 里的官方 seed（可复现子集） |
| `handover_block,stamp_seal` | 自定义 | 逗号分隔 |

50 个任务名：

```
adjust_bottle, beat_block_hammer, blocks_ranking_rgb, blocks_ranking_size,
click_alarmclock, click_bell, dump_bin_bigbin, grab_roller, handover_block,
handover_mic, hanging_mug, lift_pot, move_can_pot, move_pillbottle_pad,
move_playingcard_away, move_stapler_pad, open_laptop, open_microwave,
pick_diverse_bottles, pick_dual_bottles, place_a2b_left, place_a2b_right,
place_bread_basket, place_bread_skillet, place_burger_fries, place_can_basket,
place_cans_plasticbox, place_container_plate, place_dual_shoes, place_empty_cup,
place_fan, place_mouse_pad, place_object_basket, place_object_scale,
place_object_stand, place_phone_stand, place_shoe, press_stapler,
put_bottles_dustbin, put_object_cabinet, rotate_qrcode, scan_object,
shake_bottle, shake_bottle_horizontally, stack_blocks_three, stack_blocks_two,
stack_bowls_three, stack_bowls_two, stamp_seal, turn_switch
```

`eval_seeds.json` 里这 22 个有官方 success seed：`adjust_bottle`、`place_phone_stand`、`click_bell`、`place_a2b_left`、`move_can_pot`、`handover_mic`、`place_a2b_right`、`pick_dual_bottles`、`lift_pot`、`put_bottles_dustbin`、`stack_blocks_two`、`stack_bowls_two`、`handover_block`、`shake_bottle`、`place_empty_cup`、`place_container_plate`、`move_stapler_pad`、`beat_block_hammer`、`blocks_ranking_rgb`、`move_pillbottle_pad`、`place_mouse_pad`、`place_shoe`。其余任务随机 reset seed（`robotwin_env.py` 缺 key 时回退，不会崩）。

---

## 5. 跑全任务扫表

```bash
export ROBOTWIN_PATH=/abs/path/to/RoboTwin
export ROBOT_PLATFORM=ALOHA

MODEL_PATH=/path/to/pi05_robotwin_openpi \
  TASKS=all SETTING=both N_TRIALS=100 \
  bash evaluations/eval_pi_sft_robotwin_exp.sh
```

默认 `TASKS=all`、`SETTING=both`：50 个任务全部评，每个任务都判关键阶段；先 Easy 再 Hard。目录是同一时间戳下的 `...-clean-all` 和 `...-randomized-all`。只跑一边时设 `SETTING=clean` 或 `SETTING=hard`。

MTP 8 卡（协议不变，只加大并行 env。`TOTAL_NUM_ENVS` **必须整除 GPU 数**，`env,rollout: all` 时 `env_world_size == N_GPU`；8 卡用 16/24/32/40）：

```bash
MODEL_PATH=/path/to/pi05_robotwin_openpi \
  TASKS=all SETTING=both N_TRIALS=100 \
  bash evaluations/eval_pi_sft_robotwin_mtp.sh
```

| 变量 | 默认 | 含义 |
|---|---|---|
| `MODEL_PATH` | 必填 | 第 3 节转换后的目录 |
| `ROBOTWIN_PATH` | `$CODE_DEPS/RoboTwin` | `RLinf_support` 根目录 |
| `TASKS` | `all` | **50 个全跑**，每个都有关键阶段门控 |
| `SETTING` | `both` | `both`=Easy 再 Hard；`clean`=Easy；`randomized`/`hard`=Hard |
| `N_TRIALS` | `100` | 官方每任务 100 条 |
| `TOTAL_NUM_ENVS` | explorer `8` / MTP `N_GPU*2`（8 卡=16） | 并行 env，必须整除 `N_GPU`。公平评测只加大这个，不要靠加 seed。轮数 = `ceil(N_TRIALS / TOTAL_NUM_ENVS)` |
| `SEEDS` | `0` | 评测 RNG，可 `SEEDS="0 1 2"` |
| `SAVE_VIDEO` | `False` | 建议筛任务时先关，确认后再开 |
| `START_FROM` | 空 | 从该任务名接着跑；默认必须空，否则会跳过前面的任务 |
| `EXTRA_OVERRIDES` | 空 | 追加 Hydra。JAX 共训不必再写 `norm_stats_path` / `adapt_to_pi` |

YAML 协议（`robotwin_openpi_pi05_eval.yaml` + `env/robotwin_aloha_pi05.yaml`）：

- `embodiment: [aloha-agilex]`，腕部相机开，`center_crop: False`
- `num_action_chunks: 50`，`action_dim: 14`，`num_images_in_input: 3`
- `openpi.config_name: pi05_aloha_robotwin`，`openpi.task: eval`
- `openpi_data.adapt_to_pi: false`，`extra_delta_transform: true`（JAX AgileX 共训配方）
- Easy 关闭全部 DR；Hard 打开 `random_background` / `cluttered_table` / `random_light`

`ignore_terminations: True` 时 `episode_len` 会钉死在 `step_lim`（`adjust_bottle` 是 200）、`std=0`，这是协议不是失败。真正的成功率看 `success_once`。`success_once=0` 且 `return=0` 时先查 `adapt_to_pi` / `norm_stats`，不要先怀疑转换。

列出任务 / 查步数 / 查该任务怎么判「进了关键阶段」：

```bash
python evaluations/robotwin/robotwin_tasks.py --list all
python evaluations/robotwin/robotwin_tasks.py --step-lim handover_mic
python evaluations/robotwin/robotwin_tasks.py --gate handover_mic
```

单任务、不走扫表：

```bash
bash evaluations/run_eval.sh robotwin robotwin_openpi_pi05_eval \
  rollout.model.model_path=/path/to/pi05_robotwin_openpi \
  env.eval.assets_path=/path/to/RoboTwin \
  env.eval.task_config.task_name=handover_block \
  env.eval.task_config.step_lim=400 \
  env.eval.max_episode_steps=400
```

---

## 6. 结果与筛选

目录：`$RLinf_OUTPUTS/rlinf_outputs/logs/<timestamp>-robotwin_openpi_pi05_eval-<setting>-<tasks>/`（`REPO_PATH/logs` 只是软链；MTP `schedule-train` 上这条链经常是文件或空链，脚本不再往里面 `mkdir`）。

- `runs.tsv`：task / setting / seed / run_dir / rc
- `<task>-<setting>-seed<seed>/eval_embodiment.log`
- `rlt_screen.txt`

### 如何判断「进入关键阶段」

每个 action chunk（不是整集结束才看一次）：

1. `VectorEnv.SubEnv.step` 跑完 `check_success()`，得到整集 `success`
2. `eval_compat.patch_critical_phase_info` 在同一 live task 上调用 `critical_phase.progress_into_critical(task)`
3. 几何门控按**任务类名**查 `GATES`（50 个都有）。**复位时已经为真的姿态不算进入**：只有从 False 升到 True 才记 `in_critical`
4. 写入 `info.in_critical`（progress），`info.critical_gate` 只给 env 内部用
5. `RoboTwinEnv._record_metrics` 按 episode 锁存：
   - `reach_critical_once |= in_critical | success`（成功一定算进入）
   - `fail_in_critical = reach_critical_once & ~success_once`
6. episode done 时 Metric Table 只打 `eval/success_once`、`eval/reach_critical_once`、`eval/fail_in_critical`。**不要看 `critical_gate`**：那是「有没有检测器」，50 个任务都有，所以旧日志里永远是 1.0，不是进入率

门控是官方 Reward 树里 **Success 之前那一招的前提**，不是成功本身，也不是「靠近过目标」。例如 `handover_mic`：接触麦克风且 `z>0.85` = 进了交接；对侧臂接住才是成功。`beat_block_hammer`：抓起锤子（接触 + z>0.82）= 进了 Place；锤子砸到 block 2cm 才是成功。`adjust_bottle` 出生时 `|x|≈0.10`，旧门控 `|x|>0.05` 在 t=0 就真，所以 reach 会被刷成 1.0。

`click_*` 的最后一招就是点按钮，接触 ≈ 成功，所以 `reach ≈ success`、`failC≈0` 是对的：这类任务没有残差窗口。

exp / mtp 脚本每个任务开头会打印 `critical-gate: <task> <last_skill> <enter_when>`。查一条：

```bash
python evaluations/robotwin/robotwin_tasks.py --gate stack_bowls_three
```

整集 `eval/success_once` **不能**当 RLT 筛选：它混了「粗阶段没进关键段」和「进了但关键段失败」。筛的是下面三个数（Metric Table 会打出来）：

| 指标 | 含义 |
|---|---|
| `reach_critical_once` | P(进入关键阶段)。**筛任务看这个**，不是 `critical_gate` |
| `success_once` | P(整集成功) ≤ reach |
| `fail_in_critical` | `reach - success`，关键段失败、残差有活干 |
| `S\|R` | `success / reach`，进了之后还能不能成 |

| band | 条件 | 动作 |
|---|---|---|
| `keep` | Reach ≥ 0.70 且 S\|R ≤ 0.90 且 failC ≥ 0.10 | 留给 RLT Stage-2 |
| `coarse_fail` | Reach < 0.70 | 丢掉，粗策略到不了关键段 |
| `saturated` | 进了之后 S\|R > 0.90 | 丢掉，关键段已经满了 |
| `low_headroom` | 进得去但 failC < 0.10 | 丢掉，残差没数据 |
| `no_gate` | 该任务还没有关键阶段门控 | 先写 gate，再评 |
| `*/episode` | 旧日志只有 `success_once` | 只能预筛，不能定 keep |

已有的两趟 Easy 只有 `success_once`，**不能**据此定 RLT 任务。必须用改过的脚本把 **50 个任务 × Easy+Hard** 再跑一遍，看 `reach_critical_once`。50 个门控定义在 `rlinf/envs/sim/robotwin/critical_phase.py` 的 `GATES`。

```bash
python evaluations/robotwin/summarize_robotwin_eval.py \
  --runs-tsv logs/<sweep>/runs.tsv \
  --append logs/<sweep>/rlt_screen.txt
```

扫表必须看每个任务的 `reach_critical_once`，不要只看 Easy 成功率。Easy+Hard 都按 Reach / S|R / failC 交叉；Hard 的 Reach 也不能塌到 0.55 以下。筛表是给 **Stage-2 单任务 online RL** 用的。Stage-1 是 50 任务 `demo_clean` 混训，不必先筛完再下数据。

### Stage-1 训练（50 任务混训）

和 ManiSkill 一阶段同一条链路：`examples/sft/train_vla_sft.py` + OpenPI 官方 LeRobot loader + `openpi.use_rlt=True`。`model_type` 必须是 `openpi`。数据用官方 **LeRobot v2.1 整包**（`lerobot_dataset/RoboTwin_lerobot_v21.zip`，约 2.4 GB），不要用 v30（本环境 LeRobot 是 v2.1）。语言从每条 episode 的 task 读（`prompt_from_task`），不是单任务 prompt。

```bash
# explorer 2-GPU（micro 2 / global 32，开 grad ckpt）
bash scripts/robotwin/run_pi_rlt_stage1_exp.sh

# MTP 8 x 80 GiB（micro 2 / global 64 / 1000 step，开 grad ckpt；RoboTwin 三相机 chunk-50 不能抄 ManiSkill 的 micro 8）
bash scripts/robotwin/run_pi_rlt_stage1_mtp.sh
```

脚本会把 zip 解到 `$RLinf_OUTPUTS/robotwin_lerobot_v21/`，缓存 zip 在 `$RLinf_OUTPUTS/robotwin_data/download_cache/`。默认 `MODEL_PATH` 是已转换的 50 任务 π₀.₅，`norm_stats` 用旁边的 `physical-intelligence/robotwin/norm_stats.json`。ckpt 在 `$RLinf_OUTPUTS/rlinf_outputs/logs/<stamp>-robotwin_rlt_stage1_...-mix/`，Stage-2 单任务 online RL 的 `model_path` 指这里的 `checkpoints/global_step_<N>/actor`。

不要用 `dataset/*/demo_clean.zip` 再自己转，除非官方 v21 不可用。HDF5 转换器仍留着当兜底。

---

## 7. 本目录与相关文件

本目录：

```
evaluations/robotwin/README.md
evaluations/robotwin/robotwin_openpi_pi05_eval.yaml
evaluations/robotwin/robotwin_tasks.py
evaluations/robotwin/summarize_robotwin_eval.py
evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_eval.yaml
evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml
evaluations/robotwin/robotwin_place_empty_cup_openpi_eval.yaml
evaluations/robotwin/robotwin_place_empty_cup_openvlaoft_eval.yaml
evaluations/robotwin/robotwin_place_shoe_lingbotvla_eval.yaml
evaluations/robotwin/robotwin_click_bell_lingbotvla_eval.yaml
```

扫表与转换：

```
evaluations/eval_pi_sft_robotwin_exp.sh
evaluations/eval_pi_sft_robotwin_mtp.sh
evaluations/run_eval.sh
evaluations/eval_embodied_agent.py
examples/embodiment/config/env/robotwin_aloha_pi05.yaml
examples/embodiment/config/model/pi0_5.yaml
rlinf/envs/sim/robotwin/robotwin_env.py
rlinf/envs/sim/robotwin/eval_compat.py
rlinf/envs/sim/robotwin/critical_phase.py
rlinf/envs/sim/robotwin/seed_utils.py
rlinf/envs/sim/robotwin/seeds/eval_seeds.json
rlinf/models/embodiment/openpi/__init__.py
rlinf/models/embodiment/openpi/dataconfig/__init__.py    # pi05_aloha_robotwin
rlinf/utils/ckpt_convertor/openpi/README.md
tests/unit_tests/test_robotwin_eval.py
rlinf/data/datasets/openpi/robotwin_hdf5.py
toolkits/lerobot/convert_robotwin_demo_to_lerobot.py
scripts/download_xpolicylab_data.sh
scripts/robotwin/download_demo_clean.py
scripts/robotwin/download_lerobot_v21.py
scripts/robotwin/run_pi_rlt_stage1_exp.sh
scripts/robotwin/run_pi_rlt_stage1_mtp.sh
examples/sft/config/robotwin_rlt_stage1_sft_openpi_pi05.yaml
examples/sft/config/robotwin_rlt_stage1_sft_openpi_pi05_exp.yaml
examples/sft/config/robotwin_rlt_stage1_sft_openpi_pi05_mtp.yaml
```
