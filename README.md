# VSI-Bench Evaluation

本项目基于 `lmms-eval` 评估框架，用于在 VSI-Bench 上评估多模态模型的空间理解能力。统一评估入口为 `evaluate_all_in_one.sh`。

评估结果和 token 消耗会在评估结束后写入 `logs/`，并通过 `tools/update_eval_dashboard.py` 汇总到 `docs/eval_dashboard.html`。

## 支持模型

当前 `evaluate_all_in_one.sh` 支持以下模型名：

| 模型名 | 类型 |
| --- | --- |
| `gemini_3_1_pro` | API |
| `gemini_3_1_flash_lite` | API |
| `gpt5_4` | API |
| `llava_one_vision_1_5_8b` | 本地模型 |
| `llava_next_video_7b_qwen2` | 本地模型 |
| `internvl3_5_2b` | 本地模型 |
| `internvl3_5_8b` | 本地模型 |
| `qwen3vl_8b` | 本地模型 |
| `qwen3vl_32b` | 本地模型 |
| `qwen2_5vl_72b_api` | API |
| `qwen3vl_235b_a22b_api` | API |
| `internvideo2_5_chat_8b` | 本地模型 |

可以用逗号一次评估多个模型，也可以使用 `--model all` 评估全部支持模型。

## 支持评估模式

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| 均匀采样 | `--num_frames N` | 从视频中均匀采样 N 帧，默认 `16` 帧。 |
| 自己控制 fps 采样 | `--video_sample_fps F` | 本地按 F fps 抽帧后送给模型。 |
| 指定关键帧采样 | `--video_sampling_strategy specific` | 使用 `data/keyframe_mapping.json` 中的关键帧。 |
| 平台控制 fps 采样 | `--video_input_mode file --video_sample_fps F` | 仅 Qwen API 模型支持，上传本地视频文件，由 DashScope 按 fps 抽帧。 |
| 盲测 | `--visual_input_mode none` | 不提供图片或视频，只把问题文本送给模型。 |
| 指定 CUDA 编号 | `--cuda_visible_devices 0,1` | 指定本次评估可见的 GPU 编号，会写入 `CUDA_VISIBLE_DEVICES`。 |

## 命令示例

### 单进程评估

```bash
bash evaluate_all_in_one.sh --model qwen2_5vl_72b_api --benchmark vsibench --num_processes 1
```

### 多进程评估

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2
```

### 均匀采样 16 帧

```bash
bash evaluate_all_in_one.sh --model internvl3_5_8b --benchmark vsibench --num_processes 2 --num_frames 16
```

### 均匀采样 32 帧

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --num_frames 32
```

### 自己控制 1fps 采样

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --video_sample_fps 1
```

### 指定关键帧采样

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --video_sampling_strategy specific
```

### Qwen API 上传视频并由平台按 1fps 采样

```bash
bash evaluate_all_in_one.sh --model qwen2_5vl_72b_api --benchmark vsibench --num_processes 1 --video_input_mode file --video_sample_fps 1
```

### 盲测

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --visual_input_mode none
```

### 评估多个模型

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b,internvl3_5_8b --benchmark vsibench --num_processes 2 --num_frames 16
```

### 评估全部模型

```bash
bash evaluate_all_in_one.sh --model all --benchmark vsibench --num_processes 2 --num_frames 16
```

## 结果汇总

评估完成后，如果需要手动刷新 dashboard，可以执行：

```bash
python tools/update_eval_dashboard.py logs/YYYYMMDD/vsibench/path/to/results.json
```

生成或更新后的 dashboard 文件位于：

```text
docs/eval_dashboard.html
```
