# VSI-Bench Evaluation

本项目基于 `lmms-eval` 评估框架，用于在 VSI-Bench 上评估多模态模型的空间理解能力。统一评估入口为 `evaluate_all_in_one.sh`。

评估结果和 token 消耗会在评估结束后写入 `logs/`，并通过 `tools/update_eval_dashboard.py` 汇总到 `docs/eval_dashboard.html`。

如果开启自然输出模式，评估样本日志还会生成题目级对错矩阵，输出到 `docs/eval_question_matrix.html`。

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
| 自然输出 | `--answer_mode natural` | 允许模型自然语言解释，最后用 `Final answer: ...` 给可抽取答案。指标仍使用抽取后的受限答案计算。 |
| 评估备注 | `--run_note "文本"` | 给本次评估记录添加自由文本备注，方便记录额外变量或实验条件。 |
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

### 自然输出并生成题目级对错矩阵

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 1 --limit 2 --answer_mode natural
```

### 添加评估备注

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --num_frames 16 --run_note "prompt=v2 temperature=0 data=sampleA"
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

评估记录表会保留全部历史记录，并显示 `备注` 列。图表仍按同一模型和同一采样策略只取最新记录；如果需要对比同模型同采样策略下的其它实验变量，可以在评估时用 `--run_note` 标注，然后直接在评估记录表里对比各指标。

## 自然输出和题目级对错矩阵

默认 `--answer_mode restricted` 会保持原来的受限回答方式：选择题只输出选项，数值题只输出数字。

使用 `--answer_mode natural` 时，VSI-Bench prompt 会允许模型解释或描述推理过程，但要求最后给一行可抽取答案：

```text
Final answer: A
Final answer: 12.3
```

评估时会保存两份答案：

| 字段 | 含义 |
| --- | --- |
| `natural_prediction` | 模型原始完整输出，用于查看模型自然回答。 |
| `restricted_prediction` | 从自然输出中抽取出的选项字母或数字。 |
| `prediction` | 兼容旧流程，仍然等于 `restricted_prediction`。 |

现有 aggregate 指标只使用 `restricted_prediction` 计算，不直接使用自然输出计算指标。

评估结束写出 `vsibench.json` 样本日志后，会自动生成：

```text
docs/eval_question_matrix.json
docs/eval_question_matrix.html
```

对错矩阵每条记录包含模型名称、采样策略、备注、题型、原题目、自然输出、受限输出、真实答案、选项内容、分数，以及选择题是否正确或数值题是否得分。

如果需要从已有样本日志手动生成矩阵，可以执行：

```bash
python tools/update_eval_question_matrix.py logs/YYYYMMDD/vsibench/path/to/vsibench.json
```

## 水平基线

Dashboard 支持从 Excel 题库生成水平基线，并叠加到 `指标对比` 图表中。默认会尝试读取：

```text
C:\Users\a2818\Desktop\QA\抽样测试.xlsx
```

如果默认路径存在，评估结束更新 dashboard 时会自动计算并写入基线。也可以手动指定 Excel 文件刷新 dashboard：

```bash
python tools/update_eval_dashboard.py --baseline_excel "C:\Users\a2818\Desktop\QA\抽样测试.xlsx"
```

基线会同时缓存到：

```text
docs/eval_baselines.json
```

远程服务器没有本地 Excel 时，dashboard 会自动读取这个 JSON 缓存继续显示基线。更新题库后，可以只重新生成基线缓存：

```bash
python tools/update_eval_dashboard.py --update_baseline_cache_only --baseline_excel "C:\Users\a2818\Desktop\QA\抽样测试.xlsx"
```

或者在追加某个 `results.json` 时同时指定题库：

```bash
python tools/update_eval_dashboard.py logs/YYYYMMDD/vsibench/path/to/results.json --baseline_excel "C:\Users\a2818\Desktop\QA\抽样测试.xlsx"
```

### 基线含义

选择题类别会生成两条基线：

| 基线 | 含义 |
| --- | --- |
| 随机基线 | 每道题按选项数随机猜，分数是该类别内 `1 / 选项个数` 的平均，再乘以 100。若每题都是 4 个选项，就是 25 分。 |
| 频率基线 | 统计该类别正确答案标签分布，永远猜出现最多的那个标签，能拿到的分数比例再乘以 100。 |

数值题类别会生成一条基线：

| 基线 | 含义 |
| --- | --- |
| 常数基线MRA | 取该类别全部真值的中位数作为固定预测值，对每道题都预测这个常数，再按 VSI-Bench 的 MRA 阈值规则计算平均分并乘以 100。 |

这些基线不是模型结果，只是“地板线”：用来判断模型在某个题型上是否明显超过随机猜、答案分布偏置，或数值题的常数猜测策略。
