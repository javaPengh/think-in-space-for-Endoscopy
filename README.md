# VSI-Bench Evaluation

本项目基于 `lmms-eval` 评估框架，用于在 VSI-Bench 上评估多模态模型的空间理解能力。统一评估入口为 `evaluate_all_in_one.sh`。

评估结果和 token 消耗会在评估结束后写入 `logs/`，并通过 `tools/update_eval_dashboard.py` 汇总到 `docs/eval_dashboard.html`。

VSI-Bench 评估样本日志还会生成题目级对错矩阵，输出到 `docs/eval_question_matrix.html`。

## 数据目录

VSI-Bench 的问答对由 `lmms_eval/tasks/vsibench/vsibench.yaml` 中的 `dataset_path` 指定，当前远程服务器配置为：

```text
/home/ph/.cache/huggingface/datasets/datasets--nyu-visionx--VSI-Bench
```

视觉数据由同一配置里的 `dataset_kwargs.cache_dir: vsibench` 和 `HF_HOME` 共同决定。默认 `HF_HOME=~/.cache/huggingface/` 时，当前远程服务器上的视觉数据根目录为：

```text
/home/ph/.cache/huggingface/vsibench
```

评估时如果样本里有 `media_path`，会相对这个视觉数据根目录解析；否则按下面的规则寻找视频：

```text
${HF_HOME}/vsibench/{dataset}/{scene_name}.mp4
```

README 后文提到的 `C:\Users\a2818\Desktop\QA\抽样测试.xlsx` 是用于计算 dashboard 水平基线的本地 Excel 题库，不是远程评估时直接读取的 VSI-Bench 问答数据集目录。

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
| 保存采样帧 | `--save_sample_frames true` | 默认不保存采样帧；开启后把本次评估实际抽取的帧写入 `sample_frames/`，用于核查抽帧结果。 |
| 平台控制 fps 采样 | `--video_input_mode file --video_sample_fps F` | 仅 Qwen API 模型支持，上传本地视频文件，由 DashScope 按 fps 抽帧。 |
| 盲测 | `--visual_input_mode none` | 不提供图片或视频，只把问题文本送给模型。 |
| 自然输出 | `--answer_mode natural` | 允许模型自然语言解释，最后用 `Final answer: ...` 给可抽取答案。指标仍使用抽取后的受限答案计算。 |
| 评估备注 | `--run_note "文本"` | 给本次评估记录添加自由文本备注，方便记录额外变量或实验条件。 |
| 数据版本 | `--data_version VERSION` | 标记本次评估使用的数据源或采样版本。Dashboard 和题目级矩阵会按该字段筛选，避免新旧采样数据混在一起统计。默认 `default`。 |
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

### 保存采样帧用于核查

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --num_frames 16 --save_sample_frames true
```

`--save_sample_frames` 默认关闭，正常评估不会写出采样帧，避免大量 JPG I/O 拖慢评估。传 `true`、`1` 或 `yes` 时会开启保存，采样帧会写入 `sample_frames/{model}-{sampling}/...` 目录；关闭时可传 `false`、`0` 或 `no`。该开关只影响是否额外落盘核查帧，不改变抽帧策略本身。

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

### 使用新的采样数据版本

```bash
bash evaluate_all_in_one.sh --model qwen3vl_8b --benchmark vsibench --num_processes 2 --num_frames 16 --data_version new_sample_202606
```

`--data_version` 会写入 `results.json` 的 `config.data_version`，并同步进入 `docs/eval_dashboard_data.json` 和 `docs/eval_question_matrix.json`。同一份 dashboard 中可以保留多个数据版本的历史记录，但页面默认只展示最新评估记录对应的数据版本；需要跨版本查看时，可以在页面筛选器中选择 `全部数据版本`。

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

评估记录表会保留全部历史记录，并显示 `数据版本` 和 `备注` 列。Dashboard 默认展示最新评估记录对应的数据版本，图表按同一数据版本、同一模型和同一采样策略只取最新记录；如果需要对比不同采样数据源，可以在页面筛选器中切换数据版本或选择 `全部数据版本`。

## 自然输出和题目级对错矩阵

默认 `--answer_mode restricted` 会保持原来的受限回答方式：选择题只输出选项，数值题只输出数字。

使用 `--answer_mode natural` 时，VSI-Bench prompt 会允许模型解释或描述推理过程，但要求最后给一行可抽取答案：

```text
Final answer: A
Final answer: 12.3
```

评估时会保存两份答案，restricted 和 natural 模式的列含义不同：

| 字段 | 含义 |
| --- | --- |
| `natural_prediction` | natural 模式保存模型原始完整输出；restricted 模式为空。 |
| `restricted_prediction` | natural 模式保存抽取出的选项字母或数字；restricted 模式保存模型直接回答。 |
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

## Prompt 调试

`qwen3vl.py` 适配器支持通过环境变量把真实送入模型的 prompt 落盘，用于排查模型是否收到了正确的问题、视觉消息和 chat template。默认不开启，不设置 `VSI_DEBUG_PROMPT_DIR` 时不会写任何调试文件。

调试文件是 JSON，包含：

| 字段 | 含义 |
| --- | --- |
| `messages` | 调用 `processor.apply_chat_template` 前的结构化消息。 |
| `rendered_chat_template` | 套用模型 chat template 后的完整文本。 |
| `generation_kwargs` | 当前样本实际使用的生成参数。 |
| `media_type` | 当前样本输入类型，例如 `image`、`video` 或 `text`。 |
| `metadata` | `task`、`split`、`doc_id` 等定位信息。 |

常用环境变量：

| 环境变量 | 作用 |
| --- | --- |
| `VSI_DEBUG_PROMPT_DIR` | 调试 JSON 输出目录。设置后才会启用 prompt dump。 |
| `VSI_DEBUG_PROMPT_LIMIT` | 最多输出多少条 prompt，默认 `5`。 |
| `VSI_DEBUG_PROMPT_DOC_IDS` | 只输出指定 `doc_id`，多个 id 用英文逗号分隔，例如 `362,365,370`。 |

一次性运行时开启：

```bash
VSI_DEBUG_PROMPT_DIR=docs/prompt_debug \
VSI_DEBUG_PROMPT_DOC_IDS=362,365,370 \
VSI_DEBUG_PROMPT_LIMIT=20 \
bash evaluate_all_in_one.sh --model medmo_8b_next --answer_mode natural --limit 400
```

如果希望脚本默认允许外部开关控制，可以在 `evaluate_all_in_one.sh` 初始化区加入：

```bash
export VSI_DEBUG_PROMPT_DIR="${VSI_DEBUG_PROMPT_DIR:-}"
export VSI_DEBUG_PROMPT_DOC_IDS="${VSI_DEBUG_PROMPT_DOC_IDS:-}"
export VSI_DEBUG_PROMPT_LIMIT="${VSI_DEBUG_PROMPT_LIMIT:-5}"
```

这三行不会固定开启调试；含义是保留运行命令里传入的环境变量，如果没有传入就使用空值或默认值。若要固定开启，可以直接写：

```bash
export VSI_DEBUG_PROMPT_DIR=docs/prompt_debug
```

该功能目前只对继承 `lmms_eval/models/qwen3vl.py` 的本地模型适配器生效，包括 `qwen3vl`、`qwen3vl_32b`、`medmo_8b_next` 和 `lingshu_32b`。它不影响 API 模型，也不影响没有继承 `Qwen3VL` 的本地模型。

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
