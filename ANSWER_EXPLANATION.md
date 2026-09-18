# 回答后追加解释

开启 `--explain_answer true` 后，每个样本执行两轮对话：

1. 按原有问题、视觉输入和生成参数生成答案。
2. 保留第一轮对话与相同视觉输入，追加用户消息：`Please explain your answer step by step.`

第一轮答案仍用于评分，第二轮解释仅用于分析。各样本之间不共享历史。不需要持久化 KV cache；第二轮重新编码包含第一轮答案的完整对话。视频使用相同的已抽取帧，InternVL 复用同一份视觉张量。

目前启动脚本支持 Qwen3-VL（2B/8B/32B）、Qwen2.5-VL（7B/32B）及 InternVL3.5（2B/8B/14B/38B）。其他模型或 `--model all` 开启解释时会在加载前报错。

```bash
bash evaluate_all_in_one.sh \
  --model qwen3vl_8b \
  --num_processes 1 \
  --num_frames 16 \
  --answer_mode restricted \
  --explain_answer true \
  --explanation_max_new_tokens 2048 \
  --limit 10
```

沿用脚本现有的环境变量和本地模型路径要求。也可在 `lmms-eval` 的 `--model_args` 中加入 `explain_answer=true,explanation_max_new_tokens=2048`，并开启 `--log_samples`。开关默认关闭，不改变原单轮流程。

样本日志的 `resps`、`filtered_resps` 和分数仍对应首轮答案。新增 `answer_explanations` 字段，外层按 request 排列，内层按重复生成顺序排列：

```json
{
  "answer_explanations": [[{
    "answer": "B",
    "prompt": "Please explain your answer step by step.",
    "explanation": "...",
    "generation_kwargs": {"max_new_tokens": 2048, "do_sample": false, "num_beams": 1}
  }]]
}
```

第二轮使用独立 token 上限和确定性解码。启动脚本为日志后缀添加 `_explain`，模型参数中记录实验开关。Qwen token 统计包含两轮调用，因此 `num_requests` 统计生成调用次数，而不是题目数。响应缓存不能用于此模式，因为仅缓存答案会漏掉解释。

解释失败时会抛出错误，避免把缺少解释的运行当作完成。该功能运行新一轮评估，不会自动读取旧答案日志继续追问。

研究解释：这是回答后的自我解释，可能包含事后合理化，也可能改口，不能直接等同于第一次作答时的内部推理。首轮“只输出字母/数字”的指令保留在历史中，第二轮采用指定的原文追问；应先检查小批量输出，确认模型能按第二轮要求解释。解释中的新答案不会替换首轮答案或改变其分数。

本地回归检查使用模拟模型，验证对话顺序、视觉复用、样本隔离和评分输入。真实模型兼容性、显存占用和解释质量需在远程评估环境用上述小批量命令验证。

## 题目对错矩阵

`tools/update_eval_question_matrix.py` 生成的矩阵现在使用以下字段：

- `first_turn_prediction`：第一轮原始文本，包括历史自然回答 / CoT 输出。
- `restricted_prediction`：第一轮提取后的评分用答案。
- `natural_prediction`：第一条第二轮解释文本，只从 `answer_explanations` 获取，不回退到第一轮。
- `answer_explanations`：所有第二轮记录，保留原 request / repeat 索引、被解释答案和追问提示词。
- `explanation_status`：`available`（至少一条非空解释）、`empty`（有记录但解释均为空）、`missing`（没有解释记录）。

列表提供搜索、题型、评分结果、解释状态、数据版本、模型、采样策略、首轮模式和时间筛选。详情面板保留完整文本，可切换重复生成记录；CSV 导出包含当前筛选范围的全部记录，而非仅当前页。

模型对照视图以一次评估运行为一列，避免覆盖同一模型的多次运行。只有数据版本、题号、题目、选项和标准答案都相同才按同一道题对齐。缺失单元格显示 `—`，不算答错。选择题正确率与数值题平均 MRA 分开统计，数值题部分得分不标记为答对。统计按样本记录计数，包含各次重复评估。

刷新已有矩阵：

```bash
python tools/update_eval_question_matrix.py --refresh_only
```

此命令同时更新矩阵 JSON 与 HTML。首次迁移旧数据时，如果源日志存在，会从源日志重新构建记录；否则保留历史答案与评分，把原“自然输出”移入 `first_turn_prediction`，解释显示“未生成”。不会生成、推断或伪造第二轮解释。源日志以后恢复可用时，显式重新导入：

```bash
python tools/update_eval_question_matrix.py /path/to/run/vsibench.json
```

针对某个数据版本的矩阵，可追加 `--data_path docs/<版本目录>/eval_question_matrix.json --html_path docs/<版本目录>/eval_question_matrix.html`。

页面模板位于 `tools/templates/question_matrix.html`、`.css`、`.js`，同步服务器时需要一起带上。最终生成的 HTML 内嵌全部样式、脚本和数据，打开时无需联网。页面按页渲染，完整解释只在详情面板展开。

本地检查：

```bash
python -m unittest discover -s tests -p test_question_matrix.py -v
node tests/test_question_matrix_ui.cjs
```

以上为数据与交互逻辑检查，不包含真实浏览器的布局验证；按项目规范由用户打开生成页面确认视觉效果。
