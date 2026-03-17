# 采样帧提取支持演练文档

## 1. 所做修改
- 在 `.cursor/plans/sample-frames-extraction/` 下创建了任务跟踪目录。
- **`lmms_eval/models/internvl3_5.py`**：
  - 更新 `load_video` 函数，增加参数 `model_name`。
  - 在遍历抽取出来的帧时，使用 `cv2.imwrite` 将帧（转换为 BGR 格式后）保存至 `sample_frames/{model_name}/{num_segments}f/`。
  - 修改 `generate_until` 中的调用，提取 `self.path` 的最后一部分作为 `model_name` 并传递给 `load_video`。
- **`lmms_eval/models/qwen3vl.py` 及 `lmms_eval/models/qwen3vl_32b.py`**：
  - 将原先保存临时图片的逻辑拦截，改道将其存储在 `sample_frames/{model_name}/{max_frames_num}f/` 下。
  - 删除了原有针对 `frame_paths` 的 `os.remove(tmp_p)`，确保帧可以长期保存。

## 2. 测试内容
- 验证所有模型的评估流程是否正常运行。
- 确认评估期间是否在 `sample_frames/` 下按照 `模型名称/帧数/` 的规则生成并长期保存了 `.jpg`。

## 3. 验证结果
（待用户执行对应评测后验证输出文件目录 `sample_frames` 即可）

## 4. 2026/03/17 目录结构优化 (版本自增)
- **`lmms_eval/models/internvl3_5.py`**：
  - 修改 `load_video` 中的帧保存逻辑，使其按照 `sample_frames/{model_name}-{num_segments}f/v_{xx}` 的格式保存图片。使用环境变量 `SAMPLE_FRAMES_VERSION_{model_name}_{num_segments}` 控制单次执行跑多个视频时存放同一 `v_{xx}` 下。
- **`lmms_eval/models/qwen3vl.py`** & **`lmms_eval/models/qwen3vl_32b.py`**：
  - 使用同样逻辑，建立具有模型名和帧数的目录，并计算其下的最大 `v_xx`，创建并存入。
