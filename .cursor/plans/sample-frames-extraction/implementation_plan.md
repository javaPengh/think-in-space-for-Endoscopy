# [输入采样帧保存与提取计划]

## 目标
实现一个可控的帧保存机制，以便研究采样帧率的变化对模型评估的影响。需要将最终送入模型的图像帧存入到项目根目录 `sample_frames` 之中。

## 目录结构设计
保存的层级需要带有版本自增控制：
`sample_frames/{模型名}-{采样帧数}f/v_{xx}/...`
例如：`sample_frames/InternVL-3.5-2B-32f/v_01/video_name_frame0000.jpg`，下一次评测就会自动生成 `v_02`。

为了实现：
1. **获取模型标识**：从 `self.path` 或传入参数提取 `{模型名}`，结合 `{max_frames_num}f` 构建父目录。
2. **版本自增**：在 Python 内使用 `os.listdir` 寻找当前父目录下的 `v_xx`，取最大值 + 1 作为本次执行的存放目录。通过环境变量保持每次执行的任务级版本号一致。

## 需要修改的文件
1. **`lmms_eval/models/internvl3_5.py`**：修改 `load_video` 函数，让其能够获取当前调用它的实例的参数并保存帧。 (已完成)
2. **`lmms_eval/models/qwen3vl.py`**：在 `generate_until` 中我们已经通过 `Image.save` 写入了临时目录，现在将它的保存路径重定向到 `sample_frames/{model_name}-{max_frames_num}f/v_xx/` 中。 (已完成)
3. **`lmms_eval/models/qwen3vl_32b.py`**：同步以上 Qwen3 系列的修改。 (已完成，后在重构中移除)

## 多进程并行评估（DDP）支持与代码重构（2026/03/17）
发现使用环境变量的方式在 `accelerate` 多进程中会产生竞态条件，导致一个任务生成多个（例如 `v_01` 到 `v_04`）版本目录。
**修复计划**：
1. 取消对 `os.environ` 的依赖。
2. 在各模型的 `generate_until`（或 `__init__`）中，确保只有 `self.rank == 0` 执行 `os.listdir` 寻找最大版本，并创建新的 `v_{xx}` 目录。
3. 利用 `torch.distributed.broadcast_object_list` 将新创建的版本号（如 `"v_05"`）从 Rank 0 广播到所有的进程。
4. 将明确的版本目录直接传递给 `load_video` 或使用 `self.current_version_dir` 进行访问，确保所有进程写入完全一致的相对路径。
5. **重构统一入口**：由于 `qwen3vl_32b.py` 与 `qwen3vl.py` 逻辑完全重复，合并两者至 `qwen3vl.py`，并将类增加 `@register_model("qwen3vl", "qwen3vl_32b")` 注册别名，随后删除 `qwen3vl_32b.py`，降低维护成本。

## 注意事项
必须保证这个保存操作不对原始的评测流水线造成内存泄漏或磁盘满载的堵塞，并且保存的文件能够被人类根据原视频名称辨认。