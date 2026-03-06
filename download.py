import os
import requests
from huggingface_hub import snapshot_download
import argparse
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 处理HuggingFaceHub异常的兼容性
try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError


# 兼容旧版本：手动计算默认缓存目录（与huggingface_hub内部逻辑一致）
def get_default_cache_dir():
    """获取Hugging Face默认缓存目录，兼容旧版本"""
    if os.name == "nt":  # Windows系统
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    else:  # Linux/macOS系统
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


# 添加重试装饰器
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            HfHubHTTPError,
            requests.exceptions.Timeout
    ))
)
def download_with_retry(dataset_id, force_download):
    """使用默认缓存目录下载，带重试机制"""
    return snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        force_download=force_download,
        etag_timeout=120
    )


def download_vsi_bench(dataset_id="nyu-visionx/VSI-Bench", force_download=False):
    """下载到Hugging Face默认缓存目录（兼容旧版本）"""
    # 获取默认缓存目录
    default_cache_dir = get_default_cache_dir()
    print(f"Hugging Face默认缓存根目录: {default_cache_dir}")

    try:
        downloaded_dir = download_with_retry(dataset_id, force_download)
        print(f"下载完成！文件实际存储路径: {downloaded_dir}")
        print("\n请按以下路径上传到远程服务器：")
        print(f"远程服务器对应路径: {downloaded_dir}")
    except Exception as e:
        print(f"下载失败: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="兼容旧版本的默认路径下载脚本")
    parser.add_argument("--dataset_id", type=str, default="nyu-visionx/VSI-Bench")
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    # 检查依赖
    try:
        import huggingface_hub
    except ImportError:
        print("请先安装依赖：pip install huggingface_hub tenacity requests")
        exit(1)

    download_vsi_bench(
        dataset_id=args.dataset_id,
        force_download=args.force_download
    )
