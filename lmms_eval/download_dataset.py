# NOTE: pip install datasets

from datasets import load_dataset
vsi_bench = load_dataset("nyu-visionx/VSI-Bench")
print(vsi_bench)