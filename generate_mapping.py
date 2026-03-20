import pandas as pd
import json

file_path = r"C:\Users\a2818\Desktop\think-in-space相关文件\用于内镜空间智能评估的数据集构建相关文件\vsibench_polyp_qa(尺寸估计类问题视频帧消融实验1).xlsx"

# Read the excel file
df = pd.read_excel(file_path)

# Extract the id column, assuming 'id' or 'question_id' or 'ID'
col_name = None
for col in df.columns:
    if 'id' in str(col).lower():
        col_name = col
        break

if not col_name:
    # default to the first column if no id column found
    col_name = df.columns[0]

# Generate the dictionary mapping
mapping = {"vsibench": {}}

for index, row in df.iterrows():
    # Convert id to string or keep as int based on data, here we use str for safety
    q_id = str(row[col_name])
    mapping["vsibench"][q_id] = []

# Write to data/keyframe_mapping.json
with open("data/keyframe_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=4, ensure_ascii=False)

print(f"Successfully generated keyframe mapping for {len(mapping['vsibench'])} questions.")
