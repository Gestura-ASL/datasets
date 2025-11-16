TARGET_DIR = "/home/karthikssalian/work/RWKV-PEFT/sign/dataset/kaggle/converted"
TRAIN_BASE_DIR = "/home/karthikssalian/work/RWKV-PEFT/sign/dataset/kaggle"

import os
import numpy as np
import pandas as pd

sign_map = pd.read_csv("/home/karthikssalian/work/RWKV-PEFT/sign/dataset/kaggle/train.csv")

ROWS_PER_FRAME = 543
points_per_type = {
    "face": 468,
    "left_hand": 21,
    "right_hand": 21,
    "pose": 33
}

offset_map = {}
offset = 0
for t in ["face", "left_hand", "right_hand", "pose"]:
    count = points_per_type[t]
    offset_map[t] = offset
    offset += count

def align_dataset(frame_df):
    holder = np.zeros((543, 3), dtype=np.float32)
    for _, row in frame_df.iterrows():
        landmark_index = row["landmark_index"]
        t = row["type"]
        offset = offset_map[t]
        holder[offset + landmark_index] = [row["x"], row["y"], row["z"]]
    return np.nan_to_num(holder, nan=0.0).flatten().astype(np.float32)


if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

tasks = []

def process_one_sample(parquet_path, output_path):
    df = pd.read_parquet(parquet_path)
    aligned_data = []
    for _, group in sorted(df.groupby("frame"), key=lambda x: x[0]): # type: ignore
        aligned = align_dataset(group)
        aligned_data.append(aligned)

    keypoints_arr = np.vstack(aligned_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, keypoints=keypoints_arr)

    print(f"[OK] saved: {output_path} shape={keypoints_arr.shape}")

for idx, row in sign_map.iterrows():

    path,sign_label = row["path"], row["sign"]

    output_path = os.path.join(TARGET_DIR, sign_label, os.path.splitext(os.path.basename(path))[0] + ".npz")

    if os.path.exists(output_path):
        print(f"Skipping {output_path}, already exists.")
        continue

    tasks.append((os.path.join(TRAIN_BASE_DIR, path), output_path))


from multiprocessing import Pool, cpu_count

NUM_WORKERS = max(1, 16) 

with Pool(NUM_WORKERS) as p:
    p.starmap(process_one_sample, tasks)