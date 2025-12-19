import os
import numpy as np
import polars as pl

# ---------------- CONFIG ----------------
TRAIN_BASE_DIR = "/home/karthikssalian/work/hand sign/datasets/sign/data/asl-signs"
TARGET_DIR = "/home/karthikssalian/work/hand sign/datasets/sign/data/normalized"
REF_FILE = "/home/karthikssalian/work/hand sign/datasets/sign/data/reference_pose.npz"

ROWS = 543
POSE_START = 510
L_SHOULDER = POSE_START + 11
R_SHOULDER = POSE_START + 12

POINTS_PER_TYPE = {
    "face": 468,
    "left_hand": 21,
    "right_hand": 21,
    "pose": 33,
}

# ---------------- OFFSET MAP ----------------
offset_map = {}
o = 0
for k in ["face", "left_hand", "right_hand", "pose"]:
    offset_map[k] = o
    o += POINTS_PER_TYPE[k]

# ---------------- LOAD REFERENCE ----------------
ref = np.load(REF_FILE)["keypoints"].astype(np.float32)
REFERENCE_POSE = ref.reshape(ROWS, 3)

# ---------------- FRAME NORMALIZATION ----------------
def normalize_frame(frame_3d: np.ndarray) -> np.ndarray:
    l_sh = frame_3d[L_SHOULDER]
    r_sh = frame_3d[R_SHOULDER]

    # If shoulders missing → replace whole frame
    if np.isnan(l_sh).any() or np.isnan(r_sh).any():
        return REFERENCE_POSE.flatten()

    center = (l_sh + r_sh) / 2.0
    width = np.linalg.norm(l_sh - r_sh)
    if width < 1e-6:
        width = 1.0

    norm = (frame_3d - center) / width

    # Fill missing landmarks from reference
    nan_mask = np.isnan(norm)
    norm[nan_mask] = REFERENCE_POSE[nan_mask]

    return norm.flatten().astype(np.float32)


# ---------------- PROCESS ONE SAMPLE ----------------
def process_one_sample(parquet_path, output_path):
    df = pl.read_parquet(parquet_path).sort("frame")

    frames_out = []

    for _, group in df.group_by("frame", maintain_order=True):
        frame = np.full((ROWS, 3), np.nan, dtype=np.float32)

        type_offsets = group["type"].replace(offset_map)

        # Safety: ensure no unknown types
        if type_offsets.null_count() > 0:
            raise ValueError("Unknown landmark type found")

        idx = (
            group["landmark_index"]
            + type_offsets.cast(pl.Int32)
        ).to_numpy()


        frame[idx] = group.select(["x", "y", "z"]).to_numpy()
        frames_out.append(normalize_frame(frame))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, keypoints=np.vstack(frames_out))

    # print(f"[OK] {output_path}")

sign_map = pl.read_csv(f"{TRAIN_BASE_DIR}/train.csv")
for row in sign_map.iter_rows(named=True):
    inp = os.path.join(TRAIN_BASE_DIR, row["path"])
    out = os.path.join(
        TARGET_DIR,
        row["sign"],
        os.path.splitext(os.path.basename(row["path"]))[0] + ".npz",
    )
    if not os.path.exists(out):
        process_one_sample(inp,out)

print("Done.")