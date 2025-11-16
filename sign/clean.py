import os
import numpy as np

root = "/home/karthikssalian/work/RWKV-PEFT/sign/dataset/kaggle/converted"

broken = []

for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.endswith(".npz"):
            p = os.path.join(dirpath, f)
            try:
                _ = np.load(p)
            except:
                print("[BROKEN]", p)
                broken.append(p)

# for b in broken:
#     os.remove(b)