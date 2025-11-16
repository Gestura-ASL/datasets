import cv2
import numpy as np

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_video_opencv(video_path, output_path):
    if os.path.exists(output_path):
        print("[SKIP]:", output_path)
        return
    
    import mediapipe as mp

    EXPECTED_DIM = 1662

    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_holistic.process(rgb)

        vec = extract_keypoints(results)

        # --- CHECK SHAPE ---
        if vec.shape[0] != EXPECTED_DIM:
            print("❌ ERROR in:", video_path)
            print("Frame:", frame_index)
            print("Vector length:", vec.shape[0])
            print("Skipping this video.")
            cap.release()
            return  # Skip this video entirely

        # --- CHECK NaN ---
        if np.isnan(vec).any():
            print("❌ NaN detected in:", video_path, "frame:", frame_index)
            print("Replacing NaNs with zero.")
            vec = np.nan_to_num(vec, nan=0.0)

        keypoints_list.append(vec)
        frame_index += 1

    cap.release()

    # Safe write
    keypoints_arr = np.vstack(keypoints_list)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, keypoints=keypoints_arr)

    print("[OK]:", output_path, keypoints_arr.shape)


from multiprocessing import Pool
import os
from glob import glob

def run_multiprocess(input_root, output_root, num_workers=8):

    # recursively find all mp4 files
    video_paths = sorted(glob(os.path.join(input_root, "**/*.mp4"), recursive=True))

    tasks = []

    for vid in video_paths:

        relative = os.path.relpath(vid, input_root)         # hello/v1.mp4
        out_path = os.path.join(output_root,
                                relative.replace(".mp4", ".npz"))

        if not os.path.exists(out_path):
            tasks.append((vid, out_path))

    with Pool(num_workers) as p:
        p.starmap(process_video_opencv, tasks)


if __name__ == "__main__":
    input_root = "/home/karthikssalian/work/RWKV-PEFT/sign/dataset/videos/"
    output_root = "/home/karthikssalian/work/RWKV-PEFT/sign/dataset/keypoints_npz/"

    run_multiprocess(input_root, output_root, num_workers=12)
