import os
import json
import logging
import sys
import subprocess
from multiprocessing import Pool, cpu_count

# --- Config ---
INDEX_FILE = "WLASL_v0.3.json"
RAW_DIR = "renamed_videos"
OUT_DIR = "videos"
FAILED_FILE = "failed_videos.txt"
LOG_FILE = f"preprocess_{os.getpid()}.log"

# Retry config
ENABLE_RETRY = False
MAX_RETRIES = 2
FPS = 25

# --- Logging ---
logging.basicConfig(filename=LOG_FILE, filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def check_ffmpeg():
    try:
        out = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if out.returncode != 0:
            raise FileNotFoundError
        logging.info("FFmpeg is available.")
    except FileNotFoundError:
        logging.error("FFmpeg not found in PATH. Please install ffmpeg first.")
        sys.exit(1)


def run_ffmpeg(cmd, src, dst):
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {src} -> {dst}, error: {e}")
        return False


def ffmpeg_extract_clip(src, dst, start_frame, end_frame, fps=FPS):
    start_sec = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        dst
    ]
    return run_ffmpeg(cmd, src, dst)


def ffmpeg_convert_to_mp4(src, dst):
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-c:v", "libx264", "-c:a", "aac",
        dst
    ]
    return run_ffmpeg(cmd, src, dst)


def find_source_file(video_id):
    """
    Look for an existing source file in RAW_DIR using ONLY the video_id,
    ignoring any other naming scheme or URL patterns.
    """
    possible_exts = (".mp4", ".mkv", ".swf", ".avi", ".webm")
    for ext in possible_exts:
        path = os.path.join(RAW_DIR, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    return None



def process_instance(args):
    inst, gloss = args
    url = inst["url"]
    video_id = inst["video_id"]

    # Create folder for the gloss
    gloss_folder = os.path.join(OUT_DIR, gloss)
    os.makedirs(gloss_folder, exist_ok=True)

    dst = os.path.join(gloss_folder, f"{video_id}.mp4")
    src = find_source_file(video_id)

    # Skip if destination already exists
    if os.path.exists(dst):
        logging.info(f"Skip {video_id}, already exists at {dst}")
        return None

    if not src:
        logging.warning(f"Source missing for {video_id} ({url})")
        return f"{video_id} {url}"

    attempts = MAX_RETRIES if ENABLE_RETRY else 1
    ok = False

    for attempt in range(1, attempts + 1):
        if "youtube" in url or "youtu.be" in url:
            start_frame = inst.get("frame_start", 0) - 1
            end_frame = inst.get("frame_end", 0) - 1
            if end_frame > 0 and start_frame >= 0:
                logging.info(f"[{attempt}] Extracting {video_id} frames {start_frame}-{end_frame} from {src}")
                ok = ffmpeg_extract_clip(src, dst, start_frame, end_frame)
            else:
                logging.info(f"[{attempt}] Converting full YouTube video {video_id} -> mp4")
                ok = ffmpeg_convert_to_mp4(src, dst)
        else:
            logging.info(f"[{attempt}] Converting non-YouTube video {video_id} -> mp4")
            ok = ffmpeg_convert_to_mp4(src, dst)

        if ok:
            break

    if not ok:
        logging.error(f"Failed {video_id} {url}")
        return f"{video_id} {url}"

    return None



def main():
    check_ffmpeg()

    os.makedirs(OUT_DIR, exist_ok=True)

    content = json.load(open(INDEX_FILE))
    tasks = [(inst, entry["gloss"]) for entry in content for inst in entry["instances"]]

    logging.info(f"Total instances: {len(tasks)}")

    # Deduplicate video_ids to prevent duplicate conversions
    seen_ids = set()
    unique_tasks = []
    for inst, gloss in tasks:
        vid = inst["video_id"]
        if vid not in seen_ids:
            unique_tasks.append((inst, gloss))
            seen_ids.add(vid)

    logging.info(f"Unique tasks: {len(unique_tasks)}")
    logging.info(f"Using {cpu_count()} workers, retries: {ENABLE_RETRY}")

    with Pool(cpu_count()) as pool:
        results = pool.map(process_instance, unique_tasks)

    failed = [r for r in results if r]
    if failed:
        failed = sorted(set(failed))  # remove duplicates
        with open(FAILED_FILE, "w") as f:
            f.write("\n".join(failed))
        logging.warning(f"{len(failed)} instances failed. See {FAILED_FILE}")

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
