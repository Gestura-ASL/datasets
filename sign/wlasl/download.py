import os
import json
import time
import sys
import urllib.request
import random
import logging
import ssl
import subprocess

# --- SSL workaround ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- Config ---
youtube_downloader = "yt-dlp"
indexfile = "WLASL_v0.3.json"
saveto = "raw_videos"
failed_file = "failed.txt"

# --- Logging ---
logging.basicConfig(filename=f'download_{int(time.time())}.log',
                    filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def check_youtube_dl_version():
    try:
        ver = subprocess.check_output([youtube_downloader, "--version"]).decode().strip()
    except Exception:
        logging.error(f"{youtube_downloader} not found in PATH. Install it first.")
        sys.exit(1)
    logging.info(f"Using {youtube_downloader}, version: {ver}")


def request_video(url, referer=''):
    headers = {'User-Agent': 'Mozilla/5.0'}
    if referer:
        headers['Referer'] = referer
    req = urllib.request.Request(url, headers=headers)
    logging.info(f"Requesting {url}")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def save_video(data, path):
    with open(path, 'wb') as f:
        f.write(data)
    time.sleep(random.uniform(0.5, 1.5))


# ---------- Download Methods ----------

def download_aslpro(url, dirname, video_id):
    path = os.path.join(dirname, f"{video_id}.swf")
    if os.path.exists(path):
        logging.info(f"Already exists: {path}")
        return True
    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, path)
    return True


def download_others(url, dirname, video_id):
    path = os.path.join(dirname, f"{video_id}.mp4")
    if os.path.exists(path):
        logging.info(f"Already exists: {path}")
        return True
    data = request_video(url)
    save_video(data, path)
    return True


def download_youtube(url, dirname, video_id):
    # Save directly as video_id
    out_template = os.path.join(dirname, f"{video_id}.%(ext)s")
    if any(os.path.exists(os.path.join(dirname, f"{video_id}.{ext}")) for ext in ['mp4','webm','mkv']):
        logging.info(f"Already exists: {video_id}")
        return True

    cmd = [youtube_downloader, url, "-o", out_template]
    rv = subprocess.call(cmd)
    return rv == 0


# ---------- Dispatcher ----------

def select_download_method(url):
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_youtube
    else:
        return download_others
    

# ---------- Main Loop ----------

def download_all(indexfile, saveto, failed_file):
    with open(indexfile, "r") as f:
        content = json.load(f)
    os.makedirs(saveto, exist_ok=True)

    for entry in content:
        gloss = entry['gloss']
        for inst in entry['instances']:
            url = inst['url']
            video_id = inst['video_id']

            try:
                # Skip some domains
                for skipped_url in ["www.handspeak.com", "www.aslpro.com", "www.aslsearch.com"]:
                    if skipped_url in url:
                        raise RuntimeError("ignored url")

                logging.info(f"Gloss: {gloss}, Video: {video_id}, URL: {url}")
                download_method = select_download_method(url)
                ok = download_method(url, saveto, video_id)
                if not ok:
                    raise RuntimeError("Downloader failed")
            except Exception as e:
                logging.error(f"Failed: {video_id} {url} ({e})")
                with open(failed_file, "a") as f:
                    f.write(f"{video_id} {url}\n")


if __name__ == "__main__":
    check_youtube_dl_version()
    logging.info("Starting all downloads...")
    download_all(indexfile, saveto, failed_file)
    logging.info("Download complete.")