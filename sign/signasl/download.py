import os
import re
import time
import string
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from typing import Union
import random

BASE_URL = "https://www.signasl.org"
TARGET_DIR = "signasl"
FAILED_FILE = "failed_downloads.txt"

os.makedirs(TARGET_DIR, exist_ok=True)
pattern = re.compile(r"^video_con_signasl_\d+$")

def log_failure(url: str, error: Union[str,Exception]):
    """Append failed URL and error message to file."""
    with open(FAILED_FILE, "a") as f:
        f.write(f"{url} | {error}\n")

for letter in string.ascii_lowercase:
    dict_url = f"{BASE_URL}/dictionary/{letter}"
    print(f"\n[*] Fetching sign list for '{letter.upper()}'...")

    try:
        response = requests.get(dict_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"[!] Failed to load {dict_url}: {e}")
        log_failure(dict_url, e)
        continue

    table = soup.find("table")
    if not table:
        print(f"[!] No table found for letter '{letter}'.")
        log_failure(dict_url, "No table found")
        continue

    sign_links = [urljoin(BASE_URL, str(a["href"])) for a in table.find_all("a", href=True)]
    print(f"  Found {len(sign_links)} signs.")

    for idx, sign_url in enumerate(sign_links, 1):
        try:
            print(f"  [{idx}/{len(sign_links)}] Fetching: {sign_url}")
            resp = requests.get(sign_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')

            # Folder per sign
            sign_name = sign_url.rstrip("/").split("/")[-1]
            sign_dir = os.path.join(TARGET_DIR, sign_name)
            os.makedirs(sign_dir, exist_ok=True)

            videos = soup.find_all("video", id=pattern)
            if not videos:
                print("     No videos found.")
                continue

            for v in videos:
                src = str(v.get("src"))
                if not src:
                    continue

                video_url = urljoin(BASE_URL, src)
                filename = os.path.basename(video_url)
                filepath = os.path.join(sign_dir, filename)

                if os.path.exists(filepath):
                    print(f"     Skipping existing: {filename}")
                    continue

                print(f"     Downloading {filename} ...")
                try:
                    r = requests.get(video_url, timeout=30)
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(r.content)
                except Exception as ve:
                    print(f"     [!] Video failed: {ve}")
                    log_failure(video_url, ve)
                    continue

                time.sleep(random.uniform(1.0, 1.5))

        except Exception as e:
            print(f"     [!] Error on {sign_url}: {e}")
            log_failure(sign_url, e)
            continue

print("\n✅ All done!")
print(f"   Videos saved in: {os.path.abspath(TARGET_DIR)}")
print(f"   Failed URLs logged in: {os.path.abspath(FAILED_FILE)}")
