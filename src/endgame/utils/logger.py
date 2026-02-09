import json
import os
from datetime import datetime
import subprocess


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "unknown"


def make_run_dir(base="logs/kpk"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
