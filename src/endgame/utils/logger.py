import json
import os
from datetime import datetime
import subprocess
import numpy as np


def _json_converter(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_converter)


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