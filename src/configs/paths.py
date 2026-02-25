from pathlib import Path
import os

# Project root (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories (can be overridden by env vars)
DATA_DIR = Path(os.getenv("EGA_DATA", PROJECT_ROOT / "data"))
STORAGE_DIR = Path(os.getenv("EGA_STORAGE", PROJECT_ROOT / "storage"))
SYZYGY_PATH = STORAGE_DIR / "syzygy" / "3_4_5"

# Tablebase directory
TB_DIR = STORAGE_DIR / "syzygy"

# Processed datasets
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"