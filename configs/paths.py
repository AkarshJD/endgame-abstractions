from pathlib import Path
import os

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories (can be overridden by env vars)
DATA_DIR = Path(os.getenv("EGA_DATA", PROJECT_ROOT / "data"))
STORAGE_DIR = Path(os.getenv("EGA_STORAGE", PROJECT_ROOT / "storage"))

# Tablebase directory
TB_DIR = STORAGE_DIR / "syzygy"

# Processed datasets
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
