"""
Main research pipeline.

TB → Features → Concepts → Rules → Evaluation
"""

import sys
import os

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=== Endgame Abstractions Pipeline ===")
    print("Project root:", PROJECT_ROOT)
    print("Pipeline initialized successfully.")


if __name__ == "__main__":
    main()
