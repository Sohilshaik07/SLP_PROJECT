"""
Phase 3E: Faithfulness Evaluation Runner
=========================================
Usage:
    python scripts/run_faithfulness_evaluation.py
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.evaluation.faithfulness import run_faithfulness_evaluation
run_faithfulness_evaluation()
