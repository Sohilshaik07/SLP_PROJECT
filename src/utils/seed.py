"""
Utility: Reproducibility Seed Setting
=======================================
Sets random seeds across Python, NumPy, and PyTorch
to ensure reproducible experiments.

Usage
-----
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed value. Default is 42.

    Notes
    -----
    - Sets Python's built-in random module seed.
    - Sets NumPy random seed.
    - Sets PyTorch CPU and GPU seeds if available.
    - Sets PYTHONHASHSEED environment variable.
    - Sets transformers training seed via environment variable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy not installed; skipping NumPy seed.")

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch not installed; skipping PyTorch seed.")

    logger.info(f"Random seed set to {seed}.")


def get_device() -> "torch.device":
    """
    Detect and return the best available compute device.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("No GPU detected. Using CPU.")
        return device
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")
