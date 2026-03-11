"""Random seeding helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        if deterministic:
            # Some TF builds expose this (TF 2.13+)
            try:
                tf.config.experimental.enable_op_determinism(True)
            except Exception:
                # Fallback via env var
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        # TensorFlow not available; ignore
        pass
