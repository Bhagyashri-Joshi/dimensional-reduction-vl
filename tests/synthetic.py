"""TEST-ONLY synthetic data in the exact UCI HAR file layout.

Never used by the app itself. It lets unit/integration tests run without network access.
"""
from pathlib import Path

import numpy as np


def make_synthetic_har(root: Path, n_train: int = 420, n_test: int = 150, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = Path(root) / "UCI HAR Dataset"
    centers = rng.normal(0, 1.5, size=(6, 561))
    mixing = rng.normal(0, 0.3, size=(12, 561))

    def block(n):
        y = np.tile(np.arange(1, 7), n // 6 + 1)[:n]
        latent = rng.normal(0, 1, size=(n, 12))
        X = centers[y - 1] + latent @ mixing + rng.normal(0, 0.5, size=(n, 561))
        return np.clip(X / 6, -1, 1), y

    for split, n in (("train", n_train), ("test", n_test)):
        X, y = block(n)
        d = base / split
        d.mkdir(parents=True, exist_ok=True)
        np.savetxt(d / f"X_{split}.txt", X, fmt="%.7e")
        np.savetxt(d / f"y_{split}.txt", y, fmt="%d")
    (base / "features.txt").write_text("\n".join(f"{i + 1} feat-{i + 1}" for i in range(561)))
    return base
