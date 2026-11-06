"""UCI HAR dataset loader (TRD section 3).

Loads the predefined train/test partition, validates it, and returns plain NumPy arrays.
If the dataset is not on disk it is downloaded from the official UCI repository and extracted
into the data folder. Labels are never remapped: they stay 1..6 as in the original files.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_FEATURES = 561
ACTIVITY_NAMES: dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
EXPECTED_TRAIN_ROWS = 7352
EXPECTED_TEST_ROWS = 2947

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
DATASET_URLS = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
)


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, cannot be downloaded, or fails validation."""


@dataclass(frozen=True)
class HARData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    source_dir: str

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def classes(self) -> list[int]:
        return sorted(ACTIVITY_NAMES)

    def class_distribution(self) -> pd.DataFrame:
        """Rows = activities; columns = train/test counts."""
        tr = pd.Series(self.y_train).value_counts()
        te = pd.Series(self.y_test).value_counts()
        rows = [
            {
                "Label": c,
                "Activity": ACTIVITY_NAMES[c],
                "Train": int(tr.get(c, 0)),
                "Test": int(te.get(c, 0)),
            }
            for c in self.classes
        ]
        df = pd.DataFrame(rows)
        df["Total"] = df["Train"] + df["Test"]
        return df


# --------------------------------------------------------------------------- locating / downloading
def default_data_dir() -> Path:
    env = os.environ.get("HAR_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "data"


def find_dataset_root(base: Path) -> Path | None:
    """Return the folder that holds ``train/`` and ``test/`` (searched recursively)."""
    base = Path(base)
    if not base.exists():
        return None
    for marker in sorted(base.rglob("X_train.txt")):
        root = marker.parent.parent
        if marker.parent.name == "train" and (root / "test" / "X_test.txt").exists():
            return root
    return None


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        if member.filename.startswith("__MACOSX"):
            continue
        target = (dest / member.filename).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise DatasetError(f"Unsafe path in archive: {member.filename}")
        zf.extract(member, dest)


def _extract_archive(blob: bytes, dest: Path) -> None:
    """Extract a zip (handles the UCI archive that wraps a second zip inside it)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise DatasetError("Downloaded file is not a valid zip archive.") from exc
    with zf:
        _safe_extract(zf, dest)
    for _ in range(2):  # at most two levels of nesting
        if find_dataset_root(dest):
            return
        for inner in list(dest.rglob("*.zip")):
            try:
                with zipfile.ZipFile(inner) as zf:
                    _safe_extract(zf, dest)
            except zipfile.BadZipFile:
                continue


def download_dataset(dest: Path, urls: tuple[str, ...] = DATASET_URLS, timeout: int = 120) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dr-virtual-lab/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URLs)
                blob = resp.read()
            _extract_archive(blob, dest)
            root = find_dataset_root(dest)
            if root:
                return root
            errors.append(f"{url}: archive did not contain train/X_train.txt")
        except Exception as exc:  # network, TLS, bad zip ...
            errors.append(f"{url}: {exc}")
    raise DatasetError(
        "Could not download the UCI HAR dataset.\n" + "\n".join(f"- {e}" for e in errors)
    )


# --------------------------------------------------------------------------- reading / validating
def _read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64).to_numpy()


def _read_labels(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy().ravel().astype(int)


def _validate(X_tr, y_tr, X_te, y_te, n_features: int) -> None:
    if X_tr.shape[0] != y_tr.shape[0]:
        raise DatasetError(f"Train rows ({X_tr.shape[0]}) and labels ({y_tr.shape[0]}) are not aligned.")
    if X_te.shape[0] != y_te.shape[0]:
        raise DatasetError(f"Test rows ({X_te.shape[0]}) and labels ({y_te.shape[0]}) are not aligned.")
    if X_tr.shape[1] != n_features or X_te.shape[1] != n_features:
        raise DatasetError(
            f"Expected {n_features} features, found {X_tr.shape[1]} (train) / {X_te.shape[1]} (test)."
        )
    expected = set(ACTIVITY_NAMES)
    for name, y in (("train", y_tr), ("test", y_te)):
        found = set(np.unique(y).tolist())
        if found != expected:
            raise DatasetError(f"{name} labels {sorted(found)} do not match expected classes {sorted(expected)}.")
    if not (np.isfinite(X_tr).all() and np.isfinite(X_te).all()):
        raise DatasetError("Feature matrices contain NaN or infinite values.")


def load_har(
    data_dir: str | Path | None = None,
    auto_download: bool = True,
    n_features: int = N_FEATURES,
) -> HARData:
    """Load and validate the HAR dataset. Raises :class:`DatasetError` with an actionable message."""
    base = Path(data_dir) if data_dir else default_data_dir()
    root = find_dataset_root(base)
    if root is None:
        if not auto_download:
            raise DatasetError(
                f"HAR dataset not found under '{base}'. Extract the UCI HAR archive there "
                f"(expected '.../train/X_train.txt') or download it from {DATASET_PAGE}."
            )
        try:
            root = download_dataset(base)
        except DatasetError as exc:
            raise DatasetError(
                f"{exc}\n\nFix: download the archive manually from {DATASET_PAGE}, extract it into "
                f"'{base}' so that '{base / 'UCI HAR Dataset' / 'train' / 'X_train.txt'}' exists, then reload."
            ) from exc

    files = {
        "X_train": root / "train" / "X_train.txt",
        "y_train": root / "train" / "y_train.txt",
        "X_test": root / "test" / "X_test.txt",
        "y_test": root / "test" / "y_test.txt",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise DatasetError("Dataset files missing: " + ", ".join(missing))

    X_tr, y_tr = _read_matrix(files["X_train"]), _read_labels(files["y_train"])
    X_te, y_te = _read_matrix(files["X_test"]), _read_labels(files["y_test"])
    _validate(X_tr, y_tr, X_te, y_te, n_features)

    names_file = root / "features.txt"
    if names_file.exists():
        names = tuple(pd.read_csv(names_file, sep=r"\s+", header=None)[1].astype(str))
        if len(names) != n_features:
            names = tuple(f"feature_{i}" for i in range(n_features))
    else:
        names = tuple(f"feature_{i}" for i in range(n_features))

    for arr in (X_tr, y_tr, X_te, y_te):
        arr.setflags(write=False)  # shared across sessions -> keep immutable
    return HARData(X_tr, y_tr, X_te, y_te, names, str(root))
