import numpy as np
import pytest

from src.data_loader import ACTIVITY_NAMES, DatasetError, find_dataset_root, load_har


def test_loader_shapes_and_alignment(har):
    assert har.X_train.shape[1] == 561 == har.X_test.shape[1]
    assert har.X_train.shape[0] == har.y_train.shape[0]
    assert har.X_test.shape[0] == har.y_test.shape[0]


def test_labels_within_expected_classes(har):
    assert set(np.unique(har.y_train)) == set(ACTIVITY_NAMES)
    assert set(np.unique(har.y_test)) <= set(ACTIVITY_NAMES)


def test_arrays_are_read_only(har):
    with pytest.raises(ValueError):
        har.X_train[0, 0] = 1.0


def test_class_distribution(har):
    df = har.class_distribution()
    assert len(df) == 6 and df["Train"].sum() == har.X_train.shape[0]


def test_missing_dataset_gives_actionable_error(tmp_path):
    assert find_dataset_root(tmp_path) is None
    with pytest.raises(DatasetError, match="not found"):
        load_har(tmp_path, auto_download=False)


def test_misaligned_labels_rejected(har_dir, tmp_path):
    import shutil

    dst = tmp_path / "copy"
    shutil.copytree(har_dir, dst)
    y = dst / "UCI HAR Dataset" / "train" / "y_train.txt"
    lines = y.read_text().splitlines()[:-3]
    y.write_text("\n".join(lines))
    with pytest.raises(DatasetError, match="aligned"):
        load_har(dst, auto_download=False)


def test_wrong_feature_count_rejected(har_dir):
    with pytest.raises(DatasetError, match="features"):
        load_har(har_dir, auto_download=False, n_features=560)


def test_nested_archive_extraction(har_dir, tmp_path):
    """The official UCI archive wraps a second zip; the extractor must handle both layouts."""
    import io
    import zipfile

    from src.data_loader import _extract_archive

    src = har_dir / "UCI HAR Dataset"
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for f in src.rglob("*.txt"):
            z.write(f, f"UCI HAR Dataset/{f.relative_to(src).as_posix()}")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("UCI HAR Dataset.zip", inner.getvalue())
        z.writestr("__MACOSX/junk", "x")
    dest = tmp_path / "out"
    dest.mkdir()
    _extract_archive(outer.getvalue(), dest)
    assert find_dataset_root(dest) is not None


def test_zip_slip_rejected(tmp_path):
    import io
    import zipfile

    from src.data_loader import _extract_archive

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(DatasetError, match="Unsafe"):
        _extract_archive(buf.getvalue(), dest)
