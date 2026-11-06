import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import make_synthetic_har  # noqa: E402


@pytest.fixture(scope="session")
def har_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("har")
    make_synthetic_har(root)
    return root


@pytest.fixture(scope="session")
def har(har_dir):
    from src.data_loader import load_har

    return load_har(har_dir, auto_download=False)
