from pathlib import Path

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    for sub in ("frames", "masks", "flows"):
        (tmp_path / sub).mkdir()
    return tmp_path
