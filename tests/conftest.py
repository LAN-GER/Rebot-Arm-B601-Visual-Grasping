from pathlib import Path

import pytest

from rebot_grasp.cameras.synthetic import demo_capture
from rebot_grasp.config import load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "config/default.yaml")


@pytest.fixture
def scene():
    return demo_capture()
