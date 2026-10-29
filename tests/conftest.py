import io

import pytest
from PIL import Image

from app.config import Settings


@pytest.fixture
def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), (80, 150, 220)).save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        model_name="dummy",
        rate_limit_qps=0,
        key_rate_limit_qps=0,
        max_batch_size=4,
        max_wait_ms=10,
        warmup_runs=1,
        torch_threads="1",
        log_level="ERROR",
    )
