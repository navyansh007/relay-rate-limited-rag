import socket
import threading
import time

import httpx
import pytest
import uvicorn

from relay.standin import create_app, load_standin_config

FAST = {
    "latency": {
        "workers": 4,
        "ttft_median_ms": 20,
        "ttft_sigma": 0.1,
        "decode_tps_median": 5000,
        "prefill_ms_per_1k": 0,
    },
    "faults": {"p_server_error": 0, "p_hang": 0, "p_slow_success": 0, "p_midstream_drop": 0},
    "limits": {"default": {"rpm": 1000, "tpm": 100000, "rpd": 100000, "tpd": 10000000}},
    "content": {"replay_file": "raw/__no_replay__.jsonl"},
}


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(load_standin_config(FAST)), host="127.0.0.1", port=port, log_level="error"
        )
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def admin(base_url, patch):
    """Resets the stand-in, then applies FAST defaults deep-merged with ``patch``."""
    from relay.config import _merge

    httpx.post(f"{base_url}/admin/reset")
    httpx.post(f"{base_url}/admin/config", json=_merge(FAST, patch)).raise_for_status()
