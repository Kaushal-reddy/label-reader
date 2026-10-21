"""Shared test helpers: run the mock ERP for real, on a free localhost port.

Starlette's TestClient is deprecated with this httpx version, and a real server
is more honest anyway: timeouts and dropped connections behave as they would in
production. Every serial here is FAKE - never put real device serials in tests.
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn

import mock_erp

CLIENT_ID = "test-client"
CLIENT_SECRET = "test-secret"
JOB = "JOB-TEST"


def fake_ean(first12: str) -> str:
    """A 13-digit product number with a correct check digit (prefix 2 = in-store use, never a real product)."""
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(first12))
    return first12 + str((10 - total % 10) % 10)


FAKE_SERIAL_1 = "FAKESN0001AB"
FAKE_SERIAL_2 = "FAKESN0002CD"
FAKE_EAN = fake_ean("200000000001")
FAKE_LINES = [
    {"identifier": FAKE_SERIAL_1, "description": "Laptop A", "quantity": 1},
    {"identifier": FAKE_SERIAL_2, "description": "Mouse B", "quantity": 1},
    {"identifier": FAKE_EAN, "description": "Product C", "quantity": 2},
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RunningErp:
    """The mock ERP served by uvicorn in a background thread."""

    def __init__(self, settings: mock_erp.Settings, port: int):
        self.settings = settings
        self.url = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(mock_erp.create_app(settings), host="127.0.0.1",
                                port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("mock ERP did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def token(self) -> str:
        r = httpx.post(self.url + "/oauth/token",
                       json={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET})
        r.raise_for_status()
        return r.json()["access_token"]


@pytest.fixture
def db_path(tmp_path) -> str:
    """A fresh ERP database with one job (FAKE_LINES)."""
    path = str(tmp_path / "erp.sqlite")
    mock_erp.seed_job(path, JOB, FAKE_LINES)
    return path


@pytest.fixture
def start_erp(db_path):
    """start_erp(fail_rate=..., ...) -> RunningErp. Every server is stopped after the test.

    Calling it again (after .stop()) restarts the ERP on the same database.
    """
    servers = []

    def start(port: int | None = None, **overrides) -> RunningErp:
        settings = mock_erp.Settings(db_path=db_path, client_id=CLIENT_ID,
                                     client_secret=CLIENT_SECRET, seed=1, **overrides)
        erp = RunningErp(settings, port or free_port())
        servers.append(erp)
        return erp

    yield start
    for erp in servers:
        erp.stop()
