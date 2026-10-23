"""app.py (Streamlit) smoke tests: every screen renders, offline and online.

Streamlit's AppTest runs the script in this process without a browser, so the
register module the app imports is the same one patched here. The register is
a throwaway one in tmp_path and the ERP is the real mock (conftest.start_erp).
Every serial here is FAKE.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import CLIENT_ID, CLIENT_SECRET, FAKE_SERIAL_1, JOB, free_port  # noqa: E402
from label_reader import register as reg  # noqa: E402
from label_reader.erp.erp_client import ErpClient  # noqa: E402

SCREENS = ["Scan", "Review", "Assets", "ERP"]


@pytest.fixture
def register(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "DB_PATH", str(tmp_path / "assets.db"))
    monkeypatch.setattr(reg, "SCANS_DIR", str(tmp_path / "scans"))
    monkeypatch.setattr(reg, "OUTBOX_PATH", str(tmp_path / "outbox.sqlite"))
    monkeypatch.setattr(reg, "_client", None)
    return reg


def use_erp(register, url: str) -> None:
    register._client = ErpClient(url, CLIENT_ID, CLIENT_SECRET, timeout=2, max_attempts=1)


def run_app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def sidebar_text(at: AppTest) -> str:
    return " ".join(e.value for e in [*at.sidebar.success, *at.sidebar.warning, *at.sidebar.error])


def test_every_screen_renders_with_the_erp_offline(register):
    use_erp(register, f"http://127.0.0.1:{free_port()}")
    at = run_app()
    assert "ERP offline" in sidebar_text(at)
    for screen in SCREENS:
        at.sidebar.radio(key="page").set_value(screen).run()
        assert not at.exception, f"{screen}: {at.exception}"


def test_every_screen_renders_with_the_erp_online(register, start_erp):
    use_erp(register, start_erp().url)
    at = run_app()
    assert "ERP connected" in sidebar_text(at)
    assert at.sidebar.selectbox(key="job_id").value == JOB
    for screen in SCREENS:
        at.sidebar.radio(key="page").set_value(screen).run()
        assert not at.exception, f"{screen}: {at.exception}"


def test_uncertain_scan_waits_on_the_review_screen(register):
    use_erp(register, f"http://127.0.0.1:{free_port()}")
    register.save_item(b"not really a jpeg", "photo.jpg",
                       {"serial": FAKE_SERIAL_1, "status": "needs_review", "source": "barcode",
                        "reason": "barcode only, no OCR serial to confirm it"})
    at = run_app()
    at.sidebar.radio(key="page").set_value("Review").run()
    assert not at.exception, at.exception
    page = " ".join(str(e.value) for e in [*at.markdown, *at.caption, *at.text_input])
    assert FAKE_SERIAL_1 in page
