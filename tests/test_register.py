"""register.py: the asset register's link to the ERP (used by app.py).

Runs the real mock ERP (conftest.start_erp) and a throwaway register in
tmp_path. Every serial here is FAKE.
"""

import pytest

from label_reader import register as reg
from conftest import CLIENT_ID, CLIENT_SECRET, FAKE_EAN, FAKE_SERIAL_1, FAKE_SERIAL_2, JOB
from label_reader.erp.erp_client import ErpClient


@pytest.fixture
def register(tmp_path, monkeypatch):
    """A fresh, empty register in tmp_path (instead of data/)."""
    monkeypatch.setattr(reg, "DB_PATH", str(tmp_path / "assets.db"))
    monkeypatch.setattr(reg, "SCANS_DIR", str(tmp_path / "scans"))
    monkeypatch.setattr(reg, "OUTBOX_PATH", str(tmp_path / "outbox.sqlite"))
    monkeypatch.setattr(reg, "_client", None)
    return reg


def connect(register, erp):
    """Point the register's ERP client at this test server (no retries: fail fast)."""
    register._client = ErpClient(erp.url, CLIENT_ID, CLIENT_SECRET, timeout=2, max_attempts=1)


def saved(register, serial: str, status: str = "accepted", name: str = "photo.jpg") -> int:
    return register.save_item(b"not really a jpeg", name,
                              {"serial": serial, "status": status, "source": "barcode+ocr", "reason": "test"})


def test_accepted_item_is_received_once(register, start_erp):
    connect(register, start_erp())
    item = saved(register, FAKE_SERIAL_1)
    assert register.post_item(item, JOB)[0] == reg.ERP_SENT
    # Sending the same item again finds its outbox entry instead of a second receipt.
    assert register.post_item(item, JOB)[0] == reg.ERP_SENT
    assert len(register.erp().get_receipts(JOB)) == 1
    assert register.get_item(item)["erp_status"] == reg.ERP_SENT


def test_second_photo_of_same_item_is_already_received(register, start_erp):
    connect(register, start_erp())
    register.post_item(saved(register, FAKE_SERIAL_1, name="a.jpg"), JOB)
    status, _, _ = register.post_item(saved(register, FAKE_SERIAL_1, name="b.jpg"), JOB)
    assert status == "already_received"


def test_lookalike_is_near_match_and_can_be_corrected(register, start_erp):
    connect(register, start_erp())
    item = saved(register, FAKE_SERIAL_2.replace("0", "O"))     # OCR read O for 0
    status, _, candidates = register.post_item(item, JOB)
    assert status == "near_match"
    assert [c["identifier"] for c in candidates] == [FAKE_SERIAL_2]
    assert [i["id"] for i in register.attention_items(JOB)] == [item]
    # A person picks the line; sending again now posts it.
    register.correct_serial(item, FAKE_SERIAL_2)
    assert register.post_item(item, JOB)[0] == reg.ERP_SENT
    assert register.attention_items(JOB) == []


def test_needs_review_item_is_not_posted_until_confirmed(register, start_erp):
    connect(register, start_erp())
    item = saved(register, FAKE_EAN, status="needs_review")
    assert register.post_item(item, JOB)[0] == "not_confident"
    register.confirm_item(item, FAKE_EAN, notes="")
    assert register.post_item(item, JOB)[0] == reg.ERP_SENT
    assert register.get_item(item)["item_type"] == "product"


def test_erp_down_then_back(register, start_erp):
    erp = start_erp()
    connect(register, erp)
    port = int(erp.url.rsplit(":", 1)[1])
    erp.stop()
    assert not register.erp_online()
    item = saved(register, FAKE_SERIAL_1)
    assert register.post_item(item, JOB)[0] == reg.ERP_OFFLINE
    assert [i["id"] for i in register.unsent_items(JOB)] == [item]

    erp = start_erp(port=port)            # same database, same address
    assert register.erp_online()
    assert register.post_item(item, JOB)[0] == reg.ERP_SENT
    assert register.unsent_items(JOB) == []


def test_lost_reply_stays_pending_then_retry_sends_it_once(register, start_erp):
    # Every reply to a receipt is lost (the ERP saves it, then answers 500).
    erp = start_erp(lost_rate=1.0)
    connect(register, erp)
    item = saved(register, FAKE_SERIAL_1)
    assert register.post_item(item, JOB)[0] == reg.ERP_PENDING
    assert register.pending_count(JOB) == 1

    port = int(erp.url.rsplit(":", 1)[1])
    erp.stop()
    start_erp(port=port)                  # healthy again
    assert register.retry_pending(JOB) is None
    assert register.get_item(item)["erp_status"] == reg.ERP_SENT
    assert register.pending_count(JOB) == 0
    assert len(register.erp().get_receipts(JOB)) == 1   # the retry didn't receive it twice


def test_old_register_gets_erp_columns(tmp_path, monkeypatch):
    """A data/assets.db made before the ERP columns existed still opens."""
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT NOT NULL, "
                "photo_path TEXT NOT NULL, serial TEXT NOT NULL, item_type TEXT NOT NULL, "
                "status TEXT NOT NULL, source TEXT, reason TEXT, assigned_to TEXT DEFAULT '', "
                "notes TEXT DEFAULT '')")
    con.execute("INSERT INTO items (scanned_at, photo_path, serial, item_type, status) "
                f"VALUES ('t', 'p', '{FAKE_SERIAL_1}', 'device', 'accepted')")
    con.commit()
    con.close()
    monkeypatch.setattr(reg, "DB_PATH", str(path))
    monkeypatch.setattr(reg, "SCANS_DIR", str(tmp_path / "scans"))
    assert reg.get_item(1)["erp_status"] == ""
    assert list(reg.all_items().erp_status) == [""]
