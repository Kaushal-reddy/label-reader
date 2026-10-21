"""Stage 9: matching rules, and posting through the outbox against the real mock ERP."""

import httpx

from conftest import CLIENT_ID, CLIENT_SECRET, FAKE_EAN, FAKE_SERIAL_1, FAKE_SERIAL_2, JOB, free_port
from erp_client import ErpClient
from post_to_erp import Outbox, match_row, plan_posts, run, send_pending

# ------------------------------------------------------------ matching (no server)

LINES = [
    {"line_id": 1, "identifier": FAKE_SERIAL_1, "description": "Laptop A", "quantity": 1, "received": 0},
    {"line_id": 2, "identifier": FAKE_SERIAL_2, "description": "Mouse B", "quantity": 1, "received": 0},
    {"line_id": 3, "identifier": FAKE_EAN, "description": "Product C", "quantity": 2, "received": 0},
]


def row(serial: str, status: str = "accepted", filename: str = "a.jpeg", reason: str = "") -> dict:
    return {"filename": filename, "status": status, "serial": serial, "reason": reason}


def test_accepted_exact_match_is_posted():
    m = match_row(row(FAKE_SERIAL_1), LINES, {})
    assert m.action == "post"
    assert m.line is not None and m.line["line_id"] == 1


def test_lookalike_match_is_not_posted():
    m = match_row(row(FAKE_SERIAL_1.replace("0", "O")), LINES, {})     # FAKESNO0O1AB
    assert (m.action, m.category) == ("review", "near_match")
    assert "look-alike" in m.reason
    assert [c["line_id"] for c in m.candidates] == [1]


def test_one_character_off_is_not_posted():
    m = match_row(row(FAKE_SERIAL_1[:-1] + "X"), LINES, {})
    assert (m.action, m.category) == ("review", "near_match")
    assert "one character" in m.reason


def test_unknown_item_is_not_on_the_manifest():
    m = match_row(row("OTHERSN9999ZZ"), LINES, {})
    assert (m.action, m.category) == ("review", "not_on_manifest")


def test_several_matching_lines_go_to_review():
    lines = LINES + [{**LINES[0], "line_id": 4}]
    m = match_row(row(FAKE_SERIAL_1), lines, {})
    assert (m.action, m.category) == ("review", "several_lines")
    assert [c["line_id"] for c in m.candidates] == [1, 4]


def test_line_already_received_on_the_erp():
    lines = [{**LINES[0], "received": 1}] + LINES[1:]
    m = match_row(row(FAKE_SERIAL_1), lines, {})
    assert (m.action, m.category) == ("review", "already_received")


def test_second_photo_of_the_same_item_in_one_run():
    m = match_row(row(FAKE_SERIAL_1, filename="b.jpeg"), LINES, {1: ["a.jpeg"]})
    assert (m.action, m.category) == ("review", "already_received")
    assert "a.jpeg" in m.reason


def test_quantity_two_takes_two_photos():
    assert match_row(row(FAKE_EAN, filename="b.jpeg"), LINES, {3: ["a.jpeg"]}).action == "post"
    assert match_row(row(FAKE_EAN, filename="c.jpeg"), LINES, {3: ["a.jpeg", "b.jpeg"]}).action == "review"


def test_identifier_is_normalised_before_matching():
    assert match_row(row("fakesn-0001 ab"), LINES, {}).action == "post"


def test_unsure_pipeline_result_is_never_posted_but_gets_a_hint():
    m = match_row(row(FAKE_SERIAL_1, status="needs_review", reason="barcode only"), LINES, {})
    assert (m.action, m.category) == ("review", "not_confident")
    assert "barcode only" in m.reason
    assert "line 1 (Laptop A)" in m.reason


def test_nothing_read_goes_to_review():
    m = match_row(row("", status="needs_review", reason="nothing serial-shaped found"), LINES, {})
    assert (m.action, m.category) == ("review", "not_confident")
    assert m.candidates == []


# ------------------------------------------------------------ posting (real mock ERP)

ROWS = [
    row(FAKE_SERIAL_1, filename="laptop_a_1.jpeg"),
    row(FAKE_SERIAL_1, filename="laptop_a_2.jpeg"),                     # same item again
    row(FAKE_SERIAL_2, filename="mouse_b.jpeg"),
    row(FAKE_EAN, filename="product_c_1.jpeg"),
    row(FAKE_EAN, filename="product_c_2.jpeg"),                         # quantity 2: both post
    row("", status="needs_review", filename="blurry.jpeg", reason="nothing serial-shaped found"),
]


def make_client(url: str, **kwargs) -> ErpClient:
    kwargs.setdefault("backoff", 0.01)
    kwargs.setdefault("max_backoff", 0.05)
    return ErpClient(url, CLIENT_ID, CLIENT_SECRET, **kwargs)


def received(erp) -> dict[str, int]:
    headers = {"Authorization": f"Bearer {erp.token()}"}
    lines = httpx.get(f"{erp.url}/api/v1/jobs/{JOB}/lines", headers=headers).json()
    return {line["identifier"]: line["received"] for line in lines}


def receipt_count(erp) -> int:
    headers = {"Authorization": f"Bearer {erp.token()}"}
    return len(httpx.get(f"{erp.url}/api/v1/jobs/{JOB}/receipts", headers=headers).json())


def test_every_item_received_exactly_once_despite_failures(start_erp, tmp_path):
    erp = start_erp(fail_rate=0.5, lost_rate=0.5)
    with make_client(erp.url, max_attempts=40) as client:
        report = run(ROWS, JOB, client, Outbox(str(tmp_path / "outbox.sqlite")))
    assert sorted(p["filename"] for p in report.posted) == [
        "laptop_a_1.jpeg", "mouse_b.jpeg", "product_c_1.jpeg", "product_c_2.jpeg"]
    assert {r["filename"]: r["category"] for r in report.review} == {
        "laptop_a_2.jpeg": "already_received", "blurry.jpeg": "not_confident"}
    assert report.pending == []
    erp.settings.fail_rate = erp.settings.lost_rate = 0.0     # read the result back without failures
    assert received(erp) == {FAKE_SERIAL_1: 1, FAKE_SERIAL_2: 1, FAKE_EAN: 2}
    assert receipt_count(erp) == 4


def test_running_again_posts_nothing_new(start_erp, tmp_path):
    erp = start_erp()
    outbox = Outbox(str(tmp_path / "outbox.sqlite"))
    with make_client(erp.url) as client:
        run(ROWS, JOB, client, outbox)
        again = run(ROWS, JOB, client, outbox)
    assert len(again.posted) == 4
    assert all(p["note"] == "posted in an earlier run" for p in again.posted)
    assert {r["filename"] for r in again.review} == {"laptop_a_2.jpeg", "blurry.jpeg"}
    assert receipt_count(erp) == 4


def test_outbox_keeps_posts_while_the_erp_is_down(start_erp, tmp_path):
    erp = start_erp()
    outbox = Outbox(str(tmp_path / "outbox.sqlite"))
    with make_client(erp.url) as client:
        job_key = plan_posts(ROWS, JOB, client, outbox).job_key       # matched while the ERP was up ...

    with make_client(f"http://127.0.0.1:{free_port()}", max_attempts=2) as down:
        error = send_pending(down, outbox, job_key)           # ... then it went down
    assert error is not None
    pending = outbox.pending(job_key)
    assert len(pending) == 4
    assert "failed 2 times" in pending[0]["last_error"]   # it stops at the first: no point waiting for each
    assert receipt_count(erp) == 0

    with make_client(erp.url) as client:                      # back up: a re-run sends them
        report = run(ROWS, JOB, client, outbox)
    assert len(report.posted) == 4
    assert outbox.pending(job_key) == []
    assert received(erp) == {FAKE_SERIAL_1: 1, FAKE_SERIAL_2: 1, FAKE_EAN: 2}
    assert receipt_count(erp) == 4


def test_rejected_post_goes_to_review(start_erp, tmp_path):
    erp = start_erp()
    outbox = Outbox(str(tmp_path / "outbox.sqlite"))
    with make_client(erp.url) as client:
        job_key = plan_posts(ROWS, JOB, client, outbox).job_key
        # Someone receives the laptop by hand on the ERP before our post goes out.
        laptop = next(line for line in client.get_lines(JOB) if line["identifier"] == FAKE_SERIAL_1)
        client.post_receipt(JOB, laptop["line_id"], FAKE_SERIAL_1, "by-hand", idempotency_key="manual-1")
        send_pending(client, outbox, job_key)
        report = run(ROWS, JOB, client, outbox)
    rejected = [r for r in report.review if r["category"] == "erp_rejected"]
    assert [r["filename"] for r in rejected] == ["laptop_a_1.jpeg"]
    assert "409" in rejected[0]["reason"]
    assert receipt_count(erp) == 4      # by-hand + mouse + 2 product
