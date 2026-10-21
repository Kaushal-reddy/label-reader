"""ErpClient against the real mock ERP: tokens, retries, Retry-After, giving up."""

import time

import httpx
import pytest

from conftest import CLIENT_ID, CLIENT_SECRET, FAKE_SERIAL_1, JOB, free_port
from erp_client import ErpClient, ErpRejected, ErpUnavailable


class Recorder:
    """Remembers every request sent and every status code received."""

    def __init__(self):
        self.sent: list[str] = []          # "POST /oauth/token", ...
        self.codes: list[int] = []

    def http(self) -> httpx.Client:
        return httpx.Client(event_hooks={
            "request": [lambda req: self.sent.append(f"{req.method} {req.url.path}")],
            "response": [lambda resp: self.codes.append(resp.status_code)],
        })


def make_client(url: str, rec: Recorder | None = None, **kwargs) -> ErpClient:
    kwargs.setdefault("backoff", 0.01)
    kwargs.setdefault("max_backoff", 0.05)
    return ErpClient(url, CLIENT_ID, CLIENT_SECRET, http=(rec or Recorder()).http(), **kwargs)


def line_for(client: ErpClient, identifier: str) -> dict:
    return next(line for line in client.get_lines(JOB) if line["identifier"] == identifier)


def test_client_logs_in_and_reads_lines(start_erp):
    erp = start_erp()
    with make_client(erp.url) as client:
        assert len(client.get_lines(JOB)) == 3
        assert client.list_jobs()[0]["job_id"] == JOB


def test_wrong_secret_is_rejected_without_retrying(start_erp):
    erp = start_erp()
    rec = Recorder()
    client = ErpClient(erp.url, CLIENT_ID, "wrong", http=rec.http(), backoff=0.01)
    with pytest.raises(ErpRejected) as err:
        client.get_lines(JOB)
    assert err.value.status == 401
    assert rec.sent == ["POST /oauth/token"]


def test_token_is_reused_then_refreshed_before_it_expires(start_erp):
    erp = start_erp(token_ttl=1)
    rec = Recorder()
    with make_client(erp.url, rec) as client:
        client.get_lines(JOB)
        client.get_lines(JOB)
        time.sleep(1.0)
        client.get_lines(JOB)
    assert rec.sent.count("POST /oauth/token") == 2
    assert 401 not in rec.codes


def test_token_refreshed_once_after_a_401(start_erp):
    erp = start_erp()
    rec = Recorder()
    with make_client(erp.url, rec) as client:
        client.get_lines(JOB)
        # A restarted ERP has forgotten every token it gave out.
        erp.stop()
        start_erp(port=int(erp.url.rsplit(":", 1)[1]))
        assert len(client.get_lines(JOB)) == 3
    assert rec.codes.count(401) == 1
    assert rec.sent.count("POST /oauth/token") == 2


def test_post_receipt_survives_a_lost_response(start_erp):
    erp = start_erp(lost_rate=1.0)
    with make_client(erp.url) as client:
        line = line_for(client, FAKE_SERIAL_1)
        saved = client.post_receipt(JOB, line["line_id"], FAKE_SERIAL_1, "a.jpeg", idempotency_key="k1")
        assert saved["replayed"] is True       # the 2nd try found the receipt the 1st try saved
        assert line_for(client, FAKE_SERIAL_1)["received"] == 1


def test_retry_after_is_honoured(start_erp):
    erp = start_erp(rate_limit=1)
    rec = Recorder()
    with make_client(erp.url, rec) as client:
        start = time.monotonic()
        client.get_lines(JOB)
        client.get_lines(JOB)
        elapsed = time.monotonic() - start
    assert 429 in rec.codes
    assert elapsed >= 0.9          # waited the Retry-After second, not the tiny test backoff


def test_timeouts_are_retried_then_give_up(start_erp):
    erp = start_erp(slow_rate=1.0, slow_seconds=0.5)
    rec = Recorder()
    client = make_client(erp.url, rec, timeout=0.2, max_attempts=2)
    with pytest.raises(ErpUnavailable) as err:
        client.get_lines(JOB)
    assert rec.sent.count(f"GET /api/v1/jobs/{JOB}/lines") == 2
    assert "Timeout" in str(err.value)


def test_unreachable_erp_gives_up(tmp_path):
    client = make_client(f"http://127.0.0.1:{free_port()}", max_attempts=3)
    with pytest.raises(ErpUnavailable):
        client.get_lines(JOB)


def test_conflict_is_not_retried(start_erp):
    erp = start_erp()
    rec = Recorder()
    with make_client(erp.url, rec) as client:
        line = line_for(client, FAKE_SERIAL_1)   # quantity 1
        client.post_receipt(JOB, line["line_id"], FAKE_SERIAL_1, "a.jpeg", idempotency_key="k1")
        with pytest.raises(ErpRejected) as err:
            client.post_receipt(JOB, line["line_id"], FAKE_SERIAL_1, "b.jpeg", idempotency_key="k2")
    assert err.value.status == 409
    assert "already fully received" in err.value.detail
    assert rec.sent.count("POST /api/v1/receipts") == 2
