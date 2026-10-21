"""The mock ERP server on its own: auth, receipts, idempotency, failure injection."""

import time

import httpx
import pytest

from conftest import CLIENT_ID, FAKE_EAN, FAKE_SERIAL_1, FAKE_SERIAL_2, JOB


@pytest.fixture
def erp(start_erp):
    return start_erp()


def api(erp, token: str | None = None) -> httpx.Client:
    headers = {"Authorization": f"Bearer {token or erp.token()}"}
    return httpx.Client(base_url=erp.url, headers=headers)


def lines_by_identifier(http: httpx.Client) -> dict[str, dict]:
    r = http.get(f"/api/v1/jobs/{JOB}/lines")
    r.raise_for_status()
    return {line["identifier"]: line for line in r.json()}


def receipt(line: dict, identifier: str | None = None, capture: str = "photo1.jpeg") -> dict:
    return {"job_id": JOB, "line_id": line["line_id"],
            "identifier": identifier or line["identifier"], "capture_id": capture}


def test_health_needs_no_auth(erp):
    r = httpx.get(erp.url + "/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_requires_a_token(erp):
    assert httpx.get(erp.url + "/api/v1/jobs").status_code == 401
    bad = httpx.get(erp.url + "/api/v1/jobs", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401


def test_wrong_secret_gets_no_token(erp):
    r = httpx.post(erp.url + "/oauth/token", json={"client_id": CLIENT_ID, "client_secret": "wrong"})
    assert r.status_code == 401
    assert "access_token" not in r.json()


def test_token_gives_jobs_and_lines(erp):
    with api(erp) as http:
        jobs = http.get("/api/v1/jobs").json()
        assert [j["job_id"] for j in jobs] == [JOB]
        lines = lines_by_identifier(http)
    assert set(lines) == {FAKE_SERIAL_1, FAKE_SERIAL_2, FAKE_EAN}
    assert lines[FAKE_SERIAL_1]["identifier_type"] == "serial"
    assert lines[FAKE_EAN]["identifier_type"] == "ean"
    assert lines[FAKE_EAN]["quantity"] == 2
    assert all(line["received"] == 0 for line in lines.values())


def test_unknown_job_is_404(erp):
    with api(erp) as http:
        assert http.get("/api/v1/jobs/JOB-NONE/lines").status_code == 404


def test_receipt_needs_an_idempotency_key(erp):
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_SERIAL_1]
        r = http.post("/api/v1/receipts", json=receipt(line))
    assert r.status_code == 400


def test_receipt_is_created_once_and_replays_return_it(erp):
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_SERIAL_1]
        first = http.post("/api/v1/receipts", json=receipt(line), headers={"Idempotency-Key": "k1"})
        again = http.post("/api/v1/receipts", json=receipt(line), headers={"Idempotency-Key": "k1"})
        assert first.status_code == 201
        assert again.status_code == 200
        assert again.json()["receipt_id"] == first.json()["receipt_id"]
        assert lines_by_identifier(http)[FAKE_SERIAL_1]["received"] == 1
        assert len(http.get(f"/api/v1/jobs/{JOB}/receipts").json()) == 1


def test_same_key_with_a_different_body_is_rejected(erp):
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_EAN]
        http.post("/api/v1/receipts", json=receipt(line, capture="a.jpeg"), headers={"Idempotency-Key": "k1"})
        r = http.post("/api/v1/receipts", json=receipt(line, capture="b.jpeg"), headers={"Idempotency-Key": "k1"})
    assert r.status_code == 422


def test_identifier_must_match_the_line(erp):
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_SERIAL_1]
        r = http.post("/api/v1/receipts", json=receipt(line, identifier=FAKE_SERIAL_2),
                      headers={"Idempotency-Key": "k1"})
    assert r.status_code == 422


def test_identifier_is_compared_without_case_or_punctuation(erp):
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_SERIAL_1]
        messy = "fakesn-0001 ab"
        r = http.post("/api/v1/receipts", json=receipt(line, identifier=messy),
                      headers={"Idempotency-Key": "k1"})
    assert r.status_code == 201


def test_over_receipt_is_a_conflict(erp):
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_SERIAL_1]   # quantity 1
        ok = http.post("/api/v1/receipts", json=receipt(line, capture="a.jpeg"), headers={"Idempotency-Key": "k1"})
        r = http.post("/api/v1/receipts", json=receipt(line, capture="b.jpeg"), headers={"Idempotency-Key": "k2"})
        assert ok.status_code == 201
        assert r.status_code == 409
        assert lines_by_identifier(http)[FAKE_SERIAL_1]["received"] == 1


def test_unknown_line_is_404(erp):
    with api(erp) as http:
        body = {"job_id": JOB, "line_id": 999, "identifier": FAKE_SERIAL_1, "capture_id": "a.jpeg"}
        r = http.post("/api/v1/receipts", json=body, headers={"Idempotency-Key": "k1"})
    assert r.status_code == 404


def test_token_expires(start_erp):
    erp = start_erp(token_ttl=1)
    token = erp.token()
    with api(erp, token) as http:
        assert http.get("/api/v1/jobs").status_code == 200
        time.sleep(1.2)
        assert http.get("/api/v1/jobs").status_code == 401


def test_rate_limit_says_when_to_retry(start_erp):
    erp = start_erp(rate_limit=2)
    with api(erp) as http:
        codes = [http.get("/api/v1/jobs") for _ in range(3)]
    assert [r.status_code for r in codes] == [200, 200, 429]
    assert int(codes[2].headers["Retry-After"]) >= 1


def test_injected_failure_is_503(start_erp):
    erp = start_erp(fail_rate=1.0)
    with api(erp) as http:
        assert http.get("/api/v1/jobs").status_code == 503
    assert httpx.get(erp.url + "/health").status_code == 200   # /health is never failed


def test_lost_response_still_commits_the_receipt(start_erp):
    erp = start_erp(lost_rate=1.0)
    with api(erp) as http:
        line = lines_by_identifier(http)[FAKE_SERIAL_1]
        lost = http.post("/api/v1/receipts", json=receipt(line), headers={"Idempotency-Key": "k1"})
        assert lost.status_code == 500
        assert lines_by_identifier(http)[FAKE_SERIAL_1]["received"] == 1
        # The client retries with the same key and gets the receipt it never saw.
        replay = http.post("/api/v1/receipts", json=receipt(line), headers={"Idempotency-Key": "k1"})
        assert replay.status_code == 200
        assert lines_by_identifier(http)[FAKE_SERIAL_1]["received"] == 1


def test_slow_response(start_erp):
    erp = start_erp(slow_rate=1.0, slow_seconds=0.5)
    with api(erp) as http:
        with pytest.raises(httpx.ReadTimeout):
            http.get("/api/v1/jobs", timeout=0.2)
