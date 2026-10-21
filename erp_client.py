"""Client for the mock ERP (mock_erp.py): login, retries, idempotent receipts.

What it takes care of, so the calling code doesn't have to:
  - tokens: logs in with the client id + secret, reuses the token, gets a new
    one shortly before it expires, and once more if the ERP answers 401
    (e.g. the ERP restarted and forgot its tokens)
  - retries: network errors, timeouts, 5xx and 429 are tried again after an
    exponential backoff with jitter (random spread, so many clients don't
    retry in lock-step), or after the ERP's Retry-After when it sends one
  - giving up: after max_attempts it raises ErpUnavailable; the caller keeps
    the work and tries again later (post_to_erp.py keeps it in an outbox)
  - other 4xx answers (bad request, conflict, not found) are the ERP saying
    "no", so they are raised as ErpRejected straight away, never retried

Retrying a POST is only safe because every receipt carries an Idempotency-Key:
if the first try was saved but its reply was lost, the retry gets the saved
receipt back (status 200) instead of receiving the asset twice.

Usage (the ERP_URL, ERP_CLIENT_ID and ERP_CLIENT_SECRET environment variables):
    from erp_client import ErpClient
    with ErpClient.from_env() as erp:
        lines = erp.get_lines("JOB-0001")
"""

import os
import random
import time

import httpx

# Same development defaults as mock_erp.DEV_CLIENT_ID / DEV_CLIENT_SECRET
# (copied so the client doesn't import the server and FastAPI).
DEV_URL = "http://127.0.0.1:8000"
DEV_CLIENT_ID = "label-reader"
DEV_CLIENT_SECRET = "dev-secret-change-me"


class ErpError(Exception):
    """Anything that went wrong talking to the ERP."""


class ErpRejected(ErpError):
    """The ERP understood the request and said no (a 4xx answer). Retrying won't help."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class ErpUnavailable(ErpError):
    """The ERP couldn't be reached or kept failing, even after retrying. Try again later."""


class _Retry(Exception):
    """Internal: this attempt failed in a way worth retrying."""

    def __init__(self, reason: str, wait: float | None = None):
        super().__init__(reason)
        self.wait = wait      # seconds the ERP asked us to wait (Retry-After), if it said


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["detail"])
    except (ValueError, KeyError, TypeError):
        return resp.text[:200]


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return min(60.0, float(resp.headers["Retry-After"]))
    except (KeyError, ValueError):
        return None


class ErpClient:
    def __init__(self, base_url: str, client_id: str, client_secret: str,
                 timeout: float = 10.0, max_attempts: int = 5, http: httpx.Client | None = None,
                 backoff: float = 0.5, max_backoff: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self._client_secret = client_secret
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff = backoff            # first retry waits about this long, then 2x, 4x, ...
        self.max_backoff = max_backoff
        self.http = http or httpx.Client()
        self._token: str | None = None
        self._token_expires = 0.0         # time.monotonic() after which we get a new token

    @classmethod
    def from_env(cls, **kwargs) -> "ErpClient":
        client_id = os.environ.get("ERP_CLIENT_ID", "")
        client_secret = os.environ.get("ERP_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            print("WARNING: ERP_CLIENT_ID / ERP_CLIENT_SECRET not set, using development credentials.")
            client_id, client_secret = DEV_CLIENT_ID, DEV_CLIENT_SECRET
        return cls(os.environ.get("ERP_URL", DEV_URL), client_id, client_secret, **kwargs)

    def __enter__(self) -> "ErpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    # ------------------------------------------------------------ the API

    def list_jobs(self) -> list[dict]:
        return self._request("GET", "/api/v1/jobs").json()

    def get_lines(self, job_id: str) -> list[dict]:
        return self._request("GET", f"/api/v1/jobs/{job_id}/lines").json()

    def get_receipts(self, job_id: str) -> list[dict]:
        return self._request("GET", f"/api/v1/jobs/{job_id}/receipts").json()

    def post_receipt(self, job_id: str, line_id: int, identifier: str, capture_id: str,
                     idempotency_key: str) -> dict:
        """Receive one asset. Returns the receipt, with "replayed": True when the
        ERP already had it under this key (e.g. an earlier try's reply was lost)."""
        body = {"job_id": job_id, "line_id": line_id, "identifier": identifier, "capture_id": capture_id}
        resp = self._request("POST", "/api/v1/receipts", json=body,
                             headers={"Idempotency-Key": idempotency_key})
        return {**resp.json(), "replayed": resp.status_code == 200}

    # ------------------------------------------------------------ plumbing

    def _check(self, resp: httpx.Response) -> httpx.Response:
        """Return a successful response, or raise _Retry / ErpRejected."""
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Retry(f"HTTP {resp.status_code}: {_detail(resp)}", _retry_after(resp))
        if resp.status_code >= 400:
            raise ErpRejected(resp.status_code, _detail(resp))
        return resp

    def _auth_header(self) -> dict:
        """A valid token, getting a new one if there's none or it's about to expire."""
        if self._token is None or time.monotonic() >= self._token_expires:
            resp = self._check(self.http.post(
                self.base_url + "/oauth/token", timeout=self.timeout,
                json={"grant_type": "client_credentials", "client_id": self.client_id,
                      "client_secret": self._client_secret}))
            data = resp.json()
            ttl = float(data["expires_in"])
            self._token = data["access_token"]
            # Renew a little early (a fifth of the lifetime, at most 30 s), so a
            # token never expires between being checked here and reaching the ERP.
            self._token_expires = time.monotonic() + ttl - min(30.0, ttl / 5)
        return {"Authorization": f"Bearer {self._token}"}

    def _wait_before_retry(self, failures: int) -> float:
        """Exponential backoff with jitter: about backoff, 2x, 4x, ... capped at max_backoff."""
        base = min(self.max_backoff, self.backoff * 2 ** (failures - 1))
        return random.uniform(base / 2, base)

    def _request(self, method: str, path: str, json: dict | None = None,
                 headers: dict | None = None) -> httpx.Response:
        failures = 0
        refreshed = False
        while True:
            try:
                resp = self.http.request(method, self.base_url + path, json=json, timeout=self.timeout,
                                         headers={**self._auth_header(), **(headers or {})})
                if resp.status_code == 401 and not refreshed:
                    # Token no longer accepted: get a new one, once.
                    refreshed = True
                    self._token = None
                    continue
                return self._check(resp)
            except (httpx.TransportError, _Retry) as e:
                failures += 1
                reason = str(e) if isinstance(e, _Retry) else f"{type(e).__name__}: {e}"
                if failures >= self.max_attempts:
                    raise ErpUnavailable(f"{method} {path} failed {failures} times, last: {reason}") from e
                wait = e.wait if isinstance(e, _Retry) and e.wait is not None else None
                time.sleep(wait if wait is not None else self._wait_before_retry(failures))
