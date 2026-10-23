"""End-to-end check of Stage 9 with real processes: the mock ERP server, failing
on purpose, and post_to_erp.py posting a dataset's validate.py results to it.

Run with:  python -m scripts.erp_e2e [folder]   (default photos/synthetic; run
           validate.py on that folder first). CI runs it on the synthetic set.

Steps: seed a job from <folder>/truth.csv, start `mock_erp serve` with 503s,
lost replies and rate limiting, run post_to_erp.py until nothing is pending,
then run it once more. Checks, against the ERP's own receipts:
  - every accepted photo was received exactly once, and nothing else was
  - the extra run posted nothing new (idempotency keys + outbox)
Exits non-zero with the reasons if a check fails. Uses its own database and
port, so it doesn't touch output/erp/ or a running mock ERP.
"""

import os
import subprocess
import sys
import tempfile
import time

import httpx
import pandas as pd

from label_reader.erp.erp_client import DEV_CLIENT_ID, DEV_CLIENT_SECRET, ErpClient

DEFAULT_DIR = os.path.join("photos", "synthetic")
JOB = "JOB-E2E"
PORT = 8765
FAILURES = ["--fail-rate", "0.3", "--lost-rate", "0.3", "--rate-limit", "20", "--seed", "7"]
MAX_RUNS = 6


def module(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def wait_for_health(url: str, server: subprocess.Popen, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"mock ERP exited with code {server.returncode}")
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise SystemExit("mock ERP did not start")


def post(folder: str, env: dict) -> str:
    run = subprocess.run(module("label_reader.erp.post_to_erp", folder, "--job", JOB),
                         env=env, capture_output=True, text=True)
    print(run.stdout + run.stderr)
    return run.stdout


def main(folder: str = DEFAULT_DIR) -> None:
    out_dir = os.path.join("output", os.path.basename(os.path.normpath(folder)))
    validation = pd.read_csv(os.path.join(out_dir, "validation.csv"), dtype=str, keep_default_na=False)
    erp_dir = os.path.join(out_dir, "erp")
    os.makedirs(erp_dir, exist_ok=True)
    outbox = os.path.join(erp_dir, "outbox.sqlite")
    if os.path.exists(outbox):
        os.remove(outbox)               # a fresh job each time, so a fresh outbox

    url = f"http://127.0.0.1:{PORT}"
    env = {**os.environ, "ERP_URL": url, "ERP_CLIENT_ID": DEV_CLIENT_ID, "ERP_CLIENT_SECRET": DEV_CLIENT_SECRET}
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "erp.sqlite")
        subprocess.run(module("label_reader.erp.mock_erp", "--db", db, "seed", "--job", JOB,
                              "--from", os.path.join(folder, "truth.csv")), env=env, check=True)
        server = subprocess.Popen(module("label_reader.erp.mock_erp", "--db", db, "serve",
                                         "--port", str(PORT), *FAILURES), env=env)
        try:
            wait_for_health(url, server)
            for attempt in range(1, MAX_RUNS + 1):
                print(f"--- post_to_erp run {attempt}")
                if "still pending:          0" in post(folder, env):
                    break
            else:
                raise SystemExit(f"posts still pending after {MAX_RUNS} runs")
            print("--- one more run: must post nothing new")
            again = post(folder, env)
            with ErpClient(url, DEV_CLIENT_ID, DEV_CLIENT_SECRET, backoff=0.1) as client:
                receipts = client.get_receipts(JOB)
        finally:
            server.terminate()
            server.wait(timeout=10)

    problems = []
    accepted = sorted(validation.filename[validation.status == "accepted"])
    received = sorted(r["capture_id"] for r in receipts)
    if received != accepted:
        problems.append(f"receipts {received} != accepted photos {accepted}")
    if "posted now:             0" not in again:
        problems.append("the extra run posted something again")
    if problems:
        raise SystemExit("ERP end-to-end check FAILED:\n  " + "\n  ".join(problems))
    print(f"ERP end-to-end check passed: {len(receipts)} receipts, one per accepted photo, "
          "with 503s, lost replies and rate limiting; the extra run posted nothing.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
