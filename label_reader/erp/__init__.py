"""Stage 9: send the pipeline's decisions to an asset-recording system.

  mock_erp     a generic mock ERP (FastAPI + SQLite) with failure injection
  erp_client   tokens, retries and idempotent receipts
  post_to_erp  matching to the job's manifest, outbox, review queue

Needs the extra packages in requirements-erp.txt.
"""
