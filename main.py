# main.py
"""
Process entry point for local + production.

Render/hosts will run:
  uvicorn main:app --host 0.0.0.0 --port $PORT

Local dev (optional):
  python main.py
"""

import os
import uvicorn

# ✅ Export app for uvicorn
from api import app  # <-- IMPORTANT: app must exist in api.py


def run_local() -> None:
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    # If a host provides PORT, we run in "prod-like" mode.
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,   # hosts don't want reload
        log_level="info",
    )
