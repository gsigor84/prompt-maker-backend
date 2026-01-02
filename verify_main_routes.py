import sys
import os

sys.path.insert(0, os.getcwd())

# We import from main because that is where we mounted the router
from main import fastapi_app

print("Routes in fastapi_app (via main.py):")
found = False
for route in fastapi_app.routes:
    print(f" - {route.path}")
    if route.path == "/api/v2/agent/run":
        found = True

if found:
    print("\nSUCCESS: /api/v2/agent/run is present via main mount.")
else:
    print("\nFAILURE: Endpoint missing.")
