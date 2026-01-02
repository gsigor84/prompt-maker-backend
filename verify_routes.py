import sys
import os

# Ensure current dir is in path
sys.path.insert(0, os.getcwd())

from api import app

print("Routes found:")
found = False
for route in app.routes:
    print(f" - {route.path}")
    if route.path == "/api/v2/agent/run":
        found = True

if found:
    print("\nSUCCESS: /api/v2/agent/run is present.")
else:
    print("\nFAILURE: /api/v2/agent/run is MISSING.")
    print(f"File imported from: {app.__module__}")
