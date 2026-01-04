
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_routes():
    print("🧪 Testing Route Compatibility...")
    
    routes_to_test = [
        # Direct Routes
        "/api/v2/agent/run",
        "/api/v2/thinking-partner/analyze",
        # Legacy Prefixed Routes (Option B Patch)
        "/api/prompt/full/api/v2/agent/run",
        "/api/prompt/full/api/v2/thinking-partner/analyze",
        # Health
        "/health"
    ]
    
    for route in routes_to_test:
        url = f"{BASE_URL}{route}"
        print(f"\nChecking: {url}")
        try:
            # We use OPTIONS to check if the route exists without needing a full payload
            # or we can use a dummy POST. 
            # For Thinking Partner we can use a small payload.
            if "analyze" in route:
                resp = requests.post(url, json={"query": "test"}, timeout=5)
            elif "run" in route:
                resp = requests.post(url, json={"task": "test"}, timeout=5)
            else:
                resp = requests.get(url, timeout=5)
                
            if resp.status_code != 404:
                print(f"✅ {resp.status_code} - Route Found!")
            else:
                print(f"❌ 404 - Route NOT Found")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_routes()
