import json
import urllib.request
import urllib.error
import time

BASE = "http://127.0.0.1:8000/api/v1"

def call(method, path, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "null")

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("[" + status + "] " + label + ((" -- " + detail) if detail and status == "FAIL" else ""))

print("=" * 60)
print("Checking all 7 AI agents against the live server")
print("=" * 60)

email = "agenttest_" + str(time.time()) + "@test.com"
status, body = call("POST", "/users/register", {
    "full_name": "Agent Test Patient", "email": email,
    "password": "testpass123", "role": "patient",
})
check("Setup: register test patient", status == 201, "status=" + str(status) + " " + str(body))
patient_id = body.get("user", {}).get("id") if status == 201 else None
patient_token = body.get("access_token") if status == 201 else None

print("\n--- Agent 1: Scheme Intelligence ---")
status, body = call("GET", "/schemes/")
check("List schemes", status == 200, "status=" + str(status))
status, body = call("POST", "/schemes/check-eligibility", {"age": 30, "state": "Tamil Nadu"})
check("Check eligibility", status == 200, "status=" + str(status) + " " + str(body))

print("\n--- Agent 2: Prescription OCR ---")
if patient_id:
    status, body = call("POST", "/prescriptions/scan", {
        "patient_id": patient_id, "image_storage_ref": "test-ref-123",
    })
    check("Scan prescription returns a draft record", status == 200 and body.get("status") == "draft", "status=" + str(status) + " " + str(body))

print("\n--- Agent 3: Medicine Intelligence ---")
status, body = call("GET", "/medicines/search?q=paracetamol")
check("Search medicines", status == 200, "status=" + str(status) + " " + str(body))

print("\n--- Agent 4: Recommendation Agent ---")
if patient_id:
    status, body = call("POST", "/recommendations/", {
        "patient_id": patient_id, "intent": "find a doctor",
        "candidate_type": "doctor",
    })
    check("Get recommendations", status == 200, "status=" + str(status) + " " + str(body))

print("\n--- Agent 5: Health Assistant (emergency detection) ---")
if patient_id:
    status, body = call("POST", "/assistant/message", {
        "user_id": patient_id, "text": "I need to find a cardiologist",
    })
    check("Normal message -> is_emergency false", status == 200 and body.get("is_emergency") is False, "status=" + str(status) + " " + str(body))

    status, body = call("POST", "/assistant/message", {
        "user_id": patient_id, "text": "I have severe chest pain and cant breathe",
    })
    check("Emergency message -> is_emergency true", status == 200 and body.get("is_emergency") is True, "status=" + str(status) + " " + str(body))

print("\n--- Agent 6: Voice Assistant ---")
if patient_id:
    status, body = call("POST", "/voice/message?audio_storage_ref=test-audio-ref", {
        "user_id": patient_id, "audio_format": "wav",
        "audio_size_bytes": 50000, "audio_duration_seconds": 4.0,
    })
    check("Voice message accepted", status == 200, "status=" + str(status) + " " + str(body))

print("\n--- Agent 7: Analytics ---")
if patient_id:
    status, body = call("POST", "/analytics/events", {
        "event_type": "app_open", "user_id": patient_id, "metadata": {},
    })
    check("Log event (anonymized)", status in (200, 201), "status=" + str(status) + " " + str(body))
    if status in (200, 201):
        check("User ID is hashed, not raw", body.get("user_id_hash") != patient_id)

    status, body = call("POST", "/analytics/metrics", {
        "date_from": "2026-01-01T00:00:00Z", "date_to": "2026-12-31T23:59:59Z",
    }, token=patient_token)
    check("Get metrics (authenticated)", status == 200, "status=" + str(status) + " " + str(body))

print("=" * 60)
print("Agent check complete.")
print("=" * 60)
