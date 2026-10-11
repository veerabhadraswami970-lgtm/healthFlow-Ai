import json, os, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8000/api/v1"


def call(method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            code, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        code, text = e.code, e.read().decode()
    except Exception as e:
        code, text = 0, str(e)
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}
    return code, parsed, text


bad = 0


def step(label, method, path, body=None, token=None):
    global bad
    code, parsed, text = call(method, path, body, token)
    flag = "OK  " if 0 < code < 500 else "FAIL"
    if flag == "FAIL":
        bad += 1
    print(flag, code, label, "-", text[:110].replace("\n", " "))
    return parsed


stamp = str(time.time())
tokens = {}
ids = {}
for role in ("patient", "doctor", "admin"):
    r = step("register " + role, "POST", "/users/register", {
        "full_name": "Flow " + role, "email": role + "_" + stamp + "@test.com",
        "password": "testpass123", "role": role,
        "admin_key": os.environ.get("ADMIN_SIGNUP_KEY")})
    tokens[role] = r.get("access_token")
    ids[role] = (r.get("user") or {}).get("id")

d = step("doctor creates profile", "POST", "/doctors/me",
         {"specialties": ["Cardiology"], "experience_years": 5}, tokens["doctor"])
doc_id = d.get("id")
step("admin verifies doctor", "PATCH", "/doctors/" + str(doc_id) + "/verify?is_verified=true",
     None, tokens["admin"])
step("patient lists doctors", "GET", "/doctors/", None, tokens["patient"])
a = step("patient books appointment", "POST", "/appointments/", {
    "doctor_id": doc_id, "date": "2027-01-15", "start_time": "10:00:00",
    "end_time": "10:30:00", "reason": "checkup"}, tokens["patient"])
step("doctor confirms appointment", "PATCH", "/appointments/" + str(a.get("id")) + "/confirm",
     None, tokens["doctor"])
step("patient adds emergency contact", "POST", "/emergency/contacts",
     {"name": "Mom", "phone": "9999999999"}, tokens["patient"])
step("patient triggers emergency", "POST", "/emergency/trigger",
     {"latitude": 12.97, "longitude": 77.59, "notes": "test"}, tokens["patient"])
step("patient adds health record", "POST", "/health-records/",
     {"record_type": "lab_report", "title": "CBC", "storage_ref": "gs://b/cbc.pdf"}, tokens["patient"])
step("admin adds blood bank", "POST", "/blood-banks/",
     {"name": "City Bank", "address": "MG Road", "latitude": 12.97, "longitude": 77.59}, tokens["admin"])
step("patient searches blood banks", "GET", "/blood-banks/search", None, tokens["patient"])
print("done;", bad, "server errors")
