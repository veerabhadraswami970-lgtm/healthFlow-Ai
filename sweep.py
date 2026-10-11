import json, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8000"


def call(method, path, body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


status, text = call("GET", "/openapi.json")
if status != 200:
    raise SystemExit("Server not reachable: " + str(status) + " " + text)
spec = json.loads(text)
email = "sweep_" + str(time.time()) + "@test.com"
status, text = call("POST", "/api/v1/users/register", {
    "full_name": "Sweep Patient", "email": email,
    "password": "testpass123", "role": "patient"})
token = json.loads(text).get("access_token") if status == 201 else None
print("register patient:", status)

bad = 0
total = 0
for path, ops in sorted(spec["paths"].items()):
    if "get" not in ops or "{" in path or path.startswith("/app"):
        continue
    params = [p for p in ops["get"].get("parameters", []) if p.get("required")]
    if params:
        continue
    code, body = call("GET", path, token=token)
    total += 1
    flag = "OK  " if 0 < code < 500 else "FAIL"
    if flag == "FAIL":
        bad += 1
    print(flag, code, path)
print("checked", total, "GET routes;", bad, "server errors")
