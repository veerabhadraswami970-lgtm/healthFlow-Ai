import urllib.request, json
data = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/openapi.json").read())
schemas = data["components"]["schemas"]
for name in sorted(schemas.keys()):
    props = schemas[name].get("properties", {})
    required = schemas[name].get("required", [])
    print(f"--- {name} ---")
    for field, info in props.items():
        if "type" in info:
            ftype = info["type"]
        elif "$ref" in info:
            ftype = info["$ref"].split("/")[-1]
        elif "allOf" in info:
            ftype = info["allOf"][0].get("$ref", "?").split("/")[-1]
        else:
            ftype = "?"
        req = " (required)" if field in required else ""
        print(f"  {field}: {ftype}{req}")
