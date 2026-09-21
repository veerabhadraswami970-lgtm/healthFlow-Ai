"""One-time patch script: updates test_agents.py's analytics router test
to call build_analytics_router(get_current_user, require_role) instead of
build_analytics_router() with zero args, matching the new required signature.
Run once from the project root: python fix_test.py
"""

path = "backend/tests/test_agents.py"

with open(path, encoding="utf-8") as f:
    lines = f.readlines()

old_line = "        router = build_analytics_router()\n"
new_lines = [
    "        async def dummy_get_current_user():\n",
    "            pass\n",
    "\n",
    "        def dummy_require_role(*roles):\n",
    "            async def _dep():\n",
    "                pass\n",
    "            return _dep\n",
    "\n",
    "        router = build_analytics_router(dummy_get_current_user, dummy_require_role)\n",
]

matches = [i for i, l in enumerate(lines) if l == old_line]
if len(matches) != 1:
    raise SystemExit(f"Expected exactly 1 match for old_line, found {len(matches)}. Aborting, no changes made.")

idx = matches[0]
lines[idx:idx + 1] = new_lines

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(lines)

print(f"Patched {path} successfully at former line {idx + 1}.")
