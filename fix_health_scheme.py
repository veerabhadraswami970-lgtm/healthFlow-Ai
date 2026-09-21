"""One-time patch: hoists health_scheme.py repository/agent/service
construction out of the per-request get_scheme_service() function into
true module-level singletons, fixing the same bug class already fixed
in analytics_agent.py and health_assistant.py.
Run once from the project root: python fix_health_scheme.py
"""

path = "backend/agents/health_scheme.py"

with open(path, encoding="utf-8") as f:
    lines = f.readlines()

start_marker = "def build_router():\n"

start_matches = [i for i, l in enumerate(lines) if l == start_marker]
end_matches = [i for i, l in enumerate(lines) if l.rstrip("\n") == "    return router"]

if len(start_matches) != 1:
    raise SystemExit("Expected exactly 1 build_router marker, found " + str(len(start_matches)))
if len(end_matches) != 1:
    raise SystemExit("Expected exactly 1 return router marker, found " + str(len(end_matches)))

start_idx = start_matches[0]
end_idx = end_matches[0]

if end_idx < start_idx:
    raise SystemExit("end marker before start marker; unexpected shape")

old_block_text = "".join(lines[start_idx:end_idx + 1])

expected_old_block = (
    'def build_router():\n'
    '    """\n'
    '    Factory instead of a module-level router so dependencies (repository,\n'
    '    LLM client) can be injected per environment ' + chr(8212) + ' same DI approach used\n'
    '    across the rest of the service layer.\n'
    '    """\n'
    '    from fastapi import APIRouter, Depends\n'
    '\n'
    '    router = APIRouter(prefix="/schemes", tags=["schemes"])\n'
    '\n'
    '    def get_scheme_service() -> SchemeService:\n'
    "        # Wire real dependencies here in your app's dependency module,\n"
    '        # e.g. Firestore client + GeminiClient built from settings.\n'
    '        repository = InMemorySchemeRepository()\n'
    '        agent = SchemeIntelligenceAgent(llm_client=None)\n'
    '        return SchemeService(repository=repository, agent=agent)\n'
    '\n'
    '    @router.get("/", response_model=list[Scheme])\n'
    '    async def list_schemes(\n'
    '        state: str | None = None,\n'
    '        service: SchemeService = Depends(get_scheme_service),\n'
    '    ):\n'
    '        return await service.list_schemes(state=state)\n'
    '\n'
    '    @router.post("/check-eligibility", response_model=list[SchemeEligibilityResult])\n'
    '    async def check_eligibility(\n'
    '        req: EligibilityCheckRequest,\n'
    '        explain: bool = True,\n'
    '        service: SchemeService = Depends(get_scheme_service),\n'
    '    ):\n'
    '        return await service.check_eligibility(req, explain=explain)\n'
    '\n'
    '    return router'
)

if old_block_text != expected_old_block:
    print("MISMATCH - actual block below:")
    print(repr(old_block_text))
    raise SystemExit("Actual build_router block did not match. Aborting, no changes made.")

new_block = (
    '_repository = InMemorySchemeRepository()\n'
    '_agent = SchemeIntelligenceAgent(llm_client=None)\n'
    '_scheme_service = SchemeService(repository=_repository, agent=_agent)\n'
    '\n'
    '\n'
    'def build_router():\n'
    '    """\n'
    '    Factory instead of a module-level router so dependencies (repository,\n'
    '    LLM client) can be injected per environment ' + chr(8212) + ' same DI approach used\n'
    '    across the rest of the service layer.\n'
    '    """\n'
    '    from fastapi import APIRouter, Depends\n'
    '\n'
    '    router = APIRouter(prefix="/schemes", tags=["schemes"])\n'
    '\n'
    '    def get_scheme_service() -> SchemeService:\n'
    '        return _scheme_service\n'
    '\n'
    '    @router.get("/", response_model=list[Scheme])\n'
    '    async def list_schemes(\n'
    '        state: str | None = None,\n'
    '        service: SchemeService = Depends(get_scheme_service),\n'
    '    ):\n'
    '        return await service.list_schemes(state=state)\n'
    '\n'
    '    @router.post("/check-eligibility", response_model=list[SchemeEligibilityResult])\n'
    '    async def check_eligibility(\n'
    '        req: EligibilityCheckRequest,\n'
    '        explain: bool = True,\n'
    '        service: SchemeService = Depends(get_scheme_service),\n'
    '    ):\n'
    '        return await service.check_eligibility(req, explain=explain)\n'
    '\n'
    '    return router\n'
)

new_lines = lines[:start_idx] + [new_block] + lines[end_idx + 1:]

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(new_lines)

print("Patched " + path + " successfully.")
