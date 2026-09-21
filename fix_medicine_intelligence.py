"""One-time patch: hoists medicine_intelligence.py repository/agent/service
construction out of the per-request get_medicine_service() function into
true module-level singletons, fixing the same bug class already fixed
in analytics_agent.py, health_assistant.py, and health_scheme.py.
Run once from the project root: python fix_medicine_intelligence.py
"""

path = "backend/agents/medicine_intelligence.py"

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
    '    from fastapi import APIRouter, Depends, HTTPException\n'
    '\n'
    '    router = APIRouter(prefix="/medicines", tags=["medicines"])\n'
    '\n'
    '    def get_medicine_service() -> MedicineService:\n'
    "        # Wire real dependencies in your app's dependency module, e.g.\n"
    '        # MongoMedicineRepository(app.state.mongo_db) + GeminiClient\n'
    '        # built from settings.\n'
    '        repository = InMemoryMedicineRepository()\n'
    '        agent = MedicineExplanationAgent(llm_client=None)\n'
    '        return MedicineService(repository, agent)\n'
    '\n'
    '    @router.get("/search", response_model=list[MedicineReference])\n'
    '    async def search_medicines(\n'
    '        q: str,\n'
    '        limit: int = 10,\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        return await service.search(q, limit=limit)\n'
    '\n'
    '    @router.get("/{medicine_id}", response_model=MedicineReference)\n'
    '    async def get_medicine(\n'
    '        medicine_id: str,\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        medicine = await service.get(medicine_id)\n'
    '        if medicine is None:\n'
    '            raise HTTPException(status_code=404, detail="Not found")\n'
    '        return medicine\n'
    '\n'
    '    @router.post("/explain", response_model=MedicineExplanationResult)\n'
    '    async def explain_medicine(\n'
    '        req: MedicineExplanationRequest,\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        try:\n'
    '            return await service.explain(req)\n'
    '        except ValueError as exc:\n'
    '            raise HTTPException(status_code=404, detail=str(exc))\n'
    '\n'
    '    @router.post("/check-interactions", response_model=list[InteractionResult])\n'
    '    async def check_interactions(\n'
    '        medicine_ids: list[str],\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        return await service.check_interactions(medicine_ids)\n'
    '\n'
    '    return router'
)

if old_block_text != expected_old_block:
    print("MISMATCH - actual block below:")
    print(repr(old_block_text))
    raise SystemExit("Actual build_router block did not match. Aborting, no changes made.")

new_block = (
    '_repository = InMemoryMedicineRepository()\n'
    '_agent = MedicineExplanationAgent(llm_client=None)\n'
    '_medicine_service = MedicineService(_repository, _agent)\n'
    '\n'
    '\n'
    'def build_router():\n'
    '    from fastapi import APIRouter, Depends, HTTPException\n'
    '\n'
    '    router = APIRouter(prefix="/medicines", tags=["medicines"])\n'
    '\n'
    '    def get_medicine_service() -> MedicineService:\n'
    '        return _medicine_service\n'
    '\n'
    '    @router.get("/search", response_model=list[MedicineReference])\n'
    '    async def search_medicines(\n'
    '        q: str,\n'
    '        limit: int = 10,\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        return await service.search(q, limit=limit)\n'
    '\n'
    '    @router.get("/{medicine_id}", response_model=MedicineReference)\n'
    '    async def get_medicine(\n'
    '        medicine_id: str,\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        medicine = await service.get(medicine_id)\n'
    '        if medicine is None:\n'
    '            raise HTTPException(status_code=404, detail="Not found")\n'
    '        return medicine\n'
    '\n'
    '    @router.post("/explain", response_model=MedicineExplanationResult)\n'
    '    async def explain_medicine(\n'
    '        req: MedicineExplanationRequest,\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        try:\n'
    '            return await service.explain(req)\n'
    '        except ValueError as exc:\n'
    '            raise HTTPException(status_code=404, detail=str(exc))\n'
    '\n'
    '    @router.post("/check-interactions", response_model=list[InteractionResult])\n'
    '    async def check_interactions(\n'
    '        medicine_ids: list[str],\n'
    '        service: MedicineService = Depends(get_medicine_service),\n'
    '    ):\n'
    '        return await service.check_interactions(medicine_ids)\n'
    '\n'
    '    return router\n'
)

new_lines = lines[:start_idx] + [new_block] + lines[end_idx + 1:]

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(new_lines)

print("Patched " + path + " successfully.")
