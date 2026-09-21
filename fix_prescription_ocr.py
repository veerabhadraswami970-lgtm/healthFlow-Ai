"""One-time patch: hoists prescription_ocr.py repository/ocr_provider/agent/
service construction out of the per-request get_prescription_service()
function into true module-level singletons, fixing the same bug class
already fixed in analytics_agent.py, health_assistant.py, health_scheme.py,
and medicine_intelligence.py. This is the highest-priority fix: it has
confirmed live impact (created prescriptions were unretrievable, 404).
Run once from the project root: python fix_prescription_ocr.py
"""

path = "backend/agents/prescription_ocr.py"

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
    '    router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])\n'
    '\n'
    '    def get_prescription_service() -> PrescriptionService:\n'
    "        # Wire real dependencies in your app's dependency module, e.g.\n"
    '        # MongoPrescriptionRepository(app.state.mongo_db) + a real OCR\n'
    '        # provider + GeminiClient built from settings.\n'
    '        repository = InMemoryPrescriptionRepository()\n'
    '        ocr_provider = MockOCRProvider()\n'
    '        agent = PrescriptionExtractionAgent(llm_client=None)\n'
    '        return PrescriptionService(repository, ocr_provider, agent)\n'
    '\n'
    '    @router.post("/scan", response_model=PrescriptionRecord)\n'
    '    async def scan_prescription(\n'
    '        req: ScanRequest,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        return await service.scan(req)\n'
    '\n'
    '    @router.post("/{record_id}/verify", response_model=PrescriptionRecord)\n'
    '    async def verify_prescription(\n'
    '        record_id: str,\n'
    '        req: VerifyRequest,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        try:\n'
    '            return await service.verify(record_id, req)\n'
    '        except ValueError as exc:\n'
    '            raise HTTPException(status_code=404, detail=str(exc))\n'
    '\n'
    '    @router.get("/{record_id}", response_model=PrescriptionRecord)\n'
    '    async def get_prescription(\n'
    '        record_id: str,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        record = await service.get(record_id)\n'
    '        if record is None:\n'
    '            raise HTTPException(status_code=404, detail="Not found")\n'
    '        return record\n'
    '\n'
    '    @router.get("/patient/{patient_id}", response_model=list[PrescriptionRecord])\n'
    '    async def list_patient_prescriptions(\n'
    '        patient_id: str,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        return await service.list_for_patient(patient_id)\n'
    '\n'
    '    return router'
)

if old_block_text != expected_old_block:
    print("MISMATCH - actual block below:")
    print(repr(old_block_text))
    raise SystemExit("Actual build_router block did not match. Aborting, no changes made.")

new_block = (
    '_repository = InMemoryPrescriptionRepository()\n'
    '_ocr_provider = MockOCRProvider()\n'
    '_agent = PrescriptionExtractionAgent(llm_client=None)\n'
    '_prescription_service = PrescriptionService(_repository, _ocr_provider, _agent)\n'
    '\n'
    '\n'
    'def build_router():\n'
    '    from fastapi import APIRouter, Depends, HTTPException\n'
    '\n'
    '    router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])\n'
    '\n'
    '    def get_prescription_service() -> PrescriptionService:\n'
    '        return _prescription_service\n'
    '\n'
    '    @router.post("/scan", response_model=PrescriptionRecord)\n'
    '    async def scan_prescription(\n'
    '        req: ScanRequest,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        return await service.scan(req)\n'
    '\n'
    '    @router.post("/{record_id}/verify", response_model=PrescriptionRecord)\n'
    '    async def verify_prescription(\n'
    '        record_id: str,\n'
    '        req: VerifyRequest,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        try:\n'
    '            return await service.verify(record_id, req)\n'
    '        except ValueError as exc:\n'
    '            raise HTTPException(status_code=404, detail=str(exc))\n'
    '\n'
    '    @router.get("/{record_id}", response_model=PrescriptionRecord)\n'
    '    async def get_prescription(\n'
    '        record_id: str,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        record = await service.get(record_id)\n'
    '        if record is None:\n'
    '            raise HTTPException(status_code=404, detail="Not found")\n'
    '        return record\n'
    '\n'
    '    @router.get("/patient/{patient_id}", response_model=list[PrescriptionRecord])\n'
    '    async def list_patient_prescriptions(\n'
    '        patient_id: str,\n'
    '        service: PrescriptionService = Depends(get_prescription_service),\n'
    '    ):\n'
    '        return await service.list_for_patient(patient_id)\n'
    '\n'
    '    return router\n'
)

new_lines = lines[:start_idx] + [new_block] + lines[end_idx + 1:]

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(new_lines)

print("Patched " + path + " successfully.")
