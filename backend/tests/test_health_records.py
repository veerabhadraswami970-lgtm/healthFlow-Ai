"""
backend/tests/test_health_records.py
----------------------------------------
Tests for the Patient Health Records module (backend/routes/health_records.py).
No MongoDB connection or API keys needed — uses in-memory repositories.
"""

import pytest

from backend.routes.health_records import (
    RecordCreate,
    RecordType,
    RecordAccessAction,
    InMemoryHealthRecordRepository,
    InMemoryRecordAccessLogRepository,
    HealthRecordService,
    HealthRecordServiceError,
    build_router,
)


class TestHealthRecordsModule:
    @pytest.fixture
    def service(self) -> HealthRecordService:
        return HealthRecordService(InMemoryHealthRecordRepository(), InMemoryRecordAccessLogRepository())

    @pytest.mark.asyncio
    async def test_add_record_creates_it_for_the_patient(self, service):
        record = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.LAB_REPORT, title="Blood Test", storage_ref="storage/1.pdf"),
        )
        assert record.patient_id == "u-pat-1"
        assert record.record_type == RecordType.LAB_REPORT

    @pytest.mark.asyncio
    async def test_owner_viewing_their_own_record_is_not_logged(self, service):
        record = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.PRESCRIPTION, title="Rx", storage_ref="storage/2.pdf"),
        )
        await service.get_record(record.id, "u-pat-1", is_patient=True)
        log = await service.get_access_log_for_patient("u-pat-1")
        assert log == []

    @pytest.mark.asyncio
    async def test_someone_else_viewing_record_is_logged(self, service):
        record = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.PRESCRIPTION, title="Rx", storage_ref="storage/3.pdf"),
        )
        await service.get_record(record.id, "u-doc-1", is_patient=False)
        log = await service.get_access_log_for_patient("u-pat-1")
        assert len(log) == 1
        assert log[0].accessed_by == "u-doc-1"
        assert log[0].action == RecordAccessAction.VIEWED

    @pytest.mark.asyncio
    async def test_get_unknown_record_raises(self, service):
        with pytest.raises(HealthRecordServiceError):
            await service.get_record("does-not-exist", "u-pat-1", is_patient=True)

    @pytest.mark.asyncio
    async def test_owner_can_delete_own_record(self, service):
        record = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.DOCTOR_NOTE, title="Note", storage_ref="storage/4.pdf"),
        )
        await service.delete_record(record.id, "u-pat-1")
        records = await service.list_for_patient("u-pat-1")
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_non_owner_cannot_delete_record(self, service):
        record = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.DOCTOR_NOTE, title="Note", storage_ref="storage/5.pdf"),
        )
        with pytest.raises(HealthRecordServiceError):
            await service.delete_record(record.id, "u-pat-2")

    @pytest.mark.asyncio
    async def test_list_for_patient_filters_by_record_type(self, service):
        await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.LAB_REPORT, title="Blood Test", storage_ref="storage/6.pdf"),
        )
        await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.PRESCRIPTION, title="Rx", storage_ref="storage/7.pdf"),
        )
        lab_reports = await service.list_for_patient("u-pat-1", record_type=RecordType.LAB_REPORT)
        assert len(lab_reports) == 1
        assert lab_reports[0].title == "Blood Test"

    @pytest.mark.asyncio
    async def test_list_for_patient_returns_only_their_records(self, service):
        await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.LAB_REPORT, title="Report A", storage_ref="storage/8.pdf"),
        )
        await service.add_record(
            "u-pat-2", "u-pat-2",
            RecordCreate(record_type=RecordType.LAB_REPORT, title="Report B", storage_ref="storage/9.pdf"),
        )
        records = await service.list_for_patient("u-pat-1")
        assert len(records) == 1
        assert records[0].title == "Report A"

    @pytest.mark.asyncio
    async def test_access_log_accumulates_across_multiple_records(self, service):
        r1 = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.LAB_REPORT, title="Report A", storage_ref="storage/10.pdf"),
        )
        r2 = await service.add_record(
            "u-pat-1", "u-pat-1",
            RecordCreate(record_type=RecordType.PRESCRIPTION, title="Rx B", storage_ref="storage/11.pdf"),
        )
        await service.get_record(r1.id, "u-doc-1", is_patient=False)
        await service.get_record(r2.id, "u-doc-2", is_patient=False)

        log = await service.get_access_log_for_patient("u-pat-1")
        assert len(log) == 2

    def test_router_builds_with_expected_routes(self):
        async def dummy_get_current_user():
            pass

        def dummy_require_role(*roles):
            async def _dep():
                pass
            return _dep

        router = build_router(dummy_get_current_user, dummy_require_role)
        paths = {r.path for r in router.routes}
        assert "/health-records/" in paths
        assert "/health-records/me" in paths
        assert "/health-records/{record_id}" in paths
        assert "/health-records/me/access-log" in paths
