"""
backend/routes/health_records.py
-----------------------------------
Patient Health Records: a secure, access-controlled home for prescriptions,
lab reports, doctor notes, and other health documents.

Every access to a record by someone other than the owning patient is
logged so the patient can see who viewed their data.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RecordType(str, Enum):
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    DOCTOR_NOTE = "doctor_note"
    APPOINTMENT_SUMMARY = "appointment_summary"
    OTHER = "other"


class RecordCreate(BaseModel):
    record_type: RecordType
    title: str
    description: Optional[str] = None
    storage_ref: str
    related_doctor_id: Optional[str] = None
    related_appointment_id: Optional[str] = None


class HealthRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    record_type: RecordType
    title: str
    description: Optional[str] = None
    storage_ref: str
    related_doctor_id: Optional[str] = None
    related_appointment_id: Optional[str] = None
    uploaded_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecordAccessAction(str, Enum):
    VIEWED = "viewed"
    DOWNLOADED = "downloaded"


class RecordAccessLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str
    accessed_by: str
    action: RecordAccessAction
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthRecordServiceError(Exception):
    pass


class HealthRecordRepository:
    async def create(self, record: HealthRecord) -> HealthRecord:
        raise NotImplementedError

    async def get(self, record_id: str) -> Optional[HealthRecord]:
        raise NotImplementedError

    async def delete(self, record_id: str) -> None:
        raise NotImplementedError

    async def list_for_patient(self, patient_id: str, record_type: Optional[RecordType] = None) -> list[HealthRecord]:
        raise NotImplementedError


class InMemoryHealthRecordRepository(HealthRecordRepository):
    def __init__(self) -> None:
        self._store: dict[str, HealthRecord] = {}

    async def create(self, record: HealthRecord) -> HealthRecord:
        self._store[record.id] = record
        return record

    async def get(self, record_id: str) -> Optional[HealthRecord]:
        return self._store.get(record_id)

    async def delete(self, record_id: str) -> None:
        self._store.pop(record_id, None)

    async def list_for_patient(self, patient_id: str, record_type: Optional[RecordType] = None) -> list[HealthRecord]:
        results = [r for r in self._store.values() if r.patient_id == patient_id]
        if record_type is not None:
            results = [r for r in results if r.record_type == record_type]
        return sorted(results, key=lambda r: r.created_at, reverse=True)


class RecordAccessLogRepository:
    async def add(self, entry: RecordAccessLogEntry) -> RecordAccessLogEntry:
        raise NotImplementedError

    async def list_for_record(self, record_id: str) -> list[RecordAccessLogEntry]:
        raise NotImplementedError

    async def list_for_patient(self, patient_id: str, record_ids: list[str]) -> list[RecordAccessLogEntry]:
        raise NotImplementedError


class InMemoryRecordAccessLogRepository(RecordAccessLogRepository):
    def __init__(self) -> None:
        self._store: list[RecordAccessLogEntry] = []

    async def add(self, entry: RecordAccessLogEntry) -> RecordAccessLogEntry:
        self._store.append(entry)
        return entry

    async def list_for_record(self, record_id: str) -> list[RecordAccessLogEntry]:
        results = [e for e in self._store if e.record_id == record_id]
        return sorted(results, key=lambda e: e.accessed_at)

    async def list_for_patient(self, patient_id: str, record_ids: list[str]) -> list[RecordAccessLogEntry]:
        results = [e for e in self._store if e.record_id in record_ids]
        return sorted(results, key=lambda e: e.accessed_at, reverse=True)


class HealthRecordService:
    def __init__(
        self,
        record_repository: HealthRecordRepository,
        access_log_repository: RecordAccessLogRepository,
    ) -> None:
        self._records = record_repository
        self._log = access_log_repository

    async def add_record(self, uploader_id: str, patient_id: str, req: RecordCreate) -> HealthRecord:
        record = HealthRecord(
            patient_id=patient_id,
            uploaded_by=uploader_id,
            **req.model_dump(),
        )
        return await self._records.create(record)

    async def get_record(self, record_id: str, requester_id: str, is_patient: bool) -> HealthRecord:
        record = await self._records.get(record_id)
        if record is None:
            raise HealthRecordServiceError("Health record not found.")

        if record.patient_id != requester_id:
            await self._log.add(
                RecordAccessLogEntry(record_id=record.id, accessed_by=requester_id, action=RecordAccessAction.VIEWED)
            )

        return record

    async def delete_record(self, record_id: str, patient_id: str) -> None:
        record = await self._records.get(record_id)
        if record is None:
            raise HealthRecordServiceError("Health record not found.")
        if record.patient_id != patient_id:
            raise HealthRecordServiceError("You can only delete your own health records.")
        await self._records.delete(record_id)

    async def list_for_patient(self, patient_id: str, record_type: Optional[RecordType] = None) -> list[HealthRecord]:
        return await self._records.list_for_patient(patient_id, record_type)

    async def get_access_log_for_patient(self, patient_id: str) -> list[RecordAccessLogEntry]:
        patient_records = await self._records.list_for_patient(patient_id)
        record_ids = [r.id for r in patient_records]
        return await self._log.list_for_patient(patient_id, record_ids)


def build_router(get_current_user, require_role):
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/health-records", tags=["health-records"])

    _record_repo = InMemoryHealthRecordRepository()
    _log_repo = InMemoryRecordAccessLogRepository()

    def get_record_service() -> HealthRecordService:
        return HealthRecordService(_record_repo, _log_repo)

    @router.post("/", response_model=HealthRecord, status_code=status.HTTP_201_CREATED)
    async def add_record(
        req: RecordCreate,
        current_user=Depends(get_current_user),
        service: HealthRecordService = Depends(get_record_service),
    ):
        return await service.add_record(current_user.id, current_user.id, req)

    @router.get("/me", response_model=list[HealthRecord])
    async def my_records(
        record_type: Optional[RecordType] = None,
        current_user=Depends(get_current_user),
        service: HealthRecordService = Depends(get_record_service),
    ):
        return await service.list_for_patient(current_user.id, record_type)

    @router.get("/{record_id}", response_model=HealthRecord)
    async def get_record(
        record_id: str,
        current_user=Depends(get_current_user),
        service: HealthRecordService = Depends(get_record_service),
    ):
        try:
            record = await service.get_record(record_id, current_user.id, is_patient=True)
        except HealthRecordServiceError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        if record.patient_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this record.")
        return record

    @router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_record(
        record_id: str,
        current_user=Depends(get_current_user),
        service: HealthRecordService = Depends(get_record_service),
    ):
        try:
            await service.delete_record(record_id, current_user.id)
        except HealthRecordServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.get("/me/access-log", response_model=list[RecordAccessLogEntry])
    async def my_access_log(
        current_user=Depends(get_current_user),
        service: HealthRecordService = Depends(get_record_service),
    ):
        return await service.get_access_log_for_patient(current_user.id)

    return router
