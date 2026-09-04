"""
backend/routes/qr_sharing.py
-------------------------------
QR Prescription Sharing: doctors generate an expiring, revocable token
that authorizes access to a prescription — the QR code itself never
carries raw health data, only an opaque token.

Flow: prescription -> secure token -> authorization check -> record access
Every access attempt (successful or not) is written to an audit trail.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AccessAction(str, Enum):
    GRANTED = "granted"
    DENIED_EXPIRED = "denied_expired"
    DENIED_REVOKED = "denied_revoked"
    DENIED_NOT_FOUND = "denied_not_found"


class QRTokenCreate(BaseModel):
    prescription_id: str
    patient_id: str
    expires_in_hours: int = Field(default=24, ge=1, le=720)


class QRToken(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    prescription_id: str
    patient_id: str
    issuing_doctor_id: str
    revoked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime


class QRAccessResult(BaseModel):
    prescription_id: str
    patient_id: str
    issuing_doctor_id: str


class AccessLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    token_id: str
    action: AccessAction
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accessor_note: Optional[str] = None


class QRSharingServiceError(Exception):
    pass


class QRAccessDeniedError(QRSharingServiceError):
    def __init__(self, action: AccessAction, message: str):
        self.action = action
        super().__init__(message)


class QRTokenRepository:
    async def create(self, token: QRToken) -> QRToken:
        raise NotImplementedError

    async def get_by_id(self, token_id: str) -> Optional[QRToken]:
        raise NotImplementedError

    async def get_by_token(self, token_value: str) -> Optional[QRToken]:
        raise NotImplementedError

    async def update(self, token: QRToken) -> QRToken:
        raise NotImplementedError

    async def list_for_doctor(self, doctor_id: str) -> list[QRToken]:
        raise NotImplementedError


class InMemoryQRTokenRepository(QRTokenRepository):
    def __init__(self) -> None:
        self._store: dict[str, QRToken] = {}

    async def create(self, token: QRToken) -> QRToken:
        self._store[token.id] = token
        return token

    async def get_by_id(self, token_id: str) -> Optional[QRToken]:
        return self._store.get(token_id)

    async def get_by_token(self, token_value: str) -> Optional[QRToken]:
        for t in self._store.values():
            if t.token == token_value:
                return t
        return None

    async def update(self, token: QRToken) -> QRToken:
        self._store[token.id] = token
        return token

    async def list_for_doctor(self, doctor_id: str) -> list[QRToken]:
        results = [t for t in self._store.values() if t.issuing_doctor_id == doctor_id]
        return sorted(results, key=lambda t: t.created_at, reverse=True)


class AccessLogRepository:
    async def add(self, entry: AccessLogEntry) -> AccessLogEntry:
        raise NotImplementedError

    async def list_for_token(self, token_id: str) -> list[AccessLogEntry]:
        raise NotImplementedError


class InMemoryAccessLogRepository(AccessLogRepository):
    def __init__(self) -> None:
        self._store: list[AccessLogEntry] = []

    async def add(self, entry: AccessLogEntry) -> AccessLogEntry:
        self._store.append(entry)
        return entry

    async def list_for_token(self, token_id: str) -> list[AccessLogEntry]:
        results = [e for e in self._store if e.token_id == token_id]
        return sorted(results, key=lambda e: e.accessed_at)


class QRSharingService:
    def __init__(
        self,
        token_repository: QRTokenRepository,
        access_log_repository: AccessLogRepository,
    ) -> None:
        self._tokens = token_repository
        self._log = access_log_repository

    async def generate_token(self, doctor_id: str, req: QRTokenCreate) -> QRToken:
        token = QRToken(
            prescription_id=req.prescription_id,
            patient_id=req.patient_id,
            issuing_doctor_id=doctor_id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=req.expires_in_hours),
        )
        return await self._tokens.create(token)

    async def access(self, token_value: str, accessor_note: Optional[str] = None) -> QRAccessResult:
        token = await self._tokens.get_by_token(token_value)

        if token is None:
            await self._log.add(AccessLogEntry(token_id="unknown", action=AccessAction.DENIED_NOT_FOUND, accessor_note=accessor_note))
            raise QRAccessDeniedError(AccessAction.DENIED_NOT_FOUND, "This QR code is invalid.")

        if token.revoked:
            await self._log.add(AccessLogEntry(token_id=token.id, action=AccessAction.DENIED_REVOKED, accessor_note=accessor_note))
            raise QRAccessDeniedError(AccessAction.DENIED_REVOKED, "This QR code has been revoked by the doctor.")

        if datetime.now(timezone.utc) > token.expires_at:
            await self._log.add(AccessLogEntry(token_id=token.id, action=AccessAction.DENIED_EXPIRED, accessor_note=accessor_note))
            raise QRAccessDeniedError(AccessAction.DENIED_EXPIRED, "This QR code has expired.")

        await self._log.add(AccessLogEntry(token_id=token.id, action=AccessAction.GRANTED, accessor_note=accessor_note))
        return QRAccessResult(
            prescription_id=token.prescription_id,
            patient_id=token.patient_id,
            issuing_doctor_id=token.issuing_doctor_id,
        )

    async def revoke(self, token_id: str, doctor_id: str) -> QRToken:
        token = await self._tokens.get_by_id(token_id)
        if token is None:
            raise QRSharingServiceError("Token not found.")
        if token.issuing_doctor_id != doctor_id:
            raise QRSharingServiceError("You can only revoke QR tokens you issued.")

        token.revoked = True
        return await self._tokens.update(token)

    async def get_audit_log(self, token_id: str, doctor_id: str) -> list[AccessLogEntry]:
        token = await self._tokens.get_by_id(token_id)
        if token is None:
            raise QRSharingServiceError("Token not found.")
        if token.issuing_doctor_id != doctor_id:
            raise QRSharingServiceError("You can only view the audit log for QR tokens you issued.")

        return await self._log.list_for_token(token_id)

    async def list_for_doctor(self, doctor_id: str) -> list[QRToken]:
        return await self._tokens.list_for_doctor(doctor_id)


def build_router(get_current_user, require_role):
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/qr-sharing", tags=["qr-sharing"])

    _token_repo = InMemoryQRTokenRepository()
    _log_repo = InMemoryAccessLogRepository()

    def get_qr_service() -> QRSharingService:
        return QRSharingService(_token_repo, _log_repo)

    @router.post("/tokens", response_model=QRToken, status_code=status.HTTP_201_CREATED)
    async def generate_token(
        req: QRTokenCreate,
        current_user=Depends(get_current_user),
        service: QRSharingService = Depends(get_qr_service),
    ):
        return await service.generate_token(current_user.id, req)

    @router.get("/tokens/me", response_model=list[QRToken])
    async def my_tokens(
        current_user=Depends(get_current_user),
        service: QRSharingService = Depends(get_qr_service),
    ):
        return await service.list_for_doctor(current_user.id)

    @router.post("/access/{token_value}", response_model=QRAccessResult)
    async def access_via_token(
        token_value: str,
        accessor_note: Optional[str] = None,
        service: QRSharingService = Depends(get_qr_service),
    ):
        try:
            return await service.access(token_value, accessor_note)
        except QRAccessDeniedError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.patch("/tokens/{token_id}/revoke", response_model=QRToken)
    async def revoke_token(
        token_id: str,
        current_user=Depends(get_current_user),
        service: QRSharingService = Depends(get_qr_service),
    ):
        try:
            return await service.revoke(token_id, current_user.id)
        except QRSharingServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.get("/tokens/{token_id}/audit-log", response_model=list[AccessLogEntry])
    async def get_audit_log(
        token_id: str,
        current_user=Depends(get_current_user),
        service: QRSharingService = Depends(get_qr_service),
    ):
        try:
            return await service.get_audit_log(token_id, current_user.id)
        except QRSharingServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return router
