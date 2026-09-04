"""
backend/tests/test_qr_sharing.py
------------------------------------
Tests for the QR Prescription Sharing module (backend/routes/qr_sharing.py).
No MongoDB connection or API keys needed — uses in-memory repositories.
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.routes.qr_sharing import (
    QRTokenCreate,
    AccessAction,
    InMemoryQRTokenRepository,
    InMemoryAccessLogRepository,
    QRSharingService,
    QRSharingServiceError,
    QRAccessDeniedError,
    build_router,
)


class TestQRSharingModule:
    @pytest.fixture
    def service(self) -> QRSharingService:
        return QRSharingService(InMemoryQRTokenRepository(), InMemoryAccessLogRepository())

    @pytest.mark.asyncio
    async def test_generate_token_creates_expiring_token(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=24),
        )
        assert token.issuing_doctor_id == "u-doc-1"
        assert token.revoked is False
        assert token.expires_at > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_access_with_valid_token_succeeds_and_never_leaks_content(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=24),
        )
        result = await service.access(token.token)
        assert result.prescription_id == "rx-1"
        assert not hasattr(result, "medicines")

    @pytest.mark.asyncio
    async def test_access_with_unknown_token_denied(self, service):
        with pytest.raises(QRAccessDeniedError) as exc_info:
            await service.access("not-a-real-token")
        assert exc_info.value.action == AccessAction.DENIED_NOT_FOUND

    @pytest.mark.asyncio
    async def test_access_with_expired_token_denied(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=1),
        )
        token.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        await service._tokens.update(token)

        with pytest.raises(QRAccessDeniedError) as exc_info:
            await service.access(token.token)
        assert exc_info.value.action == AccessAction.DENIED_EXPIRED

    @pytest.mark.asyncio
    async def test_revoked_token_denies_access(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=24),
        )
        await service.revoke(token.id, "u-doc-1")

        with pytest.raises(QRAccessDeniedError) as exc_info:
            await service.access(token.token)
        assert exc_info.value.action == AccessAction.DENIED_REVOKED

    @pytest.mark.asyncio
    async def test_only_issuing_doctor_can_revoke(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=24),
        )
        with pytest.raises(QRSharingServiceError):
            await service.revoke(token.id, "u-doc-2")

    @pytest.mark.asyncio
    async def test_audit_log_records_every_access_attempt(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=24),
        )
        await service.access(token.token, accessor_note="Front desk")
        await service.revoke(token.id, "u-doc-1")
        try:
            await service.access(token.token)
        except QRAccessDeniedError:
            pass

        log = await service.get_audit_log(token.id, "u-doc-1")
        assert len(log) == 2
        assert log[0].action == AccessAction.GRANTED
        assert log[1].action == AccessAction.DENIED_REVOKED

    @pytest.mark.asyncio
    async def test_only_issuing_doctor_can_view_audit_log(self, service):
        token = await service.generate_token(
            "u-doc-1",
            QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1", expires_in_hours=24),
        )
        with pytest.raises(QRSharingServiceError):
            await service.get_audit_log(token.id, "u-doc-2")

    @pytest.mark.asyncio
    async def test_list_for_doctor_returns_only_their_tokens(self, service):
        await service.generate_token("u-doc-1", QRTokenCreate(prescription_id="rx-1", patient_id="u-pat-1"))
        await service.generate_token("u-doc-1", QRTokenCreate(prescription_id="rx-2", patient_id="u-pat-2"))
        await service.generate_token("u-doc-2", QRTokenCreate(prescription_id="rx-3", patient_id="u-pat-3"))

        tokens = await service.list_for_doctor("u-doc-1")
        assert len(tokens) == 2

    def test_router_builds_with_expected_routes(self):
        async def dummy_get_current_user():
            pass

        def dummy_require_role(*roles):
            async def _dep():
                pass
            return _dep

        router = build_router(dummy_get_current_user, dummy_require_role)
        paths = {r.path for r in router.routes}
        assert "/qr-sharing/tokens" in paths
        assert "/qr-sharing/access/{token_value}" in paths
        assert "/qr-sharing/tokens/{token_id}/revoke" in paths
        assert "/qr-sharing/tokens/{token_id}/audit-log" in paths
