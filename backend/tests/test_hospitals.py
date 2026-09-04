"""Tests for backend/routes/hospitals.py"""
from datetime import datetime, timezone

import pytest

from backend.routes.users import UserPublic, UserRole
from backend.routes.hospitals import (
    HospitalCreate,
    HospitalUpdate,
    HospitalServiceCreate,
    InMemoryHospitalRepository,
    HospitalService,
    HospitalError,
    build_router as build_hospitals_router,
)


class TestHospitalsModule:
    @pytest.fixture
    def service(self) -> HospitalService:
        return HospitalService(InMemoryHospitalRepository())

    @pytest.fixture
    def hospital_user(self) -> UserPublic:
        return UserPublic(id="u-hosp-1", email="hosp@x.com", full_name="City Care",
                           role=UserRole.HOSPITAL, created_at=datetime.now(timezone.utc))

    @pytest.fixture
    def other_hospital_user(self) -> UserPublic:
        return UserPublic(id="u-hosp-2", email="hosp2@x.com", full_name="Other",
                           role=UserRole.HOSPITAL, created_at=datetime.now(timezone.utc))

    @pytest.fixture
    def doctor_user(self) -> UserPublic:
        return UserPublic(id="u-doc-1", email="doc@x.com", full_name="Dr Rao",
                           role=UserRole.DOCTOR, created_at=datetime.now(timezone.utc))

    @pytest.mark.asyncio
    async def test_only_hospital_role_can_create(self, service, doctor_user):
        with pytest.raises(HospitalError):
            await service.create_hospital(doctor_user, HospitalCreate(name="Fake"))

    @pytest.mark.asyncio
    async def test_create_unverified_by_default(self, service, hospital_user):
        h = await service.create_hospital(hospital_user, HospitalCreate(name="City Care", departments=["Cardiology"]))
        assert h.is_verified is False

    @pytest.mark.asyncio
    async def test_duplicate_rejected(self, service, hospital_user):
        await service.create_hospital(hospital_user, HospitalCreate(name="City Care"))
        with pytest.raises(HospitalError):
            await service.create_hospital(hospital_user, HospitalCreate(name="Dup"))

    @pytest.mark.asyncio
    async def test_search_by_department_and_emergency(self, service, hospital_user):
        await service.create_hospital(hospital_user, HospitalCreate(
            name="City Care", departments=["Cardiology"], has_emergency_support=True))
        assert len(await service.search(department="Cardiology")) == 1
        assert len(await service.search(department="Dermatology")) == 0
        assert len(await service.search(emergency_only=True)) == 1

    @pytest.mark.asyncio
    async def test_only_owner_can_update(self, service, hospital_user, other_hospital_user):
        h = await service.create_hospital(hospital_user, HospitalCreate(name="City Care"))
        with pytest.raises(HospitalError):
            await service.update_own(other_hospital_user, h.id, HospitalUpdate(has_blood_bank=True))
        updated = await service.update_own(hospital_user, h.id, HospitalUpdate(has_blood_bank=True))
        assert updated.has_blood_bank is True

    @pytest.mark.asyncio
    async def test_verify_flips_flag(self, service, hospital_user):
        h = await service.create_hospital(hospital_user, HospitalCreate(name="City Care"))
        assert h.is_verified is False
        verified = await service.set_verified(h.id, True)
        assert verified.is_verified is True

    @pytest.mark.asyncio
    async def test_services_only_by_owner(self, service, hospital_user, other_hospital_user):
        h = await service.create_hospital(hospital_user, HospitalCreate(name="City Care"))
        with pytest.raises(HospitalError):
            await service.add_service(other_hospital_user, h.id, HospitalServiceCreate(name="MRI"))
        added = await service.add_service(hospital_user, h.id, HospitalServiceCreate(name="MRI"))
        assert added.hospital_id == h.id
        assert len(await service.list_services(h.id)) == 1

    @pytest.mark.asyncio
    async def test_resolve_owner(self, service, hospital_user):
        h = await service.create_hospital(hospital_user, HospitalCreate(name="City Care"))
        owner = await service.resolve_owner(h.id)
        assert owner == hospital_user.id
        assert await service.resolve_owner("nonexistent") is None

    def test_router_builds_with_expected_routes(self):
        from backend.routes.users import build_router as build_users_router
        users_router, get_current_user, require_role = build_users_router()
        router = build_hospitals_router(get_current_user, require_role)
        paths = {r.path for r in router.routes}
        assert "/hospitals/me" in paths
        assert "/hospitals/{hospital_id}/verify" in paths
        assert "/hospitals/{hospital_id}/services" in paths
