"""
backend/tests/test_emergency.py
----------------------------------
Tests for the Emergency System module (backend/routes/emergency.py).
No MongoDB connection or API keys needed — uses in-memory repositories.
"""

import pytest

from backend.routes.emergency import (
    EmergencyTrigger,
    EmergencyContact,
    EmergencyStatus,
    InMemoryEmergencyEventRepository,
    InMemoryEmergencyContactRepository,
    EmergencyService,
    EmergencyServiceError,
    NearestHospital,
    NearestHospitalFinder,
    build_router,
)


class FakeHospitalFinder(NearestHospitalFinder):
    async def find_nearest(self, latitude, longitude, limit=3):
        return [
            NearestHospital(hospital_id="h1", name="City Care Hospital", distance_km=1.2),
            NearestHospital(hospital_id="h2", name="General Hospital", distance_km=3.4),
        ]


class TestEmergencyModule:
    @pytest.fixture
    def service(self) -> EmergencyService:
        return EmergencyService(
            InMemoryEmergencyEventRepository(),
            InMemoryEmergencyContactRepository(),
            FakeHospitalFinder(),
        )

    @pytest.mark.asyncio
    async def test_trigger_creates_event_in_triggered_state(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger(latitude=13.08, longitude=80.27))
        assert event.status == EmergencyStatus.TRIGGERED
        assert event.patient_id == "u-pat-1"
        assert event.ambulance_dispatch_confirmed is False

    @pytest.mark.asyncio
    async def test_confirm_locates_hospitals_and_notifies_contacts(self, service):
        await service.add_contact("u-pat-1", EmergencyContact(name="Mom", phone="9999999999"))
        event = await service.trigger("u-pat-1", EmergencyTrigger(latitude=13.08, longitude=80.27))

        confirmed = await service.confirm(event.id, "u-pat-1")
        assert confirmed.status == EmergencyStatus.ACTIVE
        assert len(confirmed.nearest_hospitals) == 2
        assert confirmed.contacts_notified == ["Mom"]

    @pytest.mark.asyncio
    async def test_confirm_never_fabricates_ambulance_dispatch(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        confirmed = await service.confirm(event.id, "u-pat-1")
        assert confirmed.ambulance_dispatch_confirmed is False

    @pytest.mark.asyncio
    async def test_other_patient_cannot_confirm_someone_elses_event(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        with pytest.raises(EmergencyServiceError):
            await service.confirm(event.id, "u-pat-2")

    @pytest.mark.asyncio
    async def test_cannot_confirm_already_active_event(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        await service.confirm(event.id, "u-pat-1")
        with pytest.raises(EmergencyServiceError):
            await service.confirm(event.id, "u-pat-1")

    @pytest.mark.asyncio
    async def test_resolve_marks_event_resolved(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        await service.confirm(event.id, "u-pat-1")
        resolved = await service.resolve(event.id, "u-pat-1", resolution_notes="Reached hospital safely")
        assert resolved.status == EmergencyStatus.RESOLVED
        assert resolved.resolution_notes == "Reached hospital safely"

    @pytest.mark.asyncio
    async def test_cancel_marks_event_cancelled(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        cancelled = await service.cancel(event.id, "u-pat-1")
        assert cancelled.status == EmergencyStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cannot_resolve_already_resolved_event(self, service):
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        await service.resolve(event.id, "u-pat-1")
        with pytest.raises(EmergencyServiceError):
            await service.resolve(event.id, "u-pat-1")

    @pytest.mark.asyncio
    async def test_list_for_patient_returns_only_their_events(self, service):
        await service.trigger("u-pat-1", EmergencyTrigger())
        await service.trigger("u-pat-1", EmergencyTrigger())
        await service.trigger("u-pat-2", EmergencyTrigger())

        events = await service.list_for_patient("u-pat-1")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_add_and_list_emergency_contacts(self, service):
        await service.add_contact("u-pat-1", EmergencyContact(name="Dad", phone="8888888888", relationship="Father"))
        contacts = await service.list_contacts("u-pat-1")
        assert len(contacts) == 1
        assert contacts[0].name == "Dad"

    @pytest.mark.asyncio
    async def test_null_hospital_finder_returns_empty_not_fake_data(self):
        service = EmergencyService(InMemoryEmergencyEventRepository(), InMemoryEmergencyContactRepository())
        event = await service.trigger("u-pat-1", EmergencyTrigger())
        confirmed = await service.confirm(event.id, "u-pat-1")
        assert confirmed.nearest_hospitals == []

    def test_router_builds_with_expected_routes(self):
        async def dummy_get_current_user():
            pass

        def dummy_require_role(*roles):
            async def _dep():
                pass
            return _dep

        router = build_router(dummy_get_current_user, dummy_require_role)
        paths = {r.path for r in router.routes}
        assert "/emergency/trigger" in paths
        assert "/emergency/{event_id}/confirm" in paths
        assert "/emergency/{event_id}/resolve" in paths
        assert "/emergency/{event_id}/cancel" in paths
        assert "/emergency/contacts" in paths
