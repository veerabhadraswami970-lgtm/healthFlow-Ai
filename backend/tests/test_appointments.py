"""
backend/tests/test_appointments.py
------------------------------------
Tests for the Appointments module (backend/routes/appointments.py).
No MongoDB connection or API keys needed — uses in-memory repositories.

Run it with:
    pytest backend/tests/test_appointments.py -v
"""

import pytest

from backend.routes.appointments import (
    AppointmentCreate,
    AppointmentReschedule,
    AppointmentStatus,
    ConsultationType,
    InMemoryAppointmentRepository,
    AppointmentService,
    AppointmentServiceError,
    build_router,
)


class TestAppointmentsModule:
    @pytest.fixture
    def service(self) -> AppointmentService:
        async def resolve_doctor_owner(doctor_id: str):
            # In these tests, doctor_id "doc-profile-1" is owned by user "u-doc-1"
            mapping = {"doc-profile-1": "u-doc-1", "doc-profile-2": "u-doc-2"}
            return mapping.get(doctor_id)

        return AppointmentService(InMemoryAppointmentRepository(), resolve_doctor_owner)

    @pytest.mark.asyncio
    async def test_patient_can_book_appointment(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        assert appt.status == AppointmentStatus.PENDING
        assert appt.patient_id == "u-pat-1"

    @pytest.mark.asyncio
    async def test_double_booking_same_slot_rejected(self, service):
        await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        with pytest.raises(AppointmentServiceError):
            await service.book(
                "u-pat-2",
                AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:15", end_time="09:45"),
            )

    @pytest.mark.asyncio
    async def test_non_overlapping_slots_both_allowed(self, service):
        await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        appt2 = await service.book(
            "u-pat-2",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:30", end_time="10:00"),
        )
        assert appt2.status == AppointmentStatus.PENDING

    @pytest.mark.asyncio
    async def test_doctor_can_confirm_own_appointment(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        confirmed = await service.confirm(appt.id, "u-doc-1")
        assert confirmed.status == AppointmentStatus.CONFIRMED

    @pytest.mark.asyncio
    async def test_other_doctor_cannot_confirm_appointment(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        with pytest.raises(AppointmentServiceError):
            await service.confirm(appt.id, "u-doc-2")

    @pytest.mark.asyncio
    async def test_completing_appointment_requires_confirmed_first(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        with pytest.raises(AppointmentServiceError):
            await service.complete(appt.id, "u-doc-1")

        await service.confirm(appt.id, "u-doc-1")
        completed = await service.complete(appt.id, "u-doc-1")
        assert completed.status == AppointmentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_patient_can_cancel_own_appointment(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        cancelled = await service.cancel(appt.id, "u-pat-1", "patient")
        assert cancelled.status == AppointmentStatus.CANCELLED
        assert cancelled.cancelled_by == "u-pat-1"

    @pytest.mark.asyncio
    async def test_unrelated_patient_cannot_cancel(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        with pytest.raises(AppointmentServiceError):
            await service.cancel(appt.id, "u-pat-2", "patient")

    @pytest.mark.asyncio
    async def test_completed_appointment_cannot_be_cancelled(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        await service.confirm(appt.id, "u-doc-1")
        await service.complete(appt.id, "u-doc-1")
        with pytest.raises(AppointmentServiceError):
            await service.cancel(appt.id, "u-pat-1", "patient")

    @pytest.mark.asyncio
    async def test_patient_can_reschedule_to_free_slot(self, service):
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        rescheduled = await service.reschedule(
            appt.id, "u-pat-1",
            AppointmentReschedule(date="2026-09-11", start_time="14:00", end_time="14:30"),
        )
        assert rescheduled.date == "2026-09-11"
        assert rescheduled.status == AppointmentStatus.PENDING

    @pytest.mark.asyncio
    async def test_reschedule_into_conflicting_slot_rejected(self, service):
        await service.book(
            "u-pat-2",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-11", start_time="14:00", end_time="14:30"),
        )
        appt = await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        with pytest.raises(AppointmentServiceError):
            await service.reschedule(
                appt.id, "u-pat-1",
                AppointmentReschedule(date="2026-09-11", start_time="14:15", end_time="14:45"),
            )

    @pytest.mark.asyncio
    async def test_list_for_patient_and_doctor(self, service):
        await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-1", date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        await service.book(
            "u-pat-1",
            AppointmentCreate(doctor_id="doc-profile-2", date="2026-09-12", start_time="10:00", end_time="10:30"),
        )
        patient_appts = await service.list_for_patient("u-pat-1")
        assert len(patient_appts) == 2

        doctor_appts = await service.list_for_doctor("doc-profile-1")
        assert len(doctor_appts) == 1

    def test_router_builds_with_expected_routes(self):
        async def dummy_get_current_user():
            pass

        def dummy_require_role(*roles):
            async def _dep():
                pass
            return _dep

        router = build_router(dummy_get_current_user, dummy_require_role)
        paths = {r.path for r in router.routes}
        assert "/appointments/" in paths
        assert "/appointments/me" in paths
        assert "/appointments/{appointment_id}/confirm" in paths
        assert "/appointments/{appointment_id}/cancel" in paths
        assert "/appointments/{appointment_id}/reschedule" in paths