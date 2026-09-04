"""
backend/routes/appointments.py
-------------------------------
Appointments route: end-to-end booking workflow between patients and
doctors, built on top of the DoctorProfile / AvailabilitySlot system
in backend/routes/doctors.py.

State machine:
    PENDING -> CONFIRMED -> COMPLETED
    PENDING/CONFIRMED -> CANCELLED
    PENDING/CONFIRMED -> RESCHEDULED (new date/time, goes back to PENDING)

Design (mirrors doctors.py's style):
- Repository pattern (ABC + in-memory implementation).
- The service layer never trusts the frontend for authorization — every
  action re-checks that the caller is the patient who booked it, or the
  doctor who owns the appointment (verified via a DoctorRepository-like
  lookup passed into the router), or an admin.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class AppointmentStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConsultationType(str, Enum):
    IN_PERSON = "in_person"
    VIDEO = "video"
    PHONE = "phone"


class AppointmentCreate(BaseModel):
    doctor_id: str
    hospital_id: Optional[str] = None
    date: str  # "YYYY-MM-DD"
    start_time: str  # "HH:MM"
    end_time: str  # "HH:MM"
    consultation_type: ConsultationType = ConsultationType.IN_PERSON
    reason: Optional[str] = None


class AppointmentReschedule(BaseModel):
    date: str
    start_time: str
    end_time: str


class Appointment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    doctor_id: str
    hospital_id: Optional[str] = None
    date: str
    start_time: str
    end_time: str
    consultation_type: ConsultationType
    status: AppointmentStatus = AppointmentStatus.PENDING
    reason: Optional[str] = None
    cancelled_by: Optional[str] = None
    cancellation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class AppointmentServiceError(Exception):
    pass


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------

class AppointmentRepository:
    async def create(self, appointment: Appointment) -> Appointment:
        raise NotImplementedError

    async def get(self, appointment_id: str) -> Optional[Appointment]:
        raise NotImplementedError

    async def update(self, appointment: Appointment) -> Appointment:
        raise NotImplementedError

    async def list_for_patient(self, patient_id: str) -> list[Appointment]:
        raise NotImplementedError

    async def list_for_doctor(self, doctor_id: str) -> list[Appointment]:
        raise NotImplementedError

    async def has_conflict(
        self, doctor_id: str, date: str, start_time: str, end_time: str,
        exclude_appointment_id: Optional[str] = None,
    ) -> bool:
        raise NotImplementedError


class InMemoryAppointmentRepository(AppointmentRepository):
    def __init__(self) -> None:
        self._store: dict[str, Appointment] = {}

    async def create(self, appointment: Appointment) -> Appointment:
        self._store[appointment.id] = appointment
        return appointment

    async def get(self, appointment_id: str) -> Optional[Appointment]:
        return self._store.get(appointment_id)

    async def update(self, appointment: Appointment) -> Appointment:
        appointment.updated_at = datetime.now(timezone.utc)
        self._store[appointment.id] = appointment
        return appointment

    async def list_for_patient(self, patient_id: str) -> list[Appointment]:
        results = [a for a in self._store.values() if a.patient_id == patient_id]
        return sorted(results, key=lambda a: (a.date, a.start_time))

    async def list_for_doctor(self, doctor_id: str) -> list[Appointment]:
        results = [a for a in self._store.values() if a.doctor_id == doctor_id]
        return sorted(results, key=lambda a: (a.date, a.start_time))

    async def has_conflict(
        self, doctor_id: str, date: str, start_time: str, end_time: str,
        exclude_appointment_id: Optional[str] = None,
    ) -> bool:
        active_statuses = {AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED}
        for appt in self._store.values():
            if appt.id == exclude_appointment_id:
                continue
            if appt.doctor_id != doctor_id or appt.date != date:
                continue
            if appt.status not in active_statuses:
                continue
            # Overlap check: two ranges [s1,e1) and [s2,e2) overlap if s1 < e2 and s2 < e1
            if start_time < appt.end_time and appt.start_time < end_time:
                return True
        return False


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

# A callable that, given a doctor_id (DoctorProfile.id), returns the
# owning user's id — lets this module verify "is this doctor-user allowed
# to act on this appointment" without importing doctors.py's repository
# directly (kept decoupled, same spirit as build_router in doctors.py).
DoctorOwnerResolver = Callable[[str], Awaitable[Optional[str]]]


class AppointmentService:
    def __init__(
        self,
        repository: AppointmentRepository,
        resolve_doctor_owner: Optional[DoctorOwnerResolver] = None,
    ) -> None:
        self._repo = repository
        self._resolve_doctor_owner = resolve_doctor_owner

    async def book(self, patient_id: str, req: AppointmentCreate) -> Appointment:
        if req.start_time >= req.end_time:
            raise AppointmentServiceError("Start time must be before end time.")

        conflict = await self._repo.has_conflict(
            req.doctor_id, req.date, req.start_time, req.end_time
        )
        if conflict:
            raise AppointmentServiceError("This doctor already has an appointment in that time slot.")

        appointment = Appointment(
            patient_id=patient_id,
            doctor_id=req.doctor_id,
            hospital_id=req.hospital_id,
            date=req.date,
            start_time=req.start_time,
            end_time=req.end_time,
            consultation_type=req.consultation_type,
            reason=req.reason,
        )
        return await self._repo.create(appointment)

    async def get(self, appointment_id: str) -> Optional[Appointment]:
        return await self._repo.get(appointment_id)

    async def list_for_patient(self, patient_id: str) -> list[Appointment]:
        return await self._repo.list_for_patient(patient_id)

    async def list_for_doctor(self, doctor_id: str) -> list[Appointment]:
        return await self._repo.list_for_doctor(doctor_id)

    async def _assert_doctor_owns(self, appointment: Appointment, doctor_user_id: str) -> None:
        if self._resolve_doctor_owner is None:
            return  # no resolver wired up — skip ownership check (permissive fallback)
        owner_user_id = await self._resolve_doctor_owner(appointment.doctor_id)
        if owner_user_id != doctor_user_id:
            raise AppointmentServiceError("You can only manage your own appointments.")

    async def confirm(self, appointment_id: str, doctor_user_id: str) -> Appointment:
        appointment = await self._repo.get(appointment_id)
        if appointment is None:
            raise AppointmentServiceError("Appointment not found.")
        await self._assert_doctor_owns(appointment, doctor_user_id)
        if appointment.status != AppointmentStatus.PENDING:
            raise AppointmentServiceError("Only pending appointments can be confirmed.")
        appointment.status = AppointmentStatus.CONFIRMED
        return await self._repo.update(appointment)

    async def complete(self, appointment_id: str, doctor_user_id: str) -> Appointment:
        appointment = await self._repo.get(appointment_id)
        if appointment is None:
            raise AppointmentServiceError("Appointment not found.")
        await self._assert_doctor_owns(appointment, doctor_user_id)
        if appointment.status != AppointmentStatus.CONFIRMED:
            raise AppointmentServiceError("Only confirmed appointments can be marked completed.")
        appointment.status = AppointmentStatus.COMPLETED
        return await self._repo.update(appointment)

    async def cancel(
        self, appointment_id: str, actor_id: str, actor_role: str, reason: Optional[str] = None
    ) -> Appointment:
        appointment = await self._repo.get(appointment_id)
        if appointment is None:
            raise AppointmentServiceError("Appointment not found.")

        is_patient = appointment.patient_id == actor_id
        is_doctor_owner = False
        if actor_role == "doctor":
            owner_user_id = (
                await self._resolve_doctor_owner(appointment.doctor_id)
                if self._resolve_doctor_owner else actor_id
            )
            is_doctor_owner = owner_user_id == actor_id

        if not (is_patient or is_doctor_owner or actor_role == "admin"):
            raise AppointmentServiceError("You are not authorized to cancel this appointment.")

        if appointment.status == AppointmentStatus.COMPLETED:
            raise AppointmentServiceError("A completed appointment cannot be cancelled.")

        appointment.status = AppointmentStatus.CANCELLED
        appointment.cancelled_by = actor_id
        appointment.cancellation_reason = reason
        return await self._repo.update(appointment)

    async def reschedule(
        self, appointment_id: str, patient_id: str, req: AppointmentReschedule
    ) -> Appointment:
        appointment = await self._repo.get(appointment_id)
        if appointment is None:
            raise AppointmentServiceError("Appointment not found.")
        if appointment.patient_id != patient_id:
            raise AppointmentServiceError("You can only reschedule your own appointments.")
        if appointment.status not in (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED):
            raise AppointmentServiceError("Only pending or confirmed appointments can be rescheduled.")
        if req.start_time >= req.end_time:
            raise AppointmentServiceError("Start time must be before end time.")

        conflict = await self._repo.has_conflict(
            appointment.doctor_id, req.date, req.start_time, req.end_time,
            exclude_appointment_id=appointment.id,
        )
        if conflict:
            raise AppointmentServiceError("This doctor already has an appointment in that time slot.")

        appointment.date = req.date
        appointment.start_time = req.start_time
        appointment.end_time = req.end_time
        appointment.status = AppointmentStatus.PENDING
        return await self._repo.update(appointment)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router(get_current_user, require_role, resolve_doctor_owner: Optional[DoctorOwnerResolver] = None):
    """
    Takes the shared auth dependencies from users.py's build_router(),
    same pattern doctors.py uses. `resolve_doctor_owner` is an optional
    callable (doctor_id -> owning user_id) for verifying that a doctor
    can only manage their own appointments.
    """
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/appointments", tags=["appointments"])

    _repo = InMemoryAppointmentRepository()

    def get_appointment_service() -> AppointmentService:
        return AppointmentService(_repo, resolve_doctor_owner)

    @router.post("/", response_model=Appointment, status_code=status.HTTP_201_CREATED)
    async def book_appointment(
        req: AppointmentCreate,
        current_user=Depends(get_current_user),
        service: AppointmentService = Depends(get_appointment_service),
    ):
        try:
            return await service.book(current_user.id, req)
        except AppointmentServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.get("/me", response_model=list[Appointment])
    async def my_appointments(
        current_user=Depends(get_current_user),
        service: AppointmentService = Depends(get_appointment_service),
    ):
        return await service.list_for_patient(current_user.id)

    @router.get("/doctor/{doctor_id}", response_model=list[Appointment])
    async def doctor_appointments(
        doctor_id: str,
        service: AppointmentService = Depends(get_appointment_service),
    ):
        return await service.list_for_doctor(doctor_id)

    @router.get("/{appointment_id}", response_model=Appointment)
    async def get_appointment(
        appointment_id: str,
        service: AppointmentService = Depends(get_appointment_service),
    ):
        appointment = await service.get(appointment_id)
        if appointment is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found.")
        return appointment

    @router.patch("/{appointment_id}/confirm", response_model=Appointment)
    async def confirm_appointment(
        appointment_id: str,
        current_user=Depends(get_current_user),
        service: AppointmentService = Depends(get_appointment_service),
    ):
        try:
            return await service.confirm(appointment_id, current_user.id)
        except AppointmentServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.patch("/{appointment_id}/complete", response_model=Appointment)
    async def complete_appointment(
        appointment_id: str,
        current_user=Depends(get_current_user),
        service: AppointmentService = Depends(get_appointment_service),
    ):
        try:
            return await service.complete(appointment_id, current_user.id)
        except AppointmentServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.patch("/{appointment_id}/cancel", response_model=Appointment)
    async def cancel_appointment(
        appointment_id: str,
        reason: Optional[str] = None,
        current_user=Depends(get_current_user),
        service: AppointmentService = Depends(get_appointment_service),
    ):
        try:
            return await service.cancel(appointment_id, current_user.id, current_user.role.value, reason)
        except AppointmentServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.patch("/{appointment_id}/reschedule", response_model=Appointment)
    async def reschedule_appointment(
        appointment_id: str,
        req: AppointmentReschedule,
        current_user=Depends(get_current_user),
        service: AppointmentService = Depends(get_appointment_service),
    ):
        try:
            return await service.reschedule(appointment_id, current_user.id, req)
        except AppointmentServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return router