"""
backend/routes/emergency.py
-----------------------------
Emergency System: the one-tap SOS flow.

Per the project doc's "Honesty by Design" principle: this module NEVER
claims an ambulance has been dispatched or a hospital bed reserved unless
a real integration confirms it. Since no such integration exists yet,
every dispatch-related field defaults to "not confirmed" rather than a
fabricated success message.

Flow: trigger -> confirm -> locate -> notify contacts -> track status
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmergencyStatus(str, Enum):
    TRIGGERED = "triggered"
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class EmergencyContact(BaseModel):
    name: str
    phone: str
    relationship: Optional[str] = None


class EmergencyTrigger(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None


class NearestHospital(BaseModel):
    hospital_id: str
    name: str
    distance_km: float


class EmergencyEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    status: EmergencyStatus = EmergencyStatus.TRIGGERED
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None
    nearest_hospitals: list[NearestHospital] = Field(default_factory=list)
    contacts_notified: list[str] = Field(default_factory=list)
    ambulance_dispatch_confirmed: bool = False
    resolution_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmergencyServiceError(Exception):
    pass


class EmergencyEventRepository:
    async def create(self, event: EmergencyEvent) -> EmergencyEvent:
        raise NotImplementedError

    async def get(self, event_id: str) -> Optional[EmergencyEvent]:
        raise NotImplementedError

    async def update(self, event: EmergencyEvent) -> EmergencyEvent:
        raise NotImplementedError

    async def list_for_patient(self, patient_id: str) -> list[EmergencyEvent]:
        raise NotImplementedError

    async def list_active(self) -> list[EmergencyEvent]:
        raise NotImplementedError


class InMemoryEmergencyEventRepository(EmergencyEventRepository):
    def __init__(self) -> None:
        self._store: dict[str, EmergencyEvent] = {}

    async def create(self, event: EmergencyEvent) -> EmergencyEvent:
        self._store[event.id] = event
        return event

    async def get(self, event_id: str) -> Optional[EmergencyEvent]:
        return self._store.get(event_id)

    async def update(self, event: EmergencyEvent) -> EmergencyEvent:
        event.updated_at = datetime.now(timezone.utc)
        self._store[event.id] = event
        return event

    async def list_for_patient(self, patient_id: str) -> list[EmergencyEvent]:
        results = [e for e in self._store.values() if e.patient_id == patient_id]
        return sorted(results, key=lambda e: e.created_at, reverse=True)

    async def list_active(self) -> list[EmergencyEvent]:
        results = [e for e in self._store.values() if e.status == EmergencyStatus.ACTIVE]
        return sorted(results, key=lambda e: e.created_at)


class EmergencyContactRepository:
    async def list_for_patient(self, patient_id: str) -> list[EmergencyContact]:
        raise NotImplementedError

    async def add(self, patient_id: str, contact: EmergencyContact) -> EmergencyContact:
        raise NotImplementedError


class InMemoryEmergencyContactRepository(EmergencyContactRepository):
    def __init__(self) -> None:
        self._store: dict[str, list[EmergencyContact]] = {}

    async def list_for_patient(self, patient_id: str) -> list[EmergencyContact]:
        return self._store.get(patient_id, [])

    async def add(self, patient_id: str, contact: EmergencyContact) -> EmergencyContact:
        self._store.setdefault(patient_id, []).append(contact)
        return contact


class NearestHospitalFinder:
    async def find_nearest(
        self, latitude: Optional[float], longitude: Optional[float], limit: int = 3
    ) -> list[NearestHospital]:
        raise NotImplementedError


class NullHospitalFinder(NearestHospitalFinder):
    async def find_nearest(
        self, latitude: Optional[float], longitude: Optional[float], limit: int = 3
    ) -> list[NearestHospital]:
        return []


class EmergencyService:
    def __init__(
        self,
        event_repository: EmergencyEventRepository,
        contact_repository: EmergencyContactRepository,
        hospital_finder: Optional[NearestHospitalFinder] = None,
    ) -> None:
        self._events = event_repository
        self._contacts = contact_repository
        self._hospital_finder = hospital_finder or NullHospitalFinder()

    async def trigger(self, patient_id: str, req: EmergencyTrigger) -> EmergencyEvent:
        event = EmergencyEvent(
            patient_id=patient_id,
            latitude=req.latitude,
            longitude=req.longitude,
            notes=req.notes,
        )
        return await self._events.create(event)

    async def confirm(self, event_id: str, patient_id: str) -> EmergencyEvent:
        event = await self._events.get(event_id)
        if event is None:
            raise EmergencyServiceError("Emergency event not found.")
        if event.patient_id != patient_id:
            raise EmergencyServiceError("You can only confirm your own emergency event.")
        if event.status != EmergencyStatus.TRIGGERED:
            raise EmergencyServiceError("Only a freshly triggered event can be confirmed.")

        hospitals = await self._hospital_finder.find_nearest(event.latitude, event.longitude)
        contacts = await self._contacts.list_for_patient(patient_id)

        event.nearest_hospitals = hospitals
        event.contacts_notified = [c.name for c in contacts]
        event.status = EmergencyStatus.ACTIVE
        return await self._events.update(event)

    async def resolve(self, event_id: str, patient_id: str, resolution_notes: Optional[str] = None) -> EmergencyEvent:
        event = await self._events.get(event_id)
        if event is None:
            raise EmergencyServiceError("Emergency event not found.")
        if event.patient_id != patient_id:
            raise EmergencyServiceError("You can only resolve your own emergency event.")
        if event.status not in (EmergencyStatus.TRIGGERED, EmergencyStatus.ACTIVE):
            raise EmergencyServiceError("Only a triggered or active event can be resolved.")

        event.status = EmergencyStatus.RESOLVED
        event.resolution_notes = resolution_notes
        return await self._events.update(event)

    async def cancel(self, event_id: str, patient_id: str) -> EmergencyEvent:
        event = await self._events.get(event_id)
        if event is None:
            raise EmergencyServiceError("Emergency event not found.")
        if event.patient_id != patient_id:
            raise EmergencyServiceError("You can only cancel your own emergency event.")
        if event.status not in (EmergencyStatus.TRIGGERED, EmergencyStatus.ACTIVE):
            raise EmergencyServiceError("Only a triggered or active event can be cancelled.")

        event.status = EmergencyStatus.CANCELLED
        return await self._events.update(event)

    async def get(self, event_id: str) -> Optional[EmergencyEvent]:
        return await self._events.get(event_id)

    async def list_for_patient(self, patient_id: str) -> list[EmergencyEvent]:
        return await self._events.list_for_patient(patient_id)

    async def add_contact(self, patient_id: str, contact: EmergencyContact) -> EmergencyContact:
        return await self._contacts.add(patient_id, contact)

    async def list_contacts(self, patient_id: str) -> list[EmergencyContact]:
        return await self._contacts.list_for_patient(patient_id)


def build_router(get_current_user, require_role):
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/emergency", tags=["emergency"])

    _event_repo = InMemoryEmergencyEventRepository()
    _contact_repo = InMemoryEmergencyContactRepository()

    def get_emergency_service() -> EmergencyService:
        return EmergencyService(_event_repo, _contact_repo)

    @router.post("/trigger", response_model=EmergencyEvent, status_code=status.HTTP_201_CREATED)
    async def trigger_emergency(
        req: EmergencyTrigger,
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        return await service.trigger(current_user.id, req)

    @router.patch("/{event_id}/confirm", response_model=EmergencyEvent)
    async def confirm_emergency(
        event_id: str,
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        try:
            return await service.confirm(event_id, current_user.id)
        except EmergencyServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.patch("/{event_id}/resolve", response_model=EmergencyEvent)
    async def resolve_emergency(
        event_id: str,
        resolution_notes: Optional[str] = None,
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        try:
            return await service.resolve(event_id, current_user.id, resolution_notes)
        except EmergencyServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.patch("/{event_id}/cancel", response_model=EmergencyEvent)
    async def cancel_emergency(
        event_id: str,
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        try:
            return await service.cancel(event_id, current_user.id)
        except EmergencyServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.get("/{event_id}", response_model=EmergencyEvent)
    async def get_emergency(
        event_id: str,
        service: EmergencyService = Depends(get_emergency_service),
    ):
        event = await service.get(event_id)
        if event is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Emergency event not found.")
        return event

    @router.get("/me/history", response_model=list[EmergencyEvent])
    async def my_emergency_history(
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        return await service.list_for_patient(current_user.id)

    @router.post("/contacts", response_model=EmergencyContact, status_code=status.HTTP_201_CREATED)
    async def add_emergency_contact(
        contact: EmergencyContact,
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        return await service.add_contact(current_user.id, contact)

    @router.get("/contacts/me", response_model=list[EmergencyContact])
    async def my_emergency_contacts(
        current_user=Depends(get_current_user),
        service: EmergencyService = Depends(get_emergency_service),
    ):
        return await service.list_contacts(current_user.id)

    return router
