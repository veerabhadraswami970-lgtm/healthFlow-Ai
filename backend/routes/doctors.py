"""
doctors.py
----------
Core domain route: Doctors.

Depends on users.py for identity â€” a doctor profile is always linked to
a UserRole.DOCTOR account, and admin-only actions (verification) reuse
the `require_role` dependency built there rather than duplicating auth
logic.

This is the data layer the Recommendation Agent's CandidateRepository
(recommendation_agent.py) reads from in production â€” that agent ranks
what this route stores.

Design:
- A doctor can only create/edit THEIR OWN profile (enforced via the
  authenticated user's id, not a client-supplied doctor_id).
- Verification is a separate, admin-only action â€” a doctor cannot
  self-verify. This mirrors the doc's "Admin verifies doctors and
  hospitals" principle.
- Availability slots are a separate collection (`doctor_availability`,
  per the doc's database design) so booking logic (in appointments.py,
  built next) can query/update them independently of the profile.

Layering:
  Router (FastAPI)
      -> DoctorService (orchestration)
          -> DoctorRepository (Mongo: doctors)
          -> AvailabilityRepository (Mongo: doctor_availability)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.routes.users import UserPublic, UserRole


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class ConsultationType(str, Enum):
    IN_PERSON = "in_person"
    ONLINE = "online"
    EITHER = "either"


class DoctorProfileCreate(BaseModel):
    specialties: list[str] = Field(..., min_length=1)
    qualifications: list[str] = Field(default_factory=list)
    experience_years: int = Field(..., ge=0, le=70)
    hospital_id: Optional[str] = None
    consultation_type: ConsultationType = ConsultationType.IN_PERSON
    consultation_fee: Optional[float] = Field(None, ge=0)
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class DoctorProfileUpdate(BaseModel):
    specialties: Optional[list[str]] = None
    qualifications: Optional[list[str]] = None
    experience_years: Optional[int] = Field(None, ge=0, le=70)
    hospital_id: Optional[str] = None
    consultation_type: Optional[ConsultationType] = None
    consultation_fee: Optional[float] = Field(None, ge=0)
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class DoctorProfile(BaseModel):
    """Mirrors the `doctors` MongoDB collection."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str  # links to users collection
    full_name: str  # denormalised from the user account at creation time
    specialties: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)
    experience_years: int = 0
    hospital_id: Optional[str] = None
    consultation_type: ConsultationType = ConsultationType.IN_PERSON
    consultation_fee: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_verified: bool = False  # only an admin can flip this
    rating: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "DoctorProfile":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class AvailabilitySlotCreate(BaseModel):
    date: str  # ISO date, e.g. "2026-09-05" â€” kept as string to avoid timezone ambiguity
    start_time: str  # "09:00"
    end_time: str  # "09:30"


class AvailabilitySlot(BaseModel):
    """Mirrors the `doctor_availability` MongoDB collection."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    doctor_id: str
    date: str
    start_time: str
    end_time: str
    is_booked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "AvailabilitySlot":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class DoctorServiceError(Exception):
    pass


# --------------------------------------------------------------------------
# Repositories (MongoDB via motor)
# --------------------------------------------------------------------------

class DoctorRepository(ABC):
    @abstractmethod
    async def create(self, profile: DoctorProfile) -> DoctorProfile:
        ...

    @abstractmethod
    async def get(self, doctor_id: str) -> Optional[DoctorProfile]:
        ...

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> Optional[DoctorProfile]:
        ...

    @abstractmethod
    async def search(
        self, specialty: Optional[str], hospital_id: Optional[str], verified_only: bool
    ) -> list[DoctorProfile]:
        ...

    @abstractmethod
    async def update(self, profile: DoctorProfile) -> DoctorProfile:
        ...


class MongoDoctorRepository(DoctorRepository):
    def __init__(self, db) -> None:
        self._collection = db["doctors"]

    async def create(self, profile: DoctorProfile) -> DoctorProfile:
        await self._collection.insert_one(profile.to_mongo())
        return profile

    async def get(self, doctor_id: str) -> Optional[DoctorProfile]:
        doc = await self._collection.find_one({"_id": doctor_id})
        return DoctorProfile.from_mongo(doc) if doc else None

    async def get_by_user_id(self, user_id: str) -> Optional[DoctorProfile]:
        doc = await self._collection.find_one({"user_id": user_id})
        return DoctorProfile.from_mongo(doc) if doc else None

    async def search(
        self, specialty: Optional[str], hospital_id: Optional[str], verified_only: bool
    ) -> list[DoctorProfile]:
        query: dict = {}
        if specialty:
            query["specialties"] = specialty
        if hospital_id:
            query["hospital_id"] = hospital_id
        if verified_only:
            query["is_verified"] = True
        cursor = self._collection.find(query)
        return [DoctorProfile.from_mongo(doc) async for doc in cursor]

    async def update(self, profile: DoctorProfile) -> DoctorProfile:
        profile.updated_at = datetime.now(timezone.utc)
        await self._collection.replace_one({"_id": profile.id}, profile.to_mongo())
        return profile

    async def create_indexes(self) -> None:
        await self._collection.create_index("user_id", unique=True)
        await self._collection.create_index("specialties")
        await self._collection.create_index("hospital_id")


class InMemoryDoctorRepository(DoctorRepository):
    def __init__(self) -> None:
        self._store: dict[str, DoctorProfile] = {}

    async def create(self, profile: DoctorProfile) -> DoctorProfile:
        self._store[profile.id] = profile
        return profile

    async def get(self, doctor_id: str) -> Optional[DoctorProfile]:
        return self._store.get(doctor_id)

    async def get_by_user_id(self, user_id: str) -> Optional[DoctorProfile]:
        return next((d for d in self._store.values() if d.user_id == user_id), None)

    async def search(
        self, specialty: Optional[str], hospital_id: Optional[str], verified_only: bool
    ) -> list[DoctorProfile]:
        results = list(self._store.values())
        if specialty:
            results = [d for d in results if specialty in d.specialties]
        if hospital_id:
            results = [d for d in results if d.hospital_id == hospital_id]
        if verified_only:
            results = [d for d in results if d.is_verified]
        return results

    async def update(self, profile: DoctorProfile) -> DoctorProfile:
        profile.updated_at = datetime.now(timezone.utc)
        self._store[profile.id] = profile
        return profile


class AvailabilityRepository(ABC):
    @abstractmethod
    async def add_slot(self, slot: AvailabilitySlot) -> AvailabilitySlot:
        ...

    @abstractmethod
    async def list_for_doctor(self, doctor_id: str, date: Optional[str]) -> list[AvailabilitySlot]:
        ...

    @abstractmethod
    async def get_slot(self, slot_id: str) -> Optional[AvailabilitySlot]:
        ...

    @abstractmethod
    async def update_slot(self, slot: AvailabilitySlot) -> AvailabilitySlot:
        ...


class MongoAvailabilityRepository(AvailabilityRepository):
    def __init__(self, db) -> None:
        self._collection = db["doctor_availability"]

    async def add_slot(self, slot: AvailabilitySlot) -> AvailabilitySlot:
        await self._collection.insert_one(slot.to_mongo())
        return slot

    async def list_for_doctor(self, doctor_id: str, date: Optional[str]) -> list[AvailabilitySlot]:
        query: dict = {"doctor_id": doctor_id}
        if date:
            query["date"] = date
        cursor = self._collection.find(query).sort([("date", 1), ("start_time", 1)])
        return [AvailabilitySlot.from_mongo(doc) async for doc in cursor]

    async def get_slot(self, slot_id: str) -> Optional[AvailabilitySlot]:
        doc = await self._collection.find_one({"_id": slot_id})
        return AvailabilitySlot.from_mongo(doc) if doc else None

    async def update_slot(self, slot: AvailabilitySlot) -> AvailabilitySlot:
        await self._collection.replace_one({"_id": slot.id}, slot.to_mongo())
        return slot


class InMemoryAvailabilityRepository(AvailabilityRepository):
    def __init__(self) -> None:
        self._store: dict[str, AvailabilitySlot] = {}

    async def add_slot(self, slot: AvailabilitySlot) -> AvailabilitySlot:
        self._store[slot.id] = slot
        return slot

    async def list_for_doctor(self, doctor_id: str, date: Optional[str]) -> list[AvailabilitySlot]:
        results = [s for s in self._store.values() if s.doctor_id == doctor_id]
        if date:
            results = [s for s in results if s.date == date]
        return sorted(results, key=lambda s: (s.date, s.start_time))

    async def get_slot(self, slot_id: str) -> Optional[AvailabilitySlot]:
        return self._store.get(slot_id)

    async def update_slot(self, slot: AvailabilitySlot) -> AvailabilitySlot:
        self._store[slot.id] = slot
        return slot


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

class DoctorService:
    def __init__(
        self,
        doctor_repository: DoctorRepository,
        availability_repository: AvailabilityRepository,
    ) -> None:
        self._doctors = doctor_repository
        self._availability = availability_repository

    async def create_profile(self, user: UserPublic, req: DoctorProfileCreate) -> DoctorProfile:
        if user.role != UserRole.DOCTOR:
            raise DoctorServiceError("Only accounts with the 'doctor' role can create a doctor profile.")
        existing = await self._doctors.get_by_user_id(user.id)
        if existing is not None:
            raise DoctorServiceError("A doctor profile already exists for this account.")

        profile = DoctorProfile(user_id=user.id, full_name=user.full_name, **req.model_dump())
        return await self._doctors.create(profile)

    async def get(self, doctor_id: str) -> Optional[DoctorProfile]:
        return await self._doctors.get(doctor_id)

    async def search(
        self, specialty: Optional[str] = None, hospital_id: Optional[str] = None,
        verified_only: bool = False,
    ) -> list[DoctorProfile]:
        return await self._doctors.search(specialty, hospital_id, verified_only)

    async def update_own_profile(
        self, user: UserPublic, doctor_id: str, req: DoctorProfileUpdate
    ) -> DoctorProfile:
        profile = await self._doctors.get(doctor_id)
        if profile is None:
            raise DoctorServiceError("Doctor profile not found.")
        if profile.user_id != user.id:
            raise DoctorServiceError("You can only edit your own doctor profile.")

        updates = req.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(profile, key, value)
        return await self._doctors.update(profile)

    async def set_verified(self, doctor_id: str, is_verified: bool) -> DoctorProfile:
        """Admin-only action â€” the router enforces the role, not this method."""
        profile = await self._doctors.get(doctor_id)
        if profile is None:
            raise DoctorServiceError("Doctor profile not found.")
        profile.is_verified = is_verified
        return await self._doctors.update(profile)

    async def add_availability(
        self, user: UserPublic, doctor_id: str, req: AvailabilitySlotCreate
    ) -> AvailabilitySlot:
        profile = await self._doctors.get(doctor_id)
        if profile is None:
            raise DoctorServiceError("Doctor profile not found.")
        if profile.user_id != user.id:
            raise DoctorServiceError("You can only manage your own availability.")

        slot = AvailabilitySlot(doctor_id=doctor_id, **req.model_dump())
        return await self._availability.add_slot(slot)

    async def list_availability(self, doctor_id: str, date: Optional[str] = None) -> list[AvailabilitySlot]:
        return await self._availability.list_for_doctor(doctor_id, date)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router(get_current_user, require_role):
    """
    Takes the auth dependencies built in users.py so this router shares
    the same authentication/authorization mechanism rather than
    duplicating it.
    """
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/doctors", tags=["doctors"])

    # Module-level singletons so data persists across requests during
    # local/dev runs â€” swap for Mongo repositories in production.
    _doctor_repo = InMemoryDoctorRepository()
    _availability_repo = InMemoryAvailabilityRepository()

    def get_doctor_service() -> DoctorService:
        return DoctorService(_doctor_repo, _availability_repo)

    @router.post("/me", response_model=DoctorProfile, status_code=status.HTTP_201_CREATED)
    async def create_my_profile(
        req: DoctorProfileCreate,
        current_user: UserPublic = Depends(get_current_user),
        service: DoctorService = Depends(get_doctor_service),
    ):
        try:
            return await service.create_profile(current_user, req)
        except DoctorServiceError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.get("/", response_model=list[DoctorProfile])
    async def search_doctors(
        specialty: Optional[str] = None,
        hospital_id: Optional[str] = None,
        verified_only: bool = False,
        service: DoctorService = Depends(get_doctor_service),
    ):
        return await service.search(specialty, hospital_id, verified_only)

    @router.get("/{doctor_id}", response_model=DoctorProfile)
    async def get_doctor(
        doctor_id: str,
        service: DoctorService = Depends(get_doctor_service),
    ):
        profile = await service.get(doctor_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doctor not found.")
        return profile

    @router.patch("/{doctor_id}", response_model=DoctorProfile)
    async def update_my_profile(
        doctor_id: str,
        req: DoctorProfileUpdate,
        current_user: UserPublic = Depends(get_current_user),
        service: DoctorService = Depends(get_doctor_service),
    ):
        try:
            return await service.update_own_profile(current_user, doctor_id, req)
        except DoctorServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.patch(
        "/{doctor_id}/verify",
        response_model=DoctorProfile,
        dependencies=[Depends(require_role(UserRole.ADMIN))],
    )
    async def verify_doctor(
        doctor_id: str,
        is_verified: bool = True,
        service: DoctorService = Depends(get_doctor_service),
    ):
        try:
            return await service.set_verified(doctor_id, is_verified)
        except DoctorServiceError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    @router.post("/{doctor_id}/availability", response_model=AvailabilitySlot, status_code=status.HTTP_201_CREATED)
    async def add_availability(
        doctor_id: str,
        req: AvailabilitySlotCreate,
        current_user: UserPublic = Depends(get_current_user),
        service: DoctorService = Depends(get_doctor_service),
    ):
        try:
            return await service.add_availability(current_user, doctor_id, req)
        except DoctorServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.get("/{doctor_id}/availability", response_model=list[AvailabilitySlot])
    async def list_availability(
        doctor_id: str,
        date: Optional[str] = None,
        service: DoctorService = Depends(get_doctor_service),
    ):
        return await service.list_availability(doctor_id, date)

    async def resolve_doctor_owner(doctor_id: str) -> Optional[str]:
        """doctor_id -> the user_id that owns it, or None if not found.
        Used by appointments.py to verify a doctor can only manage
        their own appointments."""
        profile = await _doctor_repo.get(doctor_id)
        return profile.user_id if profile else None

    return router

