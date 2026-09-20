"""
hospitals.py
------------
Hospital profiles: registration, admin verification, department/service
listings. Follows the exact same pattern as doctors.py:

    models (Pydantic) -> Repository (ABC) -> InMemoryRepository
        -> Service (business logic + ownership checks) -> build_router()

In-memory only â€” no MongoDB, no Firebase, per project decision.
build_router(get_current_user, require_role) returns JUST the router
(not a tuple) â€” matches every module except users.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.routes.users import UserPublic, UserRole


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class HospitalCreate(BaseModel):
    name: str
    departments: list[str] = Field(default_factory=list)
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    has_emergency_support: bool = False
    has_blood_bank: bool = False


class HospitalUpdate(BaseModel):
    name: Optional[str] = None
    departments: Optional[list[str]] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    has_emergency_support: Optional[bool] = None
    has_blood_bank: Optional[bool] = None


class Hospital(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    name: str
    departments: list[str] = Field(default_factory=list)
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    has_emergency_support: bool = False
    has_blood_bank: bool = False
    is_verified: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HospitalServiceCreate(BaseModel):
    name: str
    department: Optional[str] = None
    description: Optional[str] = None


class HospitalServiceItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    hospital_id: str
    name: str
    department: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HospitalError(Exception):
    pass


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------

class HospitalRepository(ABC):
    @abstractmethod
    async def create(self, hospital: Hospital) -> Hospital: ...

    @abstractmethod
    async def get(self, hospital_id: str) -> Optional[Hospital]: ...

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> Optional[Hospital]: ...

    @abstractmethod
    async def search(
        self, department: Optional[str], verified_only: bool, emergency_only: bool
    ) -> list[Hospital]: ...

    @abstractmethod
    async def update(self, hospital: Hospital) -> Hospital: ...

    @abstractmethod
    async def add_service(self, service: HospitalServiceItem) -> HospitalServiceItem: ...

    @abstractmethod
    async def list_services(self, hospital_id: str) -> list[HospitalServiceItem]: ...


class InMemoryHospitalRepository(HospitalRepository):
    def __init__(self) -> None:
        self._hospitals: dict[str, Hospital] = {}
        self._services: list[HospitalServiceItem] = []

    async def create(self, hospital: Hospital) -> Hospital:
        self._hospitals[hospital.id] = hospital
        return hospital

    async def get(self, hospital_id: str) -> Optional[Hospital]:
        return self._hospitals.get(hospital_id)

    async def get_by_user_id(self, user_id: str) -> Optional[Hospital]:
        return next((h for h in self._hospitals.values() if h.user_id == user_id), None)

    async def search(
        self, department: Optional[str], verified_only: bool, emergency_only: bool
    ) -> list[Hospital]:
        results = list(self._hospitals.values())
        if department:
            results = [h for h in results if department in h.departments]
        if verified_only:
            results = [h for h in results if h.is_verified]
        if emergency_only:
            results = [h for h in results if h.has_emergency_support]
        return results

    async def update(self, hospital: Hospital) -> Hospital:
        hospital.updated_at = datetime.now(timezone.utc)
        self._hospitals[hospital.id] = hospital
        return hospital

    async def add_service(self, service: HospitalServiceItem) -> HospitalServiceItem:
        self._services.append(service)
        return service

    async def list_services(self, hospital_id: str) -> list[HospitalServiceItem]:
        return [s for s in self._services if s.hospital_id == hospital_id]


# --------------------------------------------------------------------------
# Service (business logic + ownership checks)
# --------------------------------------------------------------------------

class HospitalService:
    def __init__(self, repository: HospitalRepository) -> None:
        self._repo = repository

    async def create_hospital(self, user: UserPublic, req: HospitalCreate) -> Hospital:
        if user.role != UserRole.HOSPITAL:
            raise HospitalError("Only accounts with the 'hospital' role can create a hospital profile.")
        if await self._repo.get_by_user_id(user.id) is not None:
            raise HospitalError("A hospital profile already exists for this account.")
        hospital = Hospital(user_id=user.id, **req.model_dump())
        return await self._repo.create(hospital)

    async def get(self, hospital_id: str) -> Optional[Hospital]:
        return await self._repo.get(hospital_id)

    async def search(
        self, department: Optional[str] = None, verified_only: bool = False,
        emergency_only: bool = False,
    ) -> list[Hospital]:
        return await self._repo.search(department, verified_only, emergency_only)

    async def update_own(self, user: UserPublic, hospital_id: str, req: HospitalUpdate) -> Hospital:
        hospital = await self._repo.get(hospital_id)
        if hospital is None:
            raise HospitalError("Hospital not found.")
        if hospital.user_id != user.id:
            raise HospitalError("You can only edit your own hospital profile.")
        for key, value in req.model_dump(exclude_unset=True).items():
            setattr(hospital, key, value)
        return await self._repo.update(hospital)

    async def set_verified(self, hospital_id: str, is_verified: bool) -> Hospital:
        hospital = await self._repo.get(hospital_id)
        if hospital is None:
            raise HospitalError("Hospital not found.")
        hospital.is_verified = is_verified
        return await self._repo.update(hospital)

    async def add_service(
        self, user: UserPublic, hospital_id: str, req: HospitalServiceCreate
    ) -> HospitalServiceItem:
        hospital = await self._repo.get(hospital_id)
        if hospital is None:
            raise HospitalError("Hospital not found.")
        if hospital.user_id != user.id:
            raise HospitalError("You can only manage your own hospital's services.")
        service = HospitalServiceItem(hospital_id=hospital_id, **req.model_dump())
        return await self._repo.add_service(service)

    async def list_services(self, hospital_id: str) -> list[HospitalServiceItem]:
        return await self._repo.list_services(hospital_id)

    async def resolve_owner(self, hospital_id: str) -> Optional[str]:
        """hospital_id -> owning user_id, or None. For other modules to use."""
        hospital = await self._repo.get(hospital_id)
        return hospital.user_id if hospital else None


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

def build_router(get_current_user, require_role):
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/hospitals", tags=["hospitals"])

    # Module-level singleton so data persists across requests in this
    # in-memory, single-process setup.
    _repo = InMemoryHospitalRepository()

    def get_service() -> HospitalService:
        return HospitalService(_repo)

    @router.post("/me", response_model=Hospital, status_code=status.HTTP_201_CREATED)
    async def create_my_hospital(
        req: HospitalCreate,
        current_user: UserPublic = Depends(get_current_user),
        service: HospitalService = Depends(get_service),
    ):
        try:
            return await service.create_hospital(current_user, req)
        except HospitalError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.get("/", response_model=list[Hospital])
    async def search_hospitals(
        department: Optional[str] = None,
        verified_only: bool = False,
        emergency_only: bool = False,
        service: HospitalService = Depends(get_service),
    ):
        return await service.search(department, verified_only, emergency_only)

    @router.get("/{hospital_id}", response_model=Hospital)
    async def get_hospital(hospital_id: str, service: HospitalService = Depends(get_service)):
        hospital = await service.get(hospital_id)
        if hospital is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hospital not found.")
        return hospital

    @router.patch("/{hospital_id}", response_model=Hospital)
    async def update_my_hospital(
        hospital_id: str,
        req: HospitalUpdate,
        current_user: UserPublic = Depends(get_current_user),
        service: HospitalService = Depends(get_service),
    ):
        try:
            return await service.update_own(current_user, hospital_id, req)
        except HospitalError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.patch(
        "/{hospital_id}/verify",
        response_model=Hospital,
        dependencies=[Depends(require_role(UserRole.ADMIN))],
    )
    async def verify_hospital(
        hospital_id: str, is_verified: bool = True, service: HospitalService = Depends(get_service),
    ):
        try:
            return await service.set_verified(hospital_id, is_verified)
        except HospitalError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    @router.post(
        "/{hospital_id}/services", response_model=HospitalServiceItem,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_service(
        hospital_id: str,
        req: HospitalServiceCreate,
        current_user: UserPublic = Depends(get_current_user),
        service: HospitalService = Depends(get_service),
    ):
        try:
            return await service.add_service(current_user, hospital_id, req)
        except HospitalError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.get("/{hospital_id}/services", response_model=list[HospitalServiceItem])
    async def list_services(hospital_id: str, service: HospitalService = Depends(get_service)):
        return await service.list_services(hospital_id)

    return router

