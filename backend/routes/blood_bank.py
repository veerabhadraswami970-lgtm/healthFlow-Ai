"""
backend/routes/blood_bank.py
-------------------------------
Blood Bank Intelligence: search blood banks by location and blood group.

Per the project doc's "Honesty by Design" principle: "available" units are
only shown when the source provides genuinely verified, timestamped stock
data. Stock defaults to "not reported" rather than a fabricated number.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BloodGroup(str, Enum):
    A_POS = "A+"
    A_NEG = "A-"
    B_POS = "B+"
    B_NEG = "B-"
    AB_POS = "AB+"
    AB_NEG = "AB-"
    O_POS = "O+"
    O_NEG = "O-"


class StockLevel(str, Enum):
    AVAILABLE = "available"
    LOW = "low"
    UNAVAILABLE = "unavailable"
    NOT_REPORTED = "not_reported"


class BloodBankCreate(BaseModel):
    name: str
    hospital_id: Optional[str] = None
    address: str
    latitude: float
    longitude: float
    phone: Optional[str] = None


class BloodBankUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None


class StockUpdate(BaseModel):
    blood_group: BloodGroup
    level: StockLevel
    units_available: Optional[int] = None


class BloodStockEntry(BaseModel):
    blood_group: BloodGroup
    level: StockLevel = StockLevel.NOT_REPORTED
    units_available: Optional[int] = None
    last_updated: Optional[datetime] = None


class BloodBank(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner_user_id: str
    name: str
    hospital_id: Optional[str] = None
    address: str
    latitude: float
    longitude: float
    phone: Optional[str] = None
    stock: dict[str, BloodStockEntry] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BloodBankSearchResult(BaseModel):
    blood_bank: BloodBank
    distance_km: Optional[float] = None


class BloodBankServiceError(Exception):
    pass


class BloodBankRepository:
    async def create(self, bank: BloodBank) -> BloodBank:
        raise NotImplementedError

    async def get(self, bank_id: str) -> Optional[BloodBank]:
        raise NotImplementedError

    async def update(self, bank: BloodBank) -> BloodBank:
        raise NotImplementedError

    async def list_all(self) -> list[BloodBank]:
        raise NotImplementedError


class InMemoryBloodBankRepository(BloodBankRepository):
    def __init__(self) -> None:
        self._store: dict[str, BloodBank] = {}

    async def create(self, bank: BloodBank) -> BloodBank:
        self._store[bank.id] = bank
        return bank

    async def get(self, bank_id: str) -> Optional[BloodBank]:
        return self._store.get(bank_id)

    async def update(self, bank: BloodBank) -> BloodBank:
        bank.updated_at = datetime.now(timezone.utc)
        self._store[bank.id] = bank
        return bank

    async def list_all(self) -> list[BloodBank]:
        return list(self._store.values())


class BloodBankRanker:
    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))


class BloodBankService:
    def __init__(self, repository: BloodBankRepository) -> None:
        self._repo = repository

    async def register(self, owner_user_id: str, req: BloodBankCreate) -> BloodBank:
        bank = BloodBank(owner_user_id=owner_user_id, **req.model_dump())
        return await self._repo.create(bank)

    async def get(self, bank_id: str) -> Optional[BloodBank]:
        return await self._repo.get(bank_id)

    async def update_own(self, owner_user_id: str, bank_id: str, req: BloodBankUpdate) -> BloodBank:
        bank = await self._repo.get(bank_id)
        if bank is None:
            raise BloodBankServiceError("Blood bank not found.")
        if bank.owner_user_id != owner_user_id:
            raise BloodBankServiceError("You can only edit your own blood bank listing.")

        updates = req.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(bank, key, value)
        return await self._repo.update(bank)

    async def update_stock(self, owner_user_id: str, bank_id: str, req: StockUpdate) -> BloodBank:
        bank = await self._repo.get(bank_id)
        if bank is None:
            raise BloodBankServiceError("Blood bank not found.")
        if bank.owner_user_id != owner_user_id:
            raise BloodBankServiceError("You can only update stock for your own blood bank.")

        bank.stock[req.blood_group.value] = BloodStockEntry(
            blood_group=req.blood_group,
            level=req.level,
            units_available=req.units_available,
            last_updated=datetime.now(timezone.utc),
        )
        return await self._repo.update(bank)

    async def search(
        self,
        blood_group: Optional[BloodGroup] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_distance_km: Optional[float] = None,
    ) -> list[BloodBankSearchResult]:
        all_banks = await self._repo.list_all()

        if blood_group is not None:
            all_banks = [
                b for b in all_banks
                if b.stock.get(blood_group.value) is not None
                and b.stock[blood_group.value].level in (StockLevel.AVAILABLE, StockLevel.LOW)
            ]

        results: list[BloodBankSearchResult] = []
        for bank in all_banks:
            distance = None
            if latitude is not None and longitude is not None:
                distance = BloodBankRanker._haversine_km(latitude, longitude, bank.latitude, bank.longitude)
                if max_distance_km is not None and distance > max_distance_km:
                    continue
            results.append(BloodBankSearchResult(blood_bank=bank, distance_km=distance))

        if latitude is not None and longitude is not None:
            results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))

        return results


def build_router(get_current_user, require_role):
    from fastapi import APIRouter, Depends, HTTPException, status

    router = APIRouter(prefix="/blood-banks", tags=["blood-banks"])

    _repo = InMemoryBloodBankRepository()

    def get_blood_bank_service() -> BloodBankService:
        return BloodBankService(_repo)

    @router.post("/", response_model=BloodBank, status_code=status.HTTP_201_CREATED)
    async def register_blood_bank(
        req: BloodBankCreate,
        current_user=Depends(get_current_user),
        service: BloodBankService = Depends(get_blood_bank_service),
    ):
        return await service.register(current_user.id, req)

    @router.get("/search", response_model=list[BloodBankSearchResult])
    async def search_blood_banks(
        blood_group: Optional[BloodGroup] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        service: BloodBankService = Depends(get_blood_bank_service),
    ):
        return await service.search(blood_group, latitude, longitude, max_distance_km)

    @router.get("/{bank_id}", response_model=BloodBank)
    async def get_blood_bank(
        bank_id: str,
        service: BloodBankService = Depends(get_blood_bank_service),
    ):
        bank = await service.get(bank_id)
        if bank is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blood bank not found.")
        return bank

    @router.patch("/{bank_id}", response_model=BloodBank)
    async def update_blood_bank(
        bank_id: str,
        req: BloodBankUpdate,
        current_user=Depends(get_current_user),
        service: BloodBankService = Depends(get_blood_bank_service),
    ):
        try:
            return await service.update_own(current_user.id, bank_id, req)
        except BloodBankServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    @router.patch("/{bank_id}/stock", response_model=BloodBank)
    async def update_stock(
        bank_id: str,
        req: StockUpdate,
        current_user=Depends(get_current_user),
        service: BloodBankService = Depends(get_blood_bank_service),
    ):
        try:
            return await service.update_stock(current_user.id, bank_id, req)
        except BloodBankServiceError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return router
