"""
backend/tests/test_blood_bank.py
------------------------------------
Tests for the Blood Bank Intelligence module (backend/routes/blood_bank.py).
No MongoDB connection or API keys needed — uses in-memory repositories.
"""

import pytest

from backend.routes.blood_bank import (
    BloodBankCreate,
    BloodBankUpdate,
    StockUpdate,
    BloodGroup,
    StockLevel,
    InMemoryBloodBankRepository,
    BloodBankService,
    BloodBankServiceError,
    build_router,
)


class TestBloodBankModule:
    @pytest.fixture
    def service(self) -> BloodBankService:
        return BloodBankService(InMemoryBloodBankRepository())

    @pytest.mark.asyncio
    async def test_register_creates_blood_bank_with_no_reported_stock(self, service):
        bank = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="City Blood Bank", address="MG Road", latitude=13.08, longitude=80.27),
        )
        assert bank.name == "City Blood Bank"
        assert bank.stock == {}

    @pytest.mark.asyncio
    async def test_owner_can_update_own_listing(self, service):
        bank = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="City Blood Bank", address="MG Road", latitude=13.08, longitude=80.27),
        )
        updated = await service.update_own("u-hosp-1", bank.id, BloodBankUpdate(phone="9999999999"))
        assert updated.phone == "9999999999"

    @pytest.mark.asyncio
    async def test_non_owner_cannot_update_listing(self, service):
        bank = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="City Blood Bank", address="MG Road", latitude=13.08, longitude=80.27),
        )
        with pytest.raises(BloodBankServiceError):
            await service.update_own("u-hosp-2", bank.id, BloodBankUpdate(phone="8888888888"))

    @pytest.mark.asyncio
    async def test_owner_can_update_stock_with_timestamp(self, service):
        bank = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="City Blood Bank", address="MG Road", latitude=13.08, longitude=80.27),
        )
        updated = await service.update_stock(
            "u-hosp-1", bank.id,
            StockUpdate(blood_group=BloodGroup.O_NEG, level=StockLevel.AVAILABLE, units_available=12),
        )
        entry = updated.stock["O-"]
        assert entry.level == StockLevel.AVAILABLE
        assert entry.units_available == 12
        assert entry.last_updated is not None

    @pytest.mark.asyncio
    async def test_non_owner_cannot_update_stock(self, service):
        bank = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="City Blood Bank", address="MG Road", latitude=13.08, longitude=80.27),
        )
        with pytest.raises(BloodBankServiceError):
            await service.update_stock(
                "u-hosp-2", bank.id,
                StockUpdate(blood_group=BloodGroup.O_NEG, level=StockLevel.AVAILABLE),
            )

    @pytest.mark.asyncio
    async def test_search_filters_by_blood_group_availability(self, service):
        bank1 = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="City Blood Bank", address="MG Road", latitude=13.08, longitude=80.27),
        )
        await service.update_stock("u-hosp-1", bank1.id, StockUpdate(blood_group=BloodGroup.O_NEG, level=StockLevel.AVAILABLE))

        bank2 = await service.register(
            "u-hosp-2",
            BloodBankCreate(name="Other Blood Bank", address="Anna Salai", latitude=13.05, longitude=80.25),
        )
        await service.update_stock("u-hosp-2", bank2.id, StockUpdate(blood_group=BloodGroup.O_NEG, level=StockLevel.UNAVAILABLE))

        results = await service.search(blood_group=BloodGroup.O_NEG)
        assert len(results) == 1
        assert results[0].blood_bank.id == bank1.id

    @pytest.mark.asyncio
    async def test_search_sorts_by_distance_when_location_given(self, service):
        near = await service.register(
            "u-hosp-1",
            BloodBankCreate(name="Near Bank", address="Close by", latitude=13.0827, longitude=80.2707),
        )
        far = await service.register(
            "u-hosp-2",
            BloodBankCreate(name="Far Bank", address="Far away", latitude=11.0168, longitude=76.9558),
        )
        results = await service.search(latitude=13.0827, longitude=80.2707)
        assert results[0].blood_bank.id == near.id
        assert results[0].distance_km < results[1].distance_km

    @pytest.mark.asyncio
    async def test_search_excludes_banks_beyond_max_distance(self, service):
        await service.register(
            "u-hosp-1",
            BloodBankCreate(name="Near Bank", address="Close by", latitude=13.0827, longitude=80.2707),
        )
        await service.register(
            "u-hosp-2",
            BloodBankCreate(name="Far Bank", address="Far away", latitude=11.0168, longitude=76.9558),
        )
        results = await service.search(latitude=13.0827, longitude=80.2707, max_distance_km=50)
        assert len(results) == 1
        assert results[0].blood_bank.name == "Near Bank"

    @pytest.mark.asyncio
    async def test_get_unknown_bank_returns_none(self, service):
        assert await service.get("does-not-exist") is None

    def test_router_builds_with_expected_routes(self):
        async def dummy_get_current_user():
            pass

        def dummy_require_role(*roles):
            async def _dep():
                pass
            return _dep

        router = build_router(dummy_get_current_user, dummy_require_role)
        paths = {r.path for r in router.routes}
        assert "/blood-banks/" in paths
        assert "/blood-banks/search" in paths
        assert "/blood-banks/{bank_id}/stock" in paths
