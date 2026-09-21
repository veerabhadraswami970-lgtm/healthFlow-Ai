"""
medicine_intelligence.py
-------------------------
Module 04: Medicine Intelligence Agent.

Design intent (per project architecture doc):
- Reference information only — uses, precautions, side effects,
  interactions, storage, warnings — always carrying a source and a
  last-updated date, sourced from a trusted, curated database.
- NO personalised dosage recommendations, ever. The agent explains
  trusted reference data in plain language; it never tells a specific
  user how much of something to take. That decision always defers to
  the prescribing doctor.
- Drug-drug interaction checking is DETERMINISTIC (a lookup against a
  curated interaction table), same as the scheme engine's eligibility
  logic — the LLM is not trusted to decide whether two drugs interact.
- Repository is Mongo-backed (motor, async), same pattern as the other
  two agents.

Layering:
  Router (FastAPI)
      -> MedicineService (orchestration)
          -> InteractionChecker (pure, deterministic, unit-testable)
          -> MedicineExplanationAgent (LLM plain-language summary, guardrailed)
          -> MedicineRepository (Mongo, curated reference data)
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("healthflow.medicine_intelligence_agent")


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class InteractionSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class MedicineReference(BaseModel):
    """Mirrors the `medicines` MongoDB collection — curated, not user-generated."""

    id: str
    name: str
    generic_name: Optional[str] = None
    uses: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    # interacts_with: other medicine IDs this entry is known to interact with
    interacts_with: dict[str, InteractionSeverity] = Field(default_factory=dict)
    storage: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    source: str  # e.g. "CDSCO", "NHS Medicines A-Z", internal curated source
    last_updated: date

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "MedicineReference":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class InteractionResult(BaseModel):
    medicine_a: str
    medicine_b: str
    interacts: bool
    severity: Optional[InteractionSeverity] = None
    note: str


class MedicineExplanationRequest(BaseModel):
    medicine_id: str
    language: str = "en"
    question: Optional[str] = None  # e.g. "what are the side effects?"


class MedicineExplanationResult(BaseModel):
    medicine: MedicineReference
    explanation: str
    disclaimer: str = (
        "This is general reference information, not a personal dosage "
        "recommendation. Always follow your prescribing doctor's instructions."
    )


# --------------------------------------------------------------------------
# Repository (MongoDB via motor)
# --------------------------------------------------------------------------

class MedicineRepository(ABC):
    @abstractmethod
    async def get(self, medicine_id: str) -> Optional[MedicineReference]:
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[MedicineReference]:
        ...

    @abstractmethod
    async def get_many(self, medicine_ids: list[str]) -> list[MedicineReference]:
        ...


class MongoMedicineRepository(MedicineRepository):
    def __init__(self, db) -> None:
        # db: an AsyncIOMotorDatabase instance, injected at app startup
        self._collection = db["medicines"]

    async def get(self, medicine_id: str) -> Optional[MedicineReference]:
        doc = await self._collection.find_one({"_id": medicine_id})
        return MedicineReference.from_mongo(doc) if doc else None

    async def search(self, query: str, limit: int = 10) -> list[MedicineReference]:
        # Requires a text index: db.medicines.create_index([("name", "text"),
        # ("generic_name", "text")]) — set that up once via a migration script.
        cursor = self._collection.find(
            {"$text": {"$search": query}}
        ).limit(limit)
        return [MedicineReference.from_mongo(doc) async for doc in cursor]

    async def get_many(self, medicine_ids: list[str]) -> list[MedicineReference]:
        cursor = self._collection.find({"_id": {"$in": medicine_ids}})
        return [MedicineReference.from_mongo(doc) async for doc in cursor]


class InMemoryMedicineRepository(MedicineRepository):
    """For local dev / unit tests without a running Mongo instance."""

    def __init__(self, medicines: list[MedicineReference] | None = None) -> None:
        self._store = {m.id: m for m in (medicines or [])}

    async def get(self, medicine_id: str) -> Optional[MedicineReference]:
        return self._store.get(medicine_id)

    async def search(self, query: str, limit: int = 10) -> list[MedicineReference]:
        q = query.lower()
        results = [
            m for m in self._store.values()
            if q in m.name.lower() or (m.generic_name and q in m.generic_name.lower())
        ]
        return results[:limit]

    async def get_many(self, medicine_ids: list[str]) -> list[MedicineReference]:
        return [self._store[i] for i in medicine_ids if i in self._store]


# --------------------------------------------------------------------------
# Deterministic interaction checker (NO LLM calls in this class, ever)
# --------------------------------------------------------------------------

class InteractionChecker:
    """
    Pure lookup against each medicine's curated `interacts_with` map.
    Kept separate from the AI layer for the same reason as the scheme
    eligibility engine: this is a clinical-safety-adjacent decision and
    must be auditable and independent of model behaviour.
    """

    @staticmethod
    def check_pair(a: MedicineReference, b: MedicineReference) -> InteractionResult:
        severity = a.interacts_with.get(b.id) or b.interacts_with.get(a.id)
        if severity is None:
            return InteractionResult(
                medicine_a=a.name,
                medicine_b=b.name,
                interacts=False,
                note="No known interaction found in the reference database. "
                     "This does not guarantee there is no interaction — "
                     "always confirm with a pharmacist or doctor.",
            )
        return InteractionResult(
            medicine_a=a.name,
            medicine_b=b.name,
            interacts=True,
            severity=severity,
            note=f"Known {severity.value} interaction. Consult your doctor or pharmacist "
                 "before taking these together.",
        )

    @classmethod
    def check_all(cls, medicines: list[MedicineReference]) -> list[InteractionResult]:
        results = []
        for i in range(len(medicines)):
            for j in range(i + 1, len(medicines)):
                results.append(cls.check_pair(medicines[i], medicines[j]))
        return results


# --------------------------------------------------------------------------
# AI explanation layer — reference summarisation only, dosage-guardrailed
# --------------------------------------------------------------------------

class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        ...


class GeminiClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash") -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(model)

    async def generate(self, prompt: str) -> str:
        response = await self._model.generate_content_async(prompt)
        return response.text


# Heuristic guard: catches obvious personalised-dosage phrasing that may slip
# through despite prompt instructions (e.g. "you should take 2 tablets").
# This is a safety net, not the primary control — the primary control is the
# prompt itself never being asked to produce a dosage in the first place.
_DOSAGE_ADVICE_PATTERN = re.compile(
    r"\byou should take\b|\btake \d+\s*(mg|ml|tablets?|pills?|drops?)\b|\bi recommend (a )?dose\b",
    re.IGNORECASE,
)


class MedicineExplanationAgent:
    """
    Summarises MedicineReference data in plain language. Never asked to,
    and explicitly instructed not to, produce a personalised dosage.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def explain(
        self, medicine: MedicineReference, language: str = "en", question: Optional[str] = None
    ) -> MedicineExplanationResult:
        fallback = self._deterministic_summary(medicine)

        if self._llm is None:
            return MedicineExplanationResult(medicine=medicine, explanation=fallback)

        prompt = self._build_prompt(medicine, language, question)
        try:
            text = (await self._llm.generate(prompt)).strip()
            if _DOSAGE_ADVICE_PATTERN.search(text):
                logger.warning(
                    "Medicine explanation for %s contained dosage-like language; "
                    "using deterministic fallback instead.", medicine.id
                )
                text = fallback
            return MedicineExplanationResult(medicine=medicine, explanation=text)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never break the response
            logger.warning("Medicine explanation generation failed: %s", exc)
            return MedicineExplanationResult(medicine=medicine, explanation=fallback)

    @staticmethod
    def _build_prompt(medicine: MedicineReference, language: str, question: Optional[str]) -> str:
        focus = f'The user specifically asked: "{question}"\n' if question else ""
        return (
            "You are a medicine reference assistant. Summarise the reference data "
            f"below in {language}, in plain language for a general audience. "
            "STRICT RULES: do not recommend a dosage or frequency for any individual; "
            "do not tell the user how much to take or how often; do not diagnose. "
            "Only describe what the reference data already says (uses, precautions, "
            "side effects, storage, warnings). If asked about dosage, say this must "
            "come from their prescribing doctor.\n\n"
            f"{focus}"
            f"Medicine: {medicine.name} ({medicine.generic_name or 'n/a'})\n"
            f"Uses: {', '.join(medicine.uses) or 'n/a'}\n"
            f"Precautions: {', '.join(medicine.precautions) or 'n/a'}\n"
            f"Side effects: {', '.join(medicine.side_effects) or 'n/a'}\n"
            f"Storage: {medicine.storage or 'n/a'}\n"
            f"Warnings: {', '.join(medicine.warnings) or 'n/a'}\n"
            f"Source: {medicine.source} (last updated {medicine.last_updated})"
        )

    @staticmethod
    def _deterministic_summary(medicine: MedicineReference) -> str:
        parts = [f"{medicine.name} is used for: {', '.join(medicine.uses) or 'see package insert'}."]
        if medicine.side_effects:
            parts.append(f"Possible side effects include: {', '.join(medicine.side_effects)}.")
        if medicine.warnings:
            parts.append(f"Warnings: {', '.join(medicine.warnings)}.")
        parts.append(f"Source: {medicine.source}, last updated {medicine.last_updated}.")
        return " ".join(parts)


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

class MedicineService:
    def __init__(
        self,
        repository: MedicineRepository,
        agent: Optional[MedicineExplanationAgent] = None,
    ) -> None:
        self._repository = repository
        self._agent = agent or MedicineExplanationAgent(llm_client=None)

    async def get(self, medicine_id: str) -> Optional[MedicineReference]:
        return await self._repository.get(medicine_id)

    async def search(self, query: str, limit: int = 10) -> list[MedicineReference]:
        return await self._repository.search(query, limit=limit)

    async def explain(self, req: MedicineExplanationRequest) -> MedicineExplanationResult:
        medicine = await self._repository.get(req.medicine_id)
        if medicine is None:
            raise ValueError(f"Medicine {req.medicine_id} not found")
        return await self._agent.explain(medicine, language=req.language, question=req.question)

    async def check_interactions(self, medicine_ids: list[str]) -> list[InteractionResult]:
        if len(medicine_ids) < 2:
            return []
        medicines = await self._repository.get_many(medicine_ids)
        return InteractionChecker.check_all(medicines)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

_repository = InMemoryMedicineRepository()
_agent = MedicineExplanationAgent(llm_client=None)
_medicine_service = MedicineService(_repository, _agent)


def build_router():
    from fastapi import APIRouter, Depends, HTTPException

    router = APIRouter(prefix="/medicines", tags=["medicines"])

    def get_medicine_service() -> MedicineService:
        return _medicine_service

    @router.get("/search", response_model=list[MedicineReference])
    async def search_medicines(
        q: str,
        limit: int = 10,
        service: MedicineService = Depends(get_medicine_service),
    ):
        return await service.search(q, limit=limit)

    @router.get("/{medicine_id}", response_model=MedicineReference)
    async def get_medicine(
        medicine_id: str,
        service: MedicineService = Depends(get_medicine_service),
    ):
        medicine = await service.get(medicine_id)
        if medicine is None:
            raise HTTPException(status_code=404, detail="Not found")
        return medicine

    @router.post("/explain", response_model=MedicineExplanationResult)
    async def explain_medicine(
        req: MedicineExplanationRequest,
        service: MedicineService = Depends(get_medicine_service),
    ):
        try:
            return await service.explain(req)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.post("/check-interactions", response_model=list[InteractionResult])
    async def check_interactions(
        medicine_ids: list[str],
        service: MedicineService = Depends(get_medicine_service),
    ):
        return await service.check_interactions(medicine_ids)

    return router
