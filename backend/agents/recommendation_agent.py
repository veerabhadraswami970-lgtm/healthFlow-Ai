"""
recommendation_agent.py
------------------------
Module 12: AI Recommendation Engine.

Design intent (per project architecture doc):
- Combines intent, location, preferences and available providers to
  suggest relevant doctors, hospitals and schemes — WITHOUT making
  unsupported medical diagnoses.
- Ranking is a DETERMINISTIC weighted score (distance, specialty match,
  verification status, rating, availability, preference fit) — same
  reasoning as the other agents: a "why was this ranked first" question
  must be answerable from auditable numbers, not model output.
- The LLM's only job is to explain, in plain language, why the already-
  ranked list looks the way it does. It is explicitly instructed never to
  suggest a diagnosis or condition, and a keyword guard catches obvious
  slips and falls back to a deterministic explanation instead.
- Repository is Mongo-backed (motor, async), same pattern as the other
  three agents.

Layering:
  Router (FastAPI)
      -> RecommendationService (orchestration)
          -> RecommendationRanker (pure, deterministic, unit-testable)
          -> RecommendationExplanationAgent (LLM plain-language summary, guardrailed)
          -> CandidateRepository (Mongo, doctors/hospitals/schemes)
"""

from __future__ import annotations

import logging
import math
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("healthflow.recommendation_agent")


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class CandidateType(str, Enum):
    DOCTOR = "doctor"
    HOSPITAL = "hospital"
    SCHEME = "scheme"


class ConsultationType(str, Enum):
    IN_PERSON = "in_person"
    ONLINE = "online"
    EITHER = "either"


class Candidate(BaseModel):
    """A single doctor / hospital / scheme available for ranking."""

    id: str
    type: CandidateType
    name: str
    specialties: list[str] = Field(default_factory=list)  # or scheme categories
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_verified: bool = False
    rating: Optional[float] = None  # 0-5
    has_open_slots: Optional[bool] = None  # doctors only; None = not applicable
    consultation_type: Optional[ConsultationType] = None  # doctors only


class UserPreferences(BaseModel):
    max_distance_km: Optional[float] = None
    preferred_consultation_type: Optional[ConsultationType] = None
    min_rating: Optional[float] = None
    verified_only: bool = False


class RecommendationRequest(BaseModel):
    patient_id: str
    intent: str  # free-text, e.g. "find a cardiologist", used for the explanation only
    candidate_type: CandidateType
    specialty_or_category: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    language: str = "en"
    limit: int = 10


class ScoredCandidate(BaseModel):
    candidate: Candidate
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    distance_km: Optional[float] = None
    explanation: Optional[str] = None


# --------------------------------------------------------------------------
# Repository (MongoDB via motor)
# --------------------------------------------------------------------------

class CandidateRepository(ABC):
    @abstractmethod
    async def find_candidates(
        self, candidate_type: CandidateType, specialty_or_category: Optional[str]
    ) -> list[Candidate]:
        ...


class MongoCandidateRepository(CandidateRepository):
    """
    Reads from the appropriate collection depending on candidate type.
    Kept thin: no scoring or business logic lives here.
    """

    def __init__(self, db) -> None:
        self._db = db
        self._collection_by_type = {
            CandidateType.DOCTOR: "doctors",
            CandidateType.HOSPITAL: "hospitals",
            CandidateType.SCHEME: "schemes",
        }

    async def find_candidates(
        self, candidate_type: CandidateType, specialty_or_category: Optional[str]
    ) -> list[Candidate]:
        collection = self._db[self._collection_by_type[candidate_type]]
        query: dict = {}
        if specialty_or_category:
            query["specialties"] = specialty_or_category
        cursor = collection.find(query)
        docs = [doc async for doc in cursor]
        return [self._to_candidate(doc, candidate_type) for doc in docs]

    @staticmethod
    def _to_candidate(doc: dict, candidate_type: CandidateType) -> Candidate:
        return Candidate(
            id=str(doc.get("_id")),
            type=candidate_type,
            name=doc.get("name", ""),
            specialties=doc.get("specialties", []),
            latitude=doc.get("latitude"),
            longitude=doc.get("longitude"),
            is_verified=doc.get("is_verified", False),
            rating=doc.get("rating"),
            has_open_slots=doc.get("has_open_slots"),
            consultation_type=doc.get("consultation_type"),
        )


class InMemoryCandidateRepository(CandidateRepository):
    """For local dev / unit tests without a running Mongo instance."""

    def __init__(self, candidates: list[Candidate] | None = None) -> None:
        self._candidates = candidates or []

    async def find_candidates(
        self, candidate_type: CandidateType, specialty_or_category: Optional[str]
    ) -> list[Candidate]:
        results = [c for c in self._candidates if c.type == candidate_type]
        if specialty_or_category:
            results = [c for c in results if specialty_or_category in c.specialties]
        return results


# --------------------------------------------------------------------------
# Deterministic ranking engine (NO LLM calls in this class, ever)
# --------------------------------------------------------------------------

class RecommendationRanker:
    """
    Pure, synchronous, side-effect-free scoring. Weights are explicit and
    tunable constants — auditable and unit-testable without mocking a model.
    """

    # Tunable weights; keep them summing to something sane (~1.0) for
    # readability, though the ranker only needs relative ordering.
    WEIGHT_SPECIALTY_MATCH = 0.30
    WEIGHT_DISTANCE = 0.25
    WEIGHT_VERIFIED = 0.15
    WEIGHT_RATING = 0.15
    WEIGHT_AVAILABILITY = 0.10
    WEIGHT_PREFERENCE_FIT = 0.05

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    @classmethod
    def score(
        cls, candidate: Candidate, req: RecommendationRequest
    ) -> Optional[ScoredCandidate]:
        prefs = req.preferences
        breakdown: dict[str, float] = {}
        distance_km: Optional[float] = None

        # Hard filters — candidates failing these are excluded entirely
        if prefs.verified_only and not candidate.is_verified:
            return None
        if prefs.min_rating is not None and (candidate.rating or 0) < prefs.min_rating:
            return None

        # Specialty / category match
        if req.specialty_or_category:
            match = req.specialty_or_category.lower() in [
                s.lower() for s in candidate.specialties
            ]
            breakdown["specialty_match"] = cls.WEIGHT_SPECIALTY_MATCH if match else 0.0
        else:
            breakdown["specialty_match"] = cls.WEIGHT_SPECIALTY_MATCH * 0.5  # neutral

        # Distance
        if (
            req.latitude is not None and req.longitude is not None
            and candidate.latitude is not None and candidate.longitude is not None
        ):
            distance_km = cls._haversine_km(
                req.latitude, req.longitude, candidate.latitude, candidate.longitude
            )
            if prefs.max_distance_km is not None and distance_km > prefs.max_distance_km:
                return None  # hard filter: out of range
            # Closer = higher score; 0km -> full weight, 50km+ -> ~0
            distance_score = max(0.0, 1 - (distance_km / 50.0))
            breakdown["distance"] = cls.WEIGHT_DISTANCE * distance_score
        else:
            breakdown["distance"] = cls.WEIGHT_DISTANCE * 0.5  # neutral if unknown

        # Verification
        breakdown["verified"] = cls.WEIGHT_VERIFIED if candidate.is_verified else 0.0

        # Rating (normalised 0-5 -> 0-1)
        rating_score = (candidate.rating or 0) / 5.0
        breakdown["rating"] = cls.WEIGHT_RATING * rating_score

        # Availability (doctors only; neutral for hospitals/schemes)
        if candidate.has_open_slots is None:
            breakdown["availability"] = cls.WEIGHT_AVAILABILITY * 0.5
        else:
            breakdown["availability"] = cls.WEIGHT_AVAILABILITY * (
                1.0 if candidate.has_open_slots else 0.0
            )

        # Preference fit (consultation type)
        if prefs.preferred_consultation_type and candidate.consultation_type:
            fits = (
                candidate.consultation_type == prefs.preferred_consultation_type
                or candidate.consultation_type == ConsultationType.EITHER
            )
            breakdown["preference_fit"] = cls.WEIGHT_PREFERENCE_FIT if fits else 0.0
        else:
            breakdown["preference_fit"] = cls.WEIGHT_PREFERENCE_FIT * 0.5

        total = round(sum(breakdown.values()), 4)
        return ScoredCandidate(
            candidate=candidate, score=total, score_breakdown=breakdown, distance_km=distance_km
        )

    @classmethod
    def rank(
        cls, candidates: list[Candidate], req: RecommendationRequest
    ) -> list[ScoredCandidate]:
        scored = [cls.score(c, req) for c in candidates]
        scored = [s for s in scored if s is not None]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: req.limit]


# --------------------------------------------------------------------------
# AI explanation layer — plain-language "why", diagnosis-guardrailed
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


# Heuristic guard: catches obvious diagnostic-sounding phrasing that may slip
# through despite prompt instructions. Safety net, not the primary control.
_DIAGNOSIS_PATTERN = re.compile(
    r"\byou (likely|probably|may) have\b|\bthis (suggests|indicates) you have\b"
    r"|\byour diagnosis\b|\byou are suffering from\b",
    re.IGNORECASE,
)


class RecommendationExplanationAgent:
    """
    Explains why a ranked list looks the way it does, in plain language.
    Never asked to interpret symptoms or suggest a condition — only to
    describe the ranking factors (specialty match, distance, rating,
    verification, availability) that produced the order.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def explain(
        self, results: list[ScoredCandidate], req: RecommendationRequest
    ) -> list[ScoredCandidate]:
        if self._llm is None:
            for r in results:
                r.explanation = self._deterministic_explanation(r)
            return results

        for r in results:
            fallback = self._deterministic_explanation(r)
            prompt = self._build_prompt(r, req)
            try:
                text = (await self._llm.generate(prompt)).strip()
                if _DIAGNOSIS_PATTERN.search(text):
                    logger.warning(
                        "Recommendation explanation for %s contained diagnostic-like "
                        "language; using deterministic fallback instead.", r.candidate.id
                    )
                    text = fallback
                r.explanation = text
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning("Recommendation explanation generation failed: %s", exc)
                r.explanation = fallback
        return results

    @staticmethod
    def _build_prompt(result: ScoredCandidate, req: RecommendationRequest) -> str:
        c = result.candidate
        return (
            "You are a healthcare navigation assistant. Explain in "
            f"{req.language}, in 1-2 short sentences, why this option was "
            "recommended — based ONLY on the ranking factors given below "
            "(specialty match, distance, rating, verification, availability). "
            "Do NOT interpret symptoms, suggest a condition, or say what the "
            "user 'might have'. Do not invent facts not given below.\n\n"
            f"User intent (context only): {req.intent}\n"
            f"Option: {c.name} ({c.type.value})\n"
            f"Specialty/category match: {'yes' if result.score_breakdown.get('specialty_match', 0) > 0 else 'partial/unknown'}\n"
            f"Distance: {f'{result.distance_km:.1f} km' if result.distance_km is not None else 'unknown'}\n"
            f"Rating: {c.rating if c.rating is not None else 'not rated'}\n"
            f"Verified: {c.is_verified}\n"
        )

    @staticmethod
    def _deterministic_explanation(result: ScoredCandidate) -> str:
        c = result.candidate
        bits = []
        if result.score_breakdown.get("specialty_match", 0) > 0:
            bits.append("matches the specialty/category you're looking for")
        if result.distance_km is not None:
            bits.append(f"is about {result.distance_km:.1f} km away")
        if c.is_verified:
            bits.append("is a verified provider")
        if c.rating:
            bits.append(f"has a rating of {c.rating}/5")
        if not bits:
            bits.append("met the basic criteria for this search")
        return f"{c.name} " + ", ".join(bits) + "."


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

class RecommendationService:
    def __init__(
        self,
        repository: CandidateRepository,
        agent: Optional[RecommendationExplanationAgent] = None,
    ) -> None:
        self._repository = repository
        self._agent = agent or RecommendationExplanationAgent(llm_client=None)

    async def recommend(self, req: RecommendationRequest) -> list[ScoredCandidate]:
        candidates = await self._repository.find_candidates(
            req.candidate_type, req.specialty_or_category
        )
        ranked = RecommendationRanker.rank(candidates, req)
        return await self._agent.explain(ranked, req)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

_repository = InMemoryCandidateRepository()
_agent = RecommendationExplanationAgent(llm_client=None)
_recommendation_service = RecommendationService(_repository, _agent)


def build_router():
    from fastapi import APIRouter, Depends

    router = APIRouter(prefix="/recommendations", tags=["recommendations"])

    def get_recommendation_service() -> RecommendationService:
        return _recommendation_service

    @router.post("/", response_model=list[ScoredCandidate])
    async def get_recommendations(
        req: RecommendationRequest,
        service: RecommendationService = Depends(get_recommendation_service),
    ):
        return await service.recommend(req)

    return router
