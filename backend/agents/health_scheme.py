"""
health_scheme.py
-----------------
Module 02: Government Health Scheme Intelligence.

Design intent (per project architecture doc):
- Eligibility is decided ONLY by a deterministic, auditable rule engine.
  The LLM never guesses eligibility.
- The AI agent's only job is to explain an already-computed rule result
  in plain, multilingual-friendly language.
- Sits behind a repository interface so Firestore can be swapped for
  any other store later without touching business logic.
- Async throughout so it composes cleanly with FastAPI + Celery/APScheduler.

Layering:
  Router (FastAPI)
      -> SchemeService (orchestration)
          -> SchemeEligibilityEngine (pure, deterministic, unit-testable)
          -> SchemeIntelligenceAgent (LLM explanation, optional/soft-fail)
          -> SchemeRepository (data access, swappable backend)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("healthflow.scheme_agent")


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class SchemeCategory(str, Enum):
    MATERNAL = "maternal"
    CHILD = "child"
    SENIOR_CITIZEN = "senior_citizen"
    DISABILITY = "disability"
    GENERAL = "general"
    CRITICAL_ILLNESS = "critical_illness"


class Scheme(BaseModel):
    """Mirrors the `schemes` Firestore collection."""

    id: str
    name: str
    state: str  # "ALL" for central/national schemes
    category: SchemeCategory
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    max_annual_income: Optional[int] = None  # in INR, None = no income cap
    eligible_conditions: list[str] = Field(default_factory=list)  # empty = any
    benefits: str
    required_documents: list[str] = Field(default_factory=list)
    application_steps: list[str] = Field(default_factory=list)
    is_active: bool = True


class EligibilityCheckRequest(BaseModel):
    age: int = Field(..., ge=0, le=130)
    state: str
    annual_income: Optional[int] = Field(None, ge=0)
    condition: Optional[str] = None
    category_hint: Optional[SchemeCategory] = None
    language: str = "en"  # for the AI explanation, not the rule engine

    @field_validator("state")
    @classmethod
    def normalise_state(cls, v: str) -> str:
        return v.strip().title()


class RuleOutcome(str, Enum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_REVIEW = "needs_review"  # missing data prevents a clean decision


class SchemeEligibilityResult(BaseModel):
    """
    Pydantic (not a plain dataclass) so it serialises cleanly as a
    FastAPI response_model without extra adapters.
    """

    scheme: Scheme
    outcome: RuleOutcome
    reasons: list[str] = Field(default_factory=list)
    explanation: Optional[str] = None  # filled in by the AI agent, may be None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Repository interface (swap Firestore for anything else without changing
# the service layer above it)
# --------------------------------------------------------------------------

class SchemeRepository(ABC):
    @abstractmethod
    async def list_active_schemes(self, state: str | None = None) -> list[Scheme]:
        ...

    @abstractmethod
    async def get_scheme(self, scheme_id: str) -> Optional[Scheme]:
        ...


class FirestoreSchemeRepository(SchemeRepository):
    """
    Firestore-backed implementation. Kept thin: no business logic lives
    here, only data access, so it can be swapped for Postgres/Mongo later.
    """

    def __init__(self, firestore_client) -> None:
        self._db = firestore_client
        self._collection = "schemes"

    async def list_active_schemes(self, state: str | None = None) -> list[Scheme]:
        query = self._db.collection(self._collection).where("is_active", "==", True)
        if state:
            # Schemes stored with state == "ALL" are national/central schemes
            query = query.where("state", "in", [state, "ALL"])
        docs = [doc.to_dict() async for doc in query.stream()]
        return [Scheme(**doc) for doc in docs]

    async def get_scheme(self, scheme_id: str) -> Optional[Scheme]:
        doc = await self._db.collection(self._collection).document(scheme_id).get()
        return Scheme(**doc.to_dict()) if doc.exists else None


class InMemorySchemeRepository(SchemeRepository):
    """Useful for local dev, unit tests, and demos without Firebase set up."""

    def __init__(self, schemes: list[Scheme] | None = None) -> None:
        self._schemes = {s.id: s for s in (schemes or [])}

    async def list_active_schemes(self, state: str | None = None) -> list[Scheme]:
        result = [s for s in self._schemes.values() if s.is_active]
        if state:
            result = [s for s in result if s.state in (state, "ALL")]
        return result

    async def get_scheme(self, scheme_id: str) -> Optional[Scheme]:
        return self._schemes.get(scheme_id)


# --------------------------------------------------------------------------
# Deterministic eligibility engine (NO LLM calls in this class, ever)
# --------------------------------------------------------------------------

class SchemeEligibilityEngine:
    """
    Pure, synchronous, side-effect-free rule evaluation.
    Kept deliberately separate from the AI agent so eligibility logic can
    be unit tested without mocking a model, and audited independently.
    """

    @staticmethod
    def evaluate(scheme: Scheme, req: EligibilityCheckRequest) -> SchemeEligibilityResult:
        reasons: list[str] = []
        outcome = RuleOutcome.ELIGIBLE

        # State check
        if scheme.state not in (req.state, "ALL"):
            reasons.append(f"Scheme is only available in {scheme.state}.")
            outcome = RuleOutcome.NOT_ELIGIBLE

        # Age check
        if scheme.min_age is not None and req.age < scheme.min_age:
            reasons.append(f"Minimum age required is {scheme.min_age}.")
            outcome = RuleOutcome.NOT_ELIGIBLE
        if scheme.max_age is not None and req.age > scheme.max_age:
            reasons.append(f"Maximum age allowed is {scheme.max_age}.")
            outcome = RuleOutcome.NOT_ELIGIBLE

        # Income check
        if scheme.max_annual_income is not None:
            if req.annual_income is None:
                reasons.append("Annual income was not provided; cannot confirm the income limit.")
                if outcome == RuleOutcome.ELIGIBLE:
                    outcome = RuleOutcome.NEEDS_REVIEW
            elif req.annual_income > scheme.max_annual_income:
                reasons.append(
                    f"Annual income exceeds the scheme's limit of ₹{scheme.max_annual_income:,}."
                )
                outcome = RuleOutcome.NOT_ELIGIBLE

        # Condition check
        if scheme.eligible_conditions:
            if not req.condition:
                reasons.append("A medical condition was not provided; cannot confirm condition match.")
                if outcome == RuleOutcome.ELIGIBLE:
                    outcome = RuleOutcome.NEEDS_REVIEW
            elif req.condition.strip().lower() not in [c.lower() for c in scheme.eligible_conditions]:
                reasons.append("Condition is not covered by this scheme.")
                outcome = RuleOutcome.NOT_ELIGIBLE

        if outcome == RuleOutcome.ELIGIBLE and not reasons:
            reasons.append("All eligibility criteria are satisfied.")

        return SchemeEligibilityResult(scheme=scheme, outcome=outcome, reasons=reasons)

    @classmethod
    def evaluate_many(
        cls, schemes: list[Scheme], req: EligibilityCheckRequest
    ) -> list[SchemeEligibilityResult]:
        results = [cls.evaluate(s, req) for s in schemes]
        # Eligible first, then needs-review, then not-eligible
        order = {RuleOutcome.ELIGIBLE: 0, RuleOutcome.NEEDS_REVIEW: 1, RuleOutcome.NOT_ELIGIBLE: 2}
        return sorted(results, key=lambda r: order[r.outcome])


# --------------------------------------------------------------------------
# AI explanation layer — strictly explanatory, never decisional
# --------------------------------------------------------------------------

class LLMClient(ABC):
    """Minimal interface so the provider (Gemini, OpenAI, etc.) is swappable."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        ...


class GeminiClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash") -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._model_name = model
        # Import kept local so this module doesn't hard-require the SDK
        # unless a Gemini client is actually instantiated.
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(self._model_name)

    async def generate(self, prompt: str) -> str:
        response = await self._model.generate_content_async(prompt)
        return response.text


class SchemeIntelligenceAgent:
    """
    Explains an ALREADY-COMPUTED rule engine result in plain language.
    Never asked to decide eligibility itself. If the LLM call fails or
    is unavailable, the raw rule-engine reasons are still returned, so
    the feature degrades gracefully instead of breaking.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def explain(
        self, result: SchemeEligibilityResult, language: str = "en"
    ) -> SchemeEligibilityResult:
        if self._llm is None:
            logger.info("No LLM client configured; returning rule-engine reasons as-is.")
            return result

        prompt = self._build_prompt(result, language)
        try:
            explanation = await self._llm.generate(prompt)
            result.explanation = explanation.strip()
        except Exception as exc:  # noqa: BLE001 - must never break the eligibility response
            logger.warning("Scheme explanation generation failed: %s", exc)
            result.explanation = None
        return result

    @staticmethod
    def _build_prompt(result: SchemeEligibilityResult, language: str) -> str:
        reasons_text = "; ".join(result.reasons)
        return (
            "You are a healthcare scheme assistant. You must NOT change or "
            "re-decide the eligibility outcome below — only explain it clearly "
            f"in {language}, in 2-3 short sentences, for someone with low health "
            "literacy. Be warm and factual. Do not invent any new eligibility "
            "criteria, documents, or benefits beyond what is given.\n\n"
            f"Scheme: {result.scheme.name}\n"
            f"Outcome: {result.outcome.value}\n"
            f"Rule engine reasons: {reasons_text}\n"
            f"Benefits (for context only, do not restate as a decision): "
            f"{result.scheme.benefits}"
        )


# --------------------------------------------------------------------------
# Orchestration service — what the FastAPI router actually calls
# --------------------------------------------------------------------------

class SchemeService:
    def __init__(
        self,
        repository: SchemeRepository,
        agent: Optional[SchemeIntelligenceAgent] = None,
    ) -> None:
        self._repository = repository
        self._agent = agent or SchemeIntelligenceAgent(llm_client=None)

    async def check_eligibility(
        self, req: EligibilityCheckRequest, explain: bool = True
    ) -> list[SchemeEligibilityResult]:
        schemes = await self._repository.list_active_schemes(state=req.state)
        results = SchemeEligibilityEngine.evaluate_many(schemes, req)

        if explain:
            # Only explain the top few results to control latency/cost —
            # this is the kind of scaling knob mentioned in your roadmap.
            top_n = results[:5]
            for r in top_n:
                await self._agent.explain(r, language=req.language)

        return results

    async def list_schemes(self, state: str | None = None) -> list[Scheme]:
        return await self._repository.list_active_schemes(state=state)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router():
    """
    Factory instead of a module-level router so dependencies (repository,
    LLM client) can be injected per environment — same DI approach used
    across the rest of the service layer.
    """
    from fastapi import APIRouter, Depends

    router = APIRouter(prefix="/schemes", tags=["schemes"])

    def get_scheme_service() -> SchemeService:
        # Wire real dependencies here in your app's dependency module,
        # e.g. Firestore client + GeminiClient built from settings.
        repository = InMemorySchemeRepository()
        agent = SchemeIntelligenceAgent(llm_client=None)
        return SchemeService(repository=repository, agent=agent)

    @router.get("/", response_model=list[Scheme])
    async def list_schemes(
        state: str | None = None,
        service: SchemeService = Depends(get_scheme_service),
    ):
        return await service.list_schemes(state=state)

    @router.post("/check-eligibility", response_model=list[SchemeEligibilityResult])
    async def check_eligibility(
        req: EligibilityCheckRequest,
        explain: bool = True,
        service: SchemeService = Depends(get_scheme_service),
    ):
        return await service.check_eligibility(req, explain=explain)

    return router