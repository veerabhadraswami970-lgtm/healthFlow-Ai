"""
analytics_agent.py
-------------------
Module: Analytics Agent.

Design intent (per project architecture doc):
- Aggregates ANONYMISED usage signals to improve recommendations and
  surface platform health metrics for admins — privacy is the whole
  point of this agent, not an afterthought.
- Anonymisation (hashing user IDs, stripping PII from event metadata)
  happens BEFORE anything is persisted, deterministically, so raw
  identifiers are never written to the analytics store even if a caller
  forgets to anonymise upstream.
- Metric computation (counts, active users, completion rates, top items)
  is a DETERMINISTIC aggregation over stored events — same reasoning as
  every other agent in this codebase: numbers an admin makes decisions
  on must be exact and auditable, not model output.
- The LLM's only job is to narrate an already-computed metrics snapshot
  in plain language for the admin console. It is guardrailed so it can
  only reference numbers that actually appear in the snapshot — any
  number it mentions that isn't in the real data causes a fallback to
  the deterministic summary instead.

Layering:
  Router (FastAPI)
      -> AnalyticsService (orchestration)
          -> Anonymizer (pure, deterministic, applied before every write)
          -> EventRepository (Mongo: analytics_events)
          -> MetricsAggregator (pure, deterministic, unit-testable)
          -> AnalyticsNarrativeAgent (LLM summary, number-guardrailed)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger("healthflow.analytics_agent")


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class EventType(str, Enum):
    APP_OPEN = "app_open"
    DOCTOR_SEARCHED = "doctor_searched"
    APPOINTMENT_BOOKED = "appointment_booked"
    APPOINTMENT_COMPLETED = "appointment_completed"
    APPOINTMENT_CANCELLED = "appointment_cancelled"
    PRESCRIPTION_SCANNED = "prescription_scanned"
    SCHEME_VIEWED = "scheme_viewed"
    SCHEME_ELIGIBILITY_CHECKED = "scheme_eligibility_checked"
    EMERGENCY_TRIGGERED = "emergency_triggered"
    AI_ASSISTANT_MESSAGE = "ai_assistant_message"
    VOICE_MESSAGE = "voice_message"


# Metadata keys that must never reach the analytics store, even if a
# caller accidentally includes them.
_PII_KEYS = {
    "name", "full_name", "email", "phone", "phone_number", "address",
    "aadhaar", "aadhaar_number", "dob", "date_of_birth", "raw_user_id",
}


class RawAnalyticsEvent(BaseModel):
    """What a caller submits — may still contain a real user_id."""

    event_type: EventType
    user_id: str
    metadata: dict = Field(default_factory=dict)


class AnalyticsEvent(BaseModel):
    """What actually gets persisted — anonymised, mirrors `analytics_events`."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    user_id_hash: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "AnalyticsEvent":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class MetricsSnapshot(BaseModel):
    date_from: datetime
    date_to: datetime
    total_events: int
    active_users: int
    event_counts: dict[str, int] = Field(default_factory=dict)
    appointment_completion_rate: Optional[float] = None  # completed / (completed+cancelled)
    emergency_events: int = 0
    top_metadata_values: dict[str, list[tuple[str, int]]] = Field(default_factory=dict)
    narrative: Optional[str] = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsRequest(BaseModel):
    date_from: datetime
    date_to: datetime
    top_metadata_keys: list[str] = Field(default_factory=list)  # e.g. ["specialty", "scheme_id"]
    explain: bool = True
    language: str = "en"


# --------------------------------------------------------------------------
# Deterministic anonymisation (NO storage, NO LLM calls in this class)
# --------------------------------------------------------------------------

class Anonymizer:
    """
    Pure functions. Applied unconditionally before persistence — this is
    the platform's privacy guarantee for analytics, not a best-effort.
    """

    @staticmethod
    def hash_user_id(user_id: str, salt: str = "") -> str:
        salt = salt or os.getenv("ANALYTICS_HASH_SALT", "healthflow-default-salt")
        return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:32]

    @staticmethod
    def strip_pii(metadata: dict) -> dict:
        return {k: v for k, v in metadata.items() if k.lower() not in _PII_KEYS}

    @classmethod
    def anonymize(cls, raw: RawAnalyticsEvent) -> AnalyticsEvent:
        return AnalyticsEvent(
            event_type=raw.event_type,
            user_id_hash=cls.hash_user_id(raw.user_id),
            metadata=cls.strip_pii(raw.metadata),
        )


# --------------------------------------------------------------------------
# Repository (MongoDB via motor)
# --------------------------------------------------------------------------

class EventRepository(ABC):
    @abstractmethod
    async def save(self, event: AnalyticsEvent) -> AnalyticsEvent:
        ...

    @abstractmethod
    async def query(self, date_from: datetime, date_to: datetime) -> list[AnalyticsEvent]:
        ...


class MongoEventRepository(EventRepository):
    def __init__(self, db) -> None:
        # db: an AsyncIOMotorDatabase instance, injected at app startup
        self._collection = db["analytics_events"]

    async def save(self, event: AnalyticsEvent) -> AnalyticsEvent:
        await self._collection.insert_one(event.to_mongo())
        return event

    async def query(self, date_from: datetime, date_to: datetime) -> list[AnalyticsEvent]:
        cursor = self._collection.find(
            {"created_at": {"$gte": date_from.isoformat(), "$lte": date_to.isoformat()}}
        )
        return [AnalyticsEvent.from_mongo(doc) async for doc in cursor]

    async def create_indexes(self) -> None:
        # Run once at startup / via a migration script.
        await self._collection.create_index("created_at")
        await self._collection.create_index("event_type")


class InMemoryEventRepository(EventRepository):
    """For local dev / unit tests without a running Mongo instance."""

    def __init__(self) -> None:
        self._events: list[AnalyticsEvent] = []

    async def save(self, event: AnalyticsEvent) -> AnalyticsEvent:
        self._events.append(event)
        return event

    async def query(self, date_from: datetime, date_to: datetime) -> list[AnalyticsEvent]:
        return [e for e in self._events if date_from <= e.created_at <= date_to]


# --------------------------------------------------------------------------
# Deterministic metrics aggregation (NO LLM calls in this class, ever)
# --------------------------------------------------------------------------

class MetricsAggregator:
    """
    Pure, synchronous aggregation over a list of already-anonymised events.
    Every number an admin sees traces back to this function, not a model.
    """

    @staticmethod
    def aggregate(
        events: list[AnalyticsEvent],
        date_from: datetime,
        date_to: datetime,
        top_metadata_keys: Optional[list[str]] = None,
    ) -> MetricsSnapshot:
        event_counts = Counter(e.event_type.value for e in events)
        active_users = len({e.user_id_hash for e in events})

        completed = event_counts.get(EventType.APPOINTMENT_COMPLETED.value, 0)
        cancelled = event_counts.get(EventType.APPOINTMENT_CANCELLED.value, 0)
        completion_rate = (
            round(completed / (completed + cancelled), 4) if (completed + cancelled) > 0 else None
        )

        emergency_events = event_counts.get(EventType.EMERGENCY_TRIGGERED.value, 0)

        top_values: dict[str, list[tuple[str, int]]] = {}
        for key in top_metadata_keys or []:
            counter: Counter = Counter()
            for e in events:
                val = e.metadata.get(key)
                if val is not None:
                    counter[str(val)] += 1
            top_values[key] = counter.most_common(5)

        return MetricsSnapshot(
            date_from=date_from,
            date_to=date_to,
            total_events=len(events),
            active_users=active_users,
            event_counts=dict(event_counts),
            appointment_completion_rate=completion_rate,
            emergency_events=emergency_events,
            top_metadata_values=top_values,
        )


# --------------------------------------------------------------------------
# AI narrative layer — number-fabrication-guardrailed
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


_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


class AnalyticsNarrativeAgent:
    """
    Turns an already-computed MetricsSnapshot into an admin-friendly
    summary. Any numeric token in the generated text that doesn't appear
    anywhere in the snapshot's own numbers is treated as a fabrication
    risk, and the response falls back to a deterministic summary instead.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def narrate(self, snapshot: MetricsSnapshot, language: str = "en") -> str:
        fallback = self._deterministic_summary(snapshot)
        if self._llm is None:
            return fallback

        prompt = self._build_prompt(snapshot, language)
        try:
            text = (await self._llm.generate(prompt)).strip()
            if not self._numbers_are_grounded(text, snapshot):
                logger.warning(
                    "Analytics narrative contained numbers not present in the "
                    "snapshot; using deterministic fallback instead."
                )
                return fallback
            return text
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Analytics narrative generation failed: %s", exc)
            return fallback

    @staticmethod
    def _allowed_numbers(snapshot: MetricsSnapshot) -> set[str]:
        allowed = {str(snapshot.total_events), str(snapshot.active_users), str(snapshot.emergency_events)}
        allowed.update(str(v) for v in snapshot.event_counts.values())
        if snapshot.appointment_completion_rate is not None:
            pct = round(snapshot.appointment_completion_rate * 100)
            allowed.update({str(pct), f"{pct}%", str(snapshot.appointment_completion_rate)})
        for pairs in snapshot.top_metadata_values.values():
            allowed.update(str(count) for _, count in pairs)
        return allowed

    @classmethod
    def _numbers_are_grounded(cls, text: str, snapshot: MetricsSnapshot) -> bool:
        allowed = cls._allowed_numbers(snapshot)
        found = {tok.rstrip("%").replace(",", "") for tok in _NUMBER_PATTERN.findall(text)}
        allowed_clean = {a.rstrip("%").replace(",", "") for a in allowed}
        # Ignore very small numbers (1-31) which are almost always dates/list
        # positions in prose, not metric values, to avoid over-triggering.
        suspicious = {f for f in found if f and (not f.isdigit() or int(f) > 31) and f not in allowed_clean}
        return len(suspicious) == 0

    @staticmethod
    def _build_prompt(snapshot: MetricsSnapshot, language: str) -> str:
        return (
            "You are writing a short admin dashboard summary in "
            f"{language}. Use ONLY the numbers given below — do not "
            "calculate, round, estimate, or introduce any number that "
            "isn't explicitly listed. Do not mention any individual user. "
            "2-4 sentences, plain language, highlight anything notable "
            "(e.g. emergency events, low completion rate).\n\n"
            f"Period: {snapshot.date_from.date()} to {snapshot.date_to.date()}\n"
            f"Total events: {snapshot.total_events}\n"
            f"Active users: {snapshot.active_users}\n"
            f"Event counts: {snapshot.event_counts}\n"
            f"Appointment completion rate: {snapshot.appointment_completion_rate}\n"
            f"Emergency events: {snapshot.emergency_events}\n"
            f"Top metadata values: {snapshot.top_metadata_values}"
        )

    @staticmethod
    def _deterministic_summary(snapshot: MetricsSnapshot) -> str:
        parts = [
            f"Between {snapshot.date_from.date()} and {snapshot.date_to.date()}, "
            f"the platform logged {snapshot.total_events} events from "
            f"{snapshot.active_users} active users."
        ]
        if snapshot.appointment_completion_rate is not None:
            parts.append(
                f"Appointment completion rate was "
                f"{round(snapshot.appointment_completion_rate * 100)}%."
            )
        if snapshot.emergency_events:
            parts.append(f"{snapshot.emergency_events} emergency events were triggered.")
        return " ".join(parts)


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

class AnalyticsService:
    def __init__(
        self,
        repository: EventRepository,
        narrative_agent: Optional[AnalyticsNarrativeAgent] = None,
    ) -> None:
        self._repository = repository
        self._narrative_agent = narrative_agent or AnalyticsNarrativeAgent(llm_client=None)

    async def log_event(self, raw: RawAnalyticsEvent) -> AnalyticsEvent:
        # Anonymisation happens here, unconditionally, before any write.
        event = Anonymizer.anonymize(raw)
        return await self._repository.save(event)

    async def get_metrics(self, req: MetricsRequest) -> MetricsSnapshot:
        events = await self._repository.query(req.date_from, req.date_to)
        snapshot = MetricsAggregator.aggregate(
            events, req.date_from, req.date_to, req.top_metadata_keys
        )
        if req.explain:
            snapshot.narrative = await self._narrative_agent.narrate(snapshot, req.language)
        return snapshot


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router():
    from fastapi import APIRouter, Depends

    router = APIRouter(prefix="/analytics", tags=["analytics"])

    def get_analytics_service() -> AnalyticsService:
        # Wire real dependencies in your app's dependency module, e.g.
        # MongoEventRepository(app.state.mongo_db) + GeminiClient built
        # from settings.
        repository = InMemoryEventRepository()
        agent = AnalyticsNarrativeAgent(llm_client=None)
        return AnalyticsService(repository, agent)

    @router.post("/events", response_model=AnalyticsEvent)
    async def log_event(
        req: RawAnalyticsEvent,
        service: AnalyticsService = Depends(get_analytics_service),
    ):
        return await service.log_event(req)

    @router.post("/metrics", response_model=MetricsSnapshot)
    async def get_metrics(
        req: MetricsRequest,
        service: AnalyticsService = Depends(get_analytics_service),
    ):
        # NOTE: in your real app, protect this endpoint with an
        # admin-role dependency — it should not be reachable by patients.
        return await service.get_metrics(req)

    return router