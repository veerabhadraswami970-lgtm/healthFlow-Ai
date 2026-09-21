"""
health_assistant.py
--------------------
Module 01: AI Health Assistant Agent.

Design intent (per project architecture doc):
- Conversational assistant with text/voice input, multilingual, able to
  navigate the whole platform on the user's behalf.
- Answers healthcare information questions, but is clearly separated from
  professional medical advice — it must never present itself as a doctor
  or issue a diagnosis.
- Escalates urgent situations to real emergency services rather than
  attempting to handle them conversationally.
- Conversation history persisted (ai_conversations / ai_messages
  collections, per the doc's database design) so the assistant has
  context and admins have an audit trail.

Layering:
  Router (FastAPI)
      -> HealthAssistantService (orchestration)
          -> EmergencyDetector (pure, deterministic, checked BEFORE any LLM call)
          -> HealthAssistantLLM (conversational response, guardrailed)
          -> ConversationRepository (Mongo: ai_conversations + ai_messages)

Safety-critical ordering: the emergency detector runs first, on every
message, unconditionally. If it fires, the LLM is not consulted at all —
the response is a fixed, deterministic escalation message. This guarantees
escalation behaviour can't be talked out of by prompt injection, a model
having an off day, or a provider outage.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger("healthflow.health_assistant_agent")


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Mirrors the `ai_messages` MongoDB collection."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    role: MessageRole
    content: str
    is_emergency_escalation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "Message":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class Conversation(BaseModel):
    """Mirrors the `ai_conversations` MongoDB collection."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    language: str = "en"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    flagged_emergency: bool = False

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "Conversation":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class SendMessageRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None  # None = start a new conversation
    text: str
    language: str = "en"


class AssistantReply(BaseModel):
    conversation_id: str
    message: Message
    is_emergency: bool = False
    suggested_action: Optional[str] = None  # e.g. "open_emergency_flow", "open_doctor_search"


# --------------------------------------------------------------------------
# Deterministic emergency detector (NO LLM calls in this class, ever)
# --------------------------------------------------------------------------

class EmergencyDetector:
    """
    Pattern-based detection of urgent situations. Deliberately simple,
    conservative on recall (would rather over-trigger than miss a real
    emergency), and fully independent of the LLM so it can't be bypassed
    by prompt phrasing.

    This is NOT a clinical triage tool — it exists only to route the user
    to real emergency services/human help quickly, per the doc's
    "AI informs, never diagnoses, escalates urgent situations" principle.
    """

    _PATTERNS = [
        r"\bchest pain\b",
        r"\bcan'?t breathe\b|\bdifficulty breathing\b|\bshortness of breath\b",
        r"\bunconscious\b|\bnot responding\b|\bnot waking up\b",
        r"\bsevere bleeding\b|\bbleeding (a lot|heavily)\b",
        r"\bsuicid(e|al)\b|\bkill myself\b|\bend my life\b|\bwant to die\b",
        r"\bself[\s-]?harm\b|\bhurting myself\b",
        r"\bstroke\b|\bface (is )?drooping\b|\bslurred speech\b",
        r"\bheart attack\b",
        r"\bpoison(ed|ing)?\b|\boverdose\b",
        r"\bcan'?t move (my )?(arm|leg|body)\b",
        r"\bseizure\b|\bconvulsion\b",
        r"\bsevere allergic reaction\b|\banaphylaxis\b",
    ]
    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

    # Self-harm/suicide phrases route to a supportive crisis message rather
    # than the generic "go to hospital" emergency message.
    _CRISIS_PATTERNS = [
        re.compile(r"\bsuicid(e|al)\b|\bkill myself\b|\bend my life\b|\bwant to die\b", re.IGNORECASE),
        re.compile(r"\bself[\s-]?harm\b|\bhurting myself\b", re.IGNORECASE),
    ]

    @classmethod
    def check(cls, text: str) -> tuple[bool, bool]:
        """Returns (is_emergency, is_crisis) — crisis is a subtype of emergency."""
        is_crisis = any(p.search(text) for p in cls._CRISIS_PATTERNS)
        if is_crisis:
            return True, True
        is_emergency = any(p.search(text) for p in cls._COMPILED)
        return is_emergency, False


# --------------------------------------------------------------------------
# LLM conversational layer — guardrailed against diagnosis language
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


_DIAGNOSIS_PATTERN = re.compile(
    r"\byou (likely|probably|may) have\b|\bthis (suggests|indicates) you have\b"
    r"|\byour diagnosis\b|\byou are suffering from\b|\bi diagnose\b",
    re.IGNORECASE,
)

_FALLBACK_REPLY = (
    "I'm having trouble responding right now. I can still help you find a "
    "doctor, check a government scheme, or look up medicine information — "
    "or if this is urgent, please use the Emergency button."
)


class HealthAssistantLLM:
    """
    Generates the conversational reply once the emergency detector has
    already cleared the message. Kept diagnosis-guardrailed the same way
    as the medicine and recommendation agents.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def respond(
        self, history: list[Message], user_text: str, language: str
    ) -> str:
        if self._llm is None:
            return (
                "I can help you find doctors, hospitals, government schemes, "
                "or look up medicine information — what would you like to do?"
            )

        prompt = self._build_prompt(history, user_text, language)
        try:
            text = (await self._llm.generate(prompt)).strip()
            if _DIAGNOSIS_PATTERN.search(text):
                logger.warning("Assistant reply contained diagnosis-like language; using fallback.")
                return (
                    "I can't diagnose conditions, but I can help you find the right "
                    "doctor or explain general health information. Would you like me "
                    "to help you find a doctor for this?"
                )
            return text
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Health assistant generation failed: %s", exc)
            return _FALLBACK_REPLY

    @staticmethod
    def _build_prompt(history: list[Message], user_text: str, language: str) -> str:
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in history[-10:])
        return (
            "You are HealthFlow AI's health assistant. Respond in "
            f"{language}. STRICT RULES: you are an information tool, not a "
            "doctor — never diagnose a condition, never say what the user "
            "'might have', never prescribe or recommend a dosage. For medical "
            "concerns, help the user find/book a doctor instead of speculating. "
            "For platform tasks (finding doctors, hospitals, schemes, medicines, "
            "appointments), offer to help navigate there. Keep replies short "
            "and clear, suited to a general audience.\n\n"
            f"Conversation so far:\n{transcript}\n\n"
            f"user: {user_text}\nassistant:"
        )


# --------------------------------------------------------------------------
# Repository (MongoDB via motor) — ai_conversations + ai_messages
# --------------------------------------------------------------------------

class ConversationRepository(ABC):
    @abstractmethod
    async def get_or_create_conversation(
        self, user_id: str, conversation_id: Optional[str], language: str
    ) -> Conversation:
        ...

    @abstractmethod
    async def save_conversation(self, conversation: Conversation) -> Conversation:
        ...

    @abstractmethod
    async def save_message(self, message: Message) -> Message:
        ...

    @abstractmethod
    async def get_recent_messages(self, conversation_id: str, limit: int = 10) -> list[Message]:
        ...


class MongoConversationRepository(ConversationRepository):
    def __init__(self, db) -> None:
        # db: an AsyncIOMotorDatabase instance, injected at app startup
        self._conversations = db["ai_conversations"]
        self._messages = db["ai_messages"]

    async def get_or_create_conversation(
        self, user_id: str, conversation_id: Optional[str], language: str
    ) -> Conversation:
        if conversation_id:
            doc = await self._conversations.find_one({"_id": conversation_id})
            if doc:
                return Conversation.from_mongo(doc)
        conversation = Conversation(user_id=user_id, language=language)
        await self._conversations.insert_one(conversation.to_mongo())
        return conversation

    async def save_conversation(self, conversation: Conversation) -> Conversation:
        conversation.last_message_at = datetime.now(timezone.utc)
        await self._conversations.replace_one(
            {"_id": conversation.id}, conversation.to_mongo(), upsert=True
        )
        return conversation

    async def save_message(self, message: Message) -> Message:
        await self._messages.insert_one(message.to_mongo())
        return message

    async def get_recent_messages(self, conversation_id: str, limit: int = 10) -> list[Message]:
        cursor = (
            self._messages.find({"conversation_id": conversation_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        docs = [doc async for doc in cursor]
        return [Message.from_mongo(d) for d in reversed(docs)]


class InMemoryConversationRepository(ConversationRepository):
    """For local dev / unit tests without a running Mongo instance."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._messages: list[Message] = []

    async def get_or_create_conversation(
        self, user_id: str, conversation_id: Optional[str], language: str
    ) -> Conversation:
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        conversation = Conversation(user_id=user_id, language=language)
        self._conversations[conversation.id] = conversation
        return conversation

    async def save_conversation(self, conversation: Conversation) -> Conversation:
        conversation.last_message_at = datetime.now(timezone.utc)
        self._conversations[conversation.id] = conversation
        return conversation

    async def save_message(self, message: Message) -> Message:
        self._messages.append(message)
        return message

    async def get_recent_messages(self, conversation_id: str, limit: int = 10) -> list[Message]:
        matching = [m for m in self._messages if m.conversation_id == conversation_id]
        return matching[-limit:]


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

_EMERGENCY_REPLY = (
    "This sounds like it could be a medical emergency. Please use the "
    "Emergency button now to alert nearby hospitals and your emergency "
    "contacts, or call your local emergency number immediately. "
    "I'm not able to help with this conversationally — please get real "
    "help right away."
)

_CRISIS_REPLY = (
    "I'm really sorry you're going through this. You deserve support right "
    "now, and I want to make sure you get it from people who can really "
    "help. Please use the Emergency button, or reach out to a crisis "
    "helpline or someone you trust immediately. You don't have to go "
    "through this alone."
)


class HealthAssistantService:
    def __init__(
        self,
        repository: ConversationRepository,
        llm: Optional[HealthAssistantLLM] = None,
    ) -> None:
        self._repository = repository
        self._llm = llm or HealthAssistantLLM(llm_client=None)

    async def send_message(self, req: SendMessageRequest) -> AssistantReply:
        conversation = await self._repository.get_or_create_conversation(
            req.user_id, req.conversation_id, req.language
        )

        user_message = Message(
            conversation_id=conversation.id, role=MessageRole.USER, content=req.text
        )
        await self._repository.save_message(user_message)

        # Emergency check runs BEFORE anything else, unconditionally.
        is_emergency, is_crisis = EmergencyDetector.check(req.text)

        if is_emergency:
            reply_text = _CRISIS_REPLY if is_crisis else _EMERGENCY_REPLY
            assistant_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=reply_text,
                is_emergency_escalation=True,
            )
            await self._repository.save_message(assistant_message)
            conversation.flagged_emergency = True
            await self._repository.save_conversation(conversation)
            return AssistantReply(
                conversation_id=conversation.id,
                message=assistant_message,
                is_emergency=True,
                suggested_action="open_emergency_flow",
            )

        history = await self._repository.get_recent_messages(conversation.id, limit=10)
        reply_text = await self._llm.respond(history, req.text, req.language)

        assistant_message = Message(
            conversation_id=conversation.id, role=MessageRole.ASSISTANT, content=reply_text
        )
        await self._repository.save_message(assistant_message)
        await self._repository.save_conversation(conversation)

        return AssistantReply(conversation_id=conversation.id, message=assistant_message)


# --------------------------------------------------------------------------
# Module-level singletons
# --------------------------------------------------------------------------
# NOTE: created ONCE, at import time, and shared across every request.
# Previously each request built its own throwaway repository/LLM/service
# via the FastAPI Depends() function below, so conversation history never
# actually persisted between messages in the same conversation_id (every
# message effectively started a brand-new, memoryless conversation). Swap
# these two lines out for Mongo-backed equivalents when a real database is
# wired in.
_repository: ConversationRepository = InMemoryConversationRepository()
_llm = HealthAssistantLLM(llm_client=None)
_assistant_service = HealthAssistantService(_repository, _llm)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router():
    from fastapi import APIRouter, Depends

    router = APIRouter(prefix="/assistant", tags=["assistant"])

    def get_assistant_service() -> HealthAssistantService:
        return _assistant_service

    @router.post("/message", response_model=AssistantReply)
    async def send_message(
        req: SendMessageRequest,
        service: HealthAssistantService = Depends(get_assistant_service),
    ):
        return await service.send_message(req)

    return router
