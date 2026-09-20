"""
voice_assistant.py
-------------------
Module: Voice Assistant Agent (speech-to-text / text-to-speech).

Design intent (per project architecture doc):
- Speech-to-text / text-to-speech across supported languages, giving
  low-literacy and low-connectivity users a voice-first way into the
  platform.
- Same "never invent a value" principle as the Prescription OCR agent:
  a low-confidence transcription is NOT silently passed on as if it were
  understood â€” the user is asked to confirm or repeat instead.
- Audio input is validated deterministically (format, size, duration)
  BEFORE any paid STT call is made â€” cost control and abuse prevention,
  not a model decision.
- Once transcribed, the text is handed to the existing Health Assistant
  Agent (module 01) so emergency detection and conversation handling are
  not duplicated â€” this agent only adds the audio layer on top.
- Voice provider is pluggable (Whisper-style API, Google Speech, etc.)
  behind one interface, same as OCR/LLM providers in the other agents.

Layering:
  Router (FastAPI)
      -> VoiceAssistantService (orchestration)
          -> AudioValidator (pure, deterministic, checked before any provider call)
          -> VoiceProvider (STT + TTS, pluggable)
          -> HealthAssistantService (reused from health_assistant.py â€” no duplicated logic)
          -> VoiceInteractionRepository (Mongo, audit log of voice turns)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.agents.health_assistant import (
    HealthAssistantService,
    SendMessageRequest,
)

logger = logging.getLogger("healthflow.voice_assistant_agent")


# --------------------------------------------------------------------------
# Config / supported languages
# --------------------------------------------------------------------------

class SupportedLanguage(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    BENGALI = "bn"
    MARATHI = "mr"

    @classmethod
    def is_supported(cls, code: str) -> bool:
        return code in {lang.value for lang in cls}


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    WEBM = "webm"


# Deterministic, tunable limits â€” not model decisions.
MAX_AUDIO_BYTES = 10 * 1024 * 1024        # 10 MB
MAX_AUDIO_DURATION_SECONDS = 120           # 2 minutes per turn
MIN_TRANSCRIPTION_CONFIDENCE = 0.55        # below this, ask the user to repeat


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class VoiceMessageRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    audio_format: AudioFormat
    language_hint: SupportedLanguage = SupportedLanguage.ENGLISH
    audio_size_bytes: int
    audio_duration_seconds: float


class TranscriptionResult(BaseModel):
    text: Optional[str] = None
    confidence: float = 0.0
    detected_language: Optional[str] = None
    is_low_confidence: bool = True


class VoiceInteraction(BaseModel):
    """Mirrors a `voice_interactions` MongoDB collection â€” audit trail."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    conversation_id: Optional[str] = None
    audio_storage_ref: Optional[str] = None
    transcript: Optional[str] = None
    transcription_confidence: float = 0.0
    reply_text: Optional[str] = None
    reply_audio_storage_ref: Optional[str] = None
    is_emergency: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict:
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "VoiceInteraction":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class VoiceReply(BaseModel):
    conversation_id: Optional[str] = None
    transcript: Optional[str] = None
    needs_repeat: bool = False               # True = transcription too uncertain to act on
    reply_text: Optional[str] = None
    reply_audio_storage_ref: Optional[str] = None
    is_emergency: bool = False
    suggested_action: Optional[str] = None


# --------------------------------------------------------------------------
# Deterministic audio validation (NO provider/LLM calls in this class)
# --------------------------------------------------------------------------

class AudioValidationError(Exception):
    pass


class AudioValidator:
    """
    Pure checks run before spending money on an STT call. Kept separate
    and unit-testable, same reasoning as every other deterministic
    component in this codebase.
    """

    @staticmethod
    def validate(req: VoiceMessageRequest) -> None:
        if req.audio_size_bytes <= 0:
            raise AudioValidationError("Audio file is empty.")
        if req.audio_size_bytes > MAX_AUDIO_BYTES:
            raise AudioValidationError(
                f"Audio file exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)}MB limit."
            )
        if req.audio_duration_seconds <= 0:
            raise AudioValidationError("Audio duration could not be determined.")
        if req.audio_duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise AudioValidationError(
                f"Audio exceeds the {MAX_AUDIO_DURATION_SECONDS}-second limit per turn."
            )
        if not SupportedLanguage.is_supported(req.language_hint.value):
            raise AudioValidationError(f"Language '{req.language_hint.value}' is not supported yet.")


# --------------------------------------------------------------------------
# Voice provider interface (STT + TTS, pluggable)
# --------------------------------------------------------------------------

class VoiceProvider(ABC):
    @abstractmethod
    async def transcribe(
        self, audio_ref: str, language_hint: SupportedLanguage
    ) -> TranscriptionResult:
        ...

    @abstractmethod
    async def synthesize(self, text: str, language: SupportedLanguage) -> str:
        """Returns a storage ref/path to the generated audio."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class WhisperVoiceProvider(VoiceProvider):
    """
    Example hosted-API provider (OpenAI Whisper-compatible endpoint, or any
    similar STT/TTS API). Kept as a thin adapter â€” swap for Google Speech,
    Azure, etc. without touching the service layer.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("VOICE_PROVIDER_API_KEY")
        if not self._api_key:
            raise ValueError("VOICE_PROVIDER_API_KEY is not set")

    @property
    def name(self) -> str:
        return "whisper"

    async def transcribe(
        self, audio_ref: str, language_hint: SupportedLanguage
    ) -> TranscriptionResult:
        # Real implementation: fetch audio bytes from storage using audio_ref,
        # POST to the STT endpoint, map its confidence/segments into
        # TranscriptionResult. Left as an integration point.
        raise NotImplementedError("Wire this up to your chosen STT API.")

    async def synthesize(self, text: str, language: SupportedLanguage) -> str:
        # Real implementation: POST to a TTS endpoint, upload the resulting
        # audio to storage, return its ref/path.
        raise NotImplementedError("Wire this up to your chosen TTS API.")


class MockVoiceProvider(VoiceProvider):
    """For local dev / tests without a real STT/TTS integration."""

    def __init__(self, fixed_text: str = "", fixed_confidence: float = 0.9) -> None:
        self._text = fixed_text
        self._confidence = fixed_confidence

    @property
    def name(self) -> str:
        return "mock"

    async def transcribe(
        self, audio_ref: str, language_hint: SupportedLanguage
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=self._text or None,
            confidence=self._confidence,
            detected_language=language_hint.value,
            is_low_confidence=self._confidence < MIN_TRANSCRIPTION_CONFIDENCE,
        )

    async def synthesize(self, text: str, language: SupportedLanguage) -> str:
        return f"mock-audio/{uuid4()}.mp3"


# --------------------------------------------------------------------------
# Repository (MongoDB via motor)
# --------------------------------------------------------------------------

class VoiceInteractionRepository(ABC):
    @abstractmethod
    async def save(self, interaction: VoiceInteraction) -> VoiceInteraction:
        ...


class MongoVoiceInteractionRepository(VoiceInteractionRepository):
    def __init__(self, db) -> None:
        # db: an AsyncIOMotorDatabase instance, injected at app startup
        self._collection = db["voice_interactions"]

    async def save(self, interaction: VoiceInteraction) -> VoiceInteraction:
        await self._collection.insert_one(interaction.to_mongo())
        return interaction


class InMemoryVoiceInteractionRepository(VoiceInteractionRepository):
    def __init__(self) -> None:
        self.store: list[VoiceInteraction] = []

    async def save(self, interaction: VoiceInteraction) -> VoiceInteraction:
        self.store.append(interaction)
        return interaction


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

class VoiceAssistantService:
    def __init__(
        self,
        voice_provider: VoiceProvider,
        health_assistant: HealthAssistantService,
        repository: Optional[VoiceInteractionRepository] = None,
    ) -> None:
        self._voice = voice_provider
        self._assistant = health_assistant
        self._repository = repository or InMemoryVoiceInteractionRepository()

    async def handle_voice_message(
        self, req: VoiceMessageRequest, audio_storage_ref: str
    ) -> VoiceReply:
        # 1. Deterministic validation, before any provider call.
        AudioValidator.validate(req)

        # 2. Speech-to-text.
        transcription = await self._voice.transcribe(audio_storage_ref, req.language_hint)

        # 3. Never act on a low-confidence or empty transcript â€” ask the
        #    user to repeat instead of guessing what they meant.
        if not transcription.text or transcription.is_low_confidence:
            reply = VoiceReply(
                conversation_id=req.conversation_id,
                transcript=transcription.text,
                needs_repeat=True,
                reply_text=(
                    "Sorry, I didn't catch that clearly. Could you please repeat "
                    "that, or try typing it instead?"
                ),
            )
            await self._repository.save(
                VoiceInteraction(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    audio_storage_ref=audio_storage_ref,
                    transcript=transcription.text,
                    transcription_confidence=transcription.confidence,
                    reply_text=reply.reply_text,
                )
            )
            return reply

        # 4. Hand off to the existing Health Assistant Agent â€” this is
        #    where emergency detection and conversational logic live;
        #    nothing is duplicated here.
        assistant_reply = await self._assistant.send_message(
            SendMessageRequest(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                text=transcription.text,
                language=req.language_hint.value,
            )
        )

        # 5. Text-to-speech for the reply.
        reply_audio_ref = await self._voice.synthesize(
            assistant_reply.message.content, req.language_hint
        )

        interaction = VoiceInteraction(
            user_id=req.user_id,
            conversation_id=assistant_reply.conversation_id,
            audio_storage_ref=audio_storage_ref,
            transcript=transcription.text,
            transcription_confidence=transcription.confidence,
            reply_text=assistant_reply.message.content,
            reply_audio_storage_ref=reply_audio_ref,
            is_emergency=assistant_reply.is_emergency,
        )
        await self._repository.save(interaction)

        return VoiceReply(
            conversation_id=assistant_reply.conversation_id,
            transcript=transcription.text,
            needs_repeat=False,
            reply_text=assistant_reply.message.content,
            reply_audio_storage_ref=reply_audio_ref,
            is_emergency=assistant_reply.is_emergency,
            suggested_action=assistant_reply.suggested_action,
        )


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router():
    from fastapi import APIRouter, Depends, HTTPException

    from backend.agents.health_assistant import (
        HealthAssistantLLM,
        InMemoryConversationRepository,
    )

    router = APIRouter(prefix="/voice", tags=["voice"])

    def get_voice_service() -> VoiceAssistantService:
        # Wire real dependencies in your app's dependency module, e.g.
        # a real STT/TTS provider + MongoConversationRepository +
        # MongoVoiceInteractionRepository built from app.state.mongo_db.
        voice_provider = MockVoiceProvider()
        health_assistant = HealthAssistantService(
            InMemoryConversationRepository(), HealthAssistantLLM(llm_client=None)
        )
        repository = InMemoryVoiceInteractionRepository()
        return VoiceAssistantService(voice_provider, health_assistant, repository)

    @router.post("/message", response_model=VoiceReply)
    async def send_voice_message(
        req: VoiceMessageRequest,
        audio_storage_ref: str,
        service: VoiceAssistantService = Depends(get_voice_service),
    ):
        try:
            return await service.handle_voice_message(req, audio_storage_ref)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return router
