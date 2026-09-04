"""
prescription_ocr.py
--------------------
Module 03: AI Prescription Reader (Prescription OCR Agent).

Design intent (per project architecture doc):
- Flow: image -> OCR -> structured fields -> confidence per field -> user
  verification -> only THEN saved as a real prescription record.
- The engine NEVER invents a value for a field it couldn't read. Missing
  or low-confidence fields are surfaced to the user, not guessed.
- OCR provider is pluggable (Tesseract locally, Google Vision / other
  hosted OCR in production) behind one interface.
- The LLM's only job is to turn messy OCR text into structured candidate
  fields — it does not get to invalidate the confidence scores, and every
  field it returns is still subject to the same "no invented value" rule.
- Repository is Mongo-backed (motor, async), swappable like the scheme
  agent's repository.

Layering:
  Router (FastAPI)
      -> PrescriptionService (orchestration)
          -> OCRProvider (raw image -> raw text + provider confidence)
          -> PrescriptionExtractionAgent (raw text -> structured MedicineEntry list)
          -> PrescriptionRepository (Mongo, draft vs verified state)
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

logger = logging.getLogger("healthflow.prescription_ocr_agent")


# --------------------------------------------------------------------------
# Domain models
# --------------------------------------------------------------------------

class PrescriptionStatus(str, Enum):
    DRAFT = "draft"                 # freshly OCR'd, not yet confirmed by user
    PENDING_REVIEW = "pending_review"  # has at least one low-confidence field
    VERIFIED = "verified"           # user confirmed/corrected all fields
    DISCARDED = "discarded"


class FieldConfidence(str, Enum):
    HIGH = "high"       # >= 0.85
    MEDIUM = "medium"   # 0.5 - 0.85
    LOW = "low"          # < 0.5
    MISSING = "missing"  # could not be read at all


class ExtractedField(BaseModel):
    """A single structured value with its confidence — never silently guessed."""

    value: Optional[str] = None
    confidence: FieldConfidence = FieldConfidence.MISSING
    raw_score: Optional[float] = None  # underlying 0-1 score from OCR/LLM, for audit

    @classmethod
    def from_score(cls, value: Optional[str], score: Optional[float]) -> "ExtractedField":
        if not value or score is None:
            return cls(value=None, confidence=FieldConfidence.MISSING, raw_score=score)
        if score >= 0.85:
            level = FieldConfidence.HIGH
        elif score >= 0.5:
            level = FieldConfidence.MEDIUM
        else:
            level = FieldConfidence.LOW
        return cls(value=value, confidence=level, raw_score=score)


class MedicineEntry(BaseModel):
    """One medicine line item extracted from a prescription."""

    name: ExtractedField = Field(default_factory=ExtractedField)
    strength: ExtractedField = Field(default_factory=ExtractedField)
    dosage: ExtractedField = Field(default_factory=ExtractedField)
    frequency: ExtractedField = Field(default_factory=ExtractedField)
    duration: ExtractedField = Field(default_factory=ExtractedField)
    instructions: ExtractedField = Field(default_factory=ExtractedField)

    def needs_review(self) -> bool:
        fields = [self.name, self.strength, self.dosage, self.frequency, self.duration]
        return any(f.confidence in (FieldConfidence.LOW, FieldConfidence.MISSING) for f in fields)


class PrescriptionRecord(BaseModel):
    """Mirrors the `prescriptions` MongoDB collection."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    patient_id: str
    doctor_id: Optional[str] = None  # None if patient self-uploaded, e.g. an old paper prescription
    source_image_ref: str  # storage path/key, never the raw bytes in this record
    medicines: list[MedicineEntry] = Field(default_factory=list)
    status: PrescriptionStatus = PrescriptionStatus.DRAFT
    ocr_provider: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified_at: Optional[datetime] = None

    def to_mongo(self) -> dict:
        doc = self.model_dump()
        doc["_id"] = doc.pop("id")
        return doc

    @classmethod
    def from_mongo(cls, doc: dict) -> "PrescriptionRecord":
        doc = dict(doc)
        doc["id"] = str(doc.pop("_id"))
        return cls(**doc)


class ScanRequest(BaseModel):
    patient_id: str
    doctor_id: Optional[str] = None
    image_storage_ref: str  # image is uploaded separately (e.g. to S3/Firebase Storage) first


class FieldCorrection(BaseModel):
    medicine_index: int
    field_name: str  # "name" | "strength" | "dosage" | "frequency" | "duration" | "instructions"
    corrected_value: str


class VerifyRequest(BaseModel):
    corrections: list[FieldCorrection] = Field(default_factory=list)
    confirmed: bool = True  # False = user reviewed but explicitly discarded instead


# --------------------------------------------------------------------------
# OCR provider interface (pluggable, per your tech stack doc)
# --------------------------------------------------------------------------

class OCRProvider(ABC):
    @abstractmethod
    async def extract_text(self, image_ref: str) -> tuple[str, float]:
        """Returns (raw_text, overall_confidence 0-1)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class TesseractOCRProvider(OCRProvider):
    """Local OCR via pytesseract — good for dev, low-cost self-hosting."""

    def __init__(self, image_loader=None) -> None:
        # image_loader: callable(image_ref) -> PIL.Image, injected so this
        # class doesn't need to know if images live on disk, S3, Firebase, etc.
        self._image_loader = image_loader

    @property
    def name(self) -> str:
        return "tesseract"

    async def extract_text(self, image_ref: str) -> tuple[str, float]:
        import pytesseract

        if self._image_loader is None:
            raise ValueError("TesseractOCRProvider requires an image_loader function")
        image = self._image_loader(image_ref)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = [w for w in data["text"] if w.strip()]
        confidences = [int(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and c != "-1"]
        text = " ".join(words)
        avg_conf = (sum(confidences) / len(confidences) / 100) if confidences else 0.0
        return text, avg_conf


class MockOCRProvider(OCRProvider):
    """For local dev / tests without a real OCR engine or image pipeline."""

    def __init__(self, fixed_text: str = "", fixed_confidence: float = 0.9) -> None:
        self._text = fixed_text
        self._confidence = fixed_confidence

    @property
    def name(self) -> str:
        return "mock"

    async def extract_text(self, image_ref: str) -> tuple[str, float]:
        return self._text, self._confidence


# --------------------------------------------------------------------------
# Extraction agent — LLM turns raw OCR text into structured fields.
# Deterministic guardrails still apply: nothing is invented, every field
# keeps a confidence score, and low OCR confidence caps the field confidence
# regardless of how sure the LLM sounds.
# --------------------------------------------------------------------------

class LLMClient(ABC):
    @abstractmethod
    async def generate_json(self, prompt: str) -> dict:
        ...


class GeminiClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash") -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(
            model,
            generation_config={"response_mime_type": "application/json"},
        )

    async def generate_json(self, prompt: str) -> dict:
        import json

        response = await self._model.generate_content_async(prompt)
        return json.loads(response.text)


class PrescriptionExtractionAgent:
    """
    Converts raw OCR text into a structured MedicineEntry list.
    If no LLM client is configured, falls back to a naive line-splitter so
    the pipeline still produces *something* reviewable rather than failing.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def extract(self, raw_text: str, ocr_confidence: float) -> list[MedicineEntry]:
        if not raw_text.strip():
            return []

        if self._llm is None:
            return self._fallback_extract(raw_text, ocr_confidence)

        try:
            payload = await self._llm.generate_json(self._build_prompt(raw_text))
            return self._to_entries(payload, ocr_confidence)
        except Exception as exc:  # noqa: BLE001 - never break the scan, degrade instead
            logger.warning("LLM extraction failed, falling back to naive parse: %s", exc)
            return self._fallback_extract(raw_text, ocr_confidence)

    @staticmethod
    def _build_prompt(raw_text: str) -> str:
        return (
            "You are a prescription structuring assistant. Extract medicines from the "
            "OCR text below into a JSON array. For EACH field, if you are not confident "
            "or the information is simply not present in the text, set it to null — "
            "NEVER invent a plausible-sounding value. Also return your own confidence "
            "(0.0-1.0) per field based on how clearly it appeared in the text.\n\n"
            'Return ONLY JSON in this exact shape: '
            '{"medicines": [{"name": {"value": str|null, "score": float}, '
            '"strength": {...}, "dosage": {...}, "frequency": {...}, '
            '"duration": {...}, "instructions": {...}}]}\n\n'
            f"OCR TEXT:\n{raw_text}"
        )

    @staticmethod
    def _to_entries(payload: dict, ocr_confidence: float) -> list[MedicineEntry]:
        entries: list[MedicineEntry] = []
        for item in payload.get("medicines", []):
            fields = {}
            for key in ("name", "strength", "dosage", "frequency", "duration", "instructions"):
                sub = item.get(key) or {}
                # Cap the effective confidence at the OCR engine's own confidence,
                # so a garbled scan can never produce an artificially "high" field
                # just because the LLM guessed a fluent-sounding value.
                effective_score = None
                if sub.get("score") is not None:
                    effective_score = min(float(sub["score"]), ocr_confidence + 0.15)
                fields[key] = ExtractedField.from_score(sub.get("value"), effective_score)
            entries.append(MedicineEntry(**fields))
        return entries

    @staticmethod
    def _fallback_extract(raw_text: str, ocr_confidence: float) -> list[MedicineEntry]:
        """No-LLM fallback: one entry per non-empty line, name only, rest flagged missing."""
        entries = []
        for line in [l.strip() for l in raw_text.splitlines() if l.strip()]:
            entries.append(
                MedicineEntry(
                    name=ExtractedField.from_score(line, ocr_confidence),
                )
            )
        return entries


# --------------------------------------------------------------------------
# Repository (MongoDB via motor)
# --------------------------------------------------------------------------

class PrescriptionRepository(ABC):
    @abstractmethod
    async def save(self, record: PrescriptionRecord) -> PrescriptionRecord:
        ...

    @abstractmethod
    async def get(self, record_id: str) -> Optional[PrescriptionRecord]:
        ...

    @abstractmethod
    async def list_for_patient(self, patient_id: str) -> list[PrescriptionRecord]:
        ...


class MongoPrescriptionRepository(PrescriptionRepository):
    """
    Thin data-access layer over Mongo — no business logic here, so the
    backing store can change again later without touching the service.
    """

    def __init__(self, db) -> None:
        # db: an AsyncIOMotorDatabase instance, injected at app startup
        self._collection = db["prescriptions"]

    async def save(self, record: PrescriptionRecord) -> PrescriptionRecord:
        record.updated_at = datetime.now(timezone.utc)
        await self._collection.replace_one(
            {"_id": record.id}, record.to_mongo(), upsert=True
        )
        return record

    async def get(self, record_id: str) -> Optional[PrescriptionRecord]:
        doc = await self._collection.find_one({"_id": record_id})
        return PrescriptionRecord.from_mongo(doc) if doc else None

    async def list_for_patient(self, patient_id: str) -> list[PrescriptionRecord]:
        cursor = self._collection.find({"patient_id": patient_id}).sort("created_at", -1)
        return [PrescriptionRecord.from_mongo(doc) async for doc in cursor]


class InMemoryPrescriptionRepository(PrescriptionRepository):
    """For local dev / unit tests without a running Mongo instance."""

    def __init__(self) -> None:
        self._store: dict[str, PrescriptionRecord] = {}

    async def save(self, record: PrescriptionRecord) -> PrescriptionRecord:
        record.updated_at = datetime.now(timezone.utc)
        self._store[record.id] = record
        return record

    async def get(self, record_id: str) -> Optional[PrescriptionRecord]:
        return self._store.get(record_id)

    async def list_for_patient(self, patient_id: str) -> list[PrescriptionRecord]:
        return sorted(
            [r for r in self._store.values() if r.patient_id == patient_id],
            key=lambda r: r.created_at,
            reverse=True,
        )


# --------------------------------------------------------------------------
# Orchestration service
# --------------------------------------------------------------------------

class PrescriptionService:
    def __init__(
        self,
        repository: PrescriptionRepository,
        ocr_provider: OCRProvider,
        extraction_agent: Optional[PrescriptionExtractionAgent] = None,
    ) -> None:
        self._repository = repository
        self._ocr = ocr_provider
        self._agent = extraction_agent or PrescriptionExtractionAgent(llm_client=None)

    async def scan(self, req: ScanRequest) -> PrescriptionRecord:
        raw_text, ocr_confidence = await self._ocr.extract_text(req.image_storage_ref)
        medicines = await self._agent.extract(raw_text, ocr_confidence)

        status = (
            PrescriptionStatus.PENDING_REVIEW
            if any(m.needs_review() for m in medicines)
            else PrescriptionStatus.DRAFT
        )

        record = PrescriptionRecord(
            patient_id=req.patient_id,
            doctor_id=req.doctor_id,
            source_image_ref=req.image_storage_ref,
            medicines=medicines,
            status=status,
            ocr_provider=self._ocr.name,
        )
        return await self._repository.save(record)

    async def verify(self, record_id: str, req: VerifyRequest) -> PrescriptionRecord:
        record = await self._repository.get(record_id)
        if record is None:
            raise ValueError(f"Prescription {record_id} not found")

        if not req.confirmed:
            record.status = PrescriptionStatus.DISCARDED
            return await self._repository.save(record)

        # Apply user corrections. A user-confirmed value is always HIGH
        # confidence going forward — it's no longer a machine guess.
        for correction in req.corrections:
            if correction.medicine_index >= len(record.medicines):
                continue
            entry = record.medicines[correction.medicine_index]
            setattr(
                entry,
                correction.field_name,
                ExtractedField(
                    value=correction.corrected_value,
                    confidence=FieldConfidence.HIGH,
                    raw_score=1.0,
                ),
            )

        record.status = PrescriptionStatus.VERIFIED
        record.verified_at = datetime.now(timezone.utc)
        return await self._repository.save(record)

    async def get(self, record_id: str) -> Optional[PrescriptionRecord]:
        return await self._repository.get(record_id)

    async def list_for_patient(self, patient_id: str) -> list[PrescriptionRecord]:
        return await self._repository.list_for_patient(patient_id)


# --------------------------------------------------------------------------
# FastAPI router (mount under /api/v1)
# --------------------------------------------------------------------------

def build_router():
    from fastapi import APIRouter, Depends, HTTPException

    router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])

    def get_prescription_service() -> PrescriptionService:
        # Wire real dependencies in your app's dependency module, e.g.
        # MongoPrescriptionRepository(app.state.mongo_db) + a real OCR
        # provider + GeminiClient built from settings.
        repository = InMemoryPrescriptionRepository()
        ocr_provider = MockOCRProvider()
        agent = PrescriptionExtractionAgent(llm_client=None)
        return PrescriptionService(repository, ocr_provider, agent)

    @router.post("/scan", response_model=PrescriptionRecord)
    async def scan_prescription(
        req: ScanRequest,
        service: PrescriptionService = Depends(get_prescription_service),
    ):
        return await service.scan(req)

    @router.post("/{record_id}/verify", response_model=PrescriptionRecord)
    async def verify_prescription(
        record_id: str,
        req: VerifyRequest,
        service: PrescriptionService = Depends(get_prescription_service),
    ):
        try:
            return await service.verify(record_id, req)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.get("/{record_id}", response_model=PrescriptionRecord)
    async def get_prescription(
        record_id: str,
        service: PrescriptionService = Depends(get_prescription_service),
    ):
        record = await service.get(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Not found")
        return record

    @router.get("/patient/{patient_id}", response_model=list[PrescriptionRecord])
    async def list_patient_prescriptions(
        patient_id: str,
        service: PrescriptionService = Depends(get_prescription_service),
    ):
        return await service.list_for_patient(patient_id)

    return router