"""
test_agents.py
---------------
Test suite covering all seven HealthFlow AI agents:

  1. health_scheme.py            - Scheme Intelligence Agent
  2. prescription_ocr.py         - Prescription OCR Agent
  3. medicine_intelligence.py    - Medicine Intelligence Agent
  4. recommendation_agent.py     - Recommendation Agent
  5. health_assistant.py         - Health Assistant Agent
  6. voice_assistant.py          - Voice Assistant Agent
  7. analytics_agent.py          - Analytics Agent

Every agent is tested against its in-memory / mock dependencies, so this
suite needs NO MongoDB connection and NO API keys to run.

Run it with:
    pytest backend/tests/test_agents.py -v

Or run just one agent's tests, e.g.:
    pytest backend/tests/test_agents.py -v -k scheme
    pytest backend/tests/test_agents.py -v -k prescription
    pytest backend/tests/test_agents.py -v -k medicine
    pytest backend/tests/test_agents.py -v -k recommendation
    pytest backend/tests/test_agents.py -v -k assistant
    pytest backend/tests/test_agents.py -v -k voice
    pytest backend/tests/test_agents.py -v -k analytics
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest")

from datetime import date, datetime, timedelta, timezone

import pytest

# --------------------------------------------------------------------------
# 0. Users & Auth (core domain route, not an AI agent)
# --------------------------------------------------------------------------

from backend.routes.users import (
    UserCreate,
    UserLogin,
    UserPublic,
    UserRole,
    InMemoryUserRepository,
    AuthService,
    AuthError,
    build_router as build_users_router,
)


class TestUsersAndAuth:
    @pytest.mark.asyncio
    async def test_register_creates_user_and_returns_token(self):
        service = AuthService(InMemoryUserRepository())
        result = await service.register(
            UserCreate(email="asha@example.com", password="securepass123", full_name="Asha Kumar")
        )
        assert result.user.email == "asha@example.com"
        assert result.user.role == UserRole.PATIENT
        assert result.access_token

    @pytest.mark.asyncio
    async def test_duplicate_email_registration_rejected(self):
        service = AuthService(InMemoryUserRepository())
        await service.register(
            UserCreate(email="asha@example.com", password="securepass123", full_name="Asha Kumar")
        )
        with pytest.raises(AuthError):
            await service.register(
                UserCreate(email="asha@example.com", password="anotherpass123", full_name="Dup")
            )

    @pytest.mark.asyncio
    async def test_login_with_correct_password_succeeds(self):
        service = AuthService(InMemoryUserRepository())
        reg = await service.register(
            UserCreate(email="asha@example.com", password="securepass123", full_name="Asha Kumar")
        )
        login = await service.login(UserLogin(email="asha@example.com", password="securepass123"))
        assert login.user.id == reg.user.id

    @pytest.mark.asyncio
    async def test_login_with_wrong_password_rejected(self):
        service = AuthService(InMemoryUserRepository())
        await service.register(
            UserCreate(email="asha@example.com", password="securepass123", full_name="Asha Kumar")
        )
        with pytest.raises(AuthError):
            await service.login(UserLogin(email="asha@example.com", password="wrongpass"))

    @pytest.mark.asyncio
    async def test_get_current_user_resolves_from_valid_token(self):
        service = AuthService(InMemoryUserRepository())
        reg = await service.register(
            UserCreate(email="asha@example.com", password="securepass123", full_name="Asha Kumar")
        )
        current = await service.get_current_user(reg.access_token)
        assert current.email == "asha@example.com"

    @pytest.mark.asyncio
    async def test_get_current_user_rejects_invalid_token(self):
        service = AuthService(InMemoryUserRepository())
        with pytest.raises(AuthError):
            await service.get_current_user("not-a-real-token")

    def test_router_builds_with_expected_routes(self):
        router, get_current_user, require_role = build_users_router()
        paths = {r.path for r in router.routes}
        assert "/users/register" in paths
        assert "/users/login" in paths
        assert "/users/me" in paths


# --------------------------------------------------------------------------
# 0b. Doctors (core domain route, not an AI agent)
# --------------------------------------------------------------------------

from backend.routes.doctors import (
    DoctorProfileCreate,
    DoctorProfileUpdate,
    AvailabilitySlotCreate,
    ConsultationType,
    InMemoryDoctorRepository,
    InMemoryAvailabilityRepository,
    DoctorService,
    DoctorServiceError,
    build_router as build_doctors_router,
)


class TestDoctorsRoute:
    @pytest.fixture
    def service(self) -> DoctorService:
        return DoctorService(InMemoryDoctorRepository(), InMemoryAvailabilityRepository())

    @pytest.fixture
    def doctor_user(self) -> UserPublic:
        return UserPublic(
            id="u-doc-1", email="doc@example.com", full_name="Dr. Meera Rao",
            role=UserRole.DOCTOR, created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def other_doctor_user(self) -> UserPublic:
        return UserPublic(
            id="u-doc-2", email="doc2@example.com", full_name="Dr. Iyer",
            role=UserRole.DOCTOR, created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def patient_user(self) -> UserPublic:
        return UserPublic(
            id="u-pat-1", email="patient@example.com", full_name="Asha Kumar",
            role=UserRole.PATIENT, created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_only_doctor_role_can_create_profile(self, service, patient_user):
        with pytest.raises(DoctorServiceError):
            await service.create_profile(
                patient_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=5)
            )

    @pytest.mark.asyncio
    async def test_doctor_can_create_profile_unverified_by_default(self, service, doctor_user):
        profile = await service.create_profile(
            doctor_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=10)
        )
        assert profile.is_verified is False
        assert profile.full_name == "Dr. Meera Rao"

    @pytest.mark.asyncio
    async def test_cannot_create_duplicate_profile_for_same_user(self, service, doctor_user):
        await service.create_profile(
            doctor_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=10)
        )
        with pytest.raises(DoctorServiceError):
            await service.create_profile(
                doctor_user, DoctorProfileCreate(specialties=["Dermatology"], experience_years=3)
            )

    @pytest.mark.asyncio
    async def test_search_filters_by_specialty(self, service, doctor_user):
        await service.create_profile(
            doctor_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=10)
        )
        results = await service.search(specialty="Cardiology")
        assert len(results) == 1
        results = await service.search(specialty="Dermatology")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_only_owner_can_update_profile(self, service, doctor_user, other_doctor_user):
        profile = await service.create_profile(
            doctor_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=10)
        )
        with pytest.raises(DoctorServiceError):
            await service.update_own_profile(
                other_doctor_user, profile.id, DoctorProfileUpdate(experience_years=20)
            )
        updated = await service.update_own_profile(
            doctor_user, profile.id, DoctorProfileUpdate(experience_years=15)
        )
        assert updated.experience_years == 15

    @pytest.mark.asyncio
    async def test_set_verified_flips_flag(self, service, doctor_user):
        profile = await service.create_profile(
            doctor_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=10)
        )
        assert profile.is_verified is False
        verified = await service.set_verified(profile.id, True)
        assert verified.is_verified is True

    @pytest.mark.asyncio
    async def test_availability_can_only_be_added_by_owner(self, service, doctor_user, other_doctor_user):
        profile = await service.create_profile(
            doctor_user, DoctorProfileCreate(specialties=["Cardiology"], experience_years=10)
        )
        with pytest.raises(DoctorServiceError):
            await service.add_availability(
                other_doctor_user, profile.id,
                AvailabilitySlotCreate(date="2026-09-10", start_time="09:00", end_time="09:30"),
            )
        slot = await service.add_availability(
            doctor_user, profile.id,
            AvailabilitySlotCreate(date="2026-09-10", start_time="09:00", end_time="09:30"),
        )
        assert slot.is_booked is False
        slots = await service.list_availability(profile.id)
        assert len(slots) == 1

    def test_router_builds_with_expected_routes(self):
        users_router, get_current_user, require_role = build_users_router()
        router = build_doctors_router(get_current_user, require_role)
        paths = {r.path for r in router.routes}
        assert "/doctors/me" in paths
        assert "/doctors/{doctor_id}/verify" in paths
        assert "/doctors/{doctor_id}/availability" in paths


# --------------------------------------------------------------------------
# 1. Scheme Intelligence Agent
# --------------------------------------------------------------------------

from backend.agents.health_scheme import (
    Scheme,
    SchemeCategory,
    EligibilityCheckRequest,
    SchemeEligibilityEngine,
    RuleOutcome,
    InMemorySchemeRepository,
    SchemeService,
    build_router as build_scheme_router,
)


class TestSchemeIntelligenceAgent:
    @pytest.fixture
    def scheme(self) -> Scheme:
        return Scheme(
            id="s1",
            name="State Maternal Care Scheme",
            state="Tamil Nadu",
            category=SchemeCategory.MATERNAL,
            min_age=18,
            max_age=45,
            max_annual_income=250_000,
            eligible_conditions=[],
            benefits="Free maternal checkups",
            required_documents=["Aadhaar"],
        )

    def test_eligible_when_all_criteria_met(self, scheme):
        req = EligibilityCheckRequest(age=28, state="Tamil Nadu", annual_income=200_000)
        result = SchemeEligibilityEngine.evaluate(scheme, req)
        assert result.outcome == RuleOutcome.ELIGIBLE

    def test_not_eligible_when_income_exceeds_limit(self, scheme):
        req = EligibilityCheckRequest(age=28, state="Tamil Nadu", annual_income=300_000)
        result = SchemeEligibilityEngine.evaluate(scheme, req)
        assert result.outcome == RuleOutcome.NOT_ELIGIBLE
        assert any("income" in r.lower() for r in result.reasons)

    def test_not_eligible_when_age_out_of_range(self, scheme):
        req = EligibilityCheckRequest(age=15, state="Tamil Nadu", annual_income=100_000)
        result = SchemeEligibilityEngine.evaluate(scheme, req)
        assert result.outcome == RuleOutcome.NOT_ELIGIBLE

    def test_needs_review_when_income_missing(self, scheme):
        req = EligibilityCheckRequest(age=28, state="Tamil Nadu", annual_income=None)
        result = SchemeEligibilityEngine.evaluate(scheme, req)
        assert result.outcome == RuleOutcome.NEEDS_REVIEW

    def test_national_scheme_matches_any_state(self):
        national = Scheme(
            id="s2", name="National Scheme", state="ALL", category=SchemeCategory.GENERAL,
            benefits="Some benefit",
        )
        req = EligibilityCheckRequest(age=30, state="Kerala")
        result = SchemeEligibilityEngine.evaluate(national, req)
        assert result.outcome == RuleOutcome.ELIGIBLE

    @pytest.mark.asyncio
    async def test_service_check_eligibility_end_to_end(self, scheme):
        repo = InMemorySchemeRepository([scheme])
        service = SchemeService(repository=repo)
        req = EligibilityCheckRequest(age=28, state="Tamil Nadu", annual_income=200_000)
        results = await service.check_eligibility(req, explain=False)
        assert len(results) == 1
        assert results[0].outcome == RuleOutcome.ELIGIBLE

    def test_router_builds_with_expected_routes(self):
        router = build_scheme_router()
        paths = {r.path for r in router.routes}
        assert "/schemes/" in paths
        assert "/schemes/check-eligibility" in paths


# --------------------------------------------------------------------------
# 2. Prescription OCR Agent
# --------------------------------------------------------------------------

from backend.agents.prescription_ocr import (
    ScanRequest,
    VerifyRequest,
    FieldCorrection,
    MockOCRProvider,
    InMemoryPrescriptionRepository,
    PrescriptionExtractionAgent,
    PrescriptionService,
    PrescriptionStatus,
    FieldConfidence,
    build_router as build_prescription_router,
)


class TestPrescriptionOCRAgent:
    @pytest.mark.asyncio
    async def test_high_confidence_name_field_extracted_correctly(self):
        # NOTE: the no-LLM fallback extractor only ever fills the `name`
        # field (it can't structure dosage/frequency/etc. without an LLM),
        # so the record is still PENDING_REVIEW even though the name field
        # itself is HIGH confidence — the other fields are genuinely unread,
        # not guessed, which is the whole point of this agent's guardrail.
        ocr = MockOCRProvider(fixed_text="Paracetamol 500mg", fixed_confidence=0.95)
        repo = InMemoryPrescriptionRepository()
        service = PrescriptionService(repo, ocr, PrescriptionExtractionAgent(llm_client=None))

        record = await service.scan(ScanRequest(patient_id="p1", image_storage_ref="img/1.jpg"))
        assert record.medicines[0].name.confidence == FieldConfidence.HIGH
        assert record.medicines[0].dosage.confidence == FieldConfidence.MISSING
        assert record.status == PrescriptionStatus.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_low_confidence_scan_flags_pending_review(self):
        ocr = MockOCRProvider(fixed_text="Paracetamol 500mg", fixed_confidence=0.3)
        repo = InMemoryPrescriptionRepository()
        service = PrescriptionService(repo, ocr, PrescriptionExtractionAgent(llm_client=None))

        record = await service.scan(ScanRequest(patient_id="p1", image_storage_ref="img/1.jpg"))
        assert record.status == PrescriptionStatus.PENDING_REVIEW
        assert record.medicines[0].name.confidence == FieldConfidence.LOW

    @pytest.mark.asyncio
    async def test_verify_applies_corrections_and_marks_verified(self):
        ocr = MockOCRProvider(fixed_text="Amoxicillin 250mg", fixed_confidence=0.4)
        repo = InMemoryPrescriptionRepository()
        service = PrescriptionService(repo, ocr, PrescriptionExtractionAgent(llm_client=None))

        record = await service.scan(ScanRequest(patient_id="p1", image_storage_ref="img/2.jpg"))
        verified = await service.verify(
            record.id,
            VerifyRequest(corrections=[
                FieldCorrection(medicine_index=0, field_name="dosage", corrected_value="1 tablet twice daily")
            ]),
        )
        assert verified.status == PrescriptionStatus.VERIFIED
        assert verified.medicines[0].dosage.confidence == FieldConfidence.HIGH
        assert verified.medicines[0].dosage.value == "1 tablet twice daily"
        assert verified.verified_at is not None

    @pytest.mark.asyncio
    async def test_discard_marks_status_discarded(self):
        ocr = MockOCRProvider(fixed_text="Something", fixed_confidence=0.9)
        repo = InMemoryPrescriptionRepository()
        service = PrescriptionService(repo, ocr, PrescriptionExtractionAgent(llm_client=None))

        record = await service.scan(ScanRequest(patient_id="p1", image_storage_ref="img/3.jpg"))
        discarded = await service.verify(record.id, VerifyRequest(confirmed=False))
        assert discarded.status == PrescriptionStatus.DISCARDED

    @pytest.mark.asyncio
    async def test_verify_unknown_record_raises(self):
        repo = InMemoryPrescriptionRepository()
        service = PrescriptionService(repo, MockOCRProvider(), PrescriptionExtractionAgent(llm_client=None))
        with pytest.raises(ValueError):
            await service.verify("does-not-exist", VerifyRequest())

    def test_router_builds_with_expected_routes(self):
        router = build_prescription_router()
        paths = {r.path for r in router.routes}
        assert "/prescriptions/scan" in paths
        assert "/prescriptions/{record_id}/verify" in paths


# --------------------------------------------------------------------------
# 3. Medicine Intelligence Agent
# --------------------------------------------------------------------------

from backend.agents.medicine_intelligence import (
    MedicineReference,
    InteractionSeverity,
    InMemoryMedicineRepository,
    MedicineService,
    MedicineExplanationRequest,
    InteractionChecker,
    build_router as build_medicine_router,
)


class TestMedicineIntelligenceAgent:
    @pytest.fixture
    def paracetamol(self) -> MedicineReference:
        return MedicineReference(
            id="m1", name="Paracetamol", generic_name="Acetaminophen",
            uses=["Fever", "Mild pain relief"], side_effects=["Nausea (rare)"],
            interacts_with={"m2": InteractionSeverity.MODERATE},
            storage="Store below 25°C", warnings=["Avoid with liver disease"],
            source="CDSCO", last_updated=date(2026, 1, 15),
        )

    @pytest.fixture
    def warfarin(self) -> MedicineReference:
        return MedicineReference(
            id="m2", name="Warfarin", generic_name="Coumadin",
            uses=["Blood thinning"], side_effects=["Bleeding risk"],
            interacts_with={}, storage="Room temperature",
            warnings=["Regular INR monitoring required"],
            source="CDSCO", last_updated=date(2026, 1, 15),
        )

    @pytest.mark.asyncio
    async def test_search_finds_by_partial_name(self, paracetamol, warfarin):
        repo = InMemoryMedicineRepository([paracetamol, warfarin])
        service = MedicineService(repo)
        results = await service.search("parac")
        assert len(results) == 1
        assert results[0].name == "Paracetamol"

    @pytest.mark.asyncio
    async def test_explain_without_llm_returns_deterministic_summary(self, paracetamol):
        repo = InMemoryMedicineRepository([paracetamol])
        service = MedicineService(repo)
        result = await service.explain(MedicineExplanationRequest(medicine_id="m1"))
        assert "Paracetamol" in result.explanation
        assert "dosage" in result.disclaimer.lower()

    @pytest.mark.asyncio
    async def test_explain_unknown_medicine_raises(self):
        repo = InMemoryMedicineRepository([])
        service = MedicineService(repo)
        with pytest.raises(ValueError):
            await service.explain(MedicineExplanationRequest(medicine_id="does-not-exist"))

    def test_interaction_checker_detects_known_interaction(self, paracetamol, warfarin):
        result = InteractionChecker.check_pair(paracetamol, warfarin)
        assert result.interacts is True
        assert result.severity == InteractionSeverity.MODERATE

    def test_interaction_checker_no_interaction_found(self, warfarin):
        unrelated = MedicineReference(
            id="m3", name="Vitamin C", uses=["Supplement"], source="CDSCO",
            last_updated=date(2026, 1, 1),
        )
        result = InteractionChecker.check_pair(warfarin, unrelated)
        assert result.interacts is False

    @pytest.mark.asyncio
    async def test_check_interactions_needs_at_least_two_medicines(self, paracetamol):
        repo = InMemoryMedicineRepository([paracetamol])
        service = MedicineService(repo)
        results = await service.check_interactions(["m1"])
        assert results == []

    def test_router_builds_with_expected_routes(self):
        router = build_medicine_router()
        paths = {r.path for r in router.routes}
        assert "/medicines/search" in paths
        assert "/medicines/check-interactions" in paths


# --------------------------------------------------------------------------
# 4. Recommendation Agent
# --------------------------------------------------------------------------

from backend.agents.recommendation_agent import (
    Candidate,
    CandidateType,
    ConsultationType,
    UserPreferences,
    RecommendationRequest,
    InMemoryCandidateRepository,
    RecommendationService,
    RecommendationRanker,
    build_router as build_recommendation_router,
)


class TestRecommendationAgent:
    @pytest.fixture
    def candidates(self) -> list[Candidate]:
        near = Candidate(
            id="d1", type=CandidateType.DOCTOR, name="Dr. Meera Rao",
            specialties=["Cardiology"], latitude=13.0827, longitude=80.2707,
            is_verified=True, rating=4.7, has_open_slots=True,
            consultation_type=ConsultationType.IN_PERSON,
        )
        far = Candidate(
            id="d2", type=CandidateType.DOCTOR, name="Dr. Arjun Iyer",
            specialties=["Cardiology"], latitude=11.0168, longitude=76.9558,
            is_verified=False, rating=3.9, has_open_slots=False,
            consultation_type=ConsultationType.IN_PERSON,
        )
        return [near, far]

    @pytest.mark.asyncio
    async def test_closer_verified_doctor_ranks_first(self, candidates):
        repo = InMemoryCandidateRepository(candidates)
        service = RecommendationService(repo)
        req = RecommendationRequest(
            patient_id="p1", intent="find cardiologists near me",
            candidate_type=CandidateType.DOCTOR, specialty_or_category="Cardiology",
            latitude=13.0827, longitude=80.2707,
        )
        results = await service.recommend(req)
        assert results[0].candidate.id == "d1"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_max_distance_filter_excludes_far_candidates(self, candidates):
        repo = InMemoryCandidateRepository(candidates)
        service = RecommendationService(repo)
        req = RecommendationRequest(
            patient_id="p1", intent="find nearby cardiologists",
            candidate_type=CandidateType.DOCTOR, specialty_or_category="Cardiology",
            latitude=13.0827, longitude=80.2707,
            preferences=UserPreferences(max_distance_km=50),
        )
        results = await service.recommend(req)
        assert len(results) == 1
        assert results[0].candidate.id == "d1"

    @pytest.mark.asyncio
    async def test_verified_only_filter(self, candidates):
        repo = InMemoryCandidateRepository(candidates)
        service = RecommendationService(repo)
        req = RecommendationRequest(
            patient_id="p1", intent="find verified cardiologists",
            candidate_type=CandidateType.DOCTOR, specialty_or_category="Cardiology",
            preferences=UserPreferences(verified_only=True),
        )
        results = await service.recommend(req)
        assert all(r.candidate.is_verified for r in results)

    def test_haversine_distance_zero_for_same_point(self):
        dist = RecommendationRanker._haversine_km(13.0827, 80.2707, 13.0827, 80.2707)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_router_builds_with_expected_routes(self):
        router = build_recommendation_router()
        paths = {r.path for r in router.routes}
        assert "/recommendations/" in paths


# --------------------------------------------------------------------------
# 5. Health Assistant Agent
# --------------------------------------------------------------------------

from backend.agents.health_assistant import (
    SendMessageRequest,
    InMemoryConversationRepository,
    HealthAssistantService,
    EmergencyDetector,
    build_router as build_assistant_router,
)


class TestHealthAssistantAgent:
    @pytest.mark.parametrize("text", [
        "chest pain",
        "he is unconscious and not responding",
        "severe allergic reaction",
        "she had a seizure",
    ])
    def test_emergency_detector_fires_on_urgent_phrases(self, text):
        is_emergency, is_crisis = EmergencyDetector.check(text)
        assert is_emergency is True

    @pytest.mark.parametrize("text", [
        "I feel a bit tired today",
        "what time does the clinic open",
        "I need to find a cardiologist",
    ])
    def test_emergency_detector_does_not_fire_on_normal_messages(self, text):
        is_emergency, _ = EmergencyDetector.check(text)
        assert is_emergency is False

    def test_crisis_language_flagged_as_crisis_subtype(self):
        is_emergency, is_crisis = EmergencyDetector.check("I want to kill myself")
        assert is_emergency is True
        assert is_crisis is True

    @pytest.mark.asyncio
    async def test_normal_message_gets_conversational_reply(self):
        service = HealthAssistantService(InMemoryConversationRepository())
        reply = await service.send_message(
            SendMessageRequest(user_id="u1", text="I need to find a cardiologist")
        )
        assert reply.is_emergency is False
        assert reply.message.content  # non-empty

    @pytest.mark.asyncio
    async def test_emergency_message_bypasses_llm_and_escalates(self):
        service = HealthAssistantService(InMemoryConversationRepository())
        reply = await service.send_message(
            SendMessageRequest(user_id="u1", text="severe chest pain and can't breathe")
        )
        assert reply.is_emergency is True
        assert reply.suggested_action == "open_emergency_flow"
        assert reply.message.is_emergency_escalation is True

    @pytest.mark.asyncio
    async def test_conversation_history_persists_across_turns(self):
        repo = InMemoryConversationRepository()
        service = HealthAssistantService(repo)
        r1 = await service.send_message(SendMessageRequest(user_id="u1", text="hello"))
        await service.send_message(
            SendMessageRequest(user_id="u1", conversation_id=r1.conversation_id, text="thanks")
        )
        history = await repo.get_recent_messages(r1.conversation_id, limit=10)
        assert len(history) == 4  # 2 user + 2 assistant messages

    def test_router_builds_with_expected_routes(self):
        router = build_assistant_router()
        paths = {r.path for r in router.routes}
        assert "/assistant/message" in paths


# --------------------------------------------------------------------------
# 6. Voice Assistant Agent
# --------------------------------------------------------------------------

from backend.agents.voice_assistant import (
    VoiceMessageRequest,
    AudioFormat,
    SupportedLanguage,
    MockVoiceProvider,
    InMemoryVoiceInteractionRepository,
    VoiceAssistantService,
    AudioValidator,
    AudioValidationError,
    MIN_TRANSCRIPTION_CONFIDENCE,
    build_router as build_voice_router,
)


class TestVoiceAssistantAgent:
    @pytest.fixture
    def health_assistant(self):
        return HealthAssistantService(InMemoryConversationRepository())

    @pytest.fixture
    def base_request(self) -> VoiceMessageRequest:
        return VoiceMessageRequest(
            user_id="u1", audio_format=AudioFormat.WAV,
            language_hint=SupportedLanguage.ENGLISH,
            audio_size_bytes=50_000, audio_duration_seconds=4.0,
        )

    @pytest.mark.asyncio
    async def test_high_confidence_transcription_flows_through(self, health_assistant, base_request):
        provider = MockVoiceProvider(fixed_text="find me a cardiologist", fixed_confidence=0.92)
        service = VoiceAssistantService(provider, health_assistant, InMemoryVoiceInteractionRepository())
        reply = await service.handle_voice_message(base_request, "audio/1.wav")
        assert reply.needs_repeat is False
        assert reply.transcript == "find me a cardiologist"
        assert reply.reply_audio_storage_ref is not None

    @pytest.mark.asyncio
    async def test_low_confidence_transcription_asks_to_repeat(self, health_assistant, base_request):
        provider = MockVoiceProvider(fixed_text="mumble", fixed_confidence=0.2)
        assert 0.2 < MIN_TRANSCRIPTION_CONFIDENCE
        service = VoiceAssistantService(provider, health_assistant, InMemoryVoiceInteractionRepository())
        reply = await service.handle_voice_message(base_request, "audio/2.wav")
        assert reply.needs_repeat is True
        assert "repeat" in reply.reply_text.lower()

    @pytest.mark.asyncio
    async def test_emergency_phrase_still_escalates_through_voice(self, health_assistant, base_request):
        provider = MockVoiceProvider(fixed_text="severe chest pain", fixed_confidence=0.95)
        service = VoiceAssistantService(provider, health_assistant, InMemoryVoiceInteractionRepository())
        reply = await service.handle_voice_message(base_request, "audio/3.wav")
        assert reply.is_emergency is True
        assert reply.suggested_action == "open_emergency_flow"

    def test_validator_rejects_oversized_duration(self, base_request):
        bad = base_request.model_copy(update={"audio_duration_seconds": 300.0})
        with pytest.raises(AudioValidationError):
            AudioValidator.validate(bad)

    def test_validator_rejects_empty_audio(self, base_request):
        bad = base_request.model_copy(update={"audio_size_bytes": 0})
        with pytest.raises(AudioValidationError):
            AudioValidator.validate(bad)

    def test_validator_accepts_valid_audio(self, base_request):
        AudioValidator.validate(base_request)  # should not raise

    def test_router_builds_with_expected_routes(self):
        router = build_voice_router()
        paths = {r.path for r in router.routes}
        assert "/voice/message" in paths


# --------------------------------------------------------------------------
# 7. Analytics Agent
# --------------------------------------------------------------------------

from backend.agents.analytics_agent import (
    RawAnalyticsEvent,
    EventType,
    InMemoryEventRepository,
    AnalyticsService,
    MetricsRequest,
    Anonymizer,
    AnalyticsNarrativeAgent,
    MetricsSnapshot,
    build_router as build_analytics_router,
)


class TestAnalyticsAgent:
    def test_hash_user_id_is_deterministic_and_not_reversible_looking(self):
        h1 = Anonymizer.hash_user_id("patient-123")
        h2 = Anonymizer.hash_user_id("patient-123")
        assert h1 == h2
        assert h1 != "patient-123"

    def test_strip_pii_removes_known_pii_keys(self):
        metadata = {"specialty": "Cardiology", "name": "Asha Kumar", "phone": "9999999999"}
        cleaned = Anonymizer.strip_pii(metadata)
        assert cleaned == {"specialty": "Cardiology"}

    @pytest.mark.asyncio
    async def test_log_event_anonymizes_before_storage(self):
        repo = InMemoryEventRepository()
        service = AnalyticsService(repo)
        raw = RawAnalyticsEvent(
            event_type=EventType.DOCTOR_SEARCHED, user_id="real-patient-123",
            metadata={"specialty": "Cardiology", "email": "a@b.com"},
        )
        saved = await service.log_event(raw)
        assert saved.user_id_hash != "real-patient-123"
        assert "email" not in saved.metadata
        assert saved.metadata == {"specialty": "Cardiology"}

    @pytest.mark.asyncio
    async def test_metrics_aggregation_is_accurate(self):
        repo = InMemoryEventRepository()
        service = AnalyticsService(repo)
        events = [
            (EventType.APP_OPEN, "u1", {}),
            (EventType.APP_OPEN, "u2", {}),
            (EventType.APPOINTMENT_COMPLETED, "u1", {}),
            (EventType.APPOINTMENT_CANCELLED, "u2", {}),
            (EventType.EMERGENCY_TRIGGERED, "u3", {}),
        ]
        for et, uid, meta in events:
            await service.log_event(RawAnalyticsEvent(event_type=et, user_id=uid, metadata=meta))

        now = datetime.now(timezone.utc)
        req = MetricsRequest(
            date_from=now - timedelta(hours=1), date_to=now + timedelta(hours=1), explain=False
        )
        snapshot = await service.get_metrics(req)
        assert snapshot.total_events == 5
        assert snapshot.active_users == 3
        assert snapshot.appointment_completion_rate == 0.5
        assert snapshot.emergency_events == 1

    def test_narrative_guard_accepts_grounded_numbers(self):
        snapshot = MetricsSnapshot(
            date_from=datetime.now(timezone.utc), date_to=datetime.now(timezone.utc),
            total_events=10, active_users=5, event_counts={"app_open": 10}, emergency_events=0,
        )
        agent = AnalyticsNarrativeAgent(llm_client=None)
        text = "The platform saw 10 events from 5 active users."
        assert agent._numbers_are_grounded(text, snapshot) is True

    def test_narrative_guard_rejects_fabricated_numbers(self):
        snapshot = MetricsSnapshot(
            date_from=datetime.now(timezone.utc), date_to=datetime.now(timezone.utc),
            total_events=10, active_users=5, event_counts={"app_open": 10}, emergency_events=0,
        )
        agent = AnalyticsNarrativeAgent(llm_client=None)
        text = "The platform saw 47 events, a 300% increase."
        assert agent._numbers_are_grounded(text, snapshot) is False

    def test_router_builds_with_expected_routes(self):
        async def dummy_get_current_user():
            pass

        def dummy_require_role(*roles):
            async def _dep():
                pass
            return _dep

        router = build_analytics_router(dummy_get_current_user, dummy_require_role)
        paths = {r.path for r in router.routes}
        assert "/analytics/events" in paths
        assert "/analytics/metrics" in paths


# --------------------------------------------------------------------------
# Cross-agent: the wired-up FastAPI app (backend/main.py)
# --------------------------------------------------------------------------

class TestWiredApp:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_all_seven_agents_are_mounted(self, client):
        r = client.get("/openapi.json")
        paths = set(r.json()["paths"].keys())
        expected_prefixes = [
            "/api/v1/users", "/api/v1/doctors", "/api/v1/schemes", "/api/v1/prescriptions",
            "/api/v1/medicines", "/api/v1/recommendations", "/api/v1/assistant", "/api/v1/voice",
            "/api/v1/analytics",
        ]
        for prefix in expected_prefixes:
            assert any(p.startswith(prefix) for p in paths), f"No route found for {prefix}"

    def test_assistant_endpoint_reachable_through_app(self, client):
        r = client.post("/api/v1/assistant/message", json={"user_id": "u1", "text": "hello"})
        assert r.status_code == 200
        assert r.json()["is_emergency"] is False