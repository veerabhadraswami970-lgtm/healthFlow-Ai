"""
main.py
-------
Entry point that wires all HealthFlow AI routes into one FastAPI app.
Run it with:
    uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/docs for an interactive UI.
"""
from fastapi import FastAPI
from backend.routes.users import build_router as build_users_router
from backend.routes.doctors import build_router as build_doctors_router
from backend.routes.hospitals import build_router as build_hospitals_router
from backend.routes.appointments import build_router as build_appointments_router
from backend.routes.emergency import build_router as build_emergency_router
from backend.routes.blood_bank import build_router as build_blood_bank_router
from backend.routes.qr_sharing import build_router as build_qr_sharing_router
from backend.routes.health_records import build_router as build_health_records_router
from backend.agents.health_scheme import build_router as build_scheme_router
from backend.agents.prescription_ocr import build_router as build_prescription_router
from backend.agents.medicine_intelligence import build_router as build_medicine_router
from backend.agents.recommendation_agent import build_router as build_recommendation_router
from backend.agents.health_assistant import build_router as build_assistant_router
from backend.agents.voice_assistant import build_router as build_voice_router
from backend.agents.analytics_agent import build_router as build_analytics_router
app = FastAPI(
    title="HealthFlow AI",
    description="Backend API for HealthFlow AI's core routes and seven agents.",
    version="1.0.0",
)
@app.get("/health")
async def health_check():
    """Quick liveness check — hit this first to confirm the server is up."""
    return {"status": "ok", "service": "HealthFlow AI backend"}
# Core domain routes
users_router, get_current_user, require_role = build_users_router()
app.include_router(users_router, prefix="/api/v1")
app.include_router(build_doctors_router(get_current_user, require_role), prefix="/api/v1")
app.include_router(build_hospitals_router(get_current_user, require_role), prefix="/api/v1")
app.include_router(build_appointments_router(get_current_user, require_role), prefix="/api/v1")
app.include_router(build_emergency_router(get_current_user, require_role), prefix="/api/v1")
app.include_router(build_blood_bank_router(get_current_user, require_role), prefix="/api/v1")
app.include_router(build_qr_sharing_router(get_current_user, require_role), prefix="/api/v1")
app.include_router(build_health_records_router(get_current_user, require_role), prefix="/api/v1")
# AI agent routes
app.include_router(build_scheme_router(), prefix="/api/v1")
app.include_router(build_prescription_router(), prefix="/api/v1")
app.include_router(build_medicine_router(), prefix="/api/v1")
app.include_router(build_recommendation_router(), prefix="/api/v1")
app.include_router(build_assistant_router(), prefix="/api/v1")
app.include_router(build_voice_router(), prefix="/api/v1")
app.include_router(build_analytics_router(), prefix="/api/v1")
