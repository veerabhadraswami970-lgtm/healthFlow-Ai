# HealthFlow AI - Backend

FastAPI backend for HealthFlow AI, a healthcare platform connecting patients, doctors, hospitals, and admins. All data is currently in-memory (resets on server restart) - no database is required to run or test this project.

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Run the tests

python -m pytest -v

Should show 133 passed.

## Run the server

uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API explorer.

## Project structure

backend/main.py - Wires every module into one FastAPI app
backend/routes/users.py - Registration, login, JWT auth
backend/routes/doctors.py - Doctor profiles, verification, availability
backend/routes/hospitals.py - Hospital profiles, verification, services
backend/routes/appointments.py - Booking, confirm, complete, cancel, reschedule
backend/routes/emergency.py - SOS trigger/confirm/resolve flow, contacts
backend/routes/blood_bank.py - Blood bank registration, stock, search
backend/routes/qr_sharing.py - Expiring/revocable prescription-access tokens
backend/routes/health_records.py - Patient health records with access audit logging
backend/agents/ - 7 AI agents: health_scheme, prescription_ocr, medicine_intelligence, recommendation_agent, health_assistant, voice_assistant, analytics_agent
backend/tests/ - One test file per module, ~130+ tests total

## Architecture pattern

Every module follows: models (Pydantic) -> Repository (ABC) -> InMemoryRepository -> Service -> build_router()

users.py build_router() returns (router, get_current_user, require_role). Every other module's build_router(get_current_user, require_role) returns just the router. Ownership is always checked against the authenticated user, never a client-supplied ID. Admin-only actions use require_role(UserRole.ADMIN).

## User roles

patient, doctor, hospital, admin - set at registration via POST /api/v1/users/register

## Not yet built

- Persistent database (MongoDB or otherwise)
- Frontend (React, per the original project spec)
- Real LLM/OCR/Maps integrations
