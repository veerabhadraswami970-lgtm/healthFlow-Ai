"""
backend/routes/users.py
------------------------
Users & Auth route: registration, login, and current-user resolution.
Mirrors the DoctorProfile pattern in backend/routes/doctors.py so both
routers share the same authentication mechanism.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    HOSPITAL = "hospital"
    ADMIN = "admin"


class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    full_name: str
    role: UserRole = UserRole.PATIENT


class UserLogin(BaseModel):
    email: str
    password: str


class UserPublic(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole
    created_at: datetime


class UserInDB(UserPublic):
    hashed_password: str


class AuthResult(BaseModel):
    user: UserPublic
    access_token: str
    token_type: str = "bearer"


class AuthError(Exception):
    pass


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(candidate.hex(), hash_hex)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _create_token(payload: dict, secret: str, expires_in_seconds: int = 60 * 60 * 24) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    body = dict(payload)
    now = int(time.time())
    body.setdefault("iat", now)
    body.setdefault("exp", now + expires_in_seconds)

    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url_encode(json.dumps(body, separators=(",", ":")).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def _decode_token(token: str, secret: str) -> dict:
    try:
        header_b64, body_b64, sig_b64 = token.split(".")
    except ValueError:
        raise AuthError("Malformed token.")

    signing_input = f"{header_b64}.{body_b64}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        raise AuthError("Malformed token signature.")
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise AuthError("Invalid token signature.")

    try:
        body = json.loads(_b64url_decode(body_b64))
    except (ValueError, json.JSONDecodeError):
        raise AuthError("Malformed token payload.")

    exp = body.get("exp")
    if exp is not None and time.time() > exp:
        raise AuthError("Token has expired.")

    return body


class UserRepository:
    async def create(self, user: UserInDB) -> UserInDB:
        raise NotImplementedError

    async def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        raise NotImplementedError

    async def get_by_email(self, email: str) -> Optional[UserInDB]:
        raise NotImplementedError


class InMemoryUserRepository(UserRepository):
    def __init__(self) -> None:
        self._store: dict[str, UserInDB] = {}

    async def create(self, user: UserInDB) -> UserInDB:
        self._store[user.id] = user
        return user

    async def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        return self._store.get(user_id)

    async def get_by_email(self, email: str) -> Optional[UserInDB]:
        for user in self._store.values():
            if user.email == email:
                return user
        return None


class AuthService:
    def __init__(self, repository: UserRepository, jwt_secret: Optional[str] = None) -> None:
        self._repo = repository
        self._secret = jwt_secret or os.environ.get("JWT_SECRET_KEY", "dev-secret-key")

    async def register(self, req: UserCreate) -> AuthResult:
        existing = await self._repo.get_by_email(req.email)
        if existing is not None:
            raise AuthError("An account with this email already exists.")

        user = UserInDB(
            id=str(uuid.uuid4()),
            email=req.email,
            full_name=req.full_name,
            role=req.role,
            created_at=datetime.now(timezone.utc),
            hashed_password=_hash_password(req.password),
        )
        await self._repo.create(user)
        token = self._issue_token(user)
        return AuthResult(user=_to_public(user), access_token=token)

    async def login(self, req: UserLogin) -> AuthResult:
        user = await self._repo.get_by_email(req.email)
        if user is None or not _verify_password(req.password, user.hashed_password):
            raise AuthError("Incorrect email or password.")
        token = self._issue_token(user)
        return AuthResult(user=_to_public(user), access_token=token)

    async def get_current_user(self, token: str) -> UserPublic:
        payload = _decode_token(token, self._secret)
        user_id = payload.get("sub")
        if not user_id:
            raise AuthError("Token is missing a subject claim.")
        user = await self._repo.get_by_id(user_id)
        if user is None:
            raise AuthError("User no longer exists.")
        return _to_public(user)

    def _issue_token(self, user: UserInDB) -> str:
        return _create_token(
            {"sub": user.id, "email": user.email, "role": user.role.value}, self._secret
        )


def _to_public(user: UserInDB) -> UserPublic:
    return UserPublic(
        id=user.id, email=user.email, full_name=user.full_name,
        role=user.role, created_at=user.created_at,
    )


def build_router():
    from fastapi import APIRouter, Depends, HTTPException, status
    from fastapi.security import OAuth2PasswordBearer

    router = APIRouter(prefix="/users", tags=["users"])

    _repo = InMemoryUserRepository()

    def get_auth_service() -> AuthService:
        return AuthService(_repo)

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login", auto_error=False)

    async def get_current_user(
        token: Optional[str] = Depends(oauth2_scheme),
        service: AuthService = Depends(get_auth_service),
    ) -> UserPublic:
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
        try:
            return await service.get_current_user(token)
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    def require_role(*roles: UserRole):
        async def _dependency(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
            if current_user.role not in roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to perform this action.",
                )
            return current_user
        return _dependency

    @router.post("/register", response_model=AuthResult, status_code=status.HTTP_201_CREATED)
    async def register(req: UserCreate, service: AuthService = Depends(get_auth_service)):
        try:
            return await service.register(req)
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    @router.post("/login", response_model=AuthResult)
    async def login(req: UserLogin, service: AuthService = Depends(get_auth_service)):
        try:
            return await service.login(req)
        except AuthError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    @router.get("/me", response_model=UserPublic)
    async def get_me(current_user: UserPublic = Depends(get_current_user)):
        return current_user

    return router, get_current_user, require_role
