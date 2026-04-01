from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr | None = None
    first_name: str
    last_name: str
    role: str
    is_active: bool
    created_at: datetime


class AuthLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    first_name: str = Field(min_length=1, max_length=128)
    last_name: str = Field(min_length=1, max_length=128)
    email: EmailStr | None = None
    role: str = Field(default="viewer", pattern="^(admin|editor|viewer)$")
