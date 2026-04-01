from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg.errors import UniqueViolation

from db import (
    create_session,
    create_user,
    delete_session,
    get_user_by_token,
    get_user_by_username,
    list_users,
    verify_password,
)
from schemas.schema_auth import (
    AuthLoginRequest,
    AuthLoginResponse,
    UserCreateRequest,
    UserResponse,
)


router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация",
        )

    user = get_user_by_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия не найдена или истекла",
        )

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Пользователь деактивирован",
        )

    return user


def get_admin_user(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав",
        )
    return user


@router.post("/login", response_model=AuthLoginResponse)
async def login(request: AuthLoginRequest):
    user = get_user_by_username(request.username)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
        )

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Пользователь деактивирован",
        )

    access_token, expires_at = create_session(user["id"])

    return AuthLoginResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=UserResponse(**{key: value for key, value in user.items() if key != "password_hash"}),
    )


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    if credentials is not None:
        delete_session(credentials.credentials)

    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    return UserResponse(**user)


@router.get("/users", response_model=list[UserResponse])
async def users(_: dict = Depends(get_admin_user)):
    return [UserResponse(**user) for user in list_users()]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_endpoint(request: UserCreateRequest, _: dict = Depends(get_admin_user)):
    try:
        user = create_user(
            username=request.username,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            role=request.role,
        )
    except UniqueViolation:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Пользователь с таким логином или email уже существует",
        )

    return UserResponse(**user)
