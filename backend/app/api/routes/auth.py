"""认证路由。"""
from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    LoginRequest,
    PasswordChangeRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import (
    authenticate_user,
    create_token_bundle_for_user,
    change_password_and_rotate_tokens,
    refresh_token_bundle,
    require_current_user,
    register_user,
)

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """用户名密码登录，返回 JWT。"""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token_bundle = create_token_bundle_for_user(user)
    return TokenResponse(
        access_token=token_bundle["access_token"],
        refresh_token=token_bundle["refresh_token"],
        access_expires_in=token_bundle["access_expires_in"],
        refresh_expires_in=token_bundle["refresh_expires_in"],
        user=UserResponse(**token_bundle["user"]),
    )


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """创建新账号并直接登录。"""
    token_bundle = register_user(
        username=request.username,
        password=request.password,
        confirm_password=request.confirm_password,
    )
    return TokenResponse(
        access_token=token_bundle["access_token"],
        refresh_token=token_bundle["refresh_token"],
        access_expires_in=token_bundle["access_expires_in"],
        refresh_expires_in=token_bundle["refresh_expires_in"],
        user=UserResponse(**token_bundle["user"]),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """刷新登录令牌，并让旧令牌失效。"""
    token_bundle = refresh_token_bundle(request.refresh_token)
    return TokenResponse(
        access_token=token_bundle["access_token"],
        refresh_token=token_bundle["refresh_token"],
        access_expires_in=token_bundle["access_expires_in"],
        refresh_expires_in=token_bundle["refresh_expires_in"],
        user=UserResponse(**token_bundle["user"]),
    )


@router.post("/password", response_model=TokenResponse)
async def change_password(request: PasswordChangeRequest, current_user=Depends(require_current_user)):
    """修改当前用户密码，并自动旋转登录态。"""
    token_bundle = change_password_and_rotate_tokens(
        current_user=current_user,
        current_password=request.current_password,
        new_password=request.new_password,
    )
    return TokenResponse(
        access_token=token_bundle["access_token"],
        refresh_token=token_bundle["refresh_token"],
        access_expires_in=token_bundle["access_expires_in"],
        refresh_expires_in=token_bundle["refresh_expires_in"],
        user=UserResponse(**token_bundle["user"]),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user=Depends(require_current_user)):
    """返回当前登录用户。"""
    return UserResponse(**current_user)
