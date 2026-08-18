"""认证与用户账户服务。"""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from typing import Any, Dict, Optional
import uuid

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.storage.sqlite_metadata import AuthSession, SessionLocal, UserAccount
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)
_PBKDF2_ITERATIONS = 210_000
_PASSWORD_SALT_BYTES = 16
_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_TYPE = "access"
_REFRESH_TOKEN_TYPE = "refresh"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(_PASSWORD_SALT_BYTES)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return salt.hex(), password_hash.hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    """校验明文密码是否匹配数据库哈希。"""
    _, computed_hash = _hash_password(password, salt_hex=salt_hex)
    return hmac.compare_digest(computed_hash, expected_hash_hex or "")


def _serialize_user(user: UserAccount) -> Dict[str, Any]:
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "is_active": bool(user.is_active),
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login_at": user.last_login_at,
    }


def _get_user_by_username(db, username: str) -> Optional[UserAccount]:
    return (
        db.query(UserAccount)
        .filter(UserAccount.username == _normalize_username(username))
        .first()
    )


def _get_user_by_id(db, user_id: str) -> Optional[UserAccount]:
    return db.query(UserAccount).filter(UserAccount.user_id == user_id).first()


def _get_session_by_id(db, session_id: str) -> Optional[AuthSession]:
    return db.query(AuthSession).filter(AuthSession.session_id == session_id).first()


def _revoke_user_sessions(db, user_id: str, exclude_session_id: Optional[str] = None) -> None:
    now = _utcnow()
    query = db.query(AuthSession).filter(
        AuthSession.user_id == user_id,
        AuthSession.revoked_at.is_(None),
    )
    if exclude_session_id:
        query = query.filter(AuthSession.session_id != exclude_session_id)

    for session in query.all():
        session.revoked_at = now
        session.updated_at = now


def _create_user(db, username: str, password: str, role: str = "admin") -> UserAccount:
    salt_hex, password_hash = _hash_password(password)
    user = UserAccount(
        user_id=uuid.uuid4().hex,
        username=_normalize_username(username),
        password_hash=password_hash,
        password_salt=salt_hex,
        role=role or "admin",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def seed_bootstrap_admin_user() -> None:
    """按环境变量创建第一个管理员账号。"""
    username = _normalize_username(settings.BOOTSTRAP_ADMIN_USERNAME or "")
    password = settings.BOOTSTRAP_ADMIN_PASSWORD or ""
    if not username or not password:
        return

    db = SessionLocal()
    try:
        existing = _get_user_by_username(db, username)
        if existing:
            return

        _create_user(db, username=username, password=password, role=settings.BOOTSTRAP_ADMIN_ROLE)
        logger.info(f"✓ 已创建 bootstrap 管理员账号: {username}")
    finally:
        db.close()


def register_user(username: str, password: str, confirm_password: str) -> Dict[str, Any]:
    """创建普通用户账号，并直接签发登录令牌。"""
    normalized_username = _normalize_username(username)
    if not normalized_username:
        raise HTTPException(status_code=400, detail="账号不能为空")
    if not password:
        raise HTTPException(status_code=400, detail="密码不能为空")
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")

    db = SessionLocal()
    try:
        existing = _get_user_by_username(db, normalized_username)
        if existing:
            raise HTTPException(status_code=409, detail="账号已存在")

        user = _create_user(db, username=normalized_username, password=password, role="user")
        return create_token_bundle_for_user(_serialize_user(user))
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """校验用户名和密码，成功时返回用户信息。"""
    normalized_username = _normalize_username(username)
    if not normalized_username or not password:
        return None

    db = SessionLocal()
    try:
        user = _get_user_by_username(db, normalized_username)
        if not user or not user.is_active:
            return None
        if not verify_password(password, user.password_salt, user.password_hash):
            return None

        user.last_login_at = _utcnow()
        db.commit()
        db.refresh(user)
        return _serialize_user(user)
    finally:
        db.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """按用户名查询公开用户信息。"""
    db = SessionLocal()
    try:
        user = _get_user_by_username(db, username)
        return _serialize_user(user) if user else None
    finally:
        db.close()


def _build_access_token(user: Dict[str, Any], session_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user["username"],
        "uid": user["user_id"],
        "role": user.get("role", "admin"),
        "sid": session_id,
        "typ": _ACCESS_TOKEN_TYPE,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=_JWT_ALGORITHM)


def _renew_session_access_token(
    db,
    user: UserAccount,
    session: AuthSession,
    now: datetime,
) -> str:
    """续期当前会话，并生成新的访问令牌供客户端替换。"""
    session.last_used_at = now
    session.updated_at = now
    session.expires_at = now + timedelta(minutes=settings.SESSION_IDLE_TIMEOUT_MINUTES)
    access_token = _build_access_token(_serialize_user(user), session.session_id)
    return access_token


def _build_refresh_token(user: Dict[str, Any], session_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user["username"],
        "uid": user["user_id"],
        "role": user.get("role", "admin"),
        "sid": session_id,
        "typ": _REFRESH_TOKEN_TYPE,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str, expected_type: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[_JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="登录凭证无效") from exc

    if payload.get("typ") != expected_type:
        raise HTTPException(status_code=401, detail="登录凭证无效")
    if not payload.get("sub") or not payload.get("uid") or not payload.get("sid"):
        raise HTTPException(status_code=401, detail="登录凭证无效")
    return payload


def _issue_token_bundle(
    user: Dict[str, Any],
    *,
    revoke_existing_sessions: bool = True,
) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        if revoke_existing_sessions:
            _revoke_user_sessions(db, user["user_id"])

        session_id = uuid.uuid4().hex
        refresh_token = _build_refresh_token(user, session_id)
        access_token = _build_access_token(user, session_id)
        session = AuthSession(
            session_id=session_id,
            user_id=user["user_id"],
            refresh_token_hash=_hash_token(refresh_token),
            expires_at=_utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES),
            revoked_at=None,
            created_at=_utcnow(),
            updated_at=_utcnow(),
            last_used_at=_utcnow(),
        )
        db.add(session)
        db.commit()
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "access_expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "refresh_expires_in": settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
            "user": user,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_token_bundle_for_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """创建新的登录令牌，并让该用户之前的会话失效。"""
    return _issue_token_bundle(user, revoke_existing_sessions=True)


def refresh_token_bundle(refresh_token: str) -> Dict[str, Any]:
    """根据刷新令牌签发新的访问/刷新令牌。"""
    payload = _decode_token(refresh_token, _REFRESH_TOKEN_TYPE)

    db = SessionLocal()
    try:
        user = _get_user_by_id(db, str(payload.get("uid")))
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="登录已过期，请重新登录")

        session = _get_session_by_id(db, str(payload.get("sid")))
        if (
            not session
            or session.user_id != user.user_id
            or session.revoked_at is not None
            or session.expires_at <= _utcnow()
            or session.refresh_token_hash != _hash_token(refresh_token)
        ):
            raise HTTPException(status_code=401, detail="登录已过期，请重新登录")

        session.revoked_at = _utcnow()
        session.updated_at = _utcnow()
        db.commit()
        return _issue_token_bundle(_serialize_user(user), revoke_existing_sessions=False)
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def change_password_and_rotate_tokens(
    current_user: Dict[str, Any],
    current_password: str,
    new_password: str,
) -> Dict[str, Any]:
    """修改密码，并让旧令牌失效。"""
    db = SessionLocal()
    try:
        user = _get_user_by_id(db, current_user["user_id"])
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="未登录或登录已失效")
        if not verify_password(current_password, user.password_salt, user.password_hash):
            raise HTTPException(status_code=400, detail="当前密码不正确")

        salt_hex, password_hash = _hash_password(new_password)
        user.password_salt = salt_hex
        user.password_hash = password_hash
        user.updated_at = _utcnow()
        _revoke_user_sessions(db, user.user_id)

        session_id = uuid.uuid4().hex
        user_payload = _serialize_user(user)
        refresh_token = _build_refresh_token(user_payload, session_id)
        access_token = _build_access_token(user_payload, session_id)
        session = AuthSession(
            session_id=session_id,
            user_id=user.user_id,
            refresh_token_hash=_hash_token(refresh_token),
            expires_at=_utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES),
            revoked_at=None,
            created_at=_utcnow(),
            updated_at=_utcnow(),
            last_used_at=_utcnow(),
        )
        db.add(session)
        db.commit()
        db.refresh(user)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "access_expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "refresh_expires_in": settings.REFRESH_TOKEN_EXPIRE_MINUTES * 60,
            "user": _serialize_user(user),
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def decode_access_token(token: str) -> Dict[str, Any]:
    """解析并校验访问 JWT。"""
    return _decode_token(token, _ACCESS_TOKEN_TYPE)


def require_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """FastAPI 依赖：校验登录态并按交互滑动续期。"""
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="未登录或登录已失效")

    payload = decode_access_token(credentials.credentials)
    db = SessionLocal()
    try:
        user = _get_user_by_id(db, str(payload.get("uid")))
        session = _get_session_by_id(db, str(payload.get("sid")))
        if (
            not user
            or not user.is_active
            or not session
            or session.user_id != user.user_id
            or session.revoked_at is not None
            or session.expires_at <= _utcnow()
        ):
            raise HTTPException(status_code=401, detail="未登录或登录已失效")

        now = _utcnow()
        renewed_access_token = _renew_session_access_token(
            db,
            user,
            session,
            now,
        )
        db.commit()
        request.state.renewed_access_token = renewed_access_token
        return _serialize_user(user)
    finally:
        db.close()
