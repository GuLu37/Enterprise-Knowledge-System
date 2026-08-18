"""SQL 文档元数据与用户账户存储。"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """ORM 基类"""


class DocumentRecord(Base):
    """文档元数据记录"""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="processing")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class DocumentChunkRecord(Base):
    """文档正文切片，用于预览已上传文档且不依赖服务端原始文件。"""

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_document_order", "document_id", "chunk_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class UserAccount(Base):
    """登录账户记录。"""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    password_salt: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="admin")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AuthSession(Base):
    """登录会话记录。"""

    __tablename__ = "auth_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    refresh_token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class MemoryRecord(Base):
    """长期记忆生命周期元数据。"""

    __tablename__ = "memory_records"
    __table_args__ = (
        Index("ix_memory_records_user_type_status", "user_id", "memory_type", "status"),
        Index("ix_memory_records_user_key_status", "user_id", "memory_key", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String(256), unique=True, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    memory_key: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    base_importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    supersedes_memory_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


def init_metadata_db():
    """创建或校验关系型元数据表结构。"""
    if settings.DATABASE_URL.startswith("sqlite:///"):
        database_path = settings.DATABASE_URL.removeprefix("sqlite:///")
        if database_path and database_path != ":memory:" and not database_path.startswith("file:"):
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    try:
        from app.services.auth_service import seed_bootstrap_admin_user

        seed_bootstrap_admin_user()
    except Exception:
        logger.exception("初始化用户表或 bootstrap 管理员失败")
    logger.debug("✓ SQLite 元数据与认证表初始化完成")


def get_db():
    """生成一个 SQLite 会话，用于依赖注入或手动管理事务。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
