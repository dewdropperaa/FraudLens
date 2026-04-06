from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from claimguard.config import load_environment
from claimguard.db.models import Base, ClaimRecord  # noqa: F401 — register table metadata

load_environment()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./claimguard.db")

# SQLite needs check_same_thread=False for FastAPI threadpool usage
_connect_args: dict = {}
_engine_kwargs: dict = {"pool_pre_ping": True}
if "sqlite" in DATABASE_URL:
    _connect_args["check_same_thread"] = False
    if ":memory:" in DATABASE_URL:
        _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    **_engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
