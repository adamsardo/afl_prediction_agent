from __future__ import annotations

import pytest
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from afl_prediction_agent.core.db.base import Base


@pytest.fixture()
def session(tmp_path: Path) -> Session:
    db_path = tmp_path / "test.sqlite3"
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    with SessionLocal() as db:
        yield db
