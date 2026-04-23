"""initial schema

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-04-23 21:30:00.000000
"""

from __future__ import annotations

from alembic import op

from afl_prediction_agent.storage.models import Base


# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
