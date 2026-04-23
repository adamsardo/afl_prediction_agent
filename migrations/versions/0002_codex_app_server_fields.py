"""codex app-server fields

Revision ID: 0002_codex_app_server_fields
Revises: 0001_initial_schema
Create Date: 2026-04-23 22:55:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from afl_prediction_agent.core.db.types import JSON_VARIANT


# revision identifiers, used by Alembic.
revision = "0002_codex_app_server_fields"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def _column_map(table_name: str) -> dict[str, dict]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"]: column for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    run_config_columns = _column_map("run_configs")
    with op.batch_alter_table("run_configs") as batch_op:
        if "default_reasoning_effort" not in run_config_columns:
            batch_op.add_column(sa.Column("default_reasoning_effort", sa.String(length=20), nullable=True))
        if not run_config_columns.get("default_temperature", {}).get("nullable", True):
            batch_op.alter_column("default_temperature", existing_type=sa.Numeric(4, 2), nullable=True)

    agent_step_columns = _column_map("agent_steps")
    with op.batch_alter_table("agent_steps") as batch_op:
        if "reasoning_effort" not in agent_step_columns:
            batch_op.add_column(sa.Column("reasoning_effort", sa.String(length=20), nullable=True))
        if "provider_meta" not in agent_step_columns:
            batch_op.add_column(
                sa.Column(
                    "provider_meta",
                    JSON_VARIANT,
                    nullable=False,
                    server_default=sa.text("'{}'"),
                )
            )
        if not agent_step_columns.get("temperature", {}).get("nullable", True):
            batch_op.alter_column("temperature", existing_type=sa.Numeric(4, 2), nullable=True)


def downgrade() -> None:
    agent_step_columns = _column_map("agent_steps")
    with op.batch_alter_table("agent_steps") as batch_op:
        if "provider_meta" in agent_step_columns:
            batch_op.drop_column("provider_meta")
        if "reasoning_effort" in agent_step_columns:
            batch_op.drop_column("reasoning_effort")

    run_config_columns = _column_map("run_configs")
    with op.batch_alter_table("run_configs") as batch_op:
        if "default_reasoning_effort" in run_config_columns:
            batch_op.drop_column("default_reasoning_effort")
