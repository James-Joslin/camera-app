"""Create the initial camera run schema."""
from alembic import op
import sqlalchemy as sa

revision = "20260906_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "camera_runs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_camera_runs_status", "camera_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_camera_runs_status", table_name="camera_runs")
    op.drop_table("camera_runs")

