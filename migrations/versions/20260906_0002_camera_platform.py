"""Add users, cameras, stream sessions, and inference events."""
from alembic import op
import sqlalchemy as sa

revision = "20260906_0002"
down_revision = "20260906_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.create_table(
        "app_users",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("display_name", sa.String(120), nullable=False),
        sa.Column("password_hash", sa.LargeBinary(), nullable=False),
        sa.Column("password_salt", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.LargeBinary(), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "cameras",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("location", sa.String(180), nullable=False, server_default=""),
        sa.Column("host", sa.String(255), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False, server_default="554"),
        sa.Column("rtsp_path", sa.String(500), nullable=False),
        sa.Column("encrypted_username", sa.Text(), nullable=False),
        sa.Column("encrypted_password", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "stream_sessions",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("camera_id", sa.UUID(), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("started_by", sa.UUID(), sa.ForeignKey("app_users.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("stopped_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "inference_events",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("camera_id", sa.UUID(), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("detections", sa.JSON(), nullable=False),
        sa.Column("inference_ms", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_user_sessions_expires_at", "user_sessions", ["expires_at"])
    op.create_index("ix_stream_sessions_camera_started", "stream_sessions", ["camera_id", "started_at"])
    op.create_index("ix_inference_events_camera_created", "inference_events", ["camera_id", "created_at"])


def downgrade() -> None:
    op.drop_table("inference_events")
    op.drop_table("stream_sessions")
    op.drop_table("cameras")
    op.drop_table("user_sessions")
    op.drop_table("app_users")

