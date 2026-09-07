from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

url = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER', 'camera_app')}:{os.getenv('POSTGRES_PASSWORD', 'change-me')}@"
    f"{os.getenv('POSTGRES_HOST', 'db')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'cameras')}"
)
config.set_main_option('sqlalchemy.url', url.replace('%', '%%'))


def run_migrations_offline() -> None:
    context.configure(url=url, literal_binds=True, dialect_opts={'paramstyle': 'named'})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}), prefix='sqlalchemy.', poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

