"""Unit-Tests für build_checkpointer (Auswahl Redis vs. in-memory).

REDIS_URL entscheidet, ob der Chat-State in Redis (geteilt über alle
Gunicorn-Worker) oder im in-memory BoundedMemorySaver landet. Der RedisSaver
wird gemockt — hier geht es nur um die Auswahl- und Konfigurationslogik,
nicht um die Redis-Anbindung selbst.

REDIS_URL wird manuell gesichert/zurückgesetzt (via object.__getattribute__)
statt über monkeypatch.setattr: EnvHelper.__getattribute__ wirft für Felder
am "UNSET"-Sentinel AttributeError, wodurch monkeypatch das Feld beim
Teardown per delattr entfernen würde (siehe test_drupal_auth.py).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.env import env
from src.llm.assistant import BoundedMemorySaver, build_checkpointer


@pytest.fixture(autouse=True)
def restore_redis_env():
    fields = ["REDIS_URL", "CHAT_TTL_MINUTES", "MAX_CHAT_THREADS"]
    original = {name: object.__getattribute__(env, name) for name in fields}
    yield
    for name, value in original.items():
        setattr(env, name, value)


class TestBuildCheckpointer:
    def test_without_redis_url_falls_back_to_in_memory(self):
        setattr(env, "REDIS_URL", "UNSET")
        setattr(env, "MAX_CHAT_THREADS", 42)

        saver = build_checkpointer()

        assert isinstance(saver, BoundedMemorySaver)
        assert saver.max_threads == 42

    def test_with_redis_url_builds_redis_saver_with_ttl(self):
        setattr(env, "REDIS_URL", "redis://localhost:6379")
        setattr(env, "CHAT_TTL_MINUTES", 123)

        with patch("langgraph.checkpoint.redis.RedisSaver") as saver_cls:
            saver_cls.return_value = MagicMock()
            saver = build_checkpointer()

        saver_cls.assert_called_once_with(
            redis_url="redis://localhost:6379",
            ttl={"default_ttl": 123, "refresh_on_read": True},
        )
        # setup() legt die Suchindizes in Redis an — muss beim Bau passieren
        saver_cls.return_value.setup.assert_called_once_with()
        assert saver is saver_cls.return_value

    def test_managed_redis_style_url_uses_redis_saver(self):
        # rediss:// (TLS, Azure Managed Redis) läuft über denselben Pfad —
        # der Wechsel ist reine Konfiguration.
        url = "rediss://:key@example.germanywestcentral.redis.azure.net:10000"
        setattr(env, "REDIS_URL", url)

        with patch("langgraph.checkpoint.redis.RedisSaver") as saver_cls:
            build_checkpointer()

        assert saver_cls.call_args.kwargs["redis_url"] == url
