"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_ambient_database(monkeypatch):
    """The offline suite must not write to a developer's real database just because
    GG_DATABASE_URL is exported in their shell. Tests that want a store pass one."""
    monkeypatch.delenv("GG_DATABASE_URL", raising=False)
