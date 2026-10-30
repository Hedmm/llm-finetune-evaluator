"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import socket
from typing import Any

import pytest

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_INET_FAMILIES = (socket.AF_INET, socket.AF_INET6)


def _guarded_connect(self: socket.socket, address: Any) -> None:
    if self.family in _INET_FAMILIES:
        raise RuntimeError(f"tests must not open network connections (tried {address!r})")
    _REAL_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    if self.family in _INET_FAMILIES:
        raise RuntimeError(f"tests must not open network connections (tried {address!r})")
    return _REAL_CONNECT_EX(self, address)


@pytest.fixture(autouse=True)
def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to reach the network."""
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)


@pytest.fixture(autouse=True)
def _isolated_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the generation cache of every test out of the user's home directory."""
    monkeypatch.setenv("LFE_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
