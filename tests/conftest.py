"""All pytest suites prohibit network, including newly added evaluation fixtures."""
import socket
import pytest


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def blocked(*args,**kwargs):
        raise AssertionError("Offline tests prohibit network connections")
    monkeypatch.setattr(socket.socket,"connect",blocked)
    monkeypatch.setattr(socket.socket,"connect_ex",blocked)
