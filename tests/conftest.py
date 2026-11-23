"""All pytest suites prohibit network, including newly added evaluation fixtures."""
import socket
import pytest


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def blocked(*args,**kwargs):
        raise AssertionError("Offline tests prohibit network connections")
    monkeypatch.setattr(socket.socket,"connect",blocked)
    monkeypatch.setattr(socket.socket,"connect_ex",blocked)


@pytest.fixture(scope="module")
def collection():
    """Shared base contrast collection fixture (64 rows, 72 links)."""
    from cert_recovery.contrast_data import generate_collection
    return generate_collection()
