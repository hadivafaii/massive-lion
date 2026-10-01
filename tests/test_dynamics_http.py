"""Malformed local requests must return JSON and preserve the active run."""

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading

import pytest

from dynamics_lab import interactive
from dynamics_lab.engine import SimulationState


@pytest.fixture
def local_server(monkeypatch):
    state = SimulationState()
    state.reset({'max_steps': 3, 'optimizers': [{'name': 'Lion'}]})
    monkeypatch.setattr(interactive, 'STATE', state)
    server = ThreadingHTTPServer(('127.0.0.1', 0), interactive.Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    worker.start()
    try:
        yield server.server_address, state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize('body,length', [
    (b'{', '1'), (b'[]', '2'), (b'null', '4'), (b'\xff', '1'),
    (b'{"max_steps":NaN}', '17'), (b'', '-1'), (b'', 'not-a-number'),
    (b'', str(interactive.MAX_REQUEST_BYTES + 1)),
])
def test_invalid_reset_is_json_error_and_leaves_run_usable(local_server, body, length):
    address, state = local_server
    before = state.snapshot(include_landscape=True)
    connection = HTTPConnection(*address, timeout=3)
    try:
        connection.request('POST', '/api/reset', body=body,
                           headers={'Content-Length': length, 'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 400
        assert 'error' in json.loads(response.read())
    finally:
        connection.close()
    assert state.snapshot(include_landscape=True) == before
    connection = HTTPConnection(*address, timeout=3)
    try:
        connection.request('POST', '/api/step')
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())['global_step'] == 1
    finally:
        connection.close()
