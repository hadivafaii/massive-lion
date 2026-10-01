"""Publication must reject backend/source drift and unusable simulations."""

from copy import deepcopy

import pytest

from dynamics_lab import check_deployment


@pytest.fixture
def backend(monkeypatch):
    result = {'provenance': {'commit': 'abc', 'source_sha256': 'digest'},
              'snapshot': {'done': True, 'global_step': 3,
                           'learners': [{'trace': [0, 1, 2, 3]}] * 2}}
    health = {'status': 'ok', 'engine': 'repository-python', 'revision': 'abc'}
    monkeypatch.setattr(check_deployment, 'runtime_provenance', lambda: {'source_sha256': 'digest'})
    calls = []
    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return deepcopy(health if url.endswith('/health') else result)
    monkeypatch.setattr(check_deployment, 'fetch_json', fetch)
    return health, result, calls


def test_checks_matching_revision_and_real_simulation(backend):
    check_deployment.check('https://api.example', 'abc', 'https://site.example')
    calls = backend[2]
    assert len(calls) == 2
    assert calls[1][1]['payload']['max_steps'] == 3
    assert calls[1][1]['origin'] == 'https://site.example'


@pytest.mark.parametrize('damage', ['revision', 'source', 'incomplete'])
def test_drift_or_incomplete_run_blocks_publication(backend, damage):
    health, result, calls = backend
    if damage == 'revision':
        health['revision'] = 'old'
    elif damage == 'source':
        result['provenance']['source_sha256'] = 'old-digest'
    else:
        result['snapshot']['done'] = False
    with pytest.raises(ValueError):
        check_deployment.check('https://api.example', 'abc', 'https://site.example')
    if damage == 'revision':
        assert len(calls) == 1
