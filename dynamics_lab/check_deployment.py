"""Check the deployed engine before publishing a frontend against it."""

import argparse
import json
import time
from urllib.request import Request, urlopen

from dynamics_lab.web_simulation import runtime_provenance


def fetch_json(url, *, origin, payload=None, timeout=30):
    headers = {'Origin': origin, 'Accept': 'application/json'}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
        if response.headers.get('Access-Control-Allow-Origin') != origin:
            raise ValueError(f'Backend does not allow the website origin {origin}')
        return json.load(response)


def check(api_url, expected_revision, origin, *, wake_timeout=150):
    """Wake a free service, then require matching code and a completed real run."""
    api_url = api_url.rstrip('/')
    deadline = time.monotonic() + wake_timeout
    while True:
        try:
            health = fetch_json(api_url + '/api/health', origin=origin,
                                timeout=max(1, min(30, deadline - time.monotonic())))
            break
        except (OSError, json.JSONDecodeError) as error:
            if time.monotonic() >= deadline:
                raise RuntimeError('Backend did not become available before publication') from error
            time.sleep(min(3, max(0, deadline - time.monotonic())))
    if health.get('status') != 'ok' or health.get('engine') != 'repository-python':
        raise ValueError('Backend health did not identify the repository engine')
    if health.get('revision') != expected_revision:
        raise ValueError(f"Backend serves {health.get('revision')}; deploy {expected_revision} in Render first")
    result = fetch_json(api_url + '/api/simulate', origin=origin, timeout=90, payload={
        'max_steps': 3, 'theta0': [-1.8, 2.3], 'noise': {'enabled': False},
        'optimizers': [{'name': 'MassiveLion', 'lr': 0.03}, {'name': 'AdamW', 'lr': 0.03}],
    })
    provenance = result.get('provenance', {})
    if (provenance.get('commit') != expected_revision
            or provenance.get('source_sha256') != runtime_provenance()['source_sha256']):
        raise ValueError('Simulation source differs from the frontend build source')
    snapshot = result.get('snapshot', {})
    if (snapshot.get('done') is not True or snapshot.get('global_step') != 3
            or len(snapshot.get('learners', [])) != 2
            or any(len(learner.get('trace', [])) != 4 for learner in snapshot['learners'])):
        raise ValueError('Backend did not complete the smoke simulation')
    print(f'Backend verified: {expected_revision}, two optimizers, CORS {origin}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-url', required=True)
    parser.add_argument('--expected-revision', required=True)
    parser.add_argument('--origin', default='https://hadivafaii.github.io')
    args = parser.parse_args()
    check(args.api_url, args.expected_revision, args.origin)


if __name__ == '__main__':
    main()
