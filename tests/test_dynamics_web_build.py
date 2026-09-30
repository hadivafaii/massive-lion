import json

import pytest

from dynamics_lab import build_web


def test_hosted_html_keeps_renderer_but_removes_local_bootstrap():
    html = build_web.hosted_html()
    assert 'function drawLandscape2D(' in html
    assert '(async function init() {' not in html
    assert 'src="./site.js"' in html
    assert 'href="/favicon' not in html


def test_frozen_bundle_survives_build_without_recomputation(tmp_path, monkeypatch):
    published = tmp_path / 'published'
    published.mkdir()
    bundle = {'schema_version': 1, 'id': 'blog-figure-v1', 'title': 'A fixed figure',
              'config': {'schema_version': 2, 'config': {'max_steps': 2}},
              'snapshot': {'done': True, 'learners': [{'trace': [1, 2]}], 'landscape': {}},
              'provenance': {'commit': 'old-commit'}}
    (published / 'figure.json').write_text(json.dumps(bundle))
    monkeypatch.setattr(build_web, 'SCENES', ())
    monkeypatch.setattr(build_web, 'simulate', lambda _: pytest.fail('Frozen scene was recomputed'))
    output = tmp_path / 'site'
    manifest = build_web.build(output, published=published)
    assert manifest['scenes'][0]['id'] == 'blog-figure-v1'
    assert json.loads((output / 'scenes/blog-figure-v1.json').read_text()) == bundle


def test_incomplete_scene_is_not_publishable():
    with pytest.raises(ValueError, match='Finish'):
        build_web.validate_bundle({'schema_version': 1, 'id': 'unfinished', 'title': 'Test',
            'config': {}, 'snapshot': {'done': False, 'landscape': {}, 'learners': [{}]}})
