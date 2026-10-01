import json
from copy import deepcopy

import pytest

from dynamics_lab import build_web


@pytest.fixture
def frozen_bundle():
    return build_web.scene_bundle('blog-figure-v1', 'A fixed figure', '',
        {'max_steps': 2, 'optimizers': [{'name': 'MassiveLion'}]}, {'commit': 'old-commit'})


def test_hosted_html_keeps_renderer_but_removes_local_bootstrap():
    html = build_web.hosted_html()
    assert 'function drawLandscape2D(' in html
    assert '(async function init() {' not in html
    assert 'src="./site.js"' in html
    assert 'href="/favicon' not in html


def test_frozen_bundle_survives_build_without_recomputation(tmp_path, monkeypatch, frozen_bundle):
    published = tmp_path / 'published'
    published.mkdir()
    bundle = frozen_bundle
    (published / 'figure.json').write_text(json.dumps(bundle))
    monkeypatch.setattr(build_web, 'SCENES', ())
    monkeypatch.setattr(build_web, 'simulate', lambda _: pytest.fail('Frozen scene was recomputed'))
    output = tmp_path / 'site'
    manifest = build_web.build(output, published=published)
    assert manifest['scenes'][0]['id'] == 'blog-figure-v1'
    assert manifest['default_scene'] == 'blog-figure-v1'
    assert json.loads((output / 'scenes/blog-figure-v1.json').read_text()) == bundle


def test_incomplete_scene_is_not_publishable():
    with pytest.raises(ValueError, match='Finish'):
        build_web.validate_bundle({'schema_version': 1, 'id': 'unfinished', 'title': 'Test',
            'config': {}, 'snapshot': {'done': False, 'landscape': {}, 'learners': [{}]}})


@pytest.mark.parametrize('damage', ['grid', 'position', 'steps', 'id'])
def test_malformed_frozen_figures_fail_before_publication(frozen_bundle, damage):
    bundle = deepcopy(frozen_bundle)
    if damage == 'grid':
        bundle['snapshot']['landscape']['z'][0].pop()
    elif damage == 'position':
        bundle['snapshot']['learners'][0]['trace'][0]['theta'] = [1]
    elif damage == 'steps':
        bundle['snapshot']['learners'][0]['trace'][1]['step'] = 0
    else:
        bundle['id'] = 12
    with pytest.raises(ValueError):
        build_web.validate_bundle(bundle)


@pytest.mark.parametrize('url', ['https://example.com/api', 'https://user:pass@example.com',
    'https://example.com?token=x', 'https://example.com/#api', 'https://example.com:bad'])
def test_bad_api_origins_do_not_produce_broken_sites(tmp_path, url):
    with pytest.raises(ValueError):
        build_web.build(tmp_path / 'site', api_base_url=url)
    assert not (tmp_path / 'site').exists()


def test_packaged_build_provenance_without_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.delenv('RENDER_GIT_COMMIT', raising=False)
    monkeypatch.setattr('dynamics_lab.web_simulation.subprocess.run', missing_git)
    source = build_web.provenance()
    assert source['commit'] is None
    assert source['uncommitted_changes'] is None
    assert len(source['source_sha256']) == 64
