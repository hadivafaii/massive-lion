import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
import test from 'node:test';
import {JSDOM} from 'jsdom';

const html = await readFile(new URL('../dynamics_lab/interactive.html', import.meta.url), 'utf8');
const localScript = html.match(/<script>([\s\S]*?)<\/script>/)[1].split('(async function init() {')[0];
const hostedScript = await readFile(new URL('../dynamics_lab/web/site.js', import.meta.url), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));
const response = value => ({ok: true, json: async () => clone(value)});
function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
}

const defaults = {
  optimizers: {
    Lion: {lr: .03, beta1: .9, beta2: .99, mass: 0, momentum: .9},
    SGD: {lr: .03, momentum: .9, beta1: .9, beta2: .99, mass: 0},
  },
  optimizer_colors: {Lion: '#1f77b4', SGD: '#d62728'},
  landscape: {kind: 'sloped_ravine', target_x: 2.2, target_y: 0},
  noise: {mode: 'additive', distribution: 'gaussian', std: 0, seed: 0, enabled: false},
  theta0: [-1.8, 2.3], max_steps: 300,
};
function scene(id = 'first', rate = .03) {
  const optimizers = ['Lion', 'SGD'].map((name, i) => ({
    name, instance_id: `optimizer-${i}`, lr: rate, beta1: .9, beta2: .99,
    momentum: .9, color: defaults.optimizer_colors[name],
  }));
  return {
    schema_version: 1, id, title: id, description: `Example ${id}`,
    config: {schema_version: 2, config: {mode: 'parallel', max_steps: 4,
      theta0: [-1, 1], landscape: {kind: 'curved_valley'}, noise: {enabled: false, seed: 0}, optimizers},
      controls: {delay_ms: 35, shared_lr: false, shared_beta: false, shared_mass: false}},
    snapshot: {mode: 'parallel', global_step: 4, total_steps: 4, max_steps: 4, done: true,
      landscape: {curvature_ref: 1, optimum: {theta: [0, 0], loss: 0}}, noise: {enabled: false},
      learners: optimizers.map(spec => ({...spec, id: spec.instance_id, optimizer: spec.name, diverged: false,
        trace: Array.from({length: 5}, (_, step) => ({step, theta: [step / 4, 1], loss: 4 - step})),
        speed_samples: [], kinetic_samples: []}))},
    provenance: {git_commit: '0123456789'},
  };
}

async function until(predicate, message) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await delay(5);
  }
  assert.fail(message || 'UI did not reach the expected state');
}

async function setup(t, {url = 'https://example.test/lab/', hosted = true, fetch: handleFetch} = {}) {
  const dom = new JSDOM(html, {url, runScripts: 'outside-only', pretendToBeVisual: true});
  const {window} = dom;
  t.after(() => window.close());
  window.matchMedia = () => ({matches: true}); // Avoid ambient autoplay in tests.
  window.ResizeObserver = class {observe() {}};
  const calls = [];
  window.fetch = async (url, options = {}) => {
    const path = new URL(url, window.location.href).pathname;
    calls.push({path, options});
    const handled = handleFetch?.(path, options);
    if (handled !== undefined) return handled;
    if (path.endsWith('/defaults.json')) return response(defaults);
    if (path.endsWith('/scenes.json')) return response({default_scene: 'first', scenes: ['first', 'second', 'third'].map(id => ({id, title: id, file: `${id}.json`}))});
    if (path.endsWith('/site-config.json')) return response({api_base_url: '.'});
    if (path.endsWith('/api/health')) return response({status: 'ok'});
    for (const id of ['first', 'second', 'third']) if (path.endsWith(`/${id}.json`)) return response(scene(id));
    throw new Error(`Unexpected request: ${path}`);
  };
  const stubs = `
    drawLandscape = drawLoss = drawThetaPlot = drawVelocityPlot = drawHistogram = () => {};
    defaults = ${JSON.stringify(defaults)};
  `;
  window.eval(localScript + stubs + (hosted ? hostedScript : '') + `
    window.lab = {
      get state() { return {snapshot, fullSnapshot, viewStep, running, needsReset, stepping}; },
      readConfig: () => readConfig(), import: payload => applyImportedPayload(payload),
      run: () => run(), pause: () => pause(), step: () => stepForward(),
    };
  `);
  const find = selector => window.document.querySelector(selector);
  const input = (selector, value, type = 'input') => {
    const node = find(selector);
    if (node.type === 'checkbox') node.checked = value; else node.value = value;
    node.dispatchEvent(new window.Event(type, {bubbles: true}));
  };
  return {window, find, input, calls, lab: window.lab};
}

test('hosted timeline seeks to the requested step instead of its previous value', async t => {
  const {find, input, lab} = await setup(t);
  await until(() => find('#sceneTitle').textContent === 'first');
  input('#timeline', '3');
  assert.equal(lab.state.viewStep, 3);
  assert.equal(find('#timeline').value, '3');
  assert.equal(lab.state.running, false);
  assert.equal(lab.state.snapshot.learners[0].trace.length, 4);
});

test('an obsolete failed scene fetch cannot replace the newer scene status', async t => {
  const old = deferred();
  const {find, input, calls} = await setup(t, {fetch: path => path.endsWith('/second.json') ? old.promise : undefined});
  await until(() => find('#sceneTitle').textContent === 'first');
  input('#scenePicker', 'second', 'change');
  input('#scenePicker', 'third', 'change');
  await until(() => find('#sceneTitle').textContent === 'third');
  old.reject(new Error('Obsolete network error'));
  await delay(10);
  assert.equal(find('#sceneTitle').textContent, 'third');
  assert.equal(find('#siteNotice').hidden, true);
  assert.equal(calls.find(item => item.path.endsWith('/second.json')).options.signal.aborted, true);
});

test('importing during a run cancels the old request and installs only imported paths', async t => {
  const old = deferred();
  let simulationCount = 0;
  const {find, calls, lab} = await setup(t, {fetch: (path, options) => {
    if (!path.endsWith('/api/simulate')) return;
    simulationCount += 1;
    if (simulationCount === 1) return old.promise;
    const rate = JSON.parse(options.body).optimizers[0].lr;
    return response({snapshot: scene('custom', rate).snapshot});
  }});
  await until(() => find('#sceneTitle').textContent === 'first');
  find('#simulateBtn').click();
  await until(() => simulationCount === 1);
  await lab.import(scene('imported', .08).config);
  assert.equal(calls.find(item => item.path.endsWith('/api/simulate')).options.signal.aborted, true);
  assert.equal(lab.state.fullSnapshot.learners[0].lr, .08);
  old.resolve(response({snapshot: scene('obsolete', .03).snapshot}));
  await delay(10);
  assert.equal(lab.state.fullSnapshot.learners[0].lr, .08);
  assert.equal(lab.readConfig().optimizers[0].lr, .08);
});

test('display-only changes preserve an in-flight run; optimizer edits invalidate it', async t => {
  const pending = deferred();
  const {find, input, calls, lab} = await setup(t, {fetch: path => path.endsWith('/api/simulate') ? pending.promise : undefined});
  await until(() => find('#sceneTitle').textContent === 'first');
  find('#simulateBtn').click();
  const request = calls.find(item => item.path.endsWith('/api/simulate'));
  input('#delay_ms', '70');
  assert.equal(request.options.signal.aborted, false);
  input('.opt-lr', '.07');
  assert.equal(request.options.signal.aborted, true);
  pending.resolve(response({snapshot: scene().snapshot}));
  await delay(10);
  assert.equal(lab.state.needsReset, true);
  assert.match(find('#siteNotice').textContent, /Settings changed/);
});

test('custom shared links restore optimizer visibility after simulation completes', async t => {
  const shared = scene('shared', .07).config;
  const hash = encodeURIComponent(JSON.stringify(shared));
  const {find, lab} = await setup(t, {
    url: `https://example.test/lab/?embed=1&scene=first&show=optimizer-1#config=${hash}`,
    fetch: path => path.endsWith('/api/simulate') ? response({snapshot: scene('shared', .07).snapshot}) : undefined,
  });
  await until(() => find('#sceneTitle').textContent === 'Your experiment');
  assert.equal(lab.state.fullSnapshot.learners.length, 1);
  assert.equal(lab.state.fullSnapshot.learners[0].instance_id, 'optimizer-1');
  assert.match(find('.open-lab').href, /show=optimizer-1/);
});

test('downloaded scene bundles can be imported as ordinary configs', async t => {
  const {lab, find} = await setup(t, {fetch: path => path.endsWith('/api/simulate') ? response({snapshot: scene('import', .09).snapshot}) : undefined});
  await until(() => find('#sceneTitle').textContent === 'first');
  await lab.import(scene('import', .09));
  assert.equal(lab.readConfig().optimizers[0].lr, .09);
  assert.equal(lab.state.needsReset, false);
});

test('rejecting an invalid import preserves the loaded configuration and replay', async t => {
  const {lab, find} = await setup(t);
  await until(() => find('#sceneTitle').textContent === 'first');
  const before = JSON.stringify(lab.readConfig());
  await assert.rejects(lab.import({schema_version: 2, config: {optimizers: []}}), /at least one optimizer/);
  assert.equal(JSON.stringify(lab.readConfig()), before);
  assert.equal(lab.state.needsReset, false);
  assert.equal(find('#runBtn').disabled, false);
});

test('sparse imports use engine defaults instead of prior landscape and noise settings', async t => {
  const {lab, input, find} = await setup(t, {fetch: path => path.endsWith('/api/simulate') ? response({snapshot: scene().snapshot}) : undefined});
  await until(() => find('#sceneTitle').textContent === 'first');
  input('#target_x', '99');
  input('#noise_seed', '123');
  const payload = scene().config;
  payload.config.noise = {std: .2};
  await lab.import(payload);
  const imported = lab.readConfig();
  assert.equal(imported.landscape.target_x, 2.2);
  assert.equal(imported.noise.seed, 0);
  assert.equal(imported.noise.enabled, true);
  assert.equal(imported.noise.std, .2);
});

test('local Run after changing settings continues past its reset and first step', async t => {
  let step = 0;
  const {lab, input} = await setup(t, {hosted: false, fetch: path => {
    if (path.endsWith('/api/reset')) {
      step = 0;
      const data = scene().snapshot; data.global_step = 0; data.done = false;
      return response(data);
    }
    if (path.endsWith('/api/step')) {
      const data = scene().snapshot; data.global_step = ++step; data.done = step >= 4;
      return response(data);
    }
  }});
  await lab.import(scene().config);
  // Make an actual control edit: the next Run must reset then keep playing.
  input('.opt-lr', '.09');
  assert.equal(lab.state.needsReset, true);
  await lab.run();
  await until(() => step >= 2, 'Run stopped immediately after resetting');
  lab.pause();
});
