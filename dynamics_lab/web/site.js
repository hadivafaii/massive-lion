/* Hosted presentation layer. Optimizer updates are supplied by the Python engine. */
(() => {
  "use strict";
  const params = new URLSearchParams(location.search);
  const embedded = params.get("embed") === "1";
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const host = {
    manifest: null, scene: null, data: null, api: "", loading: false,
    applying: false, generation: 0, controller: null, hidden: new Set(),
    custom: false, error: "", cache: new Map(), config: null, provenance: null,
  };
  const main = document.querySelector("main");
  const header = document.querySelector("header");
  const controls = document.querySelector(".controls");
  const stage = document.querySelector(".stage");
  const plot = stage.querySelector(".panel");
  const originalAtStep = snapshotAtStep;
  const originalLegend = updateLegend;
  document.body.classList.add("hosted", ...(embedded ? ["is-embed"] : []));

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(text, fn, className = "") {
    const node = element("button", className, text);
    node.type = "button";
    node.addEventListener("click", fn);
    return node;
  }
  function details(title, nodes, className = "") {
    const node = element("details", className);
    node.append(element("summary", "", title), ...nodes);
    return node;
  }
  function rewire(id, text, fn) {
    const old = $(id);
    const node = old.cloneNode(true);
    node.textContent = text;
    node.title = text;
    node.setAttribute("aria-label", text);
    old.replaceWith(node);
    node.addEventListener("click", fn);
    return node;
  }

  const brand = element("div", "site-brand");
  brand.innerHTML = '<span class="brand-mark" aria-hidden="true">∿</span><span>Optimizer Dynamics Lab<small>AN OPTIMIZER PLAYGROUND</small></span>';
  const shareActions = element("div", "share-actions");
  shareActions.append(button("Copy link", () => share(false)), button("〈/〉 Embed", () => share(true)));
  const exportTools = document.querySelector(".title-tools");
  exportTools.querySelector("h1")?.remove();
  header.replaceChildren(brand, shareActions);

  const intro = element("section", "site-intro");
  intro.innerHTML = '<div><p class="eyebrow">SMALL LANDSCAPES. BIG DIFFERENCES.</p><h1>See how optimizers <em>move.</em></h1><p class="intro-copy">Follow different paths down the same landscape. Play, compare, and find your intuition.</p></div><span class="engine-badge"><i></i> Powered by the repo’s Python optimizers</span>';
  header.after(intro);

  const sceneSection = element("section", "scene-section");
  sceneSection.innerHTML = '<p class="eyebrow">01 / PICK A LANDSCAPE</p><label class="scene-picker" for="scenePicker">Explore an example<select id="scenePicker" aria-label="Example landscape"></select></label><p id="sceneDescription" class="scene-description">Loading the first landscape…</p><div class="scene-meta"><span id="sceneSteps"></span><span id="sceneNoise"></span></div>';
  const compareSection = element("section", "compare-section");
  compareSection.innerHTML = '<p class="eyebrow">02 / FOLLOW THE PATHS</p><h2>Show in the plot</h2><p class="section-hint">Toggle a path to get a closer look.</p><div id="optimizerToggles" class="optimizer-toggles"></div>';
  const fileTools = element("div", "file-tools");
  fileTools.append(exportTools, button("Download scene", downloadScene));
  const runActions = controls.querySelector(".actions");
  const originalSections = [...controls.children].filter(node => node !== runActions);
  const customDetails = details("Try your own settings", originalSections, "custom-controls");
  const customIntro = element("p", "custom-intro");
  customIntro.id = "customIntro";
  customDetails.querySelector("summary").after(customIntro);
  const simulateButton = button("Run these settings", () => runCustom(), "primary simulate-btn");
  simulateButton.id = "simulateBtn";
  customDetails.append(simulateButton, fileTools);
  for (const fieldset of [...customDetails.children].filter(node => node.tagName === "FIELDSET")) {
    const title = fieldset.querySelector("legend")?.textContent || "Settings";
    if (title !== "Optimizers") {
      const disclosure = details(title, [], "settings-group");
      fieldset.before(disclosure);
      disclosure.append(fieldset);
    }
  }
  const boundControls = plot.querySelector(".bounds-control");
  const bounds = details("Plot bounds & color", [boundControls], "settings-group");
  customDetails.insertBefore(bounds, simulateButton);
  controls.replaceChildren(sceneSection, compareSection, customDetails);

  const sceneTitle = element("span", "plot-title", "Loading landscape");
  sceneTitle.id = "sceneTitle";
  plot.querySelector("h2 > span:first-child").replaceWith(sceneTitle);
  plot.classList.add("landscape-panel");
  const plotFooter = element("div", "plot-footer");
  const legend = $("legend") || element("div", "legend-list");
  legend.id = "legend";
  plotFooter.append(legend);
  plot.append(plotFooter);
  const playback = element("div", "playback");
  playback.append(runActions);
  plotFooter.append(playback);
  runActions.className = "playback-buttons";
  const runButton = rewire("runBtn", "Pause", () => running ? pause() : run());
  rewire("pauseBtn", "Pause", () => pause()).hidden = true;
  rewire("resetBtn", "↺ Replay", () => { if (!host.loading) { seek(0); run(); } });
  rewire("stepBackBtn", "←", () => { pause(); seek(viewStep - 1); }).title = "Previous step";
  rewire("stepForwardBtn", "→", () => { pause(); seek(viewStep + 1); }).title = "Next step";
  const scrub = element("input", "timeline");
  scrub.id = "timeline";
  scrub.type = "range"; scrub.min = "0"; scrub.max = "1"; scrub.value = "0"; scrub.step = "1";
  scrub.setAttribute("aria-label", "Simulation step");
  scrub.addEventListener("input", () => { pause(); seek(Number(scrub.value)); });
  const stepCounter = element("output", "step-counter", "0 / 0");
  stepCounter.id = "stepCounter";
  const speed = element("select", "playback-speed");
  speed.setAttribute("aria-label", "Playback speed");
  speed.innerHTML = '<option value="70">0.5×</option><option value="35" selected>1×</option><option value="17.5">2×</option><option value="8.75">4×</option>';
  speed.addEventListener("change", () => { $("delay_ms").value = speed.value; });
  playback.append(scrub, stepCounter, speed);
  plotFooter.append(playback);
  const statusRow = element("div", "playback-caption");
  const status = element("span", "status"); status.id = "status";
  statusRow.append(element("span", "drag-hint", "Drag to pan · switch to 3D to rotate"), status);
  plotFooter.append(statusRow);
  plot.append(plotFooter);
  const notice = element("div", "site-notice");
  notice.id = "siteNotice";
  notice.setAttribute("role", "status");
  notice.setAttribute("aria-live", "polite");
  notice.hidden = true;
  stage.prepend(notice);
  const diagnostics = document.querySelector(".side-stage");
  const lossPanel = $("lossCanvas").closest(".panel");
  stage.append(lossPanel);
  lossPanel.classList.add("loss-preview");
  const diagnosticDetails = details("Inspect position, velocity & momentum", [diagnostics], "diagnostics");
  stage.append(diagnosticDetails);
  diagnosticDetails.addEventListener("toggle", () => { if (diagnosticDetails.open) drawAll(); });
  const pageFooter = element("footer", "site-footer");
  pageFooter.innerHTML = '<span>Explore a little. Build an intuition.</span><span id="provenance">Trajectories computed with the repository’s optimizer implementations.</span>';
  main.append(pageFooter);

  const openLab = element("a", "open-lab", "Open full playground ↗");
  openLab.target = "_blank"; openLab.rel = "noopener";
  if (embedded) {
    plotFooter.append(openLab);
    plot.querySelector("h2").append(element("span", "embed-brand", "Optimizer Dynamics Lab"));
  }

  // Originals are shared with the full local lab. Replace only transport/playback.
  pause = function hostedPause() {
    running = false;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    updateStatus();
  };
  run = function hostedRun() {
    if (host.loading || !fullSnapshot || needsReset || running) return;
    if (viewStep >= fullSnapshot.global_step) seek(0);
    running = true;
    updateStatus();
    let prior = performance.now();
    const tick = () => {
      if (!running || needsReset) return pause();
      const now = performance.now();
      const interval = Math.max(5, numberValue("delay_ms") || 35);
      const amount = Math.max(1, Math.floor((now - prior) / interval));
      prior += amount * interval;
      seek(viewStep + amount);
      if (viewStep >= fullSnapshot.global_step) return pause();
      timer = setTimeout(tick, Math.max(16, interval));
    };
    timer = setTimeout(tick, Math.max(16, numberValue("delay_ms") || 35));
  };
  resetSimulation = async function hostedReset() {
    if (host.applying) return;
    return runCustom();
  };
  stepForward = async function hostedStep() { pause(); seek(viewStep + 1); };
  stepBackward = async function hostedBack() { pause(); seek(viewStep - 1); };
  reportSimulationError = function hostedError(error) {
    pause();
    host.error = error?.message || String(error);
    showNotice(host.error, "error");
    updateStatus();
  };
  updateStatus = function hostedStatus() {
    runButton.textContent = running ? "Ⅱ Pause" : "▶ Play";
    runButton.title = running ? "Pause" : "Play";
    runButton.setAttribute("aria-label", running ? "Pause" : "Play");
    runButton.disabled = !fullSnapshot || host.loading || needsReset;
    $("resetBtn").disabled = !fullSnapshot || host.loading || needsReset;
    $("stepBackBtn").disabled = !fullSnapshot || host.loading || needsReset || viewStep === 0;
    $("stepForwardBtn").disabled = !fullSnapshot || host.loading || needsReset || viewStep >= fullSnapshot.global_step;
    scrub.disabled = !fullSnapshot || host.loading || needsReset;
    const end = fullSnapshot?.global_step || 0;
    scrub.max = String(end); scrub.value = String(viewStep);
    scrub.setAttribute("aria-valuetext", `Step ${viewStep} of ${end}`);
    stepCounter.textContent = `${viewStep} / ${end}`;
    status.textContent = host.loading ? "Computing…" : needsReset ? "Settings changed" : running ? "Playing" : viewStep >= end && end ? "Complete · replay to explore" : "Paused";
    simulateButton.disabled = !host.api || host.loading;
    originalLegend();
  };
  drawAll = function hostedDraw() {
    if (!snapshot || !landscape) return;
    drawLandscape();
    if (!embedded) {
      drawLoss();
      if (diagnosticDetails.open) { drawThetaPlot(); drawVelocityPlot(); drawHistogram(); }
    }
  };

  function showNotice(message, kind = "info") {
    notice.textContent = message;
    notice.dataset.kind = kind;
    notice.hidden = !message;
  }
  function seek(step) {
    if (!fullSnapshot || needsReset) return;
    viewStep = Math.max(0, Math.min(fullSnapshot.global_step, step));
    snapshot = originalAtStep(fullSnapshot, viewStep);
    landscape = snapshot.landscape;
    applyColorOverrides();
    updateStatus(); drawAll();
  }
  function learnerKey(learner) {
    return learner.instance_id || (host.data?.mode === "ensemble" ? learner.optimizer || learner.name : learner.id || learner.name);
  }
  function refreshVisible() {
    if (!host.data) return;
    fullSnapshot = {...host.data, learners: host.data.learners.filter(learner => !host.hidden.has(learnerKey(learner)))};
    seek(viewStep);
    updateOpenLink();
  }
  function populateToggles() {
    const container = $("optimizerToggles"); container.replaceChildren();
    const used = new Set();
    for (const learner of host.data.learners) {
      if (used.has(learnerKey(learner))) continue;
      used.add(learnerKey(learner));
      const label = element("label", "optimizer-toggle");
      const input = element("input"); input.type = "checkbox";
      input.checked = !host.hidden.has(learnerKey(learner));
      const swatch = element("span", "toggle-swatch"); swatch.style.background = learner.color;
      const text = element("span", "toggle-name", learner.name);
      input.addEventListener("change", () => {
        if (!input.checked && host.hidden.size >= used.size - 1) { input.checked = true; return; }
        if (input.checked) host.hidden.delete(learnerKey(learner)); else host.hidden.add(learnerKey(learner));
        label.classList.toggle("is-off", !input.checked);
        refreshVisible();
      });
      label.classList.toggle("is-off", !input.checked);
      label.append(input, swatch, text); container.append(label);
    }
  }
  function decorateOptimizerRows() {
    for (const row of document.querySelectorAll(".optimizer-row")) {
      if (row.dataset.hostedDecorated) continue;
      row.dataset.hostedDecorated = "true";
      const head = row.querySelector(".row-head");
      const body = [...row.children].filter(node => node !== head);
      row.append(details("Parameters & appearance", body, "optimizer-parameters"));
      row.querySelector(".opt-name").setAttribute("aria-label", "Optimizer algorithm");
      row.querySelector(".remove-opt").setAttribute("aria-label", "Remove optimizer");
    }
  }
  new MutationObserver(decorateOptimizerRows).observe($("optimizerRows"), {childList: true});
  function markDirty(event) {
    if (host.applying || !host.data) return;
    if (event?.target && (!customDetails.contains(event.target) || event.target.closest("#simulateBtn"))) return;
    if (needsReset) {
      if (host.loading) cancelRequest();
      pause();
      host.error = "";
      showNotice(host.api ? "Settings changed. Run these settings to compute new paths." : "Settings changed. Custom runs are unavailable in this preview. Choose an example to keep exploring.", "pending");
      updateStatus();
    }
  }
  controls.addEventListener("input", markDirty);
  controls.addEventListener("change", markDirty);
  controls.addEventListener("click", event => queueMicrotask(() => markDirty(event)));
  controls.addEventListener("drop", event => queueMicrotask(() => markDirty(event)));
  for (const input of ["x_min", "x_max", "y_min", "y_max"]) $(input).addEventListener("input", markDirty);
  $("landscape_view").addEventListener("change", updateOpenLink);
  $("scenePicker").addEventListener("change", () => loadScene($("scenePicker").value).catch(reportSimulationError));

  async function jsonFile(path) {
    const url = new URL(path, location.href);
    if (url.origin !== location.origin) throw new Error("Scene assets must be hosted on this site.");
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${path} (${response.status}). Please reload to retry.`);
    return response.json();
  }
  function cancelRequest() {
    host.generation += 1;
    host.controller?.abort(); host.controller = null;
    host.loading = false;
  }
  async function applyConfig(payload) {
    host.applying = true;
    try {
      await applyImportedPayload(payload.config ? payload : {schema_version: 2, config: payload});
      decorateOptimizerRows();
    } finally { host.applying = false; }
  }
  function installSnapshot(data, config, provenance) {
    if (!data || !Array.isArray(data.learners) || !data.landscape || !Number.isFinite(data.global_step)) throw new Error("This scene has no complete trajectory data.");
    host.data = data;
    host.config = config;
    host.provenance = provenance;
    needsReset = false;
    host.error = "";
    viewStep = 0;
    fullSnapshot = data;
    snapshot = originalAtStep(data, 0);
    landscape = data.landscape;
    syncRadiusFromEpsilon();
    populateToggles(); refreshVisible();
    $("sceneSteps").textContent = `${data.global_step} steps`;
    const flat = config.config || config;
    $("sceneNoise").textContent = flat.noise?.enabled ? "With gradient noise" : "No gradient noise";
    const revision = provenance?.git_commit || provenance?.commit || provenance?.revision;
    $("provenance").textContent = `Computed with repository optimizers${revision ? ` · ${String(revision).slice(0, 8)}` : ""}.`;
    showNotice("");
  }
  async function loadScene(id, initial = false) {
    pause(); cancelRequest();
    const generation = host.generation;
    const scene = host.manifest.scenes.find(item => item.id === id) || host.manifest.scenes[0];
    if (!scene) throw new Error("No published examples were found.");
    host.loading = true;
    showNotice("Loading example…"); updateStatus();
    try {
      const data = host.cache.get(scene.id) || await jsonFile(scene.file);
      if (generation !== host.generation) return;
      host.cache.set(scene.id, data);
      host.scene = scene; host.custom = false;
      host.hidden.clear();
      await applyConfig(data.config);
      if (generation !== host.generation) return;
      host.loading = false;
      $("scenePicker").value = scene.id;
      $("sceneTitle").textContent = data.title || scene.title;
      $("sceneDescription").textContent = data.description || scene.description || "Follow the optimizer paths as they descend the landscape.";
      installSnapshot(data.snapshot, data.config, data.provenance);
      if (initial && params.get("show")) {
        const allowed = new Set(params.get("show").split(","));
        const existing = data.snapshot.learners.filter(item => allowed.has(learnerKey(item)));
        if (existing.length) {
          data.snapshot.learners.forEach(item => { if (!allowed.has(learnerKey(item))) host.hidden.add(learnerKey(item)); });
          populateToggles(); refreshVisible();
        }
      }
      if (initial && ["2d", "3d"].includes(params.get("view"))) {
        $("landscape_view").value = params.get("view"); landscapeView.mode = params.get("view");
      }
      updateOpenLink(); drawAll();
      if (!initial) {
        const url = new URL(location.href); url.searchParams.set("scene", scene.id); url.searchParams.delete("show"); url.hash = "";
        history.replaceState(null, "", url);
      }
      if (!reducedMotion && params.get("autoplay") !== "0") run();
    } finally {
      if (generation === host.generation) { host.loading = false; updateStatus(); }
    }
  }
  async function runCustom() {
    pause();
    if (!host.api) { showNotice("Custom runs are unavailable in this preview. The published examples always work; select one above to keep exploring.", "pending"); return; }
    if (host.loading) return;
    let config, exportPayload;
    try { config = readConfig(); exportPayload = buildExportPayload("Custom scene"); } catch (error) { return reportSimulationError(error); }
    cancelRequest();
    const generation = host.generation;
    host.loading = true; needsReset = true;
    const controller = new AbortController(); host.controller = controller;
    showNotice("Computing your paths with the Python optimizers…"); updateStatus();
    const warm = setTimeout(() => {
      if (generation === host.generation) showNotice("The free simulation server may be waking up. This can take about a minute. You can choose any example while it starts.");
    }, 4500);
    const timeout = setTimeout(() => controller.abort(), 150000);
    try {
      const response = await fetch(`${host.api}/api/simulate`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(config), signal: controller.signal});
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(body?.error || `The server returned ${response.status}. Please try again, or explore a saved example.`);
      if (generation !== host.generation) return;
      host.loading = false; host.custom = true; host.hidden.clear();
      installSnapshot(body.snapshot || body, exportPayload, body.provenance);
      $("sceneTitle").textContent = "Your experiment";
      $("sceneDescription").textContent = "Your settings, computed with the repository’s Python implementations.";
      updateOpenLink();
      if (!reducedMotion) run();
    } catch (error) {
      if (generation !== host.generation) return;
      reportSimulationError(error.name === "AbortError" ? new Error("The simulation server took too long. Try again, or choose a published example to play immediately.") : error);
    } finally {
      clearTimeout(warm); clearTimeout(timeout);
      if (generation === host.generation) { host.loading = false; host.controller = null; updateStatus(); }
    }
  }
  function shareURL(embed) {
    const url = new URL(location.href); url.search = ""; url.hash = "";
    if (host.scene) url.searchParams.set("scene", host.scene.id);
    if (embed) url.searchParams.set("embed", "1");
    url.searchParams.set("view", landscapeView.mode);
    if (host.hidden.size && host.data) url.searchParams.set("show", host.data.learners.filter(item => !host.hidden.has(learnerKey(item))).map(item => learnerKey(item)).filter((id, i, ids) => ids.indexOf(id) === i).join(","));
    if (host.custom || needsReset) {
      const encoded = encodeURIComponent(JSON.stringify(buildExportPayload("Shared experiment")));
      if (encoded.length > 48000) throw new Error("These settings are too large for a share link. Use Export or Download scene instead.");
      url.hash = `config=${encoded}`;
    }
    return url.href;
  }
  function updateOpenLink() {
    try { openLab.href = shareURL(false); } catch { openLab.href = location.pathname; }
  }
  const dialog = element("dialog", "share-dialog");
  dialog.innerHTML = '<div class="dialog-heading"><h2 id="shareHeading">Share this landscape</h2><button type="button" id="closeShare" aria-label="Close share dialog">×</button></div><p id="shareHint"></p><label for="shareText" id="shareTextLabel">Link</label><textarea id="shareText" rows="5" readonly spellcheck="false"></textarea><div class="dialog-actions"><span id="copyStatus" role="status"></span><button type="button" class="primary" id="copyShare">Copy</button></div>';
  dialog.setAttribute("aria-labelledby", "shareHeading"); document.body.append(dialog);
  $("closeShare").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
  $("copyShare").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("shareText").value); $("copyStatus").textContent = "Copied"; }
    catch { $("shareText").focus(); $("shareText").select(); $("copyStatus").textContent = "Select and copy the text above."; }
  });
  function share(embed) {
    try {
      const url = shareURL(embed);
      const custom = host.custom || needsReset;
      $("shareHeading").textContent = embed ? "A little lab, in your article." : "Share this experiment";
      $("shareHint").textContent = custom ? "This custom configuration needs the Python server to compute its paths. Download the scene and publish it to make an instant, self-contained example." : embed ? "Paste this iframe into an HTML block in your blog or project page. Playback works instantly from the published trajectory; no server is needed." : "This link opens the same landscape and visible optimizer paths.";
      $("shareTextLabel").textContent = embed ? "Embed HTML" : "Share URL";
      const safeURL = url.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
      $("shareText").value = embed ? `<iframe src="${safeURL}" title="Optimizer Dynamics Lab — interactive optimizer landscape" width="100%" height="560" style="border:1px solid #d9dedb;border-radius:16px" loading="lazy" allow="clipboard-write"></iframe>` : url;
      $("copyStatus").textContent = "";
      dialog.showModal();
    } catch (error) { reportSimulationError(error); }
  }
  function downloadScene() {
    if (!host.data || needsReset) { showNotice("Run the changed settings before downloading a scene, or choose a published example.", "pending"); return; }
    const title = host.custom ? "My optimizer experiment" : host.scene.title;
    const data = {schema_version: 1, id: host.custom ? "my-experiment" : host.scene.id, title, description: $("sceneDescription").textContent, config: buildExportPayload(title), snapshot: host.data, provenance: host.provenance || {source: "dynamics_lab Python engine"}};
    downloadTextFile(`${data.id}.scene.json`, JSON.stringify(data));
    showNotice("Scene downloaded with its settings and trajectories. Publish this bundle with the site to create an instant blog embed.");
  }
  async function init() {
    const [defaultData, manifest, siteConfig] = await Promise.all([jsonFile("defaults.json"), jsonFile("scenes.json"), jsonFile("site-config.json").catch(() => ({}))]);
    defaults = defaultData;
    host.manifest = manifest;
    if (siteConfig.api_base_url) {
      const url = new URL(siteConfig.api_base_url, location.href);
      if (!["http:", "https:"].includes(url.protocol)) throw new Error("The simulation API URL must use HTTP or HTTPS.");
      host.api = url.href.replace(/\/$/, "");
      fetch(`${host.api}/api/health`, {signal: AbortSignal.timeout(90000)}).catch(() => {});
    }
    customIntro.textContent = host.api ? "Change algorithms, learning rates, or the landscape. Run once, then replay freely. The free server can take a minute to wake up." : "Custom runs are unavailable in this preview. Browse all settings or export a configuration; published examples play instantly.";
    simulateButton.disabled = !host.api;
    for (const scene of manifest.scenes) {
      const option = element("option", "", scene.title); option.value = scene.id;
      $("scenePicker").append(option);
    }
    await loadScene(params.get("scene") || manifest.default_scene || manifest.scenes[0]?.id, true);
    if (location.hash.startsWith("#config=")) {
      pause();
      if (location.hash.length > 48008) throw new Error("The shared configuration is too large. Import the settings as a JSON file instead.");
      const config = JSON.parse(decodeURIComponent(location.hash.slice(8)));
      await applyConfig(config);
      host.custom = true; needsReset = true; updateStatus();
      if (host.api) await runCustom();
      else showNotice("This link includes custom settings, but the simulation server is not configured. Choose an example to see saved paths.", "pending");
    }
    new ResizeObserver(() => { if (snapshot) drawAll(); }).observe(plot);
  }
  init().catch(reportSimulationError);
})();
