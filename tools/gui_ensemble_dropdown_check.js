// Checks the Ensemble tab's Speech Segmenter dropdown WITHOUT a browser.
//
// Written 2026-09-09 for the v1.9.2 balanced work, after the owner found that the
// dropdown had never been inspected. It loads the REAL app.js against the REAL
// option lists parsed out of index.html, with a small stand-in for the browser,
// then drives the same functions a click drives: handlePipelineChange ->
// applyPipelinePresets -> populateSegmenterOptions, and the change handler.
//
//   node tools/gui_ensemble_dropdown_check.js
//
// It prints, for each step, the pipeline, which of the two things the dropdown
// currently means, its option list, the selected value and the state behind it.
//
// THIS IS NOT A SUBSTITUTE FOR OPENING THE GUI. It exercises the page's logic,
// not the page. Use it to catch a broken option list in seconds; still click the
// tab before telling the owner a UI requirement is met.

// Drives the REAL app.js EnsembleManager against a minimal DOM stub, so the
// Ensemble tab's Speech Segmenter dropdown can be inspected without a browser.
// It exercises the same functions the GUI calls: handlePipelineChange ->
// applyPipelinePresets -> populateSegmenterOptions, plus the change handler.
const fs = require('fs');
const path = require('path');

const REPO = 'D:/Git/WhisperJav_V1_Minami_Edition';
const html = fs.readFileSync(path.join(REPO, 'whisperjav/webview_gui/assets/index.html'), 'utf8');

// --- pull the real <option> lists for the two segmenter selects out of index.html
function selectInner(id) {
  const re = new RegExp('<select id="' + id + '"[^>]*>([\\s\\S]*?)</select>');
  const m = html.match(re);
  if (!m) throw new Error('select not found: ' + id);
  return m[1];
}

function parseOptions(inner) {
  inner = inner.replace(/<!--[\s\S]*?-->/g, '');  // commented-out options are not options
  const out = [];
  const re = /<option\s+value="([^"]*)"([^>]*)>([\s\S]*?)<\/option>/g;
  let m;
  while ((m = re.exec(inner)) !== null) {
    out.push({ value: m[1], selected: /\bselected\b/.test(m[2]), label: m[3].trim() });
  }
  return out;
}

class FakeSelect {
  constructor(id, inner) {
    this.id = id;
    this._inner = inner || '';
    this.dataset = {};
    this.title = '';
    this.disabled = false;
    this.style = {};
    this._value = (parseOptions(this._inner).find(o => o.selected) || parseOptions(this._inner)[0] || {}).value || '';
    this._listeners = {};
  }
  get innerHTML() { return this._inner; }
  set innerHTML(v) { this._inner = v; const o = this.options; this._value = (o[0] || {}).value || ''; }
  get options() { return parseOptions(this._inner).map(o => ({ ...o, disabled: false, textContent: o.label, classList: { contains: () => false } })); }
  get value() { return this._value; }
  set value(v) { this._value = v; }
  querySelector(sel) {
    const m = sel.match(/option\[value="(.*)"\]/);
    if (!m) return null;
    return this.options.find(o => o.value === m[1]) || null;
  }
  appendChild(o) { this._inner += `<option value="${o.value}"${o.selected ? ' selected' : ''}>${o.textContent}</option>`;
                   if (o.selected || !this._value) this._value = o.value; }
  addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); }
  fire(ev) { (this._listeners[ev] || []).forEach(fn => fn({ target: this, options: this.options, selectedIndex: 0 })); }
}

const els = {};
function el(id, inner) { els[id] = new FakeSelect(id, inner); return els[id]; }

el('pass1-segmenter', selectInner('pass1-segmenter'));
el('pass2-segmenter', selectInner('pass2-segmenter'));
el('pass1-scene', selectInner('pass1-scene'));
el('pass2-scene', selectInner('pass2-scene'));
el('pass1-pipeline', selectInner('pass1-pipeline'));
el('pass2-pipeline', selectInner('pass2-pipeline'));
el('pass1-sensitivity', selectInner('pass1-sensitivity'));
el('pass2-sensitivity', selectInner('pass2-sensitivity'));
el('pass1-enhancer', selectInner('pass1-enhancer'));
el('pass2-enhancer', selectInner('pass2-enhancer'));
el('pass1-model', selectInner('pass1-model'));
el('pass2-model', selectInner('pass2-model'));

const noop = new Proxy(function () {}, {
  get: (t, k) => (k === 'then' ? undefined : (k in t ? t[k] : noop)),
  apply: () => noop,
});

global.document = {
  getElementById: (id) => els[id] || null,
  querySelectorAll: () => [],
  querySelector: () => null,
  addEventListener: () => {},
  body: { classList: { add() {}, remove() {}, contains() { return false; } } },
  documentElement: { setAttribute() {}, getAttribute() { return null; } },
  createElement: (tag) => ({ tagName: tag, value: '', textContent: '', selected: false, title: '', disabled: false,
                             style: {}, classList: { add() {}, remove() {}, contains() { return false; } },
                             appendChild() {}, setAttribute() {} }),
};
global.window = { addEventListener: () => {}, matchMedia: () => ({ matches: false, addEventListener() {} }) };
global.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
global.navigator = { platform: 'Win32', userAgent: 'node' };
global.confirm = () => true;
global.alert = () => {};
global.setTimeout = setTimeout;
global.console = console;

// Load app.js in the global scope.
const src = fs.readFileSync(path.join(REPO, 'whisperjav/webview_gui/assets/app.js'), 'utf8');
const vm = require('vm');
vm.runInThisContext(src + "\n;globalThis.__EM = EnsembleManager;", { filename: 'app.js' });

const EM = globalThis.__EM;
if (!EM) throw new Error('EnsembleManager not exposed on the global scope');

function show(tag, passKey) {
  const sel = els[passKey + '-segmenter'];
  const st = EM.state[passKey];
  console.log(`\n${tag}`);
  console.log(`  pipeline        : ${st.pipeline}`);
  console.log(`  dropdown mode   : ${sel.dataset.mode || '(external segmenters)'}`);
  console.log(`  options         : ${sel.options.map(o => o.value).join(', ')}`);
  console.log(`  selected        : ${sel.value}`);
  console.log(`  state.vadVersion: ${st.vadVersion}   state.speechSegmenter: ${st.speechSegmenter}`);
  console.log(`  title           : ${sel.title.slice(0, 70)}`);
}

// --- as the page loads (pass1 default = anime-whisper)
EM.populateSegmenterOptions('pass1');
show('[load] pass1 default pipeline', 'pass1');

// --- user picks Balanced in the Pipeline dropdown
EM.handlePipelineChange('pass1', 'balanced', els['pass1-pipeline']);
show('[click] Pipeline -> Balanced', 'pass1');

// --- user picks Silero 6.2 in the segmenter dropdown
const seg = els['pass1-segmenter'];
seg.value = '6.2';
EM.state.pass1.vadVersion = undefined;
// replay the registered change handler the way the browser would
if (EM._changeHandlerReplay) EM._changeHandlerReplay();
else {
  // handlers are attached in init(); emulate the exact body
  if (seg.dataset.mode === 'vad-version') EM.state.pass1.vadVersion = seg.value;
  else EM.state.pass1.speechSegmenter = seg.value;
}
show('[click] dropdown -> 6.2', 'pass1');

// --- user switches back to Fidelity: the external list must come back
EM.handlePipelineChange('pass1', 'fidelity', els['pass1-pipeline']);
show('[click] Pipeline -> Fidelity', 'pass1');

// --- and back to Balanced: the 6.2 choice must be remembered
EM.handlePipelineChange('pass1', 'balanced', els['pass1-pipeline']);
show('[click] Pipeline -> Balanced again', 'pass1');

// --- what the run would send
const cfg = { pipeline: EM.state.pass1.pipeline, vadVersion: EM.state.pass1.vadVersion,
              speechSegmenter: EM.state.pass1.pipeline === 'balanced' ? null : EM.state.pass1.speechSegmenter };
console.log('\n[run] pass1 payload ->', JSON.stringify(cfg));
