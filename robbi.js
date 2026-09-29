/* Robbi lernt schreiben: das Mini-Sprachmodell und die sieben Stufen */
(function () {
'use strict';

/* ================= helpers ================= */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const sleep = ms => new Promise(res => setTimeout(res, ms));
/* yield to the event loop without waiting for a frame: the browser may throttle frames while the window is covered */
const yieldNow = () => new Promise(res => { const ch = new MessageChannel(); ch.port1.onmessage = () => res(); ch.port2.postMessage(null); });
const nf0 = new Intl.NumberFormat('de-DE');
const nf1 = new Intl.NumberFormat('de-DE', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat('de-DE', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const visCh = c => (c === ' ' ? '␣' : c === '\n' ? '↵' : c === '\t' ? '⇥' : c);
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rng = Math.random) {
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
  return arr;
}
function niceCeil(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
function setStatus(msg, kind) {
  const el = $('#status');
  el.textContent = msg || '';
  el.classList.toggle('error', kind === 'error');
}
function charCat(c) {
  if (c === ' ' || c === '\n' || c === '\t') return 'punct';
  if (/\p{Lu}/u.test(c)) return 'upper';
  if (/[aeiouyäöü]/.test(c)) return 'vowel';
  if (/\p{Ll}/u.test(c)) return 'cons';
  if (/\p{P}/u.test(c)) return 'punct';
  return 'other';
}
const CAT_ORDER = { punct: 0, other: 1, vowel: 2, cons: 3, upper: 4 };

/* ================= configuration ================= */
const SIZES = {
  winzig: { nLayer: 1, d: 32, nHead: 2, ctx: 32, lr: 0.01, drop: 0.1 },
  klein: { nLayer: 2, d: 64, nHead: 4, ctx: 64, lr: 0.004, drop: 0.15 },
};
const BATCH = 32;
const ROUNDS = [100, 300, 800, 1200];
const EXTRA = 1500;
const GAME_ROUNDS = 3;
const CORPORA = {
  maerchen: { src: '#corpus-maerchen', lineMode: false, prompt: 'Es war einmal', size: 'klein', title: 'Das Märchenbuch', sub: '18 kurze Märchen' },
  namen: { src: '#corpus-namen', lineMode: true, prompt: '', size: 'winzig', title: 'Das Namenbuch', sub: 'rund 600 Vornamen' },
};
const LEVEL_NAMES = ['Start', 'Buchstaben', 'Raten', 'Lesen', 'Revanche', 'Schreiben', 'Riesen'];

/* ================= state ================= */
const S = {
  name: 'Robbi', corpusKey: 'maerchen', level: 0, visited: [true], done: [true],
  data: null, cfg: null, model: null, vars: [], opt: null, xVal: null, yVal: null, paramTotal: 0,
  runId: 0, step: 0, trainLoss: [], trainEma: [], ema: null, valPts: [], lastVal: -1,
  bestVal: Infinity, bestValStep: 0, overfit: false, overfitStep: 0, enough: false, warnedOverfit: false,
  reading: false, roundsDone: 0, diary: [], writing: false, l1done: false, wrote: false,
  score: { before: { kid: 0, robot: 0 }, after: { kid: 0, robot: 0 } },
  backendName: '',
};
const N = () => S.name;

/* all model work runs through one queue */
let queue = Promise.resolve();
function gpu(fn) {
  const p = queue.then(() => fn());
  queue = p.catch(() => {});
  return p;
}

/* ================= data ================= */
function prepareData(raw, lineMode) {
  let text = raw.replace(/\r\n?/g, '\n');
  let trainArr, valArr, trainLinesSet = null;
  if (lineMode) {
    const lines = Array.from(new Set(text.split('\n').map(s => s.trim()).filter(Boolean)));
    shuffle(lines, mulberry32(7));
    const nVal = Math.max(2, Math.round(lines.length * 0.1));
    const valL = lines.slice(0, nVal), trL = lines.slice(nVal);
    trainArr = Array.from('\n' + trL.join('\n') + '\n');
    valArr = Array.from('\n' + valL.join('\n') + '\n');
    trainLinesSet = new Set(trL);
    text = lines.join('\n');
  } else {
    text = text.replace(/[ \t]+/g, ' ').replace(/ *\n */g, '\n').replace(/\n{3,}/g, '\n\n').trim();
    const chars = Array.from(text);
    const BL = 400;
    trainArr = []; valArr = [];
    for (let i = 0, b = 0; i < chars.length; i += BL, b++) {
      const blk = chars.slice(i, i + BL);
      const dst = b % 10 === 9 ? valArr : trainArr;
      for (const c of blk) dst.push(c);
    }
  }
  const all = trainArr.concat(valArr);
  const chars = Array.from(new Set(all)).sort((a, b) => (CAT_ORDER[charCat(a)] - CAT_ORDER[charCat(b)]) || a.localeCompare(b, 'de'));
  const stoi = new Map(chars.map((c, i) => [c, i]));
  const V = chars.length;
  const train = Int32Array.from(trainArr, c => stoi.get(c));
  const val = Int32Array.from(valArr, c => stoi.get(c));
  const freq = new Float64Array(V);
  for (const t of train) freq[t]++;
  const letterIds = chars.map((c, i) => i).filter(i => /\p{Ll}/u.test(chars[i]));
  return {
    lineMode, text, chars, stoi, itos: chars, V, train, val, freq, letterIds,
    trainText: trainArr.join(''), trainLinesSet, totalChars: all.length,
    lines: lineMode ? text.split('\n').length : 0,
    refs: ngramRefs(train, val, V),
  };
}

/* how good would simple counting be? (evaluated on the unseen text) */
function ngramRefs(train, val, V) {
  const uni = new Float64Array(V);
  for (const t of train) uni[t]++;
  const Nn = train.length;
  const pUni = Array.from(uni, c => (c + 0.5) / (Nn + 0.5 * V));
  const bi = new Map(), biCtx = new Float64Array(V);
  for (let i = 1; i < Nn; i++) { const k = train[i - 1] * V + train[i]; bi.set(k, (bi.get(k) || 0) + 1); biCtx[train[i - 1]]++; }
  const tri = new Map(), triCtx = new Map();
  for (let i = 2; i < Nn; i++) {
    const c = train[i - 2] * V + train[i - 1];
    const k = c * V + train[i];
    tri.set(k, (tri.get(k) || 0) + 1);
    triCtx.set(c, (triCtx.get(c) || 0) + 1);
  }
  const pBi = (a, b) => { const n = biCtx[a]; const lam = n / (n + 3); return lam * (n ? (bi.get(a * V + b) || 0) / n : 0) + (1 - lam) * pUni[b]; };
  const pTri = (a, b, c) => { const ctx = a * V + b; const n = triCtx.get(ctx) || 0; const lam = n / (n + 3); return lam * (n ? (tri.get(ctx * V + c) || 0) / n : 0) + (1 - lam) * pBi(b, c); };
  let hU = 0, hB = 0, hT = 0, cnt = 0;
  for (let i = 2; i < val.length; i++) {
    hU -= Math.log(pUni[val[i]]);
    hB -= Math.log(pBi(val[i - 1], val[i]));
    hT -= Math.log(pTri(val[i - 2], val[i - 1], val[i]));
    cnt++;
  }
  return { uniform: Math.log(V), uni: hU / cnt, bi: hB / cnt, tri: hT / cnt };
}

/* ================= model (a tiny GPT) ================= */
function buildModel(cfg, V) {
  const vars = [];
  const p = { layers: [] };
  tf.tidy(() => {
    const mk = t => { const v = tf.variable(t); vars.push(v); return v; };
    const rn = (shape, std) => mk(tf.randomNormal(shape, 0, std));
    const z = shape => mk(tf.zeros(shape));
    const o = shape => mk(tf.ones(shape));
    const d = cfg.d, resStd = 0.02 / Math.sqrt(2 * cfg.nLayer);
    p.wte = rn([V, d], 0.02);
    p.wpe = rn([cfg.ctx, d], 0.02);
    for (let i = 0; i < cfg.nLayer; i++) {
      p.layers.push({
        ln1g: o([d]), ln1b: z([d]),
        wqkv: rn([d, 3 * d], 0.02), bqkv: z([3 * d]),
        wo: rn([d, d], resStd), bo: z([d]),
        ln2g: o([d]), ln2b: z([d]),
        w1: rn([d, 4 * d], 0.02), b1: z([4 * d]),
        w2: rn([4 * d, d], resStd), b2: z([d]),
      });
    }
    p.lnfg = o([d]); p.lnfb = z([d]);
  });
  return { p, vars };
}

const maskCache = new Map();
function causalMask(t) {
  let m = maskCache.get(t);
  if (!m) {
    m = tf.keep(tf.tidy(() => tf.mul(tf.sub(1, tf.linalg.bandPart(tf.ones([t, t]), -1, 0)), -1e4)));
    maskCache.set(t, m);
  }
  return m;
}
function layerNorm(x, g, b) {
  const mu = tf.mean(x, -1, true);
  const xc = tf.sub(x, mu);
  const va = tf.mean(tf.square(xc), -1, true);
  return tf.add(tf.mul(tf.div(xc, tf.sqrt(tf.add(va, 1e-5))), g), b);
}
function forward(p, cfg, idx, training) {
  const [B, t] = idx.shape;
  const d = cfg.d, H = cfg.nHead, hd = d / H;
  const drop = z => (training && cfg.drop > 0 ? tf.dropout(z, cfg.drop) : z);
  let x = tf.gather(p.wte, tf.reshape(idx, [B * t]));
  x = drop(tf.reshape(tf.add(tf.reshape(x, [B, t, d]), tf.slice(p.wpe, [0, 0], [t, d])), [B * t, d]));
  const mask = causalMask(t);
  const scale = 1 / Math.sqrt(hd);
  const heads = z => tf.transpose(tf.reshape(z, [B, t, H, hd]), [0, 2, 1, 3]);
  for (const L of p.layers) {
    const h = layerNorm(x, L.ln1g, L.ln1b);
    const [q, k, v] = tf.split(tf.add(tf.matMul(h, L.wqkv), L.bqkv), 3, 1);
    const att = tf.softmax(tf.add(tf.mul(tf.matMul(heads(q), heads(k), false, true), scale), mask));
    const y = tf.reshape(tf.transpose(tf.matMul(att, heads(v)), [0, 2, 1, 3]), [B * t, d]);
    x = tf.add(x, drop(tf.add(tf.matMul(y, L.wo), L.bo)));
    const h2 = layerNorm(x, L.ln2g, L.ln2b);
    x = tf.add(x, drop(tf.add(tf.matMul(tf.relu(tf.add(tf.matMul(h2, L.w1), L.b1)), L.w2), L.b2)));
  }
  x = layerNorm(x, p.lnfg, p.lnfb);
  return tf.matMul(x, p.wte, false, true);
}
function lossOf(x, y, training) {
  const logits = forward(S.model, S.cfg, x, training);
  const oh = tf.cast(tf.oneHot(tf.reshape(y, [-1]), S.data.V), 'float32');
  return tf.neg(tf.mean(tf.sum(tf.mul(tf.logSoftmax(logits), oh), 1)));
}
function sampleBatch() {
  const T = S.cfg.ctx, tr = S.data.train, n = tr.length - T - 1;
  const xs = new Int32Array(BATCH * T), ys = new Int32Array(BATCH * T);
  for (let b = 0; b < BATCH; b++) {
    const o = Math.floor(Math.random() * n);
    xs.set(tr.subarray(o, o + T), b * T);
    ys.set(tr.subarray(o + 1, o + T + 1), b * T);
  }
  return { x: tf.tensor2d(xs, [BATCH, T], 'int32'), y: tf.tensor2d(ys, [BATCH, T], 'int32') };
}
function buildValTensors() {
  const T = S.cfg.ctx, va = S.data.val;
  const maxW = Math.floor((va.length - 1) / T);
  const nW = Math.max(1, Math.min(64, maxW));
  const stride = nW > 1 ? Math.floor((va.length - 1 - T) / (nW - 1)) : 0;
  const starts = [];
  for (let i = 0; i < nW; i++) starts.push(Math.min(i * stride, va.length - 1 - T));
  const xs = new Int32Array(starts.length * T), ys = new Int32Array(starts.length * T);
  starts.forEach((o, i) => { xs.set(va.subarray(o, o + T), i * T); ys.set(va.subarray(o + 1, o + T + 1), i * T); });
  S.xVal = tf.tensor2d(xs, [starts.length, T], 'int32');
  S.yVal = tf.tensor2d(ys, [starts.length, T], 'int32');
}
/* probabilities for the next character. Must run inside gpu(). */
async function predict(ids, temp) {
  const T = S.cfg.ctx, V = S.data.V;
  const win = ids.slice(-T), t = win.length;
  const buf = new Int32Array(T);
  buf.set(win);
  const probs = tf.tidy(() => tf.softmax(tf.div(tf.slice(forward(S.model, S.cfg, tf.tensor2d(buf, [1, T], 'int32'), false), [t - 1, 0], [1, V]), temp)));
  const arr = await probs.data();
  probs.dispose();
  return arr;
}
function sampleIdx(probs, rng) {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r < acc) return i; }
  return probs.length - 1;
}
function seedId() {
  const D = S.data;
  return D.stoi.has('\n') ? D.stoi.get('\n') : D.stoi.has(' ') ? D.stoi.get(' ') : 0;
}
function encodePrompt(str) {
  const D = S.data;
  const chars = Array.from(str);
  if (D.lineMode && chars[0] !== '\n') chars.unshift('\n');
  const known = chars.filter(c => D.stoi.has(c));
  return { known, dropped: chars.length - known.length, ids: known.map(c => D.stoi.get(c)) };
}
/* write text character by character; the whole run is one queue task */
function generate(prompt, n, temp, rng, id, opts = {}) {
  return gpu(async () => {
    if (id !== S.runId) return null;
    const D = S.data;
    const enc = encodePrompt(prompt);
    const ids = enc.ids.length ? enc.ids.slice() : [seedId()];
    const out = [];
    let lines = 0;
    for (let i = 0; i < n; i++) {
      const p = await predict(ids, temp);
      if (id !== S.runId) return null;
      const nx = sampleIdx(p, rng);
      ids.push(nx);
      const ch = D.itos[nx];
      out.push(ch);
      if (opts.onChar) opts.onChar(out.join(''));
      if (ch === '\n' && opts.maxLines && ++lines >= opts.maxLines) break;
    }
    return { prompt: enc.known.join(''), gen: out.join(''), dropped: enc.dropped };
  });
}

/* ================= copied passages / new names ================= */
function copiedMask(full, from, source, minLen) {
  const mark = new Uint8Array(full.length);
  for (let i = 0; i + minLen <= full.length; i++) {
    if (!source.includes(full.substr(i, minLen))) continue;
    let lo = minLen, hi = full.length - i;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (source.includes(full.substr(i, mid))) lo = mid; else hi = mid - 1; }
    for (let k = i; k < i + lo; k++) mark[k] = 1;
  }
  return mark.subarray(from);
}
function renderMarked(prompt, gen) {
  const mark = copiedMask(prompt + gen, prompt.length, S.data.trainText, 12);
  let html = '', open = false, copied = 0;
  for (let i = 0; i < gen.length; i++) {
    const m = mark[i] === 1;
    if (m && !open) { html += '<span class="cp">'; open = true; }
    if (!m && open) { html += '</span>'; open = false; }
    html += esc(gen[i]);
    if (m) copied++;
  }
  if (open) html += '</span>';
  return { html: (prompt ? `<span class="pr">${esc(prompt)}</span>` : '') + html, share: gen.length ? copied / gen.length : 0 };
}
function namesFrom(gen) {
  const parts = gen.split('\n');
  if (!gen.endsWith('\n')) parts.pop();
  return parts.map(s => s.trim()).filter(Boolean);
}
function renderNames(gen) {
  const names = namesFrom(gen);
  if (!names.length) return { html: esc(gen.replace(/\n/g, ' ')), known: 0, total: 0 };
  let known = 0;
  const html = names.map(nm => {
    const k = S.data.trainLinesSet.has(nm);
    if (k) known++;
    return `<span class="nm ${k ? 'known' : 'new'}" title="${k ? 'stand so im Buch' : 'neu erfunden'}">${esc(nm)}</span>`;
  }).join(' ');
  return { html, known, total: names.length };
}
function renderOutput(prompt, gen) {
  return S.data.lineMode ? renderNames(gen).html : renderMarked(prompt, gen).html;
}

/* ================= robot ================= */
const MOUTH = { flat: 'M48 70 L72 70', smile: 'M46 68 Q60 80 74 68', grin: 'M44 66 Q60 84 76 66', o: 'M54 66 Q60 60 66 66 Q60 76 54 66 Z', sad: 'M46 74 Q60 64 74 74' };
function setRobot(mood, text) {
  const r = $('#robot');
  r.setAttribute('class', 'robot ' + mood);
  const m = { sleepy: 'flat', think: 'o', happy: 'smile', proud: 'grin', oops: 'sad', read: 'flat', idle: 'smile' }[mood] || 'flat';
  $('#mouth').setAttribute('d', MOUTH[m]);
  if (text != null) $('#bubble').innerHTML = text;
}
function say(text) { $('#bubble').innerHTML = text; }

/* ================= levels ================= */
function renderSteps() {
  $('#steps').innerHTML = LEVEL_NAMES.map((n, i) => {
    const cls = [S.level === i ? 'cur' : '', S.visited[i] ? 'visited' : '', S.done[i] ? 'done' : ''].filter(Boolean).join(' ');
    return `<li><button type="button" class="${cls}" data-go="${i}" ${S.visited[i] ? '' : 'disabled'} aria-current="${S.level === i ? 'step' : 'false'}"><span class="bar"></span><span class="t">${i === 0 ? n : i + ' ' + n}</span></button></li>`;
  }).join('');
}
function renderNav() {
  const L = S.level;
  $('#back').hidden = L === 0;
  $('#next').hidden = L === 6 || L === 0;
  $('#next').disabled = !S.done[L];
  const hints = {
    1: 'Erst einmal auf „Schreib mal was!“ drücken.',
    2: `Erst ${GAME_ROUNDS} Runden raten.`,
    3: 'Erst mindestens zwei Leserunden.',
    4: `Erst ${GAME_ROUNDS} Runden raten.`,
    5: 'Erst einmal auf „Schreib!“ drücken.',
  };
  $('#nav-hint').textContent = S.done[L] || L === 0 || L === 6 ? '' : hints[L];
  renderSteps();
}
function fillNames() {
  $$('.nm-slot').forEach(el => { el.textContent = N(); });
  $('#name-tag').textContent = N();
}
function goTo(n) {
  S.level = n;
  S.visited[n] = true;
  $$('.level').forEach(el => { el.hidden = parseInt(el.dataset.level, 10) !== n; });
  fillNames();
  enterLevel(n);
  renderNav();
  window.scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' });
}
function enterLevel(n) {
  const D = S.data;
  switch (n) {
    case 0:
      setRobot('sleepy', 'Hallo! Ich bin ein Roboter. Ich kenne alle Buchstaben, aber ich kann noch nicht schreiben. Bringst du es mir bei?');
      break;
    case 1:
      setRobot('idle', `Schau, das sind alle Zeichen, die ich kenne. Aber welcher Buchstabe nach welchem kommt? Keine Ahnung!`);
      break;
    case 2:
      setRobot('think', `Du rätst, ich rate. Wer hat öfter recht? Ich warne dich: Ich habe noch nichts gelesen.`);
      if (!G.before) startGame('before');
      break;
    case 3:
      setRobot(S.step ? 'idle' : 'think', S.step
        ? `Ich habe schon ${nf0.format(S.step)} Mal ins Buch geschaut. Soll ich noch weiterlesen?`
        : `Jetzt lese ich! Drück auf den Knopf, dann fange ich an.`);
      $('#book-title').textContent = CORPORA[S.corpusKey].title;
      $('#book-sub').textContent = `${CORPORA[S.corpusKey].sub} · ${nf0.format(D.totalChars)} Zeichen`;
      renderRead(); renderMeter(); renderLadder(); drawChart();
      break;
    case 4:
      setRobot('happy', `Jetzt bin ich bereit! Ich habe ${nf0.format(S.step)} Mal ins Buch geschaut. Noch mal du gegen mich!`);
      if (!G.after || G.afterStep !== S.step) startGame('after');
      break;
    case 5:
      setRobot('happy', D.lineMode ? `Ich erfinde jetzt Namen. Grün heißt: Den gab es im Buch noch nicht!` : `Sag mir, wie es anfangen soll, dann schreibe ich weiter.`);
      $('#l5-prompt-box').hidden = D.lineMode;
      break;
    case 6:
      setRobot('proud', `Ich bin ein winziger Roboter. Aber die Riesen lernen genau wie ich!`);
      renderGiants();
      zoomSetup();
      if (!S.confettiDone) { S.confettiDone = true; confetti(); }
      break;
  }
}

/* ================= level 1: gibberish ================= */
async function l1Write() {
  const btn = $('#l1-write'), out = $('#l1-out');
  btn.disabled = true;
  setRobot('think', 'Moment … ich schreibe.');
  const id = S.runId;
  out.innerHTML = '<span class="cursor"></span>';
  const res = await generate('', 90, 1, Math.random, id, { onChar: t => { out.innerHTML = esc(t.replace(/\n/g, ' ')) + '<span class="cursor"></span>'; } });
  btn.disabled = false;
  if (!res) return;
  out.textContent = res.gen.replace(/\n/g, ' ');
  S.l1done = true; S.done[1] = true;
  setRobot('oops', `Hm. Das ist Quatsch, oder? Ich nehme einfach irgendeinen Buchstaben. Ich muss erst lernen, welche zusammengehören.`);
  btn.textContent = 'Nochmal!';
  renderNav();
}

/* ================= game ================= */
const G = { before: null, after: null, afterStep: -1 };
function startGame(phase) {
  const g = { phase, round: 0, cur: null, kid: 0, robot: 0, finished: false };
  G[phase] = g;
  if (phase === 'after') G.afterStep = S.step;
  S.score[phase] = { kid: 0, robot: 0 };
  newRound(g);
}
function newRound(g) {
  const D = S.data, va = D.val;
  let i, tries = 0;
  do { i = 12 + Math.floor(Math.random() * (va.length - 13)); tries++; } while (!(D.letterIds.includes(va[i]) && (tries > 300 || /\p{Ll}/u.test(D.itos[va[i - 1]]) || D.lineMode)));
  let ctxIds = Array.from(va.subarray(Math.max(0, i - 36), i));
  let truncated = i - 36 > 0;
  if (D.lineMode) {
    const nl = ctxIds.lastIndexOf(D.stoi.get('\n'));
    if (nl >= 0) { ctxIds = ctxIds.slice(nl); truncated = false; }
    if (ctxIds.length < 2) return newRound(g);
  }
  const target = va[i];
  const pool = D.letterIds.filter(id => id !== target).sort((a, b) => D.freq[b] - D.freq[a]).slice(0, 14);
  const distract = shuffle(pool.slice()).slice(0, 5);
  const tiles = shuffle([target].concat(distract));
  g.round++;
  g.cur = { ctxIds, truncated, target, tiles, kid: null, robot: null, robotP: 0 };
  const id = S.runId, cur = g.cur;
  cur.ready = gpu(async () => {
    if (id !== S.runId) return;
    const p = await predict(ctxIds, 1);
    let best = tiles[0];
    for (const t of tiles) if (p[t] > p[best]) best = t;
    let sum = 0;
    for (const t of tiles) sum += p[t];
    cur.robot = best; cur.robotP = p[best] / (sum || 1);
    cur.probs = tiles.map(t => [t, p[t] / (sum || 1)]).sort((a, b) => b[1] - a[1]);
  });
  renderGame(g);
}
function ctxHtml(g) {
  const D = S.data, c = g.cur;
  const txt = c.ctxIds.map(t => D.itos[t]).join('').replace(/\n/g, ' ').replace(/^ /, '');
  const gap = c.kid == null ? '<span class="gap">?</span>' : `<span class="gap ok">${esc(D.itos[c.target])}</span>`;
  return (c.truncated ? '…' : '') + esc(txt) + gap;
}
function renderGame(g) {
  const box = $(g.phase === 'before' ? '#game-before' : '#game-after');
  const D = S.data, c = g.cur;
  const revealed = c.kid != null;
  const tiles = c.tiles.map(t => {
    const ch = D.itos[t];
    let cls = 'pick', tag = '';
    if (revealed) {
      if (t === c.target) cls += ' right';
      else if (t === c.kid || t === c.robot) cls += ' wrong';
      const isK = t === c.kid, isR = t === c.robot;
      if (isK && isR) tag = '<span class="tag rk">du + ' + esc(N()) + '</span>';
      else if (isK) tag = '<span class="tag k">du</span>';
      else if (isR) tag = '<span class="tag r">' + esc(N()) + '</span>';
    }
    return `<button type="button" class="${cls}" data-t="${t}" ${revealed ? 'disabled' : ''} aria-label="${esc(ch)}">${esc(ch)}${tag}</button>`;
  }).join('');
  let verdict = '';
  if (revealed) {
    const kidOk = c.kid === c.target, robOk = c.robot === c.target;
    const rp = Math.round(c.robotP * 100);
    if (kidOk && robOk) verdict = `<b class="you">Du</b> und <b class="rob">${esc(N())}</b> hattet beide recht! (${esc(N())} war sich zu ${rp} % sicher.)`;
    else if (kidOk) verdict = `<b class="you">Du</b> hattest recht! ${esc(N())} tippte auf „${esc(D.itos[c.robot])}“ und war sich nur zu ${rp} % sicher.`;
    else if (robOk) verdict = `<b class="rob">${esc(N())}</b> hatte recht (zu ${rp} % sicher). Du hattest „${esc(D.itos[c.kid])}“ getippt.`;
    else verdict = `Beide daneben! Richtig war „${esc(D.itos[c.target])}“.`;
  }
  const more = revealed && c.probs ? `<details class="more"><summary>Genauer hingeschaut</summary><div class="inner"><p>So sicher war sich ${esc(N())} bei den sechs Buchstaben:</p><div class="bars">${c.probs.map(([t, p]) => `<div class="bar-row"><span class="ch">${esc(D.itos[t])}</span><span class="track"><span class="fill" style="width:${Math.max(1, p * 100)}%"></span></span><span class="p">${Math.round(p * 100)} %</span></div>`).join('')}</div></div></details>` : '';
  const nextBtn = revealed ? (g.round < GAME_ROUNDS ? `<button class="btn btn-main" type="button" data-next="1">Nächste Runde</button>` : `<button class="btn" type="button" data-again="1">Nochmal spielen</button>`) : '';
  box.innerHTML = `
    <div class="score"><span><span class="who">Du</span><span class="n">${g.kid}</span></span><span class="vs">gegen</span><span><span class="who">${esc(N())}</span><span class="n">${g.robot}</span></span></div>
    <div class="round-no">Runde ${g.round} von ${GAME_ROUNDS}</div>
    <div class="ctx">${ctxHtml(g)}</div>
    ${revealed ? '' : '<p class="note" style="text-align:center">Welcher Buchstabe gehört in die Lücke? Tippe auf einen.</p>'}
    <div class="picks">${tiles}</div>
    <div class="verdict">${verdict}</div>
    <div class="row" style="justify-content:center">${nextBtn}</div>
    ${more}`;
}
async function kidPick(g, t) {
  const c = g.cur;
  if (c.kid != null) return;
  c.kid = t;
  await c.ready;
  if (c.robot == null) return;
  if (t === c.target) g.kid++;
  if (c.robot === c.target) g.robot++;
  renderGame(g);
  const kidOk = t === c.target, robOk = c.robot === c.target;
  if (robOk && !kidOk) setRobot('proud', pick(['Ha! Ich wusste es!', 'Den kannte ich!', 'Treffer!']));
  else if (kidOk && !robOk) setRobot('oops', pick(['Mist. Du bist gut!', 'Daneben. Wie machst du das?', 'Hm, ich muss noch üben.']));
  else if (kidOk && robOk) setRobot('happy', pick(['Beide richtig!', 'Wir sind ein gutes Team.']));
  else setRobot('think', pick(['Puh, schwer!', 'Den hätte keiner erraten.']));
  if (g.round >= GAME_ROUNDS) finishGame(g);
}
function finishGame(g) {
  g.finished = true;
  S.score[g.phase] = { kid: g.kid, robot: g.robot };
  S.done[g.phase === 'before' ? 2 : 4] = true;
  if (g.phase === 'before') {
    setRobot('oops', g.robot >= g.kid
      ? `${g.robot} zu ${g.kid}. Nicht schlecht für einen, der noch nie gelesen hat. Das war aber Glück, ich rate blind!`
      : `${g.kid} zu ${g.robot} für dich. Ich rate ja auch nur blind. Aber wart's ab, bis ich das Buch gelesen habe!`);
  } else {
    const b = S.score.before;
    setRobot(g.robot >= g.kid ? 'proud' : 'happy',
      `Vorher hatte ich ${b.robot} von ${GAME_ROUNDS}, jetzt ${g.robot} von ${GAME_ROUNDS}! ` +
      (g.robot > b.robot ? 'Das Lesen hat sich gelohnt.' : 'Hm, ich sollte noch mehr lesen. Geh doch zurück zu Stufe 3.'));
  }
  renderNav();
}

/* ================= level 3: reading ================= */
function trainChunk(k) {
  return gpu(async () => {
    const costs = [];
    for (let i = 0; i < k; i++) {
      const { x, y } = sampleBatch();
      costs.push(tf.tidy(() => S.opt.minimize(() => lossOf(x, y, true), true, S.vars)));
      x.dispose(); y.dispose();
    }
    const vals = await Promise.all(costs.map(c => c.data()));
    costs.forEach(c => c.dispose());
    return vals.map(v => v[0]);
  });
}
function recordTrain(l) {
  S.step++;
  S.trainLoss.push(l);
  S.ema = S.ema == null ? l : S.ema + 0.08 * (l - S.ema);
  S.trainEma.push(S.ema);
}
async function doVal(id) {
  const v = await gpu(async () => {
    if (id !== S.runId) return NaN;
    const t = tf.tidy(() => lossOf(S.xVal, S.yVal, false));
    const d = await t.data();
    t.dispose();
    return d[0];
  });
  if (id !== S.runId || !Number.isFinite(v)) return;
  S.lastVal = S.step;
  S.valPts.push({ s: S.step, v });
  if (v < S.bestVal) { S.bestVal = v; S.bestValStep = S.step; }
  const trainNow = S.ema == null ? v : S.ema;
  if (!S.overfit && S.step > 200 && v - S.bestVal > 0.07 && S.step - S.bestValStep >= 200 && trainNow < v - 0.12) {
    S.overfit = true; S.overfitStep = S.bestValStep;
  } else if (S.overfit && v <= S.bestVal + 0.01) {
    S.overfit = false;
  }
}
const lastVal = () => (S.valPts.length ? S.valPts[S.valPts.length - 1].v : null);
function stageOf(L) {
  const R = S.data.refs;
  if (L == null || L >= R.uni) return 0;
  if (L >= R.bi) return 1;
  if (L >= R.tri) return 2;
  return 3;
}
const curStage = () => stageOf(Number.isFinite(S.bestVal) ? S.bestVal : lastVal());
const STAGE_NAMES = () => ['Kauderwelsch', 'Kennt häufige Buchstaben', 'Kennt Silben', S.data.lineMode ? 'Erfindet Namen' : 'Schreibt Wörter'];
const STAGE_LINES = () => [
  ['Puh. Ich bin noch dauernd überrascht. Lass mich weiterlesen!'],
  ['Ich merke: Manche Buchstaben kommen viel öfter vor als andere. „e“ zum Beispiel!', 'Aha! Nach einem Leerzeichen kommt oft ein großer Buchstabe.'],
  ['Jetzt weiß ich, welche Buchstaben gern zusammen stehen. „sch“, „ei“, „en“ …', 'Das klingt schon fast wie Silben!'],
  S.data.lineMode ? ['Schau mal, ich erfinde Namen, die es gar nicht gibt!'] : ['Schau mal, ich schreibe schon echte Wörter!', 'Ich glaube, ich kann jetzt Märchen!'],
];

function roundLabel() {
  const r = S.roundsDone;
  if (S.enough) return 'Bitte Seite neu laden';
  if (r < ROUNDS.length) return `${r + 1}. Runde: ${nf0.format(ROUNDS[r])} Blicke ins Buch`;
  return `Noch eine Runde: ${nf0.format(EXTRA)} Blicke`;
}
function renderRead(start, target) {
  const btn = $('#read-btn');
  btn.textContent = S.reading ? 'Liest …' : roundLabel();
  btn.disabled = S.reading || S.enough || !S.model;
  const fill = $('#read-fill');
  if (S.reading && target > start) fill.style.width = Math.min(100, ((S.step - start) / (target - start)) * 100) + '%';
  else fill.style.width = S.roundsDone ? '100%' : '0%';
  $('#read-note').textContent = S.reading ? `${nf0.format(S.step)} Blicke bisher` : S.step ? `Bisher ${nf0.format(S.step)} Blicke ins Buch.` : '';
}
function renderMeter() {
  const lv = lastVal();
  const hi = S.data.refs.uniform, lo = 0.9;
  const L = lv == null ? hi : lv;
  const pct = Math.max(0, Math.min(1, (hi - L) / (hi - lo)));
  $('#knob').style.left = (2 + pct * 96) + '%';
}
function renderLadder() {
  const st = curStage();
  $('#ladder').innerHTML = STAGE_NAMES().map((n, i) => `<li class="${i < st ? 'on' : i === st ? 'cur' : ''}"><i>${i + 1}</i>${esc(n)}</li>`).join('');
}
async function readRound() {
  if (S.reading || !S.model || S.enough) return;
  const inc = S.roundsDone < ROUNDS.length ? ROUNDS[S.roundsDone] : EXTRA;
  const start = S.step, target = S.step + inc, id = S.runId;
  S.reading = true;
  setRobot('read', pick(['Ich lese … Buchstabe für Buchstabe.', 'Nicht stören, ich lese!', 'Mal sehen, was in dem Buch steht …']));
  renderRead(start, target);
  let nextDiary = Math.min(target, S.step + 500);
  try {
    while (S.step < target && id === S.runId) {
      const losses = await trainChunk(Math.min(4, target - S.step));
      if (id !== S.runId) return;
      if (losses.some(l => !Number.isFinite(l))) { setStatus('Oh nein, die Zahlen im Kopf des Roboters sind durcheinander geraten. Bitte die Seite neu laden.', 'error'); S.enough = true; break; }
      losses.forEach(recordTrain);
      if (S.step - S.lastVal >= 25) await doVal(id);
      if (S.step >= nextDiary && S.step < target) { await addDiary(id, false); nextDiary = Math.min(target, S.step + 500); }
      renderRead(start, target); renderMeter(); renderLadder();
      if (S.step % 20 === 0) drawChart();
      if (S.overfit && !S.warnedOverfit) break;
      await yieldNow();
    }
    if (id !== S.runId) return;
    await doVal(id);
    await addDiary(id, true);
    S.roundsDone++;
    if (S.roundsDone >= 2) S.done[3] = true;
    const st = curStage();
    if (S.overfit && !S.warnedOverfit) {
      /* warn once, then let the kids keep reading and watch what memorising does */
      S.warnedOverfit = true;
      setRobot('think', `Moment! Ab jetzt lerne ich das Buch eher auswendig, statt die Sprache zu verstehen. Du kannst mich trotzdem weiterlesen lassen. Schau im Tagebuch, was dann passiert.`);
    } else {
      setRobot(st >= 3 ? 'proud' : st >= 1 ? 'happy' : 'think', pick(STAGE_LINES()[st]) + (S.roundsDone < 2 ? ' Aber das war erst der Anfang. Noch eine Runde!' : ''));
    }
  } finally {
    S.reading = false;
    renderRead(); renderMeter(); renderLadder(); drawChart(); renderNav();
  }
}
async function addDiary(id, roundEnd) {
  const rng = mulberry32(1000 + S.diary.length);
  const res = S.data.lineMode
    ? await generate('', 100, 0.8, rng, id, { maxLines: 8 })
    : await generate(CORPORA.maerchen.prompt, 100, 0.8, rng, id);
  if (!res || id !== S.runId) return;
  const st = curStage();
  const prev = S.diary[S.diary.length - 1];
  const REMARKS = ['noch keine Ahnung', 'häufige Buchstaben!', 'erste Silben!', S.data.lineMode ? 'echte Namen!' : 'echte Wörter!'];
  let remark = '';
  if (!prev || st > prev.stage) remark = REMARKS[st];
  else if (S.overfit && !prev.overfit) remark = 'lernt auswendig!';
  const e = { step: S.step, stage: Math.max(st, prev ? prev.stage : 0), overfit: S.overfit, remark, prompt: res.prompt, gen: res.gen };
  S.diary.push(e);
  const box = $('#diary');
  const el = document.createElement('div');
  el.className = 'entry';
  el.innerHTML = `<div class="when">Nach ${nf0.format(e.step)} Blicken · ${esc(STAGE_NAMES()[st])} ${e.remark ? `<span class="remark">${esc(e.remark)}</span>` : ''}</div><div class="txt">${renderOutput(e.prompt, e.gen)}</div>`;
  box.prepend(el);
}

/* ================= chart (in "Genauer hingeschaut") ================= */
function downsample(arr, maxPts) {
  const n = arr.length, out = [];
  if (n <= maxPts) { for (let i = 0; i < n; i++) out.push([i + 1, arr[i]]); return out; }
  const b = n / maxPts;
  for (let k = 0; k < maxPts; k++) {
    const a = Math.floor(k * b), e = Math.floor((k + 1) * b);
    let s = 0;
    for (let i = a; i < e; i++) s += arr[i];
    out.push([e, s / (e - a)]);
  }
  return out;
}
function drawChart() {
  const svg = $('#chart');
  if (!S.data || !svg.clientWidth) return;
  const W = Math.max(280, svg.clientWidth), H = svg.clientHeight || 260;
  const m = { l: 40, r: 12, t: 12, b: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const R = S.data.refs;
  const yMax = Math.ceil(R.uniform + 0.3);
  const xMax = niceCeil(Math.max(100, S.step * 1.06));
  const X = s => m.l + (s / xMax) * iw;
  const Y = v => m.t + (1 - Math.max(0, Math.min(v, yMax)) / yMax) * ih;
  const pts = arr => arr.map(([s, v]) => X(s).toFixed(1) + ',' + Y(v).toFixed(1)).join(' ');
  let o = '';
  for (let v = 0; v <= yMax; v++) {
    o += `<line class="grid-l" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`;
    o += `<text class="ax" x="${m.l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${v}</text>`;
  }
  for (let s = 0; s <= xMax + 1e-9; s += xMax / 5) {
    o += `<text class="ax" x="${X(s)}" y="${H - m.b + 18}" text-anchor="${s === 0 ? 'start' : s >= xMax ? 'end' : 'middle'}">${nf0.format(Math.round(s))}</text>`;
  }
  o += `<text class="ax-title" x="${m.l + iw / 2}" y="${H - 2}" text-anchor="middle">Blicke ins Buch</text>`;
  o += `<text class="ax-title" transform="translate(11 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">Überraschung</text>`;
  if (S.overfit) o += `<rect class="overfit" x="${X(S.overfitStep)}" y="${m.t}" width="${Math.max(0, X(S.step) - X(S.overfitStep))}" height="${ih}"/>`;
  const refs = [['uniform', 'Blind raten'], ['uni', 'Buchstaben zählen'], ['bi', 'Paare zählen'], ['tri', 'Dreiergruppen zählen']];
  let lastY = -99, side = 'right';
  for (const [k, label] of refs) {
    const y = Y(R[k]);
    o += `<line class="ref" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/>`;
    side = Math.abs(y - lastY) < 15 ? (side === 'right' ? 'left' : 'right') : 'right';
    o += `<text class="ref-t" x="${side === 'right' ? W - m.r - 4 : m.l + 6}" y="${y - 5}" text-anchor="${side === 'right' ? 'end' : 'start'}">${label}</text>`;
    lastY = y;
  }
  if (S.trainEma.length) o += `<polyline class="train" points="${pts(downsample(S.trainEma, Math.floor(iw / 2)))}"/>`;
  if (S.valPts.length) o += `<polyline class="val" points="${pts(S.valPts.map(p => [p.s, p.v]))}"/>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = o;
}

/* ================= level 5: writing ================= */
async function writeNow() {
  if (!S.model || S.writing) return;
  S.writing = true;
  const btn = $('#w-go'), out = $('#w-out');
  btn.disabled = true;
  setRobot('think', pick(['Ich denke nach …', 'Buchstabe für Buchstabe …', 'Moment, ich schreibe!']));
  const prompt = S.data.lineMode ? '' : $('#w-prompt').value;
  const temp = parseFloat($('#w-temp').value);
  const id = S.runId;
  const lead = S.data.lineMode ? '' : `<span class="pr">${esc(encodePrompt(prompt).known.join(''))}</span>`;
  out.innerHTML = lead + '<span class="cursor"></span>';
  const res = await generate(prompt, S.data.lineMode ? 120 : 220, temp, Math.random, id, {
    maxLines: S.data.lineMode ? 10 : 0,
    onChar: t => { out.innerHTML = lead + esc(t) + '<span class="cursor"></span>'; },
  });
  S.writing = false;
  btn.disabled = false;
  if (!res) return;
  out.innerHTML = renderOutput(res.prompt, res.gen);
  S.wrote = true; S.done[5] = true;
  const st = curStage();
  let note = '';
  if (S.data.lineMode) {
    const r = renderNames(res.gen);
    note = r.total ? `${r.total - r.known} von ${r.total} Namen sind neu erfunden.` : '';
    setRobot(st >= 3 ? 'proud' : 'happy', st >= 3 ? 'Na? Klingen die nicht wie echte Namen?' : 'Hm, noch etwas holprig. In Stufe 3 könnte ich noch mehr lesen.');
  } else {
    const r = renderMarked(res.prompt, res.gen);
    note = r.share > 0.05 ? `${Math.round(r.share * 100)} % davon stehen genau so im Buch (gelb).` : 'Fast alles selbst erfunden!';
    setRobot(st >= 3 ? 'proud' : st >= 2 ? 'happy' : 'think',
      st >= 3 ? 'Na? Ein paar echte Wörter sind dabei, oder? Perfekt ist es nicht. Dafür bin ich zu klein.' :
      st >= 2 ? 'Es klingt schon fast wie Deutsch, findest du nicht?' : 'Hm, noch ziemlich holprig. In Stufe 3 könnte ich noch mehr lesen.');
  }
  $('#w-note').textContent = note;
  btn.textContent = 'Nochmal!';
  renderNav();
}

/* ================= level 6 ================= */
function renderGiants() {
  const P = S.paramTotal, C = S.data.totalChars;
  $('#g-ours-params').textContent = nf0.format(P);
  const kg = P * 0.025 / 1000;
  $('#g-ours-rice').textContent = `Wären das Reiskörner, wären es ${kg < 1 ? Math.round(kg * 1000) + ' Gramm, etwa eine Handvoll' : nf1.format(kg) + ' Kilo, etwa ein Sack'}.`;
  $('#g-ours-read').textContent = S.data.lineMode ? `${nf0.format(S.data.lines)} Namen` : `${nf0.format(C)} Zeichen`;
  $('#g-ours-pages').textContent = S.data.lineMode ? 'Das passt auf zwei Buchseiten.' : `Das sind ungefähr ${Math.max(1, Math.round(C / 1800))} Buchseiten.`;
}

/* ---- zoom: squares whose area is the number of parameters; zooming out shrinks the robot to nothing ---- */
const ZOOM = { stage: 0, ppu: 0, raf: 0 };
const zoomModels = () => [
  { name: N(), n: S.paramTotal, col: '--robot' },
  { name: 'GPT-2 (2019)', n: 1.5e9, col: '--blue' },
  { name: 'GPT-3 (2020)', n: 175e9, col: '--green' },
  { name: 'Die größten heute', n: 1.2e12, col: '--red' },
];
function fmtTimes(r) {
  if (r >= 1e9) return nf1.format(r / 1e9).replace(',0', '') + ' Milliarden-mal';
  if (r >= 1e6) return nf1.format(r / 1e6).replace(',0', '') + ' Millionen-mal';
  return nf0.format(Math.round(r / 1000) * 1000) + '-mal';
}
function zoomTexts() {
  const M = zoomModels(), me = M[0].n, t = j => fmtTimes(M[j].n / me);
  return [
    `Das orangefarbene Quadrat ist ${N()}: ${nf0.format(me)} Zahlen im Kopf. Die Fläche jedes Quadrats zeigt, wie viele Zahlen ein Modell im Kopf hat. Jetzt zoomen wir raus.`,
    `GPT-2 aus dem Jahr 2019 hat ${t(1)} so viele. ${N()} ist nur noch das kleine Kästchen unten links.`,
    `GPT-3 aus dem Jahr 2020 hat ${t(2)} so viele wie ${N()}. ${N()} ist jetzt kleiner als ein einziger Bildpunkt.`,
    `Die größten Modelle heute haben ${t(3)} so viele wie ${N()}. Selbst GPT-2 ist nur noch ein Kästchen. Und ${N()}? Längst unsichtbar.`,
  ];
}
function zoomPpu(k) {
  const cv = $('#zoom');
  const R = Math.max(40, Math.min(cv.clientWidth - 170, cv.clientHeight - 60));
  return R / Math.sqrt(zoomModels()[k].n);
}
function zoomUi() {
  $('#zoom-btn').textContent = ZOOM.stage >= 3 ? 'Nochmal von vorn' : 'Rauszoomen';
  $('#zoom-text').textContent = zoomTexts()[ZOOM.stage];
}
function zoomSetup() {
  clearTimeout(ZOOM.raf);
  ZOOM.stage = 0;
  ZOOM.ppu = zoomPpu(0);
  zoomUi();
  zoomDraw();
}
/* driven by a timer, not requestAnimationFrame: frames stall while the window is covered */
function zoomNext() {
  clearTimeout(ZOOM.raf);
  if (ZOOM.stage >= 3) { zoomSetup(); return; }
  ZOOM.stage++;
  zoomUi();
  const from = ZOOM.ppu, to = zoomPpu(ZOOM.stage);
  if (reduced) { ZOOM.ppu = to; zoomDraw(); return; }
  const t0 = performance.now(), dur = 1600;
  const step = () => {
    const u = Math.min(1, (performance.now() - t0) / dur), e = 1 - Math.pow(1 - u, 3);
    ZOOM.ppu = from * Math.pow(to / from, e);
    zoomDraw();
    if (u < 1) ZOOM.raf = setTimeout(step, 16);
  };
  step();
}
function zoomRedraw() {
  if (!S.data || S.level !== 6) return;
  ZOOM.ppu = zoomPpu(ZOOM.stage);
  zoomDraw();
}
function zoomDraw() {
  const cv = $('#zoom');
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const cs = getComputedStyle(document.documentElement);
  const col = v => cs.getPropertyValue(v).trim();
  ctx.font = '700 15px ' + (cs.getPropertyValue('--f-body').trim() || 'sans-serif');
  ctx.textBaseline = 'middle';
  const M = zoomModels();
  const x0 = 20, y0 = H - 30;
  const sides = M.map(m => Math.sqrt(m.n) * ZOOM.ppu);
  /* biggest first, so the small ones stay on top; only models up to the current stage */
  for (let j = Math.min(ZOOM.stage, M.length - 1); j >= 0; j--) {
    const s = Math.max(sides[j], 2), top = y0 - s;
    ctx.fillStyle = col(M[j].col);
    ctx.fillRect(x0, top, s, s);
    if (s >= 36) {
      const tw = ctx.measureText(M[j].name).width;
      if (x0 + s + 10 + tw < W - 6) { ctx.fillStyle = col('--ink'); ctx.fillText(M[j].name, x0 + s + 10, Math.max(12, top + 10)); }
      else { ctx.fillStyle = '#fff'; ctx.fillText(M[j].name, x0 + 10, Math.max(14, top + 16)); }
    } else if (j > 0 && s >= 6) {
      ctx.fillStyle = col('--ink');
      ctx.fillText(M[j].name, x0 + s + 8, y0 - s / 2);
    }
  }
  if (sides[0] < 36) {
    ctx.fillStyle = col('--robot-dark');
    ctx.fillText('▲ ' + (sides[0] < 1 ? `${N()}: kleiner als ein Bildpunkt` : N()), x0, y0 + 16);
  }
}
function confetti() {
  if (reduced) return;
  const cv = $('#confetti');
  cv.hidden = false;
  cv.width = innerWidth; cv.height = innerHeight;
  const ctx = cv.getContext('2d');
  const cs = getComputedStyle(document.documentElement);
  const cols = ['--robot', '--blue', '--green', '--yellow', '--red'].map(v => cs.getPropertyValue(v).trim());
  const parts = Array.from({ length: 110 }, () => ({ x: Math.random() * cv.width, y: -20 - Math.random() * cv.height * 0.5, vy: 2 + Math.random() * 3, vx: (Math.random() - 0.5) * 1.5, r: Math.random() * Math.PI, w: 6 + Math.random() * 6, c: pick(cols) }));
  const t0 = performance.now();
  (function frame(now) {
    ctx.clearRect(0, 0, cv.width, cv.height);
    let alive = false;
    for (const p of parts) {
      p.y += p.vy; p.x += p.vx; p.r += 0.05;
      if (p.y < cv.height + 20) alive = true;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r); ctx.fillStyle = p.c; ctx.fillRect(-p.w / 2, -p.w / 4, p.w, p.w / 2); ctx.restore();
    }
    if (alive && now - t0 < 4500) requestAnimationFrame(frame); else { ctx.clearRect(0, 0, cv.width, cv.height); cv.hidden = true; }
  })(t0);
}

/* ================= corpus + model lifecycle ================= */
function loadCorpus(key) {
  const C = CORPORA[key];
  S.corpusKey = key;
  /* the texts come from robbi-texte.js when the page is split into files, otherwise from the embedded blocks */
  const raw = (window.ROBBI_TEXTE && window.ROBBI_TEXTE[key]) || $(C.src).textContent;
  S.data = prepareData(raw, C.lineMode);
  const D = S.data;
  $('#vocab').innerHTML = D.itos.map(c => `<span class="tile" data-cat="${charCat(c)}">${esc(visCh(c))}</span>`).join('');
  const sample = Array.from(D.trainText.replace(/^\n/, '')).slice(0, 24);
  $('#tokens').innerHTML = sample.map(c => `<span class="tile" data-cat="${charCat(c)}" title="${esc(visCh(c))}">${D.stoi.get(c)}</span>`).join('');
  $('#w-prompt').value = C.prompt;
}
async function newModel() {
  await gpu(() => {
    S.runId++;
    if (S.model) {
      S.vars.forEach(v => v.dispose());
      S.opt.dispose(); S.xVal.dispose(); S.yVal.dispose();
    }
    S.cfg = SIZES[CORPORA[S.corpusKey].size];
    const { p, vars } = buildModel(S.cfg, S.data.V);
    S.model = p; S.vars = vars;
    S.opt = tf.train.adam(S.cfg.lr, 0.9, 0.99);
    buildValTensors();
  });
  S.paramTotal = S.vars.reduce((a, v) => a + v.size, 0);
  Object.assign(S, {
    step: 0, trainLoss: [], trainEma: [], ema: null, valPts: [], lastVal: -1,
    bestVal: Infinity, bestValStep: 0, overfit: false, overfitStep: 0, enough: false, warnedOverfit: false,
    reading: false, roundsDone: 0, diary: [], writing: false, l1done: false, wrote: false, confettiDone: false,
    score: { before: { kid: 0, robot: 0 }, after: { kid: 0, robot: 0 } },
    visited: [true], done: [true],
  });
  G.before = null; G.after = null; G.afterStep = -1;
  $('#diary').innerHTML = '';
  $('#l1-out').innerHTML = '<span class="ph">Hier schreibt der Roboter.</span>';
  $('#w-out').innerHTML = '<span class="ph">Hier schreibt der Roboter.</span>';
  $('#w-note').textContent = '';
  $('#l1-write').textContent = 'Schreib mal was!';
  $('#w-go').textContent = 'Schreib!';
  $('#m-params').textContent = nf0.format(S.paramTotal);
  $('#m-ctx').textContent = S.cfg.ctx;
  await doVal(S.runId);
  await addDiary(S.runId, true);
  renderRead(); renderMeter(); renderLadder();
}

/* ================= events ================= */
function wire() {
  $('#corpus-choice').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    $$('#corpus-choice button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    S.corpusKey = b.dataset.corpus;
  });
  $('#name-in').addEventListener('input', e => {
    S.name = e.target.value.trim() || 'Robbi';
    $('#name-tag').textContent = S.name;
  });
  $('#start-btn').addEventListener('click', async () => {
    S.name = $('#name-in').value.trim() || 'Robbi';
    const btn = $('#start-btn');
    btn.disabled = true; btn.textContent = 'Einen Moment …';
    loadCorpus(S.corpusKey);
    await newModel();
    btn.disabled = false; btn.textContent = 'Los geht’s!';
    goTo(1);
  });
  $('#l1-write').addEventListener('click', l1Write);
  for (const [sel, phase] of [['#game-before', 'before'], ['#game-after', 'after']]) {
    $(sel).addEventListener('click', e => {
      const g = G[phase];
      if (!g) return;
      const pk = e.target.closest('.pick');
      if (pk && !pk.disabled) { kidPick(g, parseInt(pk.dataset.t, 10)); return; }
      if (e.target.closest('[data-next]')) { newRound(g); setRobot('think', pick(['Nächste Runde!', 'Weiter geht’s.', 'Jetzt aber.'])); return; }
      if (e.target.closest('[data-again]')) { startGame(phase); setRobot('think', 'Noch mal von vorn!'); }
    });
  }
  $('#read-btn').addEventListener('click', readRound);
  $('#w-go').addEventListener('click', writeNow);
  $('#next').addEventListener('click', () => { if (S.done[S.level] && S.level < 6) goTo(S.level + 1); });
  $('#back').addEventListener('click', () => { if (S.level > 0) goTo(S.level - 1); });
  $('#steps').addEventListener('click', e => {
    const b = e.target.closest('button[data-go]');
    if (!b || b.disabled) return;
    goTo(parseInt(b.dataset.go, 10));
  });
  $('#restart').addEventListener('click', async () => {
    $('#restart').disabled = true;
    await newModel();
    $('#restart').disabled = false;
    goTo(0);
  });
  $('#zoom-btn').addEventListener('click', zoomNext);
  new ResizeObserver(() => zoomRedraw()).observe($('#zoom'));
  new ResizeObserver(() => drawChart()).observe($('#chart'));
  $$('details.more').forEach(d => d.addEventListener('toggle', () => { if (d.open) drawChart(); }));
}

/* ================= boot ================= */
async function boot() {
  wire();
  renderSteps();
  renderNav();
  if (!window.tf) {
    setStatus('Der Roboter konnte nicht geladen werden. Bitte Internetverbindung prüfen und die Seite neu laden.', 'error');
    say('Oh nein, ich komme nicht in Gang. Bitte lade die Seite neu.');
    return;
  }
  try {
    tf.enableProdMode();
    await tf.ready();
    const be = tf.getBackend();
    S.backendName = be === 'webgl' ? 'der Grafikkarte' : 'dem Prozessor';
    setRobot('sleepy', 'Hallo! Ich bin ein Roboter. Ich kenne alle Buchstaben, aber ich kann noch nicht schreiben. Bringst du es mir bei?');
    const btn = $('#start-btn');
    btn.disabled = false; btn.textContent = 'Los geht’s!';
    if (be !== 'webgl') setStatus('Hinweis: Dieser Computer rechnet ohne Grafikkarte, das Lesen dauert länger.');
  } catch (e) {
    console.error(e);
    setStatus('Start fehlgeschlagen: ' + e.message, 'error');
  }
}
boot();
})();
