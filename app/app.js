'use strict';

const SR = 4000;  // model's sample rate
const RECORD_S = 15;
const COUNTDOWN_S = 3;
const MIN_CLARITY = 0.25;
// a heart through a phone is almost all under 400 hz, talking and rubbing aren't
const MIN_LOW_SHARE = 0.7;
const SPOTS = [
  { key: 'PV', title: 'Top of the breastbone',
    text: 'Feel the collarbone, go 1 in to 2 in down, and then press against the bones of your ribs.' },
  { key: 'TV', title: 'Bottom of the breastbone',
    text: 'Find the V at the bottom where the ribs meet, go 1 inch up, and press against the bone.' },
  { key: 'MV', title: 'Under the chest muscle',
    text: 'Lift up the left chest muscle and press against the bone directly under it.' },
];
const ZOOM_S = 4;

const $ = id => document.getElementById(id);

const state = { spot: 0, results: [], note: '', share: null };
let cfg,
  session,
  modelReady,
  metrics,
  cancelRecording = null;

function show(id) {
  if (id !== 'result') stopPlayer();
  if (['home', 'samples', 'about', 'result'].includes(id)) closeMic();
  document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === id));
  document.body.dataset.screen = id;
  window.scrollTo(0, 0);
}

function showSpot() {
  const s = SPOTS[state.spot];
  $('spot-done').textContent = state.spot ? `✓ ${SPOTS[state.spot - 1].title} recorded` : '';
  $('step-n').textContent = state.spot + 1;
  $('spot-title').textContent = s.title;
  $('spot-text').textContent = s.text;
  document.querySelectorAll('#spot .diagram .spot').forEach((el, i) => {
    el.classList.toggle('current', i === state.spot);
    el.classList.toggle('done', i < state.spot);
    el.querySelector('text').textContent = i < state.spot ? '✓' : i + 1;
  });
  $('mic-error').textContent = '';
  show('spot');
}

function showResult(note = '') {
  const over = state.results.map(r => r.prob >= cfg.threshold),
    heard = SPOTS.filter((_, i) => over[i]);
  $('result-card').className = 'card ' + (heard.length ? 'murmur' : 'clear');
  $('result-title').textContent = heard.length ? 'Murmur heard: get an echocardiogram' : 'No murmur heard in these recordings';
  $('result-where').textContent = heard.length
    ? `Heard at ${heard.length} of ${SPOTS.length} spots: ${heard.map(s => s.title.toLowerCase()).join(', ')}.`
    : `Not heard at any of the ${SPOTS.length} spots.`;
  $('result-text').textContent = heard.length
    ? 'This phone heard a sound that may be a heart murmur. Many murmurs in children are harmless, but some are signs of a heart problem that is easier to treat when found early. An echocardiogram, an ultrasound of the heart, is how doctors tell the difference. Share this result with a health worker or doctor and ask for one.'
    : "This check didn't hear a murmur, but it could still be inaccurate. If you're worried about someone's heart, go see a doctor immediately.";
  state.note = note;
  $('result-note').textContent = `${note ? note + ' ' : ''}This isn't medical advice or a diagnosis.`;
  $('result-spots').innerHTML = state.results.map((r, i) => spotCard(i, r)).join('');
  show('result');
  drawRecordings();
  // make the image now, phones only let you share right after a tap
  state.share = null;
  resultFile().then(f => { state.share = f; });
}

// stretched so the line lands at 50
function murmurScore(p) {
  const t = cfg.threshold,
    s = p < t ? 50 * p / t : 50 + 50 * (p - t) / (1 - t);
  return p < t ? Math.min(49, Math.round(s)) : Math.max(50, Math.round(s));
}

function spotCard(i, r) {
  const over = r.prob >= cfg.threshold,
    score = murmurScore(r.prob);
  return `<div class="spot-card">
    <div class="spot-head"><span class="n">${i + 1}</span><b>${SPOTS[i].title}</b>
      <span class="tag ${over ? 'murmur' : 'clear'}">${over ? 'Murmur' : 'No murmur'}</span></div>
    <div class="zoom" data-zoom="${i}"><canvas></canvas><div class="playhead" id="head-${i}"></div>
      <span class="axis high">high pitch</span><span class="axis low">low</span></div>
    <div class="player">
      <button class="play" id="play-${i}" data-play="${i}" aria-label="Play">${ICON.play}</button>
      <div class="overview" data-seek="${i}"><canvas></canvas><div class="window" id="win-${i}"></div></div>
    </div>
    <div class="score"><div>Murmur score <b>${score}</b></div>
      <div class="track"><div class="fill${over ? ' over' : ''}" style="width:${score}%"></div><div class="mark" style="left:50%"></div></div></div>
    <div class="facts">${facts(r)}</div>
  </div>`;
}

const facts = r =>
  `${r.bpm ? `Heart rate about ${Math.round(r.bpm)} beats a minute` : 'Heart rate unclear'} · ${r.clarity >= 0.35 ? 'Clear' : 'Faint'} heartbeat`;

function drawRecordings() {
  state.results.forEach((r, i) => {
    drawWave(document.querySelector(`[data-seek="${i}"] canvas`), r.wave, 1.3 * r.level);
    drawZoom(i, player?.i === i ? player.audio.currentTime : 0);
  });
}
addEventListener('resize', () => $('result').classList.contains('active') && drawRecordings());

async function renderAbout() {
  const pct = v => Math.round(v * 100) + '%';
  try {
    metrics ??= await (await fetch('model/metrics.json')).json();
    const all = metrics.every_child;
    $('about-results').textContent =
      `We tested it on ${metrics.tested_on}, using recordings changed to sound like a phone. Each child was scored by ` +
      `networks that never trained on them. It was right for ${pct(all.accuracy)} of children overall.`;
  } catch {
    $('about-results').textContent = 'Results unavailable.';
  }
}

document.addEventListener('click', e => {
  const el = e.target.closest('[data-go], [data-action], [data-sample], [data-play], [data-seek], [data-zoom]');
  if (!el) return;
  e.preventDefault();
  const along = () => {
    const box = el.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
  };
  if (el.dataset.play) playSpot(+el.dataset.play);
  if (el.dataset.seek) playSpot(+el.dataset.seek, along());
  if (el.dataset.zoom) {
    const i = +el.dataset.zoom,
      { start, len, dur } = zoomWindow(i);
    playSpot(i, (start + len * along()) / dur);
  }
  if (el.dataset.go === 'about') renderAbout();
  if (el.dataset.go) show(el.dataset.go);
  if (el.dataset.sample) runSample(el.dataset.sample);
  const action = el.dataset.action;
  if (action === 'start') show('before');
  if (action === 'begin') beginCheck();
  if (action === 'record' || action === 'retry') recordSpot();
  if (action === 'mic-retry' || action === 'mic-close') $('mic-dialog').close();
  if (action === 'mic-retry') micRetry?.();
  if (action === 'cancel' && cancelRecording) cancelRecording();
  if (action === 'share') shareResult();
});

let mic = null;
let micRetry = null;

// voice settings treat the heartbeat as noise, so turn them all off
async function openMic() {
  if (mic?.stream.getAudioTracks().some(t => t.readyState === 'live')) return mic;
  closeMic();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: {
    echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
  // older safari only has webkitAudioContext
  mic = { stream, ctx: new (window.AudioContext || window.webkitAudioContext)() };
  return mic;
}

function closeMic() {
  if (!mic) return;
  const { stream, ctx } = mic;
  mic = null;
  stream.getTracks().forEach(t => t.stop());
  setTimeout(() => ctx.close().catch(() => {}), 1000);  // after the last chime
}

function reloadAndAsk() {
  location.hash = 'ask-mic';
  location.reload();
}

async function askMic(retry) {
  const t0 = performance.now();
  try {
    await openMic();
    sessionStorage.removeItem('micReloaded');
    return true;
  } catch (e) {
    // a no this fast means the phone never asked, reload once so it does
    const ms = Math.round(performance.now() - t0),
      denied = ['NotAllowedError', 'PermissionDeniedError', 'SecurityError'].includes(e?.name);
    if (denied && ms < 800 && !sessionStorage.getItem('micReloaded')) {
      sessionStorage.setItem('micReloaded', '1');
      reloadAndAsk();
    } else {
      micProblem(e, denied ? reloadAndAsk : retry);
    }
    return false;
  }
}

function micProblem(e, retry) {
  const name = e?.name || '',
    iphone = /iPhone|iPad|iPod/.test(navigator.userAgent);
  let title = "Couldn't start the microphone",
    text = 'Tap Try again. If it keeps happening, close the browser and open this page again.',
    button = 'Try again';
  if (!navigator.mediaDevices?.getUserMedia) {
    title = 'Open this page in Safari or Chrome';
    text = "This browser can't use the microphone. Apps like Instagram and WhatsApp open links in their own browser: open this page in Safari (iPhone) or Chrome (Android) instead.";
  } else if (['NotAllowedError', 'PermissionDeniedError', 'SecurityError'].includes(name)) {
    title = 'Allow the microphone';
    text = 'Murmur Check needs the microphone to hear the heart. Tap Ask again, then tap Allow. ' + (iphone
      ? "If your phone doesn't ask, open the page menu in Safari's address bar (aA or ···), tap Website Settings, and set Microphone to Allow."
      : "If your phone doesn't ask, tap the icon left of the address, then Permissions, and turn on Microphone.");
    button = 'Ask again';
  } else if (name === 'NotFoundError') {
    title = 'No microphone found';
    text = "This device doesn't seem to have a microphone. Use a phone.";
  } else if (['NotReadableError', 'AbortError', 'TrackStartError'].includes(name)) {
    title = 'The microphone is busy';
    text = 'Another app may be using it, like a call, a voice memo or another tab. Close it, then tap Try again.';
  }
  micRetry = retry;
  $('mic-title').textContent = title;
  $('mic-text').textContent = text;
  $('mic-retry').textContent = button;
  $('mic-detail').textContent = `Details: ${name || 'error'}${e?.message ? `: ${e.message}` : ''}`;
  const dialog = $('mic-dialog');
  if (dialog.showModal) dialog.showModal();
  else if (confirm(`${title}\n\n${text}`)) retry?.();
}

function clearResults() {
  stopPlayer();
  state.results.forEach(r => r.url && URL.revokeObjectURL(r.url));
  state.results = [];
}

async function beginCheck() {
  state.spot = 0;
  clearResults();
  if (await askMic(beginCheck)) showSpot();
}

async function recordSpot() {
  if (!(await askMic(recordSpot))) return;
  $('rec-step').textContent = `Spot ${state.spot + 1} of ${SPOTS.length}: ${SPOTS[state.spot].title}`;
  let rec;
  try {
    rec = await record();
  } catch (e) {
    showSpot();
    if (e.message !== 'cancelled') {
      micProblem(e, recordSpot);
    }
    return;
  }
  show('busy');
  try {
    const r = await analyze(rec.data, rec.sr, { phone: true });
    if (r.prob === null) {
      $('unclear-step').textContent = `Spot ${state.spot + 1} of ${SPOTS.length}: ${SPOTS[state.spot].title}`;
      return show('unclear');
    }
    state.results[state.spot] = r;
    if (state.spot < SPOTS.length - 1) {
      state.spot += 1;
      showSpot();
    } else {
      showResult();
    }
  } catch (e) {
    showSpot();
    $('mic-error').textContent = `Something went wrong: ${e.message}`;
  }
}

async function runSample(kind) {
  show('busy');
  try {
    clearResults();
    for (const [i, s] of SPOTS.entries()) {
      const { data, sr } = await loadWav(`samples/${kind}_${s.key}.wav`);
      const r = await analyze(data, sr);
      if (r.prob === null) throw new Error('sample recording is unclear');
      state.results[i] = r;
    }
    showResult(`Sample: a child whose doctor ${kind === 'murmur' ? 'heard a murmur' : 'heard no murmur'}.`);
  } catch (e) {
    show('samples');
    alert(`Couldn't run the sample: ${e.message}`);
  }
}

async function analyze(raw, sr, { phone = false } = {}) {
  await loadModel();
  // skip the start, phone is still settling and the beep plays here
  const x = resample(raw, sr, SR),
    usable = x.subarray(Math.round(cfg.skip_seconds * SR));
  const { clarity, bpm } = beatClarity(x, SR, cfg.skip_seconds),
    share = phone ? lowShare(usable, SR) : 1;
  if (clarity < MIN_CLARITY || share < MIN_LOW_SHARE) return { clarity, share, prob: null };
  const wave = bandpass(usable, SR, 25, 400);
  return {
    clarity, bpm, share,
    prob: await murmurProbability(usable),
    spec: logSpectrogram(usable),
    wave,
    level: typicalLevel(wave),
  };
}

// multiple threads needs special server headers so just use one
function loadModel() {
  modelReady ??= (async () => {
    cfg = await (await fetch('model/config.json')).json();
    if (cfg.sample_rate !== SR) throw new Error('model expects a different sample rate');
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create('model/murmur.onnx', { executionProviders: ['wasm'] });
  })().catch(e => { modelReady = null; throw e; });
  return modelReady;
}

// 4 s windows 2 s apart, same as training
async function murmurProbability(x) {
  const W = cfg.window, H = cfg.hop;
  if (x.length < W) x = Float32Array.from({ length: W }, (_, i) => x[i % x.length]);
  const n = Math.floor((x.length - W) / H) + 1,
    batch = new Float32Array(n * W);
  for (let i = 0; i < n; i++) batch.set(x.subarray(i * H, i * H + W), i * W);
  const { prob } = await session.run({ audio: new ort.Tensor('float32', batch, [n, W]) });
  return prob.data.reduce((a, b) => a + b, 0) / n;
}

// same as LogSpectrogram in ml/model.py
const BANDS = 31,
  FRAME = 256,
  FRAME_HOP = 40;
let kernels;
// hann window so the frame edges don't add fake pitches
function kernelsReady() {
  kernels ??= Array.from({ length: BANDS }, (_, b) => {
    const hz = 25 + b * 375 / (BANDS - 1),
      k = new Float32Array(2 * FRAME);
    for (let n = 0; n < FRAME; n++) {
      const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (FRAME - 1)),
        phase = 2 * Math.PI * hz * n / SR;
      k[n] = w * Math.cos(phase);
      k[FRAME + n] = w * Math.sin(phase);
    }
    return k;
  });
}

function logSpectrogram(x) {
  kernelsReady();
  const y = highpass(highpass(x)),
    frames = Math.floor((y.length - FRAME) / FRAME_HOP) + 1;
  const data = new Float32Array(BANDS * frames);
  let total = 0;
  for (let b = 0; b < BANDS; b++) {
    const k = kernels[b];
    for (let f = 0; f < frames; f++) {
      let re = 0, im = 0;
      for (let n = 0, o = f * FRAME_HOP; n < FRAME; n++, o++) {
        re += k[n] * y[o];
        im += k[FRAME + n] * y[o];
      }
      total += data[b * frames + f] = re * re + im * im;
    }
  }
  // small floor so near-silent parts don't blow up after the log
  const floor = 1e-4 * total / data.length + 1e-12;
  for (let b = 0; b < BANDS; b++) {
    const row = data.subarray(b * frames, (b + 1) * frames);
    let mean = 0;
    for (let f = 0; f < frames; f++) mean += row[f] = Math.log(row[f] + floor);
    mean /= frames;
    for (let f = 0; f < frames; f++) row[f] -= mean;
  }
  return { data, frames };
}

// running sum, window gets cut short near the ends
function highpass(x) {
  const half = 80,
    y = new Float32Array(x.length);
  let sum = 0, lo = 0, hi = -1;
  for (let i = 0; i < x.length; i++) {
    while (hi < Math.min(x.length - 1, i + half)) sum += x[++hi];
    while (lo < i - half) sum -= x[lo++];
    y[i] = x[i] - sum / (hi - lo + 1);
  }
  return y;
}

const HEAT = (() => {
  const stops = [[0, 0x1b1311], [0.38, 0x5e1813], [0.62, 0xb3261e], [0.82, 0xec7a2a], [1, 0xfde7b0]],
    lut = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255,
      j = stops.findIndex(s => s[0] >= t),
      [t1, c1] = stops[Math.max(0, j - 1)],
      [t2, c2] = stops[j];
    const f = t2 > t1 ? (t - t1) / (t2 - t1) : 0;
    lut.push([16, 8, 0].map(s => Math.round(((c1 >> s) & 255) * (1 - f) + ((c2 >> s) & 255) * f)));
  }
  return lut;
})();

// ignore the loudest tenth so one bump doesn't make everything dim
function specImage({ data, frames }) {
  const loud = new Float32Array(frames);
  for (let b = 0; b < BANDS; b++) for (let f = 0; f < frames; f++) loud[f] += data[b * frames + f];
  const cut = Float32Array.from(loud).sort()[Math.floor(frames * 0.9)],
    kept = [];
  for (let f = 0; f < frames; f += 2) if (loud[f] <= cut) for (let b = 0; b < BANDS; b++) kept.push(data[b * frames + f]);
  const sorted = Float32Array.from(kept).sort();
  const lo = sorted[Math.floor(sorted.length * 0.5)],
    hi = sorted[Math.floor(sorted.length * 0.99)];
  const img = new ImageData(frames, BANDS);
  for (let b = 0; b < BANDS; b++) {
    for (let f = 0; f < frames; f++) {
      const t = Math.min(1, Math.max(0, (data[b * frames + f] - lo) / (hi - lo)));
      // band 0 is lowest pitch so it goes at the bottom
      const c = HEAT[Math.round(255 * t)],
        o = 4 * ((BANDS - 1 - b) * frames + f);
      img.data.set([c[0], c[1], c[2], 255], o);
    }
  }
  const canvas = document.createElement('canvas');
  canvas.width = frames;
  canvas.height = BANDS;
  canvas.getContext('2d').putImageData(img, 0, 0);
  return canvas;
}

function zoomWindow(i, t = 0) {
  const r = state.results[i],
    dur = r.wave.length / SR,
    len = Math.min(ZOOM_S, dur);
  return { start: Math.min(dur - len, Math.max(0, t - len / 4)), len, dur };
}

function drawZoom(i, t) {
  const r = state.results[i],
    box = document.querySelector(`[data-zoom="${i}"]`),
    canvas = box.querySelector('canvas');
  const { start, len, dur } = zoomWindow(i, t),
    frames = FRAME_HOP / SR;
  r.img ??= specImage(r.spec);
  if (canvas.width !== canvas.clientWidth * devicePixelRatio) fitCanvas(canvas);
  const g = canvas.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(r.img, start / frames, 0, len / frames, BANDS, 0, 0, canvas.width, canvas.height);
  drawWave(canvas, r.wave.subarray(Math.round(start * SR), Math.round((start + len) * SR)), 1.6 * r.level, 'rgba(255,255,255,.85)', false);
  $(`head-${i}`).style.left = 100 * (t - start) / len + '%';
  Object.assign($(`win-${i}`).style, { left: 100 * start / dur + '%', width: 100 * len / dur + '%' });
}

// low-pass first so high pitches don't fold back in as fake low sounds
function resample(x, from, to) {
  if (from === to) return Float32Array.from(x);
  const cutoff = 0.4 * to / from,
    half = Math.ceil(4 / cutoff),
    taps = new Float32Array(2 * half + 1);
  let sum = 0;
  for (let k = -half; k <= half; k++) {
    const sinc = k === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * k) / (Math.PI * k);
    sum += taps[k + half] = sinc * (0.54 + 0.46 * Math.cos(Math.PI * k / half));
  }
  const filtered = i => {
    let s = 0;
    for (let k = Math.max(-half, -i), end = Math.min(half, x.length - 1 - i); k <= end; k++) s += x[i + k] * taps[k + half];
    return s / sum;
  };
  const n = Math.floor(x.length * to / from),
    y = new Float32Array(n),
    step = from / to;
  for (let m = 0; m < n; m++) {
    const t = m * step,
      i = Math.floor(t),
      frac = t - i;
    y[m] = filtered(i) * (1 - frac) + (frac ? filtered(i + 1) * frac : 0);
  }
  return y;
}

function biquad(x, type, f0, fs, q = Math.SQRT1_2) {
  const w = 2 * Math.PI * f0 / fs,
    c = Math.cos(w),
    alpha = Math.sin(w) / (2 * q);
  const b0 = type === 'lowpass' ? (1 - c) / 2 : (1 + c) / 2,
    b1 = type === 'lowpass' ? 1 - c : -(1 + c);
  const a0 = 1 + alpha,
    a1 = -2 * c,
    a2 = 1 - alpha,
    y = new Float32Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = (b0 * x[i] + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1;
    x1 = x[i];
    y2 = y1;
    y1 = v;
    y[i] = v;
  }
  return y;
}

function bandpass(x, fs, lo, hi) {
  let f = x;
  for (const [type, hz] of [['highpass', lo], ['highpass', lo], ['lowpass', hi], ['lowpass', hi]])
    f = biquad(f, type, hz, fs);
  return f;
}

function typicalLevel(x) {
  const peaks = [];
  for (let i = 0; i + SR <= x.length; i += SR) {
    let m = 0;
    for (let j = i; j < i + SR; j++) m = Math.max(m, Math.abs(x[j]));
    peaks.push(m);
  }
  return peaks.sort((a, b) => a - b)[peaks.length >> 1] || 1e-9;
}

function lowShare(x, fs) {
  const energy = f => f.reduce((a, v) => a + v * v, 0);
  return energy(bandpass(x, fs, 25, 400)) / (energy(bandpass(x, fs, 25, 1900)) || 1);
}

function beatClarity(x, fs, skipS) {
  const f = bandpass(x, fs, 25, 400);
  const hop = Math.round(fs / 100),
    env = [];
  for (let i = 0; i + hop <= f.length; i += hop) {
    let s = 0;
    for (let j = i; j < i + hop; j++) s += f[j] * f[j];
    env.push(Math.sqrt(s / hop));
  }
  const smooth = env.map((_, i) => {
    let s = 0, n = 0;
    for (let k = Math.max(0, i - 3); k <= Math.min(env.length - 1, i + 3); k++, n++) s += env[k];
    return s / n;
  });
  const tail = smooth.slice(Math.round(skipS * 100));
  if (tail.length < 300) return { clarity: 0 };
  // cap at 95th percentile so one bump can't dominate
  const cap = [...tail].sort((a, b) => a - b)[Math.floor(tail.length * 0.95)];
  const e = tail.map(v => Math.min(v, cap)),
    mean = e.reduce((a, b) => a + b, 0) / e.length,
    z = e.map(v => v - mean);
  const r0 = z.reduce((a, v) => a + v * v, 0) || 1;
  // autocorrelation, a steady heartbeat lines up at one beat length
  const ac = [];
  for (let lag = 32; lag <= 151; lag++) {
    let s = 0;
    for (let i = 0; i + lag < z.length; i++) s += z[i] * z[i + lag];
    ac[lag] = s / r0;
  }
  let best = 0, peak = 0;
  for (let lag = 33; lag <= 150; lag++) {
    best = Math.max(best, ac[lag]);
    if (ac[lag] > ac[lag - 1] && ac[lag] >= ac[lag + 1] && (!peak || ac[lag] > ac[peak])) peak = lag;
  }
  // peak is in 10 ms steps
  return { clarity: best, bpm: peak ? 6000 / peak : 0 };
}

function cue(ctx, kind) {
  const notes = { go: [[1320, 0, 0.2]], done: [[880, 0, 0.14], [1320, 0.16, 0.32]] }[kind];
  for (const [hz, at, len] of notes) {
    const t = ctx.currentTime + at,
      osc = ctx.createOscillator(),
      gain = ctx.createGain();
    osc.frequency.value = hz;
    // fade in/out so the note doesn't click
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.5, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + len);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + len + 0.05);
  }
  navigator.vibrate?.(kind === 'go' ? 150 : [100, 80, 100]);
}

function liveView(sr) {
  const step = sr / SR,
    wave = new Float32Array(ZOOM_S * SR),
    frame = new Float32Array(FRAME);
  // wraps around when full
  const img = document.createElement('canvas'),
    mean = new Float32Array(BANDS),
    column = new ImageData(1, BANDS);
  img.width = 2048;
  img.height = BANDS;
  const ig = img.getContext('2d');
  let acc = 0, count = 0, phase = 0, filled = 0, hop = 0, frames = 0, settled = 0, power = 0;
  kernelsReady();
  return {
    push(chunk) {
      for (const v of chunk) {
        acc += v;
        count++;
        if (++phase < step) continue;
        phase -= step;
        const y = acc / count;
        acc = count = 0;
        wave.copyWithin(0, 1);
        wave[wave.length - 1] = y;
        frame.copyWithin(0, 1);
        frame[FRAME - 1] = y;
        filled++;
        if (filled >= FRAME && ++hop >= FRAME_HOP) {
          hop = 0;
          addColumn();
        }
      }
    },
    draw(canvas) {
      if (canvas.width !== canvas.clientWidth * devicePixelRatio) fitCanvas(canvas);
      const g = canvas.getContext('2d'),
        w = canvas.width,
        h = canvas.height,
        shown = Math.min(frames, ZOOM_S * 100);
      g.fillStyle = '#1b1311';
      g.fillRect(0, 0, w, h);
      const x = frames % img.width, from = x - shown;
      g.imageSmoothingQuality = 'high';
      const right = w, perCol = w / (ZOOM_S * 100);
      if (from >= 0) g.drawImage(img, from, 0, shown, BANDS, right - shown * perCol, 0, shown * perCol, h);
      else {
        g.drawImage(img, img.width + from, 0, -from, BANDS, right - shown * perCol, 0, -from * perCol, h);
        g.drawImage(img, 0, 0, x, BANDS, right - x * perCol, 0, x * perCol, h);
      }
      const seen = wave.subarray(wave.length - Math.min(filled, wave.length));
      drawWave(canvas, seen, seen.length >= SR ? 2 * typicalLevel(seen) : 0, '#ff5a47', false, right - seen.length / wave.length * w);
    },
  };

  function addColumn() {
    let total = 0;
    const logs = new Float32Array(BANDS);
    for (let b = 0; b < BANDS; b++) {
      const k = kernels[b];
      let re = 0, im = 0;
      for (let n = 0; n < FRAME; n++) {
        re += k[n] * frame[n];
        im += k[FRAME + n] * frame[n];
      }
      logs[b] = re * re + im * im;
      total += logs[b];
    }
    const a = Math.max(0.01, 1 / ++settled);
    power = (1 - a) * power + a * total / BANDS;
    for (let b = 0; b < BANDS; b++) {
      const v = Math.log(logs[b] + 1e-4 * power + 1e-12);
      mean[b] = (1 - a) * mean[b] + a * v;
      const c = HEAT[Math.round(255 * Math.min(1, Math.max(0, (v - mean[b]) / 4)))],
        o = 4 * (BANDS - 1 - b);
      column.data.set([c[0], c[1], c[2], 255], o);
    }
    if (settled > 50) {
      ig.putImageData(column, frames % img.width, 0);
      frames++;
    }
  }
}

function record() {
  return new Promise(async (resolve, reject) => {
    let stream, ctx;
    try {
      ({ stream, ctx } = await openMic());
    } catch (e) {
      return reject(e);
    }
    show('recording');
    // iphone needs a tap before audio can start
    await ctx.resume();
    // these only run when connected to the speaker, so they output silence
    const sr = ctx.sampleRate,
      src = ctx.createMediaStreamSource(stream),
      proc = ctx.createScriptProcessor(4096, 1, 1);
    const hp = ctx.createBiquadFilter(),
      lp = ctx.createBiquadFilter(),
      tap = ctx.createScriptProcessor(1024, 1, 1);
    hp.type = 'highpass';
    hp.frequency.value = 25;
    lp.type = 'lowpass';
    lp.frequency.value = 400;
    src.connect(hp).connect(lp).connect(tap);
    tap.connect(ctx.destination);
    src.connect(proc);
    proc.connect(ctx.destination);

    const chunks = [], live = liveView(sr);
    let listening = false;
    proc.onaudioprocess = e => {
      if (listening) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      e.outputBuffer.getChannelData(0).fill(0);
    };
    tap.onaudioprocess = e => {
      live.push(e.inputBuffer.getChannelData(0));
      e.outputBuffer.getChannelData(0).fill(0);
    };
    let frame;
    const draw = () => {
      live.draw($('live'));
      frame = requestAnimationFrame(draw);
    };
    draw();

    const t0 = performance.now();
    let lastCheck = 0;
    $('progress').style.width = '0';
    $('beat').className = 'beat';
    $('beat-text').textContent = '';
    $('countdown').textContent = COUNTDOWN_S;
    $('count-label').textContent = 'Starting in';
    $('rec-title').textContent = 'Place the phone and hold still';
    const timer = setInterval(() => {
      const t = (performance.now() - t0) / 1000;
      if (t < COUNTDOWN_S) {
        $('countdown').textContent = Math.ceil(COUNTDOWN_S - t);
        return;
      }
      if (!listening) {
        listening = true;
        cue(ctx, 'go');
        $('rec-title').textContent = 'Listening. Hold still.';
        $('count-label').textContent = 'seconds left';
      }
      const elapsed = t - COUNTDOWN_S;
      $('countdown').textContent = Math.max(0, Math.ceil(RECORD_S - elapsed));
      $('progress').style.width = Math.min(100, elapsed / RECORD_S * 100) + '%';
      if (elapsed >= 4 && elapsed - lastCheck >= 1) {
        lastCheck = elapsed;
        liveBeatCheck(chunks, sr);
      }
      if (elapsed >= RECORD_S) {
        stop(true);
        resolve({ data: concat(chunks), sr });
      }
    }, 100);

    function stop(done = false) {
      clearInterval(timer);
      cancelAnimationFrame(frame);
      [src, hp, lp, proc, tap].forEach(n => n.disconnect());
      if (done) cue(ctx, 'done');
      cancelRecording = null;
    }
    cancelRecording = () => {
      stop();
      reject(new Error('cancelled'));
    };
  });
}

function liveBeatCheck(chunks, sr) {
  const parts = [];
  for (let i = chunks.length - 1, n = 0; i >= 0 && n < 6 * sr; i--) {
    parts.unshift(chunks[i]);
    n += chunks[i].length;
  }
  const { clarity, bpm } = beatClarity(resample(concat(parts), sr, SR), SR, 0);
  const el = $('beat'), text = $('beat-text');
  el.querySelector('.pulse').style.animationDuration = bpm ? `${60 / bpm}s` : '';
  // 0.35 matches clear on the result screen
  if (clarity >= 0.35) [text.textContent, el.className] = ['Heartbeat found. Keep holding still.', 'beat good'];
  else if (clarity >= MIN_CLARITY) [text.textContent, el.className] = ['Heartbeat faint. Press a little firmer.', 'beat weak'];
  else [text.textContent, el.className] = ["Can't hear the heartbeat yet. Check the spot and press firmly.", 'beat none'];
}

function fitCanvas(canvas) {
  if (!canvas.isConnected) return;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
}

function drawWave(canvas, data, full = 0, color = '#b3261e', fit = true, x0 = 0) {
  if (fit) fitCanvas(canvas);
  const g = canvas.getContext('2d');
  if (!full) for (const v of data) full = Math.max(full, Math.abs(v));
  const w = canvas.width - x0,
    mid = canvas.height / 2,
    y = v => mid - Math.max(-1, Math.min(1, v / (full || 1e-9))) * mid;
  g.strokeStyle = color;
  g.lineWidth = 1;
  g.beginPath();
  for (let px = 0; px < w; px++) {
    let lo = 0, hi = 0;
    for (let i = Math.floor(px * data.length / w), end = Math.floor((px + 1) * data.length / w); i < end; i++) {
      lo = Math.min(lo, data[i]);
      hi = Math.max(hi, data[i]);
    }
    g.moveTo(x0 + px + 0.5, y(hi));
    g.lineTo(x0 + px + 0.5, y(lo) + 1);
  }
  g.stroke();
}

function resultImage() {
  const W = 1080,
    P = 72,
    over = state.results.map(r => r.prob >= cfg.threshold),
    heard = over.some(Boolean);
  const c = document.createElement('canvas'), g = c.getContext('2d');
  const font = (weight, px) => `${weight} ${px}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
  const text = (s, x, y, weight, px, color, align = 'left') => {
    g.font = font(weight, px);
    g.fillStyle = color;
    g.textAlign = align;
    g.fillText(s, x, y);
  };
  const lines = (s, width, px) => {
    g.font = font(400, px);
    const out = [''];
    for (const word of s.split(' ')) {
      const next = out[out.length - 1] ? `${out[out.length - 1]} ${word}` : word;
      if (out[out.length - 1] && g.measureText(next).width > width) out.push(word);
      else out[out.length - 1] = next;
    }
    return out;
  };
  const where = lines($('result-where').textContent, W - 2 * P - 88, 32);
  const footer = lines(`${state.note ? state.note + ' ' : ''}Recorded at three spots on the chest. A murmur score of 50 or more at any ` +
    'spot means we advise getting an echocardiogram. This is a screening result from an app, not a diagnosis or medical advice.',
    W - 2 * P, 30);
  const boxH = 130 + 42 * where.length;
  c.width = W;
  c.height = P + 90 + boxH + 60 + 300 * state.results.length + 42 * footer.length + P;
  g.fillStyle = '#fff';
  g.fillRect(0, 0, W, c.height);
  text('Murmur Check', P, P + 36, 700, 38, '#b3261e');
  text(new Date().toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }), W - P, P + 36, 400, 32, '#6b6461', 'right');

  let y = P + 90;
  g.fillStyle = heard ? '#c2410c' : '#3e5466';
  g.beginPath();
  if (g.roundRect) g.roundRect(P, y, W - 2 * P, boxH, 24);
  else g.rect(P, y, W - 2 * P, boxH);  // older browsers without roundRect
  g.fill();
  text($('result-title').textContent, P + 44, y + 88, 700, 48, '#fff');
  where.forEach((line, k) => text(line, P + 44, y + 146 + 42 * k, 400, 32, '#fff'));

  y += boxH + 60;
  state.results.forEach((r, i) => {
    text(`${i + 1}. ${SPOTS[i].title}`, P, y + 36, 700, 38, '#1d1b1a');
    text(`${over[i] ? 'Murmur' : 'No murmur'}, score ${murmurScore(r.prob)}`, W - P, y + 36, 700, 34, over[i] ? '#c2410c' : '#3e5466', 'right');
    const strip = document.createElement('canvas');
    strip.width = W - 2 * P;
    strip.height = 120;
    const sg = strip.getContext('2d');
    sg.fillStyle = '#fbf8f6';
    sg.fillRect(0, 0, strip.width, strip.height);
    drawWave(strip, r.wave, 1.3 * r.level);
    g.drawImage(strip, P, y + 62);
    text(facts(r), P, y + 232, 400, 30, '#6b6461');
    y += 300;
  });
  footer.forEach((line, k) => text(line, P, y + 30 + 42 * k, 400, 30, '#6b6461'));
  return c;
}

async function resultFile() {
  const blob = await new Promise(r => resultImage().toBlob(r, 'image/png'));
  const d = new Date(), pad = n => String(n).padStart(2, '0');
  const name = `murmur-check-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.png`;
  return new File([blob], name, { type: 'image/png' });
}

function shareResult() {
  const file = state.share;
  if (!file) return;
  if (navigator.canShare?.({ files: [file] })) return navigator.share({ files: [file] }).catch(() => {});
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(file), download: file.name });
  a.click();
}

const ICON = {
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>',
};
let player = null;

function playSpot(i, at) {
  if (player?.i !== i) {
    stopPlayer();
    const r = state.results[i],
      audio = new Audio(r.url ??= wavUrl(r.wave, 0.5 / r.level));
    player = { i, audio };
    audio.onplay = () => setPlayIcon(i, 'pause');
    audio.onpause = () => setPlayIcon(i, 'play');
    audio.onended = stopPlayer;
    $(`head-${i}`).classList.add('on');
    movePlayhead();
  } else if (at === undefined && !player.audio.paused) {
    return player.audio.pause();
  }
  const { audio } = player;
  if (at !== undefined) {
    // can't seek until it knows the length
    const seek = () => { audio.currentTime = at * state.results[i].wave.length / SR; };
    if (audio.readyState >= 1) seek();
    else audio.addEventListener('loadedmetadata', seek, { once: true });
  }
  audio.play().catch(() => {});
}

function movePlayhead() {
  const { i, audio } = player;
  drawZoom(i, audio.currentTime);
  player.frame = requestAnimationFrame(movePlayhead);
}

// remove handlers first so the icon doesn't flip twice
function stopPlayer() {
  if (!player) return;
  const { i, audio, frame } = player;
  player = null;
  cancelAnimationFrame(frame);
  audio.onplay = audio.onpause = null;
  audio.pause();
  setPlayIcon(i, 'play');
  $(`head-${i}`)?.classList.remove('on');
}

function setPlayIcon(i, icon) {
  const b = $(`play-${i}`);
  if (!b) return;
  b.innerHTML = ICON[icon];
  b.setAttribute('aria-label', icon === 'play' ? 'Play' : 'Pause');
}

// <audio> still plays with the iphone silent switch on
function wavUrl(x, gain) {
  const up = 2,
    sr = SR * up,
    n = x.length * up,
    v = new DataView(new ArrayBuffer(44 + 2 * n));
  const text = (o, str) => [...str].forEach((c, k) => v.setUint8(o + k, c.charCodeAt(0)));
  // 44-byte wav header, mono 16-bit
  text(0, 'RIFF');
  v.setUint32(4, 36 + 2 * n, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, 2 * sr, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  text(36, 'data');
  v.setUint32(40, 2 * n, true);
  // tanh makes it louder without clipping bumps
  for (let m = 0; m < n; m++) {
    const i = Math.floor(m / up),
      a = x[i],
      b = x[Math.min(i + 1, x.length - 1)];
    v.setInt16(44 + 2 * m, 32767 * Math.tanh((a + (b - a) * (m % up) / up) * gain), true);
  }
  return URL.createObjectURL(new Blob([v.buffer], { type: 'audio/wav' }));
}

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

// wav is chunks with a 4-letter name and size, odd sizes get a pad byte
async function loadWav(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const v = new DataView(await res.arrayBuffer());
  let off = 12, sr = 0, data = null;
  while (off + 8 <= v.byteLength) {
    const id = String.fromCharCode(...[0, 1, 2, 3].map(i => v.getUint8(off + i))),
      size = v.getUint32(off + 4, true);
    if (id === 'fmt ') sr = v.getUint32(off + 12, true);
    if (id === 'data') data = Float32Array.from({ length: size / 2 }, (_, i) => v.getInt16(off + 8 + 2 * i, true) / 32768);
    off += 8 + size + (size & 1);
  }
  return { data, sr };
}

// back from the reload to ask for the mic
if (location.hash === '#ask-mic') {
  history.replaceState(null, '', location.pathname + location.search);
  show('before');
  beginCheck();
}

loadModel().catch(() => {});
