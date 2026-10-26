// run: npm install && npm test (set CHROME_PATH if chrome isn't found)
const http = require('http');
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const puppeteer = require('puppeteer-core');

const APP = path.join(__dirname, '..', 'app');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.wav': 'audio/wav', '.onnx': 'application/octet-stream', '.svg': 'image/svg+xml', '.png': 'image/png' };
const CHROME = process.env.CHROME_PATH || [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
].find(p => fs.existsSync(p));

function serve() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    const file = path.join(APP, url.endsWith('/') ? url + 'index.html' : url);
    if (!file.startsWith(APP) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function runSample(page, kind) {
  await page.click('[data-go="samples"]');
  await page.waitForSelector('#samples.active');
  await page.click(`[data-sample="${kind}"]`);
  await page.waitForSelector('#result.active', { timeout: 60000 });
  return page.evaluate(() => ({
    title: document.getElementById('result-title').textContent,
    line: cfg.threshold,
    scores: state.results.map(r => r.prob),
  }));
}

const tests = {
  'the page loads with no errors and the model and its line load': async page => {
    await page.evaluate(() => loadModel());
    const { line, expected } = await page.evaluate(async () => ({
      line: cfg.threshold, expected: (await (await fetch('model/config.json')).json()).threshold }));
    assert.ok(line > 0 && line < 1, `line ${line}`);
    assert.strictEqual(line, expected);
  },

  'the child with a murmur gets "get an echocardiogram"': async page => {
    const r = await runSample(page, 'murmur');
    assert.strictEqual(r.title, 'Murmur heard: get an echocardiogram');
    assert.strictEqual(r.scores.length, 3, 'all three spots scored');
    assert.ok(r.scores.some(p => p >= r.line), `scores ${r.scores} vs line ${r.line}`);
    return `scores ${r.scores.map(p => p.toFixed(2)).join(', ')} (line ${r.line.toFixed(2)})`;
  },

  'the child without a murmur gets "no murmur heard"': async page => {
    const r = await runSample(page, 'normal');
    assert.strictEqual(r.title, 'No murmur heard in these recordings');
    assert.strictEqual(r.scores.length, 3, 'all three spots scored');
    assert.ok(r.scores.every(p => p < r.line), `scores ${r.scores} vs line ${r.line}`);
    return `scores ${r.scores.map(p => p.toFixed(2)).join(', ')} (line ${r.line.toFixed(2)})`;
  },

  'a recording with no heartbeat is refused instead of scored': async page => {
    const r = await page.evaluate(async () => {
      // seeded so it's the same every run
      let seed = 1;
      const noise = Float32Array.from({ length: 15 * 4000 }, () => ((seed = seed * 16807 % 2147483647) / 2147483647 - 0.5) * 0.1);
      return analyze(noise, 4000, { phone: true });
    });
    assert.strictEqual(r.prob, null, `noise was scored ${r.prob}`);
  },
};

(async () => {
  assert.ok(CHROME, 'Chrome not found: set CHROME_PATH');
  const server = await serve();
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--disable-audio-output'] });
  let failed = 0;
  try {
    for (const [name, test] of Object.entries(tests)) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('dialog', d => { errors.push(`alert: ${d.message()}`); d.dismiss(); });
      await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: 'networkidle0' });
      await page.waitForFunction(() => typeof analyze === 'function' && typeof ort !== 'undefined', { timeout: 60000 });
      try {
        const note = await test(page);
        assert.deepStrictEqual(errors, [], 'errors on the page');
        console.log(`ok    ${name}${note ? ` — ${note}` : ''}`);
      } catch (e) {
        failed++;
        console.log(`FAIL  ${name}\n      ${e.message}`);
      }
      await page.close();
    }
  } finally {
    await browser.close();
    server.close();
  }
  console.log(failed ? `\n${failed} failed` : '\nall passed');
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
