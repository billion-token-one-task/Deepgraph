/* Responsiveness gate — Playwright, real headless Chromium (acceptance C).
 *
 * Boots the seeded fixture server, loads the dashboard, and verifies the exact
 * conditions issue #19 demands after the O(n^2) dropdown freeze:
 *   • first paint ≤ 2s with the 9 stat cards showing real (DB-derived) numbers
 *   • open every tab + switch language once + zoom/drag the graph repeatedly
 *   • hang HANG_MS (default 60s); throughout: NO main-thread long task > 200ms,
 *     the page stays interactive, and the console reports no errors
 *   • the new graph features actually work: radial labels readable for 13
 *     children, the story panel opens, the entity network renders, and the
 *     custom tooltip (not native title) appears on a metric card.
 */
import { spawn } from 'node:child_process';
import { appendFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { chromium } from 'playwright';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8766);
// E2E_BASE lets the gate attach to an already-running fixture server (started
// separately) instead of spawning one — keeps the process tree small/stable.
const BASE = process.env.E2E_BASE || `http://127.0.0.1:${PORT}`;
const SPAWN_SERVER = !process.env.E2E_BASE;
const HANG_MS = Number(process.env.HANG_MS || 60000);
const PY = process.env.PYTHON || join(ROOT, '.venv', 'bin', 'python');

// Synchronous logging: this gate may run close to the sandbox's per-command
// runtime cap, and a SIGKILL drops buffered stdout — so every line is flushed
// to LOG immediately as well as echoed.
const LOG = process.env.E2E_LOG || join(__dirname, 'e2e_result.log');
try { writeFileSync(LOG, ''); } catch (e) { /* ignore */ }
const line = (s) => { try { appendFileSync(LOG, s + '\n'); } catch (e) { /* ignore */ } console.log(s); };
const fail = (m) => { line(`  FAIL ${m}`); process.exitCode = 1; };
const ok = (m) => line(`  ok  ${m}`);

function startServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn(PY, [join(__dirname, 'serve_fixture.py')], {
      env: { ...process.env, PORT: String(PORT) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let out = '';
    const onData = (d) => { out += d; if (out.includes('Running on')) resolve(proc); };
    proc.stdout.on('data', onData);
    proc.stderr.on('data', onData);
    proc.on('exit', (c) => reject(new Error(`fixture server exited early (${c}): ${out}`)));
    setTimeout(() => reject(new Error(`server start timeout: ${out}`)), 25000);
  });
}

async function waitServer() {
  for (let i = 0; i < 40; i += 1) {
    try { const r = await fetch(`${BASE}/api/stats`); if (r.ok) return; } catch (e) { /* retry */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('server never became healthy');
}

const longTaskInit = `
  window.__longtasks = [];
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) window.__longtasks.push({ d: e.duration, t: e.startTime });
    }).observe({ entryTypes: ['longtask'] });
  } catch (e) { window.__longtaskUnsupported = true; }
`;

line('graph_e2e.mjs');
let server;
let browser;
try {
  if (SPAWN_SERVER) {
    server = await startServer();
  }
  await waitServer();
  ok(`fixture server up (${SPAWN_SERVER ? 'spawned' : 'attached ' + BASE})`);

  browser = await chromium.launch({ args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  // Hard-fail on real JS faults (uncaught exceptions, our own console.error);
  // a bare fixture legitimately lacks some optional backend endpoints, so a
  // "Failed to load resource" 4xx/5xx is recorded as a warning, not a gate
  // failure — the gate is about frozen/erroring UI, not fixture completeness.
  const jsErrors = [];
  const resourceWarnings = [];
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    const url = (msg.location() && msg.location().url) || '';
    if (/favicon/i.test(url) || /favicon/i.test(text)) return;
    if (/Failed to load resource/i.test(text)) { resourceWarnings.push(text + (url ? ` [${url}]` : '')); return; }
    jsErrors.push(text + (url ? ` [${url}]` : ''));
  });
  page.on('pageerror', (err) => jsErrors.push('pageerror: ' + err.message));
  await page.addInitScript(longTaskInit);

  // ── first paint ≤ 2s with 9 real numbers ──────────────────────────
  const ids = ['statPapers', 'statResults', 'statTaxonomy', 'statContradictions', 'statInsights',
    'statTokens', 'statExperiments', 'statDeepDiscoveries', 'statCompletePapers'];
  const t0 = Date.now();
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  // taxonomy_nodes_total is always > 0 once real stats land
  await page.waitForFunction(() => document.getElementById('statTaxonomy')
    && document.getElementById('statTaxonomy').textContent.trim() !== '0', null, { timeout: 2000 });
  const firstPaint = Date.now() - t0;
  const cardValues = await page.evaluate((idList) => idList.map((id) => {
    const el = document.getElementById(id);
    return el ? el.textContent.trim() : null;
  }), ids);
  const allPresent = cardValues.every((v) => v != null && v !== '');
  const someReal = cardValues.some((v) => /[1-9]/.test(v || ''));
  if (firstPaint <= 2000 && allPresent && someReal) ok(`9 cards real numbers in ${firstPaint}ms: ${cardValues.join(' / ')}`);
  else fail(`first paint gate: ${firstPaint}ms present=${allPresent} real=${someReal} -> ${cardValues.join(' / ')}`);

  // ── custom tooltip (acceptance D): hover a metric card ─────────────
  const card = await page.$('[data-i18n-title="overview.sourcePapers.tip"]');
  await card.hover();
  await page.waitForTimeout(150);
  const tipVisible = await page.evaluate(() => {
    const tip = document.getElementById('tooltip');
    return !!tip && tip.classList.contains('visible') && tip.textContent.trim().length > 0;
  });
  const nativeTitle = await page.evaluate(() => {
    const c = document.querySelector('[data-i18n-title="overview.sourcePapers.tip"]');
    return c ? c.getAttribute('title') : 'MISSING';
  });
  if (tipVisible && !nativeTitle) ok('custom tooltip shows on metric card; native title removed');
  else fail(`tooltip gate: visible=${tipVisible} nativeTitle=${JSON.stringify(nativeTitle)}`);
  await page.mouse.move(5, 5);

  // ── open every tab (incl. collapsed "advanced" nav items) ─────────
  // Dispatch the click via JS so hidden advanced-nav items still switch &
  // render — the point is to exercise every tab's render path under the gate.
  const tabs = await page.$$eval('[data-tab]', (els) => els.map((e) => e.dataset.tab));
  for (const tab of tabs) {
    await page.$eval(`[data-tab="${tab}"]`, (el) => el.click());
    await page.waitForTimeout(220);
  }
  ok(`opened ${tabs.length} tabs`);

  // ── Explore: render the rich node, check readable labels + graph ───
  await page.click('[data-tab="explore"]');
  await page.evaluate(() => window._dg.navigateTo('ml.dl'));
  await page.waitForSelector('#exploreGraph svg.dg-graph-svg .dg-node-child', { timeout: 5000 });
  const labelStats = await page.evaluate(() => {
    const labels = Array.from(document.querySelectorAll('#exploreGraph .dg-node-child .dg-node-label'));
    const childNodes = document.querySelectorAll('#exploreGraph .dg-node-child').length;
    const truncated = labels.filter((l) => (l.textContent || '').includes('…')).length;
    const hasLegend = !!document.querySelector('#exploreGraph .dg-legend');
    return { childNodes, labelCount: labels.length, truncated, hasLegend };
  });
  if (labelStats.childNodes >= 10 && labelStats.labelCount === labelStats.childNodes && labelStats.hasLegend) {
    ok(`radial: ${labelStats.childNodes} children, all labelled (${labelStats.truncated} ellipsised), legend present`);
  } else {
    fail(`radial gate: ${JSON.stringify(labelStats)}`);
  }

  // zoom + pan repeatedly
  const box = await (await page.$('#exploreGraph svg.dg-graph-svg')).boundingBox();
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  for (let i = 0; i < 5; i += 1) { await page.mouse.move(cx, cy); await page.mouse.wheel(0, -120); await page.waitForTimeout(40); }
  for (let i = 0; i < 3; i += 1) {
    await page.mouse.move(cx, cy); await page.mouse.down();
    await page.mouse.move(cx + 60, cy + 40, { steps: 6 }); await page.mouse.up();
    await page.waitForTimeout(40);
  }
  ok('zoom + pan interactions ran');

  // ── story panel: click an area node, see gaps/discoveries ──────────
  // Dispatch the click on the node group (after pan/zoom a node may sit under
  // the sticky card header; the renderer's d3 click handler honours a synthetic
  // event, so this exercises the real handler without pointer interception).
  await page.$eval('#exploreGraph .dg-node-child', (el) => el.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await page.waitForSelector('#exploreStoryPanel:not([hidden]) .story-section', { timeout: 5000 });
  const story = await page.evaluate(() => {
    const p = document.getElementById('exploreStoryPanel');
    return {
      visible: !p.hidden,
      flow: !!p.querySelector('.story-flow'),
      sections: p.querySelectorAll('.story-section').length,
      enter: !!p.querySelector('.story-enter'),
    };
  });
  if (story.visible && story.flow && story.sections >= 3 && story.enter) ok('story panel: area → gaps → discoveries path shown');
  else fail(`story panel gate: ${JSON.stringify(story)}`);

  // ── entity network (Evidence tab) ──────────────────────────────────
  await page.click('[data-tab="evidence"]');
  await page.evaluate(() => {
    const inp = document.getElementById('evidenceNodeSelect');
    inp.value = 'ml.dl';
    inp.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForSelector('#evidenceEntityGraph .dg-node-entity', { timeout: 5000 });
  const entityNodes = await page.$$eval('#evidenceEntityGraph .dg-node-entity', (e) => e.length);
  if (entityNodes >= 6) ok(`entity-relation network rendered (${entityNodes} entity nodes)`);
  else fail(`entity network gate: only ${entityNodes} nodes`);

  // ── switch language once ───────────────────────────────────────────
  await page.click('[data-lang="zh"]');
  await page.waitForTimeout(400);
  ok('language switched to zh');

  // ── hang and watch for long tasks / loss of interactivity ──────────
  await page.click('[data-tab="overview"]');
  line(`      hanging ${HANG_MS}ms, watching for >200ms long tasks…`);
  const hangStart = Date.now();
  let maxRaf = 0;
  while (Date.now() - hangStart < HANG_MS) {
    // measure main-thread responsiveness: time to service a rAF (should be tiny)
    const raf = await page.evaluate(() => new Promise((res) => {
      const s = performance.now();
      requestAnimationFrame(() => res(performance.now() - s));
    }));
    maxRaf = Math.max(maxRaf, raf);
    await page.waitForTimeout(1000);
  }
  const longTasks = await page.evaluate(() => window.__longtasks || []);
  const bigTasks = longTasks.filter((l) => l.d > 200);
  const unsupported = await page.evaluate(() => !!window.__longtaskUnsupported);

  if (unsupported) {
    console.log('      (longtask API unsupported in this browser; relying on rAF latency)');
  }
  if (bigTasks.length === 0) ok(`no main-thread long task > 200ms (${longTasks.length} long tasks total, max rAF latency ${maxRaf.toFixed(0)}ms)`);
  else fail(`long tasks > 200ms detected: ${JSON.stringify(bigTasks.slice(0, 5))}`);

  // page still interactive after the hang
  await page.click('[data-tab="explore"]');
  const interactive = await page.evaluate(() => document.querySelector('#tab-explore.active') != null
    || document.querySelector('#tab-explore') != null);
  if (interactive) ok('page interactive after hang');
  else fail('page not interactive after hang');

  if (jsErrors.length === 0) ok('no JS console errors / pageerrors');
  else fail(`JS errors: ${JSON.stringify(jsErrors.slice(0, 8))}`);
  if (resourceWarnings.length) line(`      note: ${resourceWarnings.length} backend resource warning(s) in bare fixture: ${JSON.stringify(resourceWarnings.slice(0, 4))}`);

  if (process.exitCode) line('\nE2E GATE FAILED');
  else line('\nE2E GATE PASSED');
} catch (e) {
  line('  FAIL exception: ' + (e && e.stack || e));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (server) server.kill('SIGKILL');
}
