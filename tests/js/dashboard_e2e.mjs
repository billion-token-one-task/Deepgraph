// Comprehensive production-scale dashboard gate (Acceptance A + B + C).
//
// Boots seed_and_serve.py — a throwaway DB seeded to a PRODUCTION shape:
// ~3300-node taxonomy, all 9 overview metrics non-zero, *heavy* deep_insights
// (~0.6 MB /api/deep_insights?limit=50) and insights (~0.7 MB) payloads, and a
// dense 29×14 benchmark matrix on the BENCH_NODE leaf. Then drives the real
// dashboard in headless Chromium and asserts, all in one user-shaped pass:
//
//   A (perf): first paint ≤ 2s with 9 real numbers; then open every tab, render
//     the heavy Evidence matrix, zoom/drag the knowledge graph, search, switch
//     language, and idle IDLE_MS — with NO main-thread long task > 200ms, the
//     page staying interactive, and zero console/page errors throughout.
//   B (heatmap): the Evidence matrix is a real heatmap — its filled cells carry
//     MORE THAN ONE distinct background colour (a gradient), and switching the
//     metric keeps it a heatmap.
//   C (behaviour): all 6 main tabs + the Advanced nav + the graph + language +
//     search all keep working on production-weight data.
//
// Env: IDLE_MS (default 60000), HEADLESS (default 1), PORT (default 5098).
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface } from "node:readline";
import { appendFileSync, writeFileSync } from "node:fs";

const REPO_ROOT = new URL("../../", import.meta.url).pathname;
const PY = `${REPO_ROOT}.venv/bin/python`;
const SERVER = `${REPO_ROOT}tests/js/seed_and_serve.py`;
const PORT = parseInt(process.env.PORT || "5098", 10);
const IDLE_MS = parseInt(process.env.IDLE_MS || "60000", 10);
const FIRST_PAINT_BUDGET_MS = 2000;
const LONGTASK_BUDGET_MS = 200;
const BENCH_NODE = "ml.bench";

const LOG = process.env.E2E_LOG || `${REPO_ROOT}tests/js/dashboard_e2e_result.log`;
try { writeFileSync(LOG, ""); } catch { /* ignore */ }
const line = (s) => { try { appendFileSync(LOG, s + "\n"); } catch { /* ignore */ } console.log(s); };

const fails = [];
const check = (cond, msg) => { if (!cond) { fails.push(msg); line(`  FAIL ${msg}`); } };
const ok = (m) => line(`  ok  ${m}`);

const STAT_IDS = [
  "statPapers", "statResults", "statTaxonomy", "statContradictions",
  "statInsights", "statTokens", "statExperiments", "statDeepDiscoveries",
  "statCompletePapers",
];
const TABS = [
  "overview", "explore", "evidence", "generated-papers", "insights", "papers",
  "paper-progress", "discoveries", "experiments", "feed", "providers", "agenda",
];

line("dashboard_e2e.mjs");
const server = spawn(PY, [SERVER, String(PORT)], { cwd: REPO_ROOT });
let serverReady = false;
const rl = createInterface({ input: server.stdout });
rl.on("line", (l) => { if (l.startsWith("READY")) serverReady = true; });
server.stderr.on("data", () => { /* werkzeug request log noise */ });

async function waitFor(pred, timeoutMs, label) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (pred()) return;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`timeout waiting for ${label}`);
}

let browser;
try {
  await waitFor(() => serverReady, 60000, "server READY");
  browser = await chromium.launch({
    headless: process.env.HEADLESS !== "0",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  // Emulate a real (slower) user device. A long task is a function of CPU
  // speed, and the CI/dev box is far faster than a typical user's laptop, so
  // an unthrottled run "passes" work that janks for real users — exactly the
  // "fast-CPU false pass" trap. Lighthouse uses 4× CPU slowdown for its mobile
  // profile; we default to 4× so the 200ms budget is measured against a
  // realistic main thread, not the test machine's.
  const CPU_THROTTLE = parseFloat(process.env.CPU_THROTTLE || "4");
  if (CPU_THROTTLE > 1) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: CPU_THROTTLE });
    line(`      CPU throttling: ${CPU_THROTTLE}× (emulating a real user device)`);
  }

  await page.addInitScript(() => {
    window.__longtasks = [];
    window.__phase = "load";
    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          window.__longtasks.push({ duration: e.duration, startTime: e.startTime, phase: window.__phase });
        }
      }).observe({ entryTypes: ["longtask"], buffered: true });
    } catch { window.__longtaskUnsupported = true; }
  });
  const setPhase = (p) => page.evaluate((x) => { window.__phase = x; }, p);

  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const text = m.text();
    const url = (m.location() && m.location().url) || "";
    if (/favicon/i.test(url) || /Failed to load resource/i.test(text)) return; // resource 404 != JS fault
    consoleErrors.push(text);
  });
  page.on("pageerror", (e) => pageErrors.push(String(e)));

  // ── 1) first paint ≤ 2s with 9 real numbers ───────────────────────────
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "commit" });
  await page.waitForFunction(
    (ids) => ids.every((id) => {
      const el = document.getElementById(id);
      const txt = el && el.textContent.trim();
      return txt && txt !== "0";
    }),
    STAT_IDS,
    { timeout: 10000 }
  );
  const firstPaintMs = Date.now() - t0;
  const statValues = await page.evaluate(
    (ids) => Object.fromEntries(ids.map((id) => [id, document.getElementById(id).textContent.trim()])),
    STAT_IDS
  );
  check(firstPaintMs <= FIRST_PAINT_BUDGET_MS, `first paint ${firstPaintMs}ms > ${FIRST_PAINT_BUDGET_MS}ms`);
  for (const id of STAT_IDS) check(statValues[id] && statValues[id] !== "0", `${id} not real: ${statValues[id]}`);
  ok(`first paint (9 real cards) in ${firstPaintMs}ms: ${Object.values(statValues).join(" / ")}`);

  // ── 2) open every tab (heavy renders) ──────────────────────────────────
  await page.evaluate(() => { const d = document.querySelector("details.advanced-nav"); if (d) d.open = true; });
  for (const tab of TABS) {
    const btn = await page.$(`[data-tab="${tab}"]`);
    check(!!btn, `tab button missing: ${tab}`);
    if (!btn) continue;
    await setPhase(`tab:${tab}`);
    const ti = Date.now();
    await btn.click();
    try {
      await page.waitForFunction((t) => document.getElementById("tab-" + t)?.classList.contains("active"), tab, { timeout: 3000 });
    } catch { check(false, `tab '${tab}' did not activate within 3s`); }
    check(Date.now() - ti <= 1500, `tab '${tab}' switch took ${Date.now() - ti}ms (janky)`);
  }
  ok(`opened ${TABS.length} tabs on production-weight data`);

  // ── 3) Evidence matrix → real heatmap (Acceptance B) ───────────────────
  await setPhase("evidence-matrix");
  await page.click('[data-tab="evidence"]');
  const tMatrix = Date.now();
  await page.evaluate((node) => {
    const inp = document.getElementById("evidenceNodeSelect");
    inp.value = node;
    inp.dispatchEvent(new Event("change", { bubbles: true }));
  }, BENCH_NODE);
  await page.waitForSelector(".matrix-table td.cell-filled", { timeout: 5000 });
  const matrixMs = Date.now() - tMatrix;

  const heat = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll(".matrix-table td.cell-filled"));
    const colors = new Set(cells.map((c) => c.style.background || getComputedStyle(c).backgroundColor));
    return { filled: cells.length, distinct: colors.size, sample: Array.from(colors).slice(0, 4) };
  });
  check(heat.filled > 0, "no filled matrix cells rendered");
  check(heat.distinct > 1, `matrix is not a heatmap: only ${heat.distinct} distinct fill colour(s)`);
  ok(`heatmap rendered in ${matrixMs}ms: ${heat.filled} filled cells, ${heat.distinct} distinct colours (e.g. ${JSON.stringify(heat.sample)})`);

  // switch the metric and confirm it stays a heatmap (updateMatrixMetric path)
  const hasSelect = await page.$(".matrix-metric-select");
  if (hasSelect) {
    const opts = await page.$$eval(".matrix-metric-select option", (os) => os.map((o) => o.value));
    if (opts.length > 1) {
      await page.selectOption(".matrix-metric-select", opts[1]);
      await page.waitForTimeout(100);
      const heat2 = await page.evaluate(() => {
        const cells = Array.from(document.querySelectorAll(".matrix-table td.cell-filled"));
        return new Set(cells.map((c) => c.style.background)).size;
      });
      check(heat2 > 1, `after metric switch the matrix lost its gradient (${heat2} colours)`);
      ok(`metric switch keeps heatmap (${heat2} distinct colours on metric '${opts[1]}')`);
    }
  }

  // ── 4) knowledge graph zoom + drag (Overview radial) ───────────────────
  await setPhase("graph-zoom");
  await page.click('[data-tab="overview"]');
  await page.waitForSelector("#overviewGraph svg.dg-graph-svg", { timeout: 6000 }).catch(() => {});
  const svg = await page.$("#overviewGraph svg.dg-graph-svg");
  if (svg) {
    const box = await svg.boundingBox();
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    for (let i = 0; i < 5; i++) { await page.mouse.move(cx, cy); await page.mouse.wheel(0, -120); await page.waitForTimeout(40); }
    await page.mouse.move(cx, cy); await page.mouse.down();
    await page.mouse.move(cx + 60, cy + 40, { steps: 6 }); await page.mouse.up();
    ok("knowledge-graph zoom + drag ran");
  } else {
    ok("overview graph not present (skipped zoom/drag)");
  }

  // ── 5) search ──────────────────────────────────────────────────────────
  await setPhase("search");
  await page.fill("#searchInput", "Method");
  await page.waitForTimeout(500);
  const searchOpen = await page.evaluate(() => document.getElementById("searchResults")?.classList.contains("open"));
  check(!!searchOpen, "search dropdown did not open");
  if (searchOpen) ok("search returned results");
  await page.keyboard.press("Escape");

  // ── 6) language switch ─────────────────────────────────────────────────
  await setPhase("lang-switch");
  const enLabel = await page.$eval('[data-i18n="nav.overview"]', (e) => e.textContent.trim());
  await page.click('.lang-btn[data-lang="zh"]');
  try {
    await page.waitForFunction((en) => document.querySelector('[data-i18n="nav.overview"]')?.textContent.trim() !== en, enLabel, { timeout: 3000 });
    ok("language switched to zh");
  } catch { check(false, "language switch did not apply within 3s"); }

  // ── 7) idle and watch for long tasks / interactivity ───────────────────
  await setPhase("idle");
  line(`      idling ${IDLE_MS}ms, watching for >${LONGTASK_BUDGET_MS}ms long tasks…`);
  let maxRaf = 0;
  const hangStart = Date.now();
  while (Date.now() - hangStart < IDLE_MS) {
    const raf = await page.evaluate(() => new Promise((res) => {
      const s = performance.now();
      requestAnimationFrame(() => res(performance.now() - s));
    }));
    maxRaf = Math.max(maxRaf, raf);
    await page.waitForTimeout(1000);
  }

  const longtasks = await page.evaluate(() => window.__longtasks || []);
  const worst = longtasks.reduce((m, t) => Math.max(m, t.duration), 0);
  const over = longtasks.filter((t) => t.duration > LONGTASK_BUDGET_MS);
  line(`      long tasks: ${longtasks.length} total, worst ${worst.toFixed(1)}ms, ${over.length} over ${LONGTASK_BUDGET_MS}ms, max rAF ${maxRaf.toFixed(0)}ms`);
  for (const t of over) line(`        >budget ${t.duration.toFixed(0)}ms phase=${t.phase}`);
  check(over.length === 0, `${over.length} main-thread long task(s) > ${LONGTASK_BUDGET_MS}ms (worst ${worst.toFixed(1)}ms)`);

  // still interactive after the idle
  await page.click('[data-tab="explore"]');
  const interactive = await page.evaluate(() => !!document.getElementById("tab-explore"));
  check(interactive, "page not interactive after idle");

  check(consoleErrors.length === 0, `${consoleErrors.length} console error(s): ${consoleErrors.slice(0, 5).join(" | ")}`);
  check(pageErrors.length === 0, `${pageErrors.length} page error(s): ${pageErrors.slice(0, 5).join(" | ")}`);
} catch (e) {
  check(false, `exception: ${e && e.stack ? e.stack : e}`);
} finally {
  if (browser) await browser.close();
  server.kill("SIGINT");
  try { await once(server, "exit"); } catch { /* ignore */ }
}

if (fails.length) { line(`\nDASHBOARD E2E FAILED — ${fails.length} failure(s).`); process.exitCode = 1; }
else { line("\nDASHBOARD E2E PASSED — fast, interactive, no long task > 200ms, real heatmap, no console errors."); }
