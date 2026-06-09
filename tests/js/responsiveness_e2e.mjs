// Acceptance A (primary) — real-browser end-to-end responsiveness test.
//
// Drives the actual dashboard in headless Chromium against a Flask server
// seeded with a production-shaped dataset (~3300 taxonomy nodes; all 9 metrics
// non-zero). Verifies:
//   1. First paint: all 9 overview stat cards show real (non-zero) numbers
//      within FIRST_PAINT_BUDGET_MS.
//   2. After load: idle for IDLE_MS, then click every tab and switch language
//      once — with NO main-thread long task > LONGTASK_BUDGET_MS, the page
//      staying interactive, and no console error / page error throughout.
//
// The old O(n^2) taxonomy-dropdown build froze the main thread for seconds
// during the idle prefetch; this test fails on that code and passes on the fix.
//
// Env: IDLE_MS (default 60000), HEADLESS (default 1).
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface } from "node:readline";
import { appendFileSync } from "node:fs";

// Optional: mirror report lines to a file (set TRACE_FILE) so results survive
// even if a CI wrapper truncates stdout for a long-running browser+server run.
const TRACE = process.env.TRACE_FILE;
function trace(m) {
  if (TRACE) { try { appendFileSync(TRACE, `${m}\n`); } catch { /* best effort */ } }
}

const REPO_ROOT = new URL("../../", import.meta.url).pathname;
const PY = `${REPO_ROOT}.venv/bin/python`;
const SERVER = `${REPO_ROOT}tests/js/seed_and_serve.py`;
const PORT = 5099;

const IDLE_MS = parseInt(process.env.IDLE_MS || "60000", 10);
const FIRST_PAINT_BUDGET_MS = 2000;
const LONGTASK_BUDGET_MS = 200;

const STAT_IDS = [
  "statPapers", "statResults", "statTaxonomy", "statContradictions",
  "statInsights", "statTokens", "statExperiments", "statDeepDiscoveries",
  "statCompletePapers",
];
// Every tab in the dashboard (main nav + collapsed advanced nav).
const TABS = [
  "overview", "explore", "evidence", "generated-papers", "insights", "papers",
  "paper-progress", "discoveries", "experiments", "feed", "providers", "agenda",
];

const fails = [];
function check(cond, msg) {
  if (!cond) { fails.push(msg); console.error(`FAIL: ${msg}`); trace("FAIL " + msg); }
}
// Mirror every report line to the trace file too: the sustained chromium+server
// run can have its shell killed before stdout is captured, but appendFileSync
// survives, so results are never lost.
function report(m) { console.log(m); trace("REPORT " + m); }

// ── start seeded server ────────────────────────────────────────────────
const server = spawn(PY, [SERVER, String(PORT)], { cwd: REPO_ROOT });
let serverReady = false;
const rl = createInterface({ input: server.stdout });
rl.on("line", (l) => { if (l.startsWith("READY")) serverReady = true; console.log(`[server] ${l}`); });
server.stderr.on("data", (d) => process.stderr.write(`[server:err] ${d}`));

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
  const page = await browser.newPage();

  // Install a long-task observer BEFORE any page script runs.
  await page.addInitScript(() => {
    window.__longtasks = [];
    window.__phase = "load";
    try {
      const obs = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          window.__longtasks.push({ duration: e.duration, startTime: e.startTime, name: e.name, phase: window.__phase });
        }
      });
      obs.observe({ entryTypes: ["longtask"], buffered: true });
    } catch (_) { /* longtask unsupported */ }
  });
  const setPhase = (p) => page.evaluate((x) => { window.__phase = x; }, p);

  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => pageErrors.push(String(e)));

  // ── 1) first paint ───────────────────────────────────────────────────
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
  report(`first paint (9 cards populated): ${firstPaintMs}ms  ${JSON.stringify(statValues)}`);
  check(firstPaintMs <= FIRST_PAINT_BUDGET_MS, `first paint ${firstPaintMs}ms > ${FIRST_PAINT_BUDGET_MS}ms`);
  for (const id of STAT_IDS) {
    check(statValues[id] && statValues[id] !== "0", `${id} not a real number: ${JSON.stringify(statValues[id])}`);
  }

  // ── 1b) tooltips (Acceptance B): every stat card has a real, visible
  //        hover explanation rendered into its title attribute. ─────────────
  const tooltips = await page.$$eval(".stat-card[data-i18n-title]", (cards) =>
    cards.map((c) => ({ key: c.getAttribute("data-i18n-title"), title: (c.getAttribute("title") || "").trim() }))
  );
  check(tooltips.length === 9, `expected 9 stat-card tooltips, found ${tooltips.length}`);
  for (const t of tooltips) {
    check(t.title.length > 0, `stat card ${t.key} has no visible title tooltip`);
  }
  report(`stat-card tooltips rendered: ${tooltips.length}/9 (e.g. ${JSON.stringify(tooltips[0])})`);

  // ── 2a) idle (lets the idle prefetch build the 3300-node dropdown) ─────
  await setPhase("idle-prefetch");
  console.log(`idling ${IDLE_MS}ms (prefetch builds the taxonomy dropdown here)...`);
  await page.waitForTimeout(IDLE_MS);

  // Make sure the prefetch actually ran (dropdown filled) — otherwise we
  // would not be exercising the path that used to freeze.
  const optionCount = await page.$eval("#evidenceNodeOptions", (s) => s.options.length).catch(() => 0);
  report(`taxonomy dropdown options after idle: ${optionCount}`);
  check(optionCount >= 3000, `taxonomy dropdown not prefetched (only ${optionCount} options)`);

  // ── 2b) click every tab; assert each activates quickly (interactive) ───
  await page.evaluate(() => {
    const d = document.querySelector("details.advanced-nav");
    if (d) d.open = true; // reveal advanced tabs so they are clickable
  });
  for (const tab of TABS) {
    const sel = `[data-tab="${tab}"]`;
    const btn = await page.$(sel);
    check(!!btn, `tab button missing: ${tab}`);
    if (!btn) continue;
    await setPhase(`tab:${tab}`);
    const ti = Date.now();
    await btn.click();
    try {
      await page.waitForFunction(
        (t) => document.getElementById("tab-" + t)?.classList.contains("active"),
        tab,
        { timeout: 3000 }
      );
    } catch {
      check(false, `tab '${tab}' did not become active within 3s (unresponsive)`);
    }
    const dt = Date.now() - ti;
    trace(`tab '${tab}' switch took ${dt}ms`);
    check(dt <= 1500, `tab '${tab}' switch took ${dt}ms (>1500ms — janky)`);
  }

  // ── 2c) switch language once; assert it takes effect ───────────────────
  await setPhase("lang-switch");
  const navOverviewEn = await page.$eval('[data-i18n="nav.overview"]', (e) => e.textContent.trim());
  const tl = Date.now();
  await page.click('.lang-btn[data-lang="zh"]');
  try {
    await page.waitForFunction(
      (en) => document.querySelector('[data-i18n="nav.overview"]')?.textContent.trim() !== en,
      navOverviewEn,
      { timeout: 3000 }
    );
  } catch {
    check(false, "language switch did not update labels within 3s (unresponsive)");
  }
  console.log(`language switch applied in ${Date.now() - tl}ms`);

  // ── 3) verdicts: long tasks + console errors ───────────────────────────
  const longtasks = await page.evaluate(() => window.__longtasks || []);
  const worst = longtasks.reduce((m, t) => Math.max(m, t.duration), 0);
  const over = longtasks.filter((t) => t.duration > LONGTASK_BUDGET_MS);
  report(`long tasks: ${longtasks.length} total, worst ${worst.toFixed(1)}ms, ${over.length} over ${LONGTASK_BUDGET_MS}ms`);
  for (const t of over) report(`  >budget: ${t.duration.toFixed(0)}ms  phase=${t.phase}  start=${t.startTime.toFixed(0)}`);
  check(over.length === 0, `${over.length} main-thread long task(s) > ${LONGTASK_BUDGET_MS}ms (worst ${worst.toFixed(1)}ms)`);

  if (consoleErrors.length) report(`console errors:\n  ${consoleErrors.join("\n  ")}`);
  if (pageErrors.length) report(`page errors:\n  ${pageErrors.join("\n  ")}`);
  check(consoleErrors.length === 0, `${consoleErrors.length} console error(s)`);
  check(pageErrors.length === 0, `${pageErrors.length} page error(s)`);
} catch (e) {
  check(false, `exception: ${e && e.stack ? e.stack : e}`);
} finally {
  if (browser) await browser.close();
  server.kill("SIGINT");
  try { await once(server, "exit"); } catch { /* ignore */ }
}

if (fails.length) {
  report(`VERDICT: ${fails.length} failure(s).`);
  process.exitCode = 1;
} else {
  report("VERDICT: PASS — first paint fast, page stayed interactive, no long task > 200ms, no console errors.");
}
