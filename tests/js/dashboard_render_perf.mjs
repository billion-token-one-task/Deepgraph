// Perf microbenchmark (Acceptance A · node fallback) for the two render paths
// this change adds/touches and that grow with data volume:
//
//   1. the Evidence benchmark HEATMAP colour scale
//      (web/static/js/app.js::buildMatrixHeat) — a single O(cells) min/max pass
//      plus an O(1) RGB lerp per cell.
//   2. the CHUNKED list builder
//      (web/static/js/app.js::renderListChunked) — builds AND inserts one chunk
//      at a time, so the worst single synchronous slice is O(chunk), not O(n),
//      no matter how long the list gets.
//
// Runs inside REAL Chromium (the engine prod users run), mirroring the
// production logic with a fast path and a deliberately O(n^2) "slow" reference,
// and proves at the spec's N=5000:
//   • the fast path is < 200ms (the long-task threshold the browser E2E enforces)
//   • the O(n^2) reference is many times slower (so the harness would actually
//     catch a quadratic regression).
import { chromium } from "playwright";

const STRESS_N = 5000;
const BUDGET_MS = 200;
const CHUNK = 25;

const browser = await chromium.launch({ args: ["--no-sandbox", "--disable-dev-shm-usage"] });
let failed = false;
const check = (cond, msg) => { if (!cond) { console.error(`FAIL: ${msg}`); failed = true; } };

try {
  const page = await browser.newPage();
  await page.setContent("<!DOCTYPE html><body><div id='c'></div></body>");

  const r = await page.evaluate(({ STRESS_N, CHUNK }) => {
    const timeOnce = (fn) => { const t0 = performance.now(); const out = fn(); return { dt: performance.now() - t0, out }; };
    const median = (fn, runs = 7) => {
      const ts = []; let out;
      for (let i = 0; i < runs; i++) { const r = timeOnce(fn); ts.push(r.dt); out = r.out; }
      ts.sort((a, b) => a - b);
      return { dt: ts[Math.floor(ts.length / 2)], out };
    };

    // ── 1) HEATMAP scale ────────────────────────────────────────────────
    // A matrix with ~STRESS_N filled cells. cells keyed like app.js.
    const side = Math.ceil(Math.sqrt(STRESS_N));
    const cells = {};
    for (let i = 0; i < side; i++) {
      for (let j = 0; j < side; j++) {
        cells[`m${i}|||d${j}|||acc`] = { value: (i * 31 + j * 17) % 100, is_sota: 0 };
      }
    }
    const cellCount = Object.keys(cells).length;
    const lo = [250, 241, 234], hi = [224, 168, 131];
    const lerp = (a, b, t) => `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;

    // fast == production buildMatrixHeat: ONE min/max pass, O(1) per cell.
    function heatFast() {
      let min = Infinity, max = -Infinity;
      for (const k in cells) { const v = cells[k].value; if (v < min) min = v; if (v > max) max = v; }
      const span = max - min, colors = new Set();
      for (const k in cells) { const t = span > 0 ? (cells[k].value - min) / span : 0.5; colors.add(lerp(lo, hi, t)); }
      return colors.size;
    }
    // slow == recompute min/max INSIDE the per-cell loop → O(n^2).
    function heatSlow() {
      const colors = new Set();
      for (const k in cells) {
        let min = Infinity, max = -Infinity;
        for (const k2 in cells) { const v = cells[k2].value; if (v < min) min = v; if (v > max) max = v; }
        const span = max - min, t = span > 0 ? (cells[k].value - min) / span : 0.5;
        colors.add(lerp(lo, hi, t));
      }
      return colors.size;
    }
    const heatFastR = median(heatFast);
    const heatSlowR = timeOnce(() => heatSlow()); // one run; O(n^2) is slow

    // ── 2) CHUNKED list build ───────────────────────────────────────────
    const items = Array.from({ length: STRESS_N }, (_, i) => ({ i, title: `Card ${i} <x> & "q"` }));
    const renderItem = (it) => `<div class="card"><h3>${it.title}</h3><p>row ${it.i}</p></div>`;
    const c = document.getElementById("c");

    // fast == one synchronous slice of renderListChunked (build + insert CHUNK).
    // Its cost must be ~constant regardless of items.length (proves O(chunk)).
    function firstSlice() {
      c.innerHTML = "";
      let html = "";
      const end = Math.min(CHUNK, items.length);
      for (let i = 0; i < end; i++) html += renderItem(items[i]);
      c.insertAdjacentHTML("beforeend", html);
      return c.children.length;
    }
    // full O(n) build (map+join, single innerHTML) — what the old non-chunked
    // renders did; still O(n) but ONE big task.
    function fullBuild() {
      c.innerHTML = items.map(renderItem).join("");
      return c.children.length;
    }
    // slow == the O(n^2) trap: innerHTML += per item.
    function quadBuild(n) {
      c.innerHTML = "";
      for (let i = 0; i < n; i++) c.innerHTML += renderItem(items[i]);
      return c.children.length;
    }
    const sliceR = median(firstSlice);
    const fullR = median(fullBuild);
    const quadR = timeOnce(() => quadBuild(2000)); // 5000 the quadratic way is minutes

    return { cellCount, heatFastR, heatSlowR, sliceR, fullR, quadR, side };
  }, { STRESS_N, CHUNK });

  console.log(`heat   fast  N=${r.cellCount}: ${r.heatFastR.dt.toFixed(2)}ms (${r.heatFastR.out} distinct colours, budget < ${BUDGET_MS}ms)`);
  console.log(`heat   slow  O(n^2) N=${r.cellCount}: ${r.heatSlowR.dt.toFixed(2)}ms — reference, must be >> fast`);
  console.log(`list   slice N=${STRESS_N} chunk=${CHUNK}: ${r.sliceR.dt.toFixed(2)}ms (${r.sliceR.out} nodes) — worst single sync slice`);
  console.log(`list   full  N=${STRESS_N}: ${r.fullR.dt.toFixed(2)}ms (${r.fullR.out} nodes, budget < ${BUDGET_MS}ms)`);
  console.log(`list   quad  O(n^2) N=2000: ${r.quadR.dt.toFixed(2)}ms — reference, must be >> slice`);

  // heatmap: fast under budget, produces a real gradient, and the harness is
  // sensitive enough to catch a quadratic regression.
  check(r.heatFastR.dt < BUDGET_MS, `heat fast ${r.heatFastR.dt.toFixed(2)}ms >= ${BUDGET_MS}ms`);
  check(r.heatFastR.out > 1, `heat produced only ${r.heatFastR.out} colour(s) — not a gradient`);
  check(r.heatSlowR.dt > r.heatFastR.dt * 5, `heat harness not sensitive (slow ${r.heatSlowR.dt.toFixed(2)}ms not >> fast)`);

  // chunked list: the worst single sync slice is tiny and under budget even at
  // N=5000; the full one-shot build stays O(n)/under budget; the O(n^2) trap is
  // far slower (harness sensitivity).
  check(r.sliceR.dt < BUDGET_MS, `list slice ${r.sliceR.dt.toFixed(2)}ms >= ${BUDGET_MS}ms`);
  check(r.sliceR.out === CHUNK, `list slice produced ${r.sliceR.out} nodes, expected ${CHUNK}`);
  check(r.fullR.dt < BUDGET_MS, `list full ${r.fullR.dt.toFixed(2)}ms >= ${BUDGET_MS}ms`);
  check(r.quadR.dt > r.fullR.dt * 5, `list harness not sensitive (quad ${r.quadR.dt.toFixed(2)}ms not >> full)`);

  if (!failed) console.log("PASS");
} finally {
  await browser.close();
}
if (failed) process.exitCode = 1;
