# Dashboard responsiveness test harness (Acceptance A)

Real-browser gates for the dashboard's load responsiveness, used to prove the
fix for the "loaded then frozen" bug (PR: dashboard responsiveness + metric
tooltips).

## Setup

```bash
cd tests/js
npm install          # playwright reuses the already-cached Chromium
```

Chromium is driven headless with `--no-sandbox --disable-dev-shm-usage`.

## Gates

### `npm run e2e` — end-to-end responsiveness (primary)

`responsiveness_e2e.mjs` boots `seed_and_serve.py` (a Flask server on a throwaway
SQLite DB seeded to a production shape: a ~3300-node taxonomy **tree** and all 9
overview metrics non-zero), drives the real dashboard in Chromium, and asserts:

- **First paint ≤ 2s**: all 9 overview stat cards show real (non-zero) numbers.
- **After load**: idle `IDLE_MS` (default 60000), click every tab, switch language
  once — with **no main-thread long task > 200ms**, the page staying interactive
  (every tab activates, language toggle takes effect), and **no console/page error**.

Run a faster iteration with a shorter idle:

```bash
IDLE_MS=6000 node responsiveness_e2e.mjs
```

This test **fails on the pre-fix code** (the O(n²) taxonomy-dropdown build froze
the main thread for seconds during the idle prefetch) and **passes on the fix**.

### `npm run perf` — deterministic perf microbenchmark

`taxonomy_dropdown_perf.mjs` builds the taxonomy option list inside real Chromium
two ways and asserts the production path (DocumentFragment, single attach) is fast
while the old `innerHTML +=` path is catastrophically slower:

- N=3300 (real tree size): fast build **< 50ms**
- N=5000 (stress): fast build **< 200ms** (the long-task threshold)
- the O(n²) `innerHTML +=` reference is many times slower (seconds), proving the
  harness catches a quadratic regression.

## Notes

- `node_modules/` and `package-lock.json` are gitignored; run `npm install` first.
- The Python server uses the repo's `.venv` (Flask). It sets a dummy
  `MINIMAX_API_KEY` and seeds an active research agenda so `/api/providers` and
  `/api/research_agenda/*` return 200 the way a healthy prod deployment does.
