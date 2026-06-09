/* Renderer perf gate for web/static/js/graph/renderer.js (issue #19 acceptance C).
 *
 * The renderer is the only module that touches D3. We mount it in jsdom against
 * real d3 v7 and assert it renders large inputs (N=5000) well under budget — an
 * O(n^2) layout/label/link bug would blow up here exactly like the dropdown
 * freeze that motivated this gate, without needing a browser.
 *
 * It also asserts the dependency-injection contract: the renderer must reach the
 * DOM, tooltip, and navigation ONLY through injected options (never #tooltip /
 * navigateTo / switchTab), so the renderer file can be swapped wholesale.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import * as d3 from 'd3';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const rendererSrc = readFileSync(join(ROOT, 'web', 'static', 'js', 'graph', 'renderer.js'), 'utf8');
const adapterSrc = readFileSync(join(ROOT, 'web', 'static', 'js', 'graph', 'adapter.js'), 'utf8');

// Forbidden global references — the renderer must not reach these directly.
// Strip comments first: the contract is about code, not the doc-block that
// names the very globals it forbids.
const rendererCode = rendererSrc
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/(^|[^:])\/\/.*$/gm, '$1');
for (const banned of ['#tooltip', 'navigateTo', 'switchTab', 'window._dg', 'getElementById']) {
  assert.ok(!rendererCode.includes(banned), `renderer.js must not reference global "${banned}" — inject it via options`);
}

const dom = new JSDOM('<!doctype html><html><body><div id="host"></div></body></html>', {
  pretendToBeVisual: true,
});
const { window } = dom;
// jsdom doesn't implement layout; give containers a non-zero width.
Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', { value: 640, configurable: true });
Object.defineProperty(window.HTMLElement.prototype, 'clientHeight', { value: 480, configurable: true });
window.d3 = d3;

function loadGlobal(src) {
  // eslint-disable-next-line no-new-func
  new Function('module', 'window', 'globalThis', src)({ exports: {} }, window, window);
}
loadGlobal(adapterSrc);
loadGlobal(rendererSrc);
const Renderer = window.DGGraphRenderer;
const Adapter = window.DGGraphAdapter;
assert.ok(Renderer && typeof Renderer.renderRadial === 'function', 'DGGraphRenderer.renderRadial missing');
assert.ok(typeof Renderer.renderNetwork === 'function', 'DGGraphRenderer.renderNetwork missing');

let passed = 0;
function test(name, fn) {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
}
function host() {
  const h = window.document.getElementById('host');
  h.innerHTML = '';
  return h;
}

console.log('graph_renderer_perf.mjs');

test('renderRadial draws nodes and wires injected click/tooltip', () => {
  const clicks = [];
  const tipCalls = { show: 0, hide: 0 };
  const model = Adapter.taxonomyToModel(
    { id: 'ml', name: 'Machine Learning' },
    [
      { id: 'ml.dl', name: 'Deep Learning', paper_count: 10, gap_count: 3, method_count: 2 },
      { id: 'ml.rl', name: 'Reinforcement Learning', paper_count: 4, gap_count: 0, method_count: 1 },
    ],
  );
  const handle = Renderer.renderRadial(host(), model, {
    height: 400,
    onNodeClick: (n) => clicks.push(n.id),
    nodeTooltipHtml: (n) => `<b>${n.name}</b>`,
    tooltip: { show: () => { tipCalls.show += 1; }, move: () => {}, hide: () => { tipCalls.hide += 1; } },
    legendItems: [{ kind: 'papers', label: 'Papers' }, { kind: 'gaps', label: 'Gaps' }],
  });
  const svg = window.document.querySelector('#host svg');
  assert.ok(svg, 'an svg is created');
  const circles = svg.querySelectorAll('circle');
  assert.ok(circles.length >= 3, `expected >=3 circles, got ${circles.length}`);
  // legend present
  assert.ok(svg.querySelectorAll('.dg-legend, [data-legend]').length >= 1 || svg.textContent.includes('Papers'),
    'legend rendered');
  // click on a child node group fires injected callback
  const childGroup = svg.querySelector('.dg-node-child');
  assert.ok(childGroup, 'child node group present');
  childGroup.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  assert.deepEqual(clicks.length, 1, 'injected onNodeClick fired exactly once');
  childGroup.dispatchEvent(new window.MouseEvent('mouseover', { bubbles: true }));
  assert.ok(tipCalls.show >= 1, 'injected tooltip.show fired on hover');
  assert.equal(typeof handle.destroy, 'function');
  handle.destroy();
});

test('renderNetwork draws entity nodes + links and fires entity click', () => {
  const gs = {
    top_entities: [
      { name: 'BERT', entity_type: 'method', paper_count: 8, mention_count: 40 },
      { name: 'GLUE', entity_type: 'dataset', paper_count: 6, mention_count: 22 },
    ],
    top_relations: [{ subject: 'BERT', predicate: 'evaluated_on', object: 'GLUE', paper_count: 5, relation_count: 9 }],
  };
  const model = Adapter.entityGraphToModel(gs);
  const clicked = [];
  Renderer.renderNetwork(host(), model, {
    height: 360,
    onNodeClick: (n) => clicked.push(n.name),
    nodeTooltipHtml: (n) => n.name,
  });
  const svg = window.document.querySelector('#host svg');
  assert.ok(svg.querySelectorAll('line, path.dg-link').length >= 1, 'links drawn');
  const node = svg.querySelector('.dg-node-entity');
  assert.ok(node, 'entity node present');
  node.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  assert.equal(clicked.length, 1);
});

test('renderRadial empty state does not throw', () => {
  const model = Adapter.taxonomyToModel({ id: 'leaf', name: 'Leaf', description: 'no kids' }, []);
  Renderer.renderRadial(host(), model, { height: 300, emptyText: 'Leaf domain' });
  assert.ok(window.document.querySelector('#host svg'));
});

// Real data is tiny (backend caps to ~12). These N=5000 cases are the O(n^2)
// tripwire: a quadratic bug (or an attempt to draw thousands of DOM nodes, which
// is what froze the dropdown) blows the budget. The renderer must cap the drawn
// node count defensively AND surface the overflow ("+N more") rather than
// silently truncate — then stay well under 200ms even when fed 5000.
test('renderRadial N=5000 caps drawn nodes, surfaces overflow, < 200ms', () => {
  const children = [];
  for (let i = 0; i < 5000; i += 1) {
    children.push({ id: `ml.n${i}`, name: `Subarea number ${i}`, paper_count: i % 50, gap_count: i % 7, method_count: i % 5 });
  }
  const model = Adapter.taxonomyToModel({ id: 'ml', name: 'root' }, children);
  const t0 = performance.now();
  Renderer.renderRadial(host(), model, { height: 600, isPreview: false });
  const dt = performance.now() - t0;
  const svg = window.document.querySelector('#host svg');
  const drawn = svg.querySelectorAll('.dg-node-child').length;
  console.log(`      renderRadial N=5000: ${dt.toFixed(1)}ms, drew ${drawn} child nodes`);
  assert.ok(drawn <= 200, `radial should cap drawn nodes, drew ${drawn}`);
  assert.ok(/\+\s*\d|more|更多/i.test(svg.textContent), 'overflow indicator surfaced (not silent)');
  assert.ok(dt < 200, `renderRadial too slow: ${dt.toFixed(1)}ms`);
});

test('renderNetwork N=5000 caps drawn nodes, < 200ms (no unbounded force)', () => {
  const top_entities = [];
  const top_relations = [];
  for (let i = 0; i < 5000; i += 1) {
    top_entities.push({ name: `e${i}`, entity_type: 'method', paper_count: i % 30, mention_count: i });
    top_relations.push({ subject: `e${i}`, predicate: 'rel', object: `e${(i + 1) % 5000}`, paper_count: 1, relation_count: 1 });
  }
  // bypass the adapter cap to stress the renderer's own large-N guard
  const model = Adapter.entityGraphToModel({ top_entities, top_relations }, { maxNodes: 100000 });
  const t0 = performance.now();
  Renderer.renderNetwork(host(), model, { height: 600 });
  const dt = performance.now() - t0;
  const drawn = window.document.querySelectorAll('#host .dg-node-entity').length;
  console.log(`      renderNetwork N=5000: ${dt.toFixed(1)}ms, drew ${drawn} nodes (model=${model.nodes.length})`);
  assert.ok(drawn <= 200, `network should cap drawn nodes, drew ${drawn}`);
  assert.ok(dt < 200, `renderNetwork too slow: ${dt.toFixed(1)}ms`);
});

console.log(`\n${passed} renderer tests passed.`);
