/* Unit + perf tests for the graph data adapter (web/static/js/graph/adapter.js).
 *
 * The adapter is the ONLY place that turns raw API payloads into the unified
 * {nodes, links} model the renderer consumes. It is pure (no D3, no DOM), so we
 * exercise it directly in Node. The N=5000 perf cases are the O(n^2) tripwire
 * required by issue #19 acceptance C: a quadratic merge/dedup bug would blow the
 * budget here long before it ever reached a browser.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const __dirname = dirname(fileURLToPath(import.meta.url));
const adapterSrc = readFileSync(
  join(__dirname, '..', '..', 'web', 'static', 'js', 'graph', 'adapter.js'),
  'utf8',
);

// adapter.js is a UMD-ish module: assigns to module.exports when present.
const mod = { exports: {} };
new Function('module', 'exports', 'window', adapterSrc)(mod, mod.exports, undefined);
const Adapter = mod.exports;

let passed = 0;
function test(name, fn) {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
}

console.log('graph_adapter.test.mjs');

// ── taxonomy model ───────────────────────────────────────────────────
test('taxonomyToModel builds parent + child nodes and links', () => {
  const parent = { id: 'ml', name: 'Machine Learning', description: 'root' };
  const children = [
    { id: 'ml.dl', name: 'Deep Learning', paper_count: 10, gap_count: 2, method_count: 4 },
    { id: 'ml.rl', name: 'Reinforcement Learning', paper_count: 5, gap_count: 0, method_count: 1 },
  ];
  const m = Adapter.taxonomyToModel(parent, children);
  assert.equal(m.kind, 'taxonomy');
  assert.equal(m.nodes.length, 3);
  assert.equal(m.links.length, 2);
  const root = m.nodes.find((n) => n.id === 'ml');
  assert.equal(root.role, 'parent');
  const dl = m.nodes.find((n) => n.id === 'ml.dl');
  assert.equal(dl.role, 'child');
  assert.equal(dl.paper_count, 10);
  assert.equal(dl.gap_count, 2);
  assert.equal(dl.method_count, 4);
  assert.deepEqual(
    m.links.map((l) => [l.source, l.target]).sort(),
    [['ml', 'ml.dl'], ['ml', 'ml.rl']],
  );
});

test('taxonomyToModel tolerates empty / null children', () => {
  const parent = { id: 'leaf', name: 'Leaf' };
  for (const children of [null, undefined, []]) {
    const m = Adapter.taxonomyToModel(parent, children);
    assert.equal(m.nodes.length, 1);
    assert.equal(m.links.length, 0);
    assert.equal(m.nodes[0].role, 'parent');
  }
});

// ── entity-relation model ────────────────────────────────────────────
test('entityGraphToModel merges entity attrs onto relation endpoints', () => {
  const gs = {
    top_entities: [
      { name: 'BERT', entity_type: 'method', paper_count: 8, mention_count: 40 },
      { name: 'GLUE', entity_type: 'dataset', paper_count: 6, mention_count: 22 },
    ],
    top_relations: [
      { subject: 'BERT', predicate: 'evaluated_on', object: 'GLUE', paper_count: 5, relation_count: 9 },
      // object 'SQuAD' is NOT in top_entities -> must still become a node
      { subject: 'BERT', predicate: 'evaluated_on', object: 'SQuAD', paper_count: 3, relation_count: 4 },
    ],
  };
  const m = Adapter.entityGraphToModel(gs);
  assert.equal(m.kind, 'entity');
  const names = m.nodes.map((n) => n.name).sort();
  assert.deepEqual(names, ['BERT', 'GLUE', 'SQuAD']);
  const bert = m.nodes.find((n) => n.name === 'BERT');
  assert.equal(bert.entity_type, 'method');
  assert.equal(bert.paper_count, 8);
  assert.equal(bert.degree, 2); // two relations touch BERT
  const squad = m.nodes.find((n) => n.name === 'SQuAD');
  assert.ok(squad, 'relation-only endpoint becomes a node');
  assert.equal(squad.paper_count, 0);
  assert.equal(m.links.length, 2);
});

test('entityGraphToModel dedupes identical relations', () => {
  const gs = {
    top_entities: [],
    top_relations: [
      { subject: 'A', predicate: 'p', object: 'B', paper_count: 1, relation_count: 1 },
      { subject: 'A', predicate: 'p', object: 'B', paper_count: 1, relation_count: 1 },
    ],
  };
  const m = Adapter.entityGraphToModel(gs);
  assert.equal(m.links.length, 1);
  assert.equal(m.nodes.length, 2);
});

test('entityGraphToModel tolerates empty / missing input', () => {
  for (const gs of [null, undefined, {}, { top_entities: [], top_relations: [] }]) {
    const m = Adapter.entityGraphToModel(gs);
    assert.equal(m.kind, 'entity');
    assert.equal(m.nodes.length, 0);
    assert.equal(m.links.length, 0);
  }
});

test('entityGraphToModel caps nodes and reports truncation, dropping dangling links', () => {
  const top_relations = [];
  for (let i = 0; i < 200; i += 1) {
    top_relations.push({ subject: `s${i}`, predicate: 'rel', object: `o${i}`, paper_count: 1, relation_count: 1 });
  }
  const m = Adapter.entityGraphToModel({ top_entities: [], top_relations }, { maxNodes: 40 });
  assert.equal(m.nodes.length, 40);
  assert.ok(m.truncated > 0, 'reports dropped node count');
  // every link must reference surviving nodes only
  const live = new Set(m.nodes.map((n) => n.id));
  for (const l of m.links) {
    assert.ok(live.has(l.source) && live.has(l.target), 'no dangling links after cap');
  }
});

// ── perf tripwire (O(n^2) catcher) ───────────────────────────────────
test('taxonomyToModel is sub-quadratic at N=5000 (<200ms)', () => {
  const children = [];
  for (let i = 0; i < 5000; i += 1) {
    children.push({ id: `ml.n${i}`, name: `Node ${i}`, paper_count: i % 50, gap_count: i % 7, method_count: i % 5 });
  }
  const t0 = performance.now();
  const m = Adapter.taxonomyToModel({ id: 'ml', name: 'root' }, children);
  const dt = performance.now() - t0;
  assert.equal(m.nodes.length, 5001);
  console.log(`      taxonomyToModel N=5000: ${dt.toFixed(1)}ms`);
  assert.ok(dt < 200, `taxonomyToModel too slow: ${dt.toFixed(1)}ms`);
});

test('entityGraphToModel is sub-quadratic at N=5000 (<200ms)', () => {
  const top_entities = [];
  const top_relations = [];
  for (let i = 0; i < 5000; i += 1) {
    top_entities.push({ name: `e${i}`, entity_type: 'method', paper_count: i % 30, mention_count: i });
    top_relations.push({ subject: `e${i}`, predicate: 'rel', object: `e${(i + 1) % 5000}`, paper_count: i % 9, relation_count: i % 4 });
  }
  const t0 = performance.now();
  const m = Adapter.entityGraphToModel({ top_entities, top_relations }, { maxNodes: 100000 });
  const dt = performance.now() - t0;
  console.log(`      entityGraphToModel N=5000: ${dt.toFixed(1)}ms (nodes=${m.nodes.length}, links=${m.links.length})`);
  assert.ok(dt < 200, `entityGraphToModel too slow: ${dt.toFixed(1)}ms`);
});

console.log(`\n${passed} adapter tests passed.`);
