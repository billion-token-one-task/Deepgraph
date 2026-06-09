/* Acceptance D: the new tooltip/legend/story strings reuse the i18n system and
 * exist in BOTH languages with identical key sets. A missing zh (or en) key
 * would silently fall back to the raw key id in the UI. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const __dirname = dirname(fileURLToPath(import.meta.url));
const i18nSrc = readFileSync(join(__dirname, '..', '..', 'web', 'static', 'js', 'i18n.js'), 'utf8');

const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/' });
const { window } = dom;
new Function('window', 'document', 'localStorage', i18nSrc)(window, window.document, window.localStorage);

const I18N = window.dgI18n.I18N;
const en = Object.keys(I18N.en).sort();
const zh = Object.keys(I18N.zh).sort();

const missingZh = en.filter((k) => !(k in I18N.zh));
const missingEn = zh.filter((k) => !(k in I18N.en));
assert.deepEqual(missingZh, [], `keys present in en but missing in zh: ${missingZh.join(', ')}`);
assert.deepEqual(missingEn, [], `keys present in zh but missing in en: ${missingEn.join(', ')}`);

// the new graph/story keys must actually be present
const required = [
  'graph.zoomHint', 'graph.entityHint', 'graph.legendPapers', 'graph.legendGaps', 'graph.legendMethods',
  'graph.clickForStory', 'graph.clickEntity', 'graph.moreNodes',
  'explore.entityNetwork', 'evidence.entityGraph',
  'story.gaps', 'story.contradictions', 'story.discoveries', 'story.enterArea',
];
for (const k of required) {
  assert.ok(k in I18N.en && k in I18N.zh, `required key "${k}" must exist in both languages`);
}
// %d placeholder preserved in both
assert.ok(I18N.en['graph.moreNodes'].includes('%d') && I18N.zh['graph.moreNodes'].includes('%d'),
  'graph.moreNodes must keep the %d placeholder in both languages');

console.log(`graph_i18n.test.mjs\n  ok  en/zh key sets equal (${en.length} keys)\n  ok  required graph/story keys present in both languages`);
