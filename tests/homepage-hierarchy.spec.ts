// Front-page contract.
//
// The page it used to describe no longer exists: the stage legend, activity
// row, credibility block and volume strip were operational readouts and
// repeated figures, and they were removed. What replaced them is a single
// chain -- read, raise, run, conclude -- the newest conclusion with its trail,
// and one collapsed section holding the map and the counts.
//
// Two things here are worth more than the rest. The first is the field table:
// every rendered figure is pinned to the /api/stats field it claims to be,
// because deep_insights, contradictions and experiment counts all sit within
// a few dozen of each other and a wrong wiring renders a believable number,
// never an error. The second is that the aggregate must not carry operational
// internals -- that is a disclosure boundary, not a layout preference.
import { expect, test, type Locator, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import spec from './homepage-hierarchy.spec.json';
import processingV3 from './fixtures/processing_status_v3.json';

const tid = (name: string) => `[data-testid="${name}"]`;
const px = (value: string) => Number.parseFloat(value) || 0;

async function size(locator: Locator) {
  return px(await locator.evaluate(node => getComputedStyle(node).fontSize));
}

function formatCount(value: number) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

async function gotoHome(page: Page, language: 'zh' | 'en' = 'zh') {
  await page.addInitScript(lang => localStorage.setItem('deepgraph.lang', lang), language);
  // The overview holds a long-polling status stream open, so networkidle is
  // neither reachable nor a useful readiness condition.
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.locator(tid('hero-title')).waitFor();
  await page.locator('#chainReadValue').filter({ hasNotText: '—' }).waitFor();
}

test.describe('homepage', () => {
  test('pins the frozen v3 fixture, which this change does not touch', () => {
    // Access to /api/processing was restricted; its shape was not.
    const fixture = readFileSync('tests/fixtures/processing_status_v3.json');
    expect(createHash('sha256').update(fixture).digest('hex')).toBe(spec.v3.fixtureSha256);
    expect(processingV3.contract_version).toBe(spec.v3.contractVersion);
  });

  test('supplies the required static test hooks', async ({ page }) => {
    await gotoHome(page);
    await page.locator(tid('map-toggle')).click();
    const dynamic = new Set(['map-node']);
    const missing: string[] = [];
    for (const name of spec.requiredTestIds) {
      if (!dynamic.has(name) && await page.locator(tid(name)).count() === 0) missing.push(name);
    }
    expect(missing).toEqual([]);
  });

  test('every chain figure equals the stats field it claims to be', async ({ page, request }) => {
    await gotoHome(page);
    const stats = await (await request.get('/api/stats')).json();
    for (const [id, field] of Object.entries(spec.chain.fields)) {
      expect(await page.locator(`#${id}`).textContent(), `${id} <- ${field}`)
        .toBe(formatCount(stats[field as string]));
    }
    // Attempted runs and completed runs differ by more than three times.
    // Showing one where the other belongs is the failure this guards.
    expect(await page.locator('#chainRunValue').textContent())
      .not.toBe(formatCount(stats.experiment_runs_total));
  });

  test('the verdict split is the real one, refutations included', async ({ page, request }) => {
    await gotoHome(page);
    const stats = await (await request.get('/api/stats')).json();
    const line = await page.locator('#chainVerdictLine').textContent() ?? '';
    for (const value of [stats.decisions_supported, stats.decisions_refuted, stats.decisions_inconclusive]) {
      expect(line).toContain(String(value));
    }
    const segments = page.locator(tid('evidence-bar-segment'));
    expect(await segments.count()).toBe(3);
  });

  test('the aggregate publishes findings and no operational internals', async ({ request }) => {
    const payload = JSON.stringify(await (await request.get('/api/homepage')).json());
    for (const leaked of ['active_grants', 'controller', 'max_active', 'halt_reason',
                          'stale_work_items', 'interval_seconds', 'papers_error',
                          'papers_pending', 'adjudication_candidates']) {
      expect(payload, leaked).not.toContain(leaked);
    }
  });

  test('an idle runtime still says why it is idle', async ({ page }) => {
    await gotoHome(page);
    const pill = (await page.locator('#statusPillText').textContent() ?? '').trim();
    expect(pill.length).toBeGreaterThan(0);
    // A bare state with no reason is the thing this replaced.
    expect(pill).not.toMatch(/^(IDLE|空闲)$/);
  });

  test('the three chain rows share one grid', async ({ page }) => {
    await gotoHome(page);
    const lefts = await page.evaluate(() => [...document.querySelectorAll('.hero-chain .chain-row')]
      .map(row => [...row.querySelectorAll(':scope > .chain-cell')]
        .map(cell => Math.round(cell.getBoundingClientRect().left * 100) / 100)));
    expect(lefts).toHaveLength(3);
    for (const row of lefts) {
      expect(row).toHaveLength(spec.chain.steps);
      expect(row).toEqual(lefts[0]);
    }
  });

  test('nothing in the hero is clipped by its own frame', async ({ page }) => {
    await gotoHome(page);
    const clipped = await page.evaluate(() => {
      const hero = document.querySelector('.research-hero')!;
      const frame = hero.getBoundingClientRect();
      return [...hero.querySelectorAll('.hero-chain *')].filter(node => {
        const box = node.getBoundingClientRect();
        return box.width > 0 && (box.right > frame.right + 0.5 || box.bottom > frame.bottom + 0.5);
      }).length;
    });
    expect(clipped).toBe(0);
  });

  test('keeps the first fold focused, legible and singly-actioned', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await gotoHome(page);
    expect(await size(page.locator(tid('hero-title')))).toBeGreaterThanOrEqual(spec.thresholds.h1MinFontSize);
    expect((await page.locator(tid('global-search')).boundingBox())?.width)
      .toBeLessThanOrEqual(spec.thresholds.searchMaxWidth);
    for (const value of await page.locator(tid('chain-value')).all()) {
      expect(await size(value)).toBeLessThanOrEqual(spec.thresholds.heroMaxNumber);
    }
    // One solid accent button above the fold, so there is one obvious action.
    const accent = await page.evaluate(() => [...document.querySelectorAll('button')].filter(button => {
      const box = button.getBoundingClientRect();
      if (!box.width || box.top > window.innerHeight || box.bottom < 0) return false;
      return getComputedStyle(button).backgroundColor === 'rgb(194, 86, 42)';
    }).length);
    expect(accent).toBe(1);
  });

  test('the latest conclusion is a statement, not a counter', async ({ page }) => {
    await gotoHome(page);
    const statement = page.locator('#latestStatement');
    expect((await statement.textContent() ?? '').trim().length).toBeGreaterThan(0);
    expect(await size(statement)).toBeGreaterThanOrEqual(spec.thresholds.conclusionMinFontSize);
    await expect(page.locator(tid('verdict-pill'))).toBeVisible();
  });

  test('the trail reads as the chain continuing, not as a card', async ({ page }) => {
    await gotoHome(page);
    const framed = await page.evaluate(() =>
      [...document.querySelectorAll('.latest-trail .chain-cell')].filter(cell => {
        const style = getComputedStyle(cell);
        const filled = style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent';
        const bordered = ['Top', 'Right', 'Bottom', 'Left'].some(side =>
          Number.parseFloat(style[`border${side}Width` as any]) > 0
          && style[`border${side}Style` as any] !== 'none');
        return filled || bordered;
      }).length);
    expect(framed).toBe(0);
  });

  test('carries no wall-clock timestamps and no retired vocabulary', async ({ page }) => {
    await gotoHome(page);
    await page.locator(tid('map-toggle')).click();
    const body = await page.locator('body').innerText();
    expect(body).not.toContain('裁定');
    expect(body).not.toMatch(/\d\d-\d\d \d\d:\d\d/);
  });

  test('states a figure once', async ({ page }) => {
    await gotoHome(page);
    await page.locator(tid('map-toggle')).click();
    const body = await page.locator('body').innerText();
    for (const value of ['245.7K', '711.9K']) {
      expect((body.match(new RegExp(value.replace('.', '\\.'), 'g')) ?? []).length).toBeLessThanOrEqual(1);
    }
  });

  test('collapsed means a title and one control', async ({ page }) => {
    await gotoHome(page);
    const section = page.locator(tid('map-section'));
    await expect(section).toHaveAttribute('data-expanded', 'false');
    const leaves = await section.evaluate(root => [...root.querySelectorAll('*')].filter(node => {
      const box = node.getBoundingClientRect();
      return box.width > 0 && box.height > 0 && (node.textContent ?? '').trim()
        && ![...node.children].some(child => (child.textContent ?? '').trim());
    }).length);
    expect(leaves).toBeLessThanOrEqual(2);
  });

  test('opens both halves in place, with no request and no layout solve', async ({ page }) => {
    await gotoHome(page);
    const scrollBefore = await page.evaluate(() => window.scrollY);
    const requests: string[] = [];
    page.on('request', request => requests.push(request.url()));
    await page.locator(tid('map-toggle')).click();
    await page.waitForTimeout(600);
    expect(requests).toEqual([]);
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);
    expect(await page.locator('#researchMapSvg .map-node').count()).toBeGreaterThan(1);
    expect(await page.locator('#researchMapRows .map-row').count()).toBeGreaterThan(1);
    expect(await page.locator('.full-counts-grid strong').count()).toBe(5);
  });

  test('keeps every map node inside the viewBox', async ({ page }) => {
    await gotoHome(page);
    await page.locator(tid('map-toggle')).click();
    const outside = await page.evaluate(() => {
      const svg = document.querySelector('#researchMapSvg')!;
      const [, , width, height] = svg.getAttribute('viewBox')!.split(/\s+/).map(Number);
      return [...svg.querySelectorAll('circle')].filter(node => {
        const cx = Number(node.getAttribute('cx'));
        const cy = Number(node.getAttribute('cy'));
        const r = Number(node.getAttribute('r'));
        return cx - r < 0 || cx + r > width || cy - r < 0 || cy + r > height;
      }).length;
    });
    expect(outside).toBe(0);
  });

  test('the fixed-width map panel measures its stated width', async ({ page }) => {
    await gotoHome(page);
    await page.locator(tid('map-toggle')).click();
    const box = await page.locator('.map-canvas-panel').boundingBox();
    expect(Math.round(box!.width)).toBe(spec.thresholds.mapPanelWidth);
  });

  test('uses the specified bilingual hero copy', async ({ page }) => {
    await gotoHome(page, 'zh');
    await expect(page.locator(tid('hero-title'))).toHaveText(spec.copy.zh.h1);
    await expect(page.locator(tid('hero-sub'))).toHaveText(spec.copy.zh.sub);
    await gotoHome(page, 'en');
    await expect(page.locator(tid('hero-title'))).toHaveText(spec.copy.en.h1);
    await expect(page.locator(tid('hero-sub'))).toHaveText(spec.copy.en.sub);
  });

  test.describe('phone', () => {
    test.use({ viewport: { width: 390, height: 844 } });

    test('turns the track on its side and never scrolls sideways', async ({ page }) => {
      await gotoHome(page);
      expect(await page.evaluate(() =>
        document.documentElement.scrollWidth - document.documentElement.clientWidth))
        .toBeLessThanOrEqual(1);
      // Vertical track: the four steps stack rather than sharing a row.
      const tops = await page.evaluate(() => [...document.querySelectorAll('.hero-chain .chain-values .chain-cell')]
        .map(cell => Math.round(cell.getBoundingClientRect().top)));
      expect(new Set(tops).size).toBe(spec.chain.steps);
    });

    test('holds every tap target on the front page at the minimum', async ({ page }) => {
      await gotoHome(page);
      const small = await page.evaluate(min => {
        const found = new Set<string>();
        for (const selector of ['#tab-overview', 'header#topBar', '.sidebar']) {
          document.querySelector(selector)?.querySelectorAll('button, a[href]').forEach(node => {
            const box = node.getBoundingClientRect();
            if (box.width > 0 && box.height > 0 && box.height < min) found.add(node.className);
          });
        }
        return [...found];
      }, spec.thresholds.minTapTarget);
      expect(small).toEqual([]);
    });

    test('leaves the dense table behind "open map"', async ({ page }) => {
      await gotoHome(page);
      await page.locator(tid('map-toggle')).click();
      await expect(page.locator('.map-table-panel')).toBeHidden();
      await expect(page.locator('#researchMapSvg')).toBeVisible();
    });
  });

  test('has no serious axe violations', async ({ page }) => {
    await gotoHome(page);
    const results = await new AxeBuilder({ page })
      .disableRules(['color-contrast'])
      .analyze();
    expect(results.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')).toEqual([]);
  });
});
