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

async function gotoHome(page: Page, language: 'zh' | 'en' = 'zh', processingSnapshot?: typeof processingV3) {
  await page.addInitScript(lang => localStorage.setItem('deepgraph.lang', lang), language);
  await page.unroute('**/api/processing');
  if (processingSnapshot) {
    await page.route('**/api/processing', route => route.fulfill({ json: processingSnapshot }));
  }
  // The overview opens an existing long-polling status stream, so networkidle
  // is neither reachable nor a useful readiness condition here.
  await page.goto('/?ff=homepage_hierarchy_v2', { waitUntil: 'domcontentloaded' });
  await page.locator(tid('hero-title')).waitFor();
}

test.describe('homepage hierarchy', () => {
  test('pins the frozen v3 fixture before using it for UI coverage', () => {
    const fixture = readFileSync('tests/fixtures/processing_status_v3.json');
    expect(createHash('sha256').update(fixture).digest('hex')).toBe(spec.v3.fixtureSha256);
    expect(processingV3.contract_version).toBe(spec.v3.contractVersion);
  });

  test('supplies the required static test hooks', async ({ page }) => {
    await gotoHome(page);
    const dynamic = new Set(['map-node']);
    const missing: string[] = [];
    for (const name of spec.requiredTestIds) {
      if (!dynamic.has(name) && await page.locator(tid(name)).count() === 0) missing.push(name);
    }
    expect(missing).toEqual([]);
  });

  test('keeps the first fold focused and legible', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await gotoHome(page);
    const hero = page.locator(tid('hero'));
    const title = page.locator(tid('hero-title'));
    const titleBox = await title.boundingBox();
    expect(await size(title)).toBeGreaterThanOrEqual(spec.thresholds.h1MinFontSize);
    expect((titleBox?.y ?? 900) + (titleBox?.height ?? 1)).toBeLessThan(900);
    expect(await page.locator(`${tid('hero')} .hero-cta.primary`).count()).toBe(1);
    expect(await hero.evaluate(node => getComputedStyle(node).backgroundColor)).not.toBe('rgba(0, 0, 0, 0)');
    expect((await page.locator(tid('global-search')).boundingBox())?.width).toBeLessThanOrEqual(spec.thresholds.searchMaxWidth);
    expect(await page.locator(tid('stage-legend-item')).count()).toBe(7);
    for (const item of await page.locator(tid('stage-legend-item')).all()) {
      expect(await size(item)).toBeGreaterThanOrEqual(spec.thresholds.minLegibleFontSize);
    }
    const stageStatuses = page.locator(tid('stage-status'));
    expect(await stageStatuses.count()).toBe(7);
  });

  test('starts with labelled desktop navigation and can collapse it', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await gotoHome(page);
    const rail = page.locator(tid('nav-rail'));
    const firstLabel = page.locator(`${tid('nav-rail-item')} span`).first();
    expect((await rail.boundingBox())?.width).toBeGreaterThanOrEqual(200);
    await expect(firstLabel).toBeVisible();
    await page.locator('#sidebarToggle').click();
    await expect(rail).toHaveClass(/collapsed/);
    await expect.poll(async () => (await rail.boundingBox())?.width ?? Infinity).toBeLessThanOrEqual(60);
  });

  test('keeps verdict ratios and number tiers coherent with live API values', async ({ page }) => {
    await gotoHome(page);
    const values = (await page.locator(`${tid('hero')} ${tid('verdict-value')}`).allInnerTexts())
      .map(value => Number(value.replace(/[^\d.]/g, '')) || 0);
    const total = values.reduce((sum, value) => sum + value, 0);
    const segments = page.locator(`${tid('hero')} ${tid('evidence-bar-segment')}`);
    expect(await segments.count()).toBe(3);
    if (total) {
      const widths = await Promise.all((await segments.all()).map(async segment => (await segment.boundingBox())?.width ?? 0));
      const widthTotal = widths.reduce((sum, value) => sum + value, 0);
      widths.forEach((width, index) => expect(Math.abs(width / widthTotal - values[index] / total)).toBeLessThan(0.015));
    } else {
      await expect(page.locator(tid('hero-proof'))).toContainText('—');
    }
    for (const value of await page.locator(`${tid('hero')} ${tid('verdict-value')}`).all()) expect(await size(value)).toBeLessThanOrEqual(spec.thresholds.heroMaxNumber);
    for (const value of await page.locator(`${tid('volume-item')} strong`).all()) expect(await size(value)).toBeLessThanOrEqual(spec.thresholds.inlineMaxNumber);
    for (const value of await page.locator(`${tid('credibility-section')} ${tid('verdict-value')}`).all()) expect(await size(value)).toBeGreaterThanOrEqual(spec.thresholds.sectionMinNumber);
  });

  test('maps frozen v3 truth independently by domain', async ({ page }) => {
    await gotoHome(page, 'en', processingV3);
    await expect(page.locator(tid('research-runtime-status'))).toHaveAttribute('data-display-state', 'idle');
    await expect(page.locator(tid('research-runtime-status'))).toContainText('11 unapproved active-agenda jobs');
    await expect(page.locator(tid('scoped-ingestion-status'))).toHaveAttribute('data-display-state', 'halted');
    await expect(page.locator(tid('scoped-ingestion-status'))).toContainText('9 reconciliation');
    await expect(page.locator(tid('legacy-ingestion-status'))).toHaveAttribute('data-display-state', 'halted');
    await expect(page.locator(tid('legacy-ingestion-status'))).toContainText('disabled');
    await expect(page.locator(tid('corpus-status'))).toHaveAttribute('data-display-state', 'idle');
    await expect(page.locator(tid('corpus-status'))).toContainText('24.2K total');
    await expect(page.locator(tid('harvest-status'))).toHaveAttribute('data-display-state', 'idle');
    await expect(page.locator(tid('harvest-status'))).toContainText('0 new');
    await expect(page.locator(tid('backfill-status'))).toHaveAttribute('data-display-state', 'halted');
    await expect(page.locator(tid('backfill-status'))).toContainText('300.0K');
    await expect(page.locator(tid('status-pill'))).toContainText('Research runtime · idle');
    for (const selector of ['corpus-status', 'research-runtime-status', 'scoped-ingestion-status', 'legacy-ingestion-status', 'harvest-status', 'backfill-status']) {
      const detail = page.locator(`${tid(selector)} small`);
      await expect(detail).toHaveCSS('white-space', 'normal');
      await expect(detail).toHaveCSS('text-overflow', 'clip');
    }
  });

  test('renders all research lifecycle states without borrowing another domain state', async ({ page }) => {
    const cases: Array<[string, string]> = [
      ['authorized_idle', 'authorized'],
      ['queued', 'queued'],
      ['running', 'running'],
      ['halted', 'halted'],
      ['failed', 'failed'],
    ];
    for (const [state, expected] of cases) {
      const snapshot = structuredClone(processingV3);
      snapshot.research_runtime.state = state;
      snapshot.research_runtime.available = state !== 'failed';
      if (state === 'queued') snapshot.research_runtime.queued_authorized_work_items = 1;
      await gotoHome(page, 'en', snapshot);
      await expect(page.locator(tid('research-runtime-status'))).toHaveAttribute('data-display-state', expected);
      await expect(page.locator(tid('status-pill'))).toContainText(`Research runtime · ${expected === 'failed' ? 'unavailable' : expected}`);
    }
  });

  test('keeps unavailable v3 domain counts unknown instead of manufacturing zeroes', async ({ page }) => {
    const snapshot = structuredClone(processingV3);
    Object.assign(snapshot.corpus, { available: false, state: 'failed', total: null, pending: null, processed: null, error: null });
    Object.assign(snapshot.harvest, { available: false, state: 'failed', last_success_at: null, last_new_count: null, last_attempt_at: null, failed_categories: null, diagnostic: 'snapshot_error' });
    Object.assign(snapshot.backfill, { available: false, state: 'failed', last_progress_at: null, age_seconds: null, halt_reason: 'snapshot_error' });
    await gotoHome(page, 'en', snapshot);
    for (const domain of ['corpus-status', 'harvest-status', 'backfill-status']) {
      await expect(page.locator(tid(domain))).toHaveAttribute('data-display-state', 'failed');
      await expect(page.locator(tid(domain))).toContainText('unavailable');
      await expect(page.locator(tid(domain))).not.toContainText('0 total');
    }
  });

  test('maps homepage actions into existing single-page views', async ({ page }) => {
    await gotoHome(page);
    await page.locator(tid('cta-runtime')).click();
    await expect(page.locator('#tab-office')).toHaveClass(/active/);
    await page.locator(`${tid('nav-rail-item')}[data-tab="overview"]`).click();
    await page.locator(tid('cta-map')).click();
    await expect(page.locator('#tab-explore')).toHaveClass(/active/);
    await expect(page.locator(tid('cta-submit-idea'))).toHaveAttribute('aria-disabled', 'true');
  });

  test('keeps every rendered map node within its canvas', async ({ page }) => {
    await gotoHome(page);
    const canvas = page.locator(tid('map-canvas'));
    const nodes = page.locator(tid('map-node'));
    test.skip(await nodes.count() === 0, 'No taxonomy nodes returned by this disposable test database.');
    await canvas.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    const canvasBox = await canvas.boundingBox();
    for (const node of await nodes.all()) {
      const box = await node.boundingBox();
      expect(box?.x ?? -1).toBeGreaterThanOrEqual((canvasBox?.x ?? 0) - 1);
      expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual((canvasBox?.x ?? 0) + (canvasBox?.width ?? 0) + 1);
      expect(box?.y ?? -1).toBeGreaterThanOrEqual((canvasBox?.y ?? 0) - 1);
      expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual((canvasBox?.y ?? 0) + (canvasBox?.height ?? 0) + 1);
    }
  });

  test('refits the research map when the viewport changes', async ({ page }) => {
    await page.route('**/api/taxonomy/ml', route => route.fulfill({ json: {
      node: { id: 'ml', name: 'Machine learning' },
      children: [
        { id: 'ml.a', name: 'Area A', paper_count: 12, gap_count: 2, method_count: 4 },
        { id: 'ml.b', name: 'Area B', paper_count: 8, gap_count: 0, method_count: 3 },
      ],
    } }));
    await page.setViewportSize({ width: 1440, height: 900 });
    await gotoHome(page);
    const graph = page.locator(tid('map-canvas'));
    await expect(page.locator(tid('map-node')).first()).toBeVisible();
    const before = await graph.getAttribute('viewBox');
    await page.setViewportSize({ width: 1024, height: 900 });
    await expect.poll(() => graph.getAttribute('viewBox')).not.toBe(before);
    const width = (value: string | null) => Number(value?.split(' ')[2] || 0);
    expect(width(await graph.getAttribute('viewBox'))).toBeLessThan(width(before));
  });

  test('uses the specified bilingual hero copy and equivalent structure', async ({ page }) => {
    await gotoHome(page, 'zh');
    const zhHooks = await page.locator('[data-testid]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-testid')).sort());
    await expect(page.locator(tid('hero-title'))).toHaveText(spec.copy.zh.h1);
    await gotoHome(page, 'en');
    await expect(page.locator(tid('hero-title'))).toHaveText(spec.copy.en.h1);
    await expect(page.locator(tid('hero-sub'))).toHaveText(spec.copy.en.sub);
    const enHooks = await page.locator('[data-testid]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-testid')).sort());
    expect(enHooks).toEqual(zhHooks);
  });

  test('has no serious axe violations and the disabled footer control is skipped', async ({ page }) => {
    await gotoHome(page);
    const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze();
    const serious = results.violations.filter(v => ['serious', 'critical'].includes(v.impact ?? ''));
    expect(serious.map(v => `${v.id}: ${v.nodes.map(node => node.target.join(' ')).join(' | ')}`)).toEqual([]);
    const focusable = await page.locator(tid('cta-submit-idea')).evaluate(node => (node as HTMLButtonElement).tabIndex);
    expect(focusable).toBe(-1);
  });

  test('keeps keyboard order from chrome through the two hero actions', async ({ page }) => {
    await gotoHome(page);
    const order: string[] = [];
    for (let index = 0; index < 28; index += 1) {
      await page.keyboard.press('Tab');
      const hook = await page.evaluate(() => document.activeElement?.closest('[data-testid]')?.getAttribute('data-testid') ?? '');
      if (hook) order.push(hook);
      if (order.includes('cta-map')) break;
    }
    expect(order.indexOf('status-pill')).toBeGreaterThan(-1);
    expect(order.indexOf('nav-rail-item')).toBeGreaterThan(order.indexOf('status-pill'));
    expect(order.indexOf('cta-runtime')).toBeGreaterThan(order.indexOf('nav-rail-item'));
    expect(order.indexOf('cta-map')).toBeGreaterThan(order.indexOf('cta-runtime'));
    expect(order).not.toContain('cta-submit-idea');
  });
});

for (const language of ['zh', 'en'] as const) {
  for (const width of [1440, 1280, 1024]) {
    test(`${language} hero visual baseline @${width}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await gotoHome(page, language);
      await expect(page.locator(tid('hero'))).toHaveScreenshot(`hero-${language}-${width}.png`, {
        animations: 'disabled',
        mask: [page.locator(tid('runtime-stage'))],
        maxDiffPixelRatio: 0.01,
      });
    });
  }
  for (const [hook, name] of [['credibility-section', 'credibility'], ['site-footer', 'footer']] as const) {
    test(`${language} ${name} visual baseline`, async ({ page }) => {
      await gotoHome(page, language);
      const locator = page.locator(tid(hook));
      await locator.scrollIntoViewIfNeeded();
      await expect(locator).toHaveScreenshot(`${name}-${language}.png`, { animations: 'disabled', maxDiffPixelRatio: 0.01 });
    });
  }
}
