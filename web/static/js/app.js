/* ═══════════════════════════════════════════════════════════════════
   DeepGraph — Single-Page Application
   Pure JS, no build tools. Requires D3.js v7.
   ═══════════════════════════════════════════════════════════════════ */

(() => {
'use strict';

// ── State ────────────────────────────────────────────────────────────

const ROOT_NODE = document.body.dataset.rootNode || 'ml';

let activeTab       = 'overview';
let exploreNodeId   = ROOT_NODE;
let exploreData     = null;      // cached /api/taxonomy/<id> response
let eventsSince     = 0;
let events          = [];        // max 50
let activePapers    = {};        // paper_id -> {title, step, startTime}
let statsCache      = null;
let allPapers       = [];
let taxonomyFlat    = [];        // flat list for Evidence dropdown
let searchTimer     = null;
let statsTimer      = null;
let providerTimer   = null;
let papersLoaded    = false;
let providersLoaded = false;
let taxonomyLoaded  = false;
let paperProgressLoaded = false;
let generatedPapersLoaded = false;
let discoveriesLoaded = false;
let experimentsLoaded = false;
let insightsLoaded = false;
let overviewGraphLoaded = false;
let inactiveTabsPrefetched = false;
let sidebarCollapsed = false;

// ── Helpers ──────────────────────────────────────────────────────────

function fmt(n) {
    if (n == null) return '0';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
}

function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function trunc(str, max) {
    if (!str) return '';
    return str.length > max ? str.slice(0, max - 1) + '\u2026' : str;
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }
function el(id) { return document.getElementById(id); }
function tr(key, vars) { return window.t ? window.t(key, vars) : key; }
function setText(id, value) {
    const node = el(id);
    if (node) node.textContent = value;
}

async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`API ${path} returned ${r.status}`);
    return r.json();
}

function timeAgo(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const s = Math.floor((Date.now() - d.getTime()) / 1000);
    if (s < 60)   return tr('common.secondsAgo', { count: s });
    if (s < 3600) return tr('common.minutesAgo', { count: Math.floor(s / 60) });
    if (s < 86400) return tr('common.hoursAgo', { count: Math.floor(s / 3600) });
    return tr('common.daysAgo', { count: Math.floor(s / 86400) });
}

function fmtDateTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return String(ts);
    return d.toLocaleString();
}

// ── Tab Navigation ───────────────────────────────────────────────────

function switchTab(tab) {
    if (tab === activeTab) return;
    activeTab = tab;

    // Update nav items
    $$('.nav-item, .advanced-nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Update panels
    $$('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === 'tab-' + tab);
    });

    // Lazy-load data for tabs that need it
    onTabActivated(tab);
}

function onTabActivated(tab) {
    switch (tab) {
        case 'explore':
            // Navigate to current explore node if not yet loaded
            if (!exploreData) navigateTo(exploreNodeId);
            break;
        case 'evidence':
            if (!taxonomyLoaded) loadTaxonomyDropdown();
            break;
        case 'papers':
            if (!papersLoaded) loadPapers();
            break;
        case 'paper-progress':
            if (!paperProgressLoaded) loadPaperProgressTab();
            break;
        case 'generated-papers':
            if (!generatedPapersLoaded) loadGeneratedPapersTab();
            break;
        case 'discoveries':
            if (!discoveriesLoaded) loadDiscoveriesTab();
            break;
        case 'experiments':
            if (!experimentsLoaded) loadExperimentsTab();
            break;
        case 'insights':
            if (!insightsLoaded) loadInsightsTab();
            break;
        case 'feed':
            scrollFeedToBottom();
            break;
        case 'providers':
            if (!providersLoaded) loadProviders();
            startProviderRefresh();
            break;
    }
}

// ── Sidebar Toggle ───────────────────────────────────────────────────

function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed;
    el('sidebar').classList.toggle('collapsed', sidebarCollapsed);
}

// ── Stats ────────────────────────────────────────────────────────────

async function refreshStats() {
    try {
        const s = await api('/api/stats');
        statsCache = s;

        // Overview stat cards
        setText('statPapers', fmt(s.papers_processed || 0));
        setText('statResults', fmt(s.results_total || 0));
        setText('statTaxonomy', fmt(s.taxonomy_nodes_total || 0));
        setText('statContradictions', fmt(s.contradictions_total || 0));
        setText('statInsights', fmt(s.insights_total || 0));
        setText('statTokens', fmt(s.tokens_consumed || 0));
        setText('statExperiments', fmt(s.experiment_runs_total || 0));
        setText('statDeepDiscoveries', fmt(s.deep_insights_total || 0));
        setText('statCompletePapers', fmt(s.submission_bundles_total || 0));
    } catch (e) {
        console.error('Stats error:', e);
    }
}

// ── Event Polling ────────────────────────────────────────────────────

async function fetchEvents() {
    try {
        const payload = await api(`/api/events?since=${eventsSince}`);
        eventsSince = payload.next_seq || eventsSince;
        for (const ev of payload.events || []) {
            events.push(ev);
            if (events.length > 50) events.shift();

            trackPaperEvent(ev);
            updateLiveBadge(ev);
            appendFeedEvent(ev);
        }
    } catch (e) {
        console.error('Event polling error:', e);
    }
}

let pipelineRunning = false;

function updateLiveBadge(ev) {
    if (ev) {
        if (ev.type === 'pipeline_start') pipelineRunning = true;
        if (ev.type === 'pipeline_done' || ev.type === 'pipeline_crash') pipelineRunning = false;
    }
    const badge = el('liveBadge');
    const activeCount = Object.values(activePapers).filter(p => !p.done).length;
    const running = pipelineRunning || activeCount > 0;
    badge.textContent = running ? tr('app.live.live') : tr('app.live.idle');
    badge.classList.toggle('running', running);
}

function trackPaperEvent(ev) {
    const pid = ev.data && ev.data.paper_id;
    if (!pid) return;

    if (ev.type === 'step') {
        if (!activePapers[pid]) {
            activePapers[pid] = { title: ev.data.title || pid, step: '', startTime: Date.now() };
        }
        activePapers[pid].step = ev.data.step || '';
        activePapers[pid].done = false;
        if (ev.data.title) activePapers[pid].title = ev.data.title;
    } else if (ev.type === 'paper_done' || ev.type === 'error') {
        if (activePapers[pid]) {
            activePapers[pid].done = true;
            activePapers[pid].doneAt = Date.now();
            activePapers[pid].step = ev.type === 'error' ? 'error' : 'done';
        }
    }

    renderProcessingList();
}

async function loadProcessingPapers() {
    try {
        const data = await api('/api/processing');
        const rows = data.papers || data;
        if (data.pipeline_running != null) pipelineRunning = data.pipeline_running;

        for (const r of rows) {
            const isDone = r.status === 'reasoned' || r.status === 'error';
            if (!activePapers[r.id]) {
                activePapers[r.id] = {
                    title: r.title || r.id,
                    step: isDone ? (r.status === 'error' ? 'error' : 'done') : (r.status || 'processing'),
                    startTime: Date.now(),
                    done: isDone,
                    doneAt: isDone ? Date.now() : null
                };
            } else if (isDone && !activePapers[r.id].done) {
                activePapers[r.id].done = true;
                activePapers[r.id].doneAt = Date.now();
                activePapers[r.id].step = r.status === 'error' ? 'error' : 'done';
            } else if (!isDone) {
                activePapers[r.id].step = r.status || 'processing';
            }
        }
        // Remove papers no longer in the API response and already done for > 10s
        const activeIds = new Set(rows.map(r => r.id));
        const now = Date.now();
        for (const [pid, info] of Object.entries(activePapers)) {
            if (!activeIds.has(pid) && info.done && now - info.doneAt > 10000) {
                delete activePapers[pid];
            } else if (!activeIds.has(pid) && !info.done) {
                activePapers[pid].done = true;
                activePapers[pid].doneAt = Date.now();
                activePapers[pid].step = 'done';
            }
        }
        renderProcessingList();
        updateLiveBadge();
    } catch (e) { /* ignore */ }
}

function renderProcessingList() {
    const listEl = el('processingList');
    const countEl = el('processingCount');
    const entries = Object.entries(activePapers);
    const activeCount = entries.filter(([, info]) => !info.done).length;
    countEl.textContent = activeCount || entries.length;

    if (entries.length === 0) {
        listEl.innerHTML = `<p class="empty-msg">${esc(tr('overview.idle'))}</p>`;
        return;
    }

    listEl.innerHTML = entries.map(([pid, info]) => {
        const elapsed = Math.round((Date.now() - info.startTime) / 1000);
        const doneClass = info.done ? ' proc-done' : '';
        const stepLabel = info.done
            ? (info.step === 'error' ? 'error' : `done (${elapsed}s)`)
            : `${info.step} (${elapsed}s)`;
        return `<div class="proc-item${doneClass}">
            <span class="proc-id">${esc(pid)}</span>
            <span class="proc-title">${esc(trunc(info.title, 50))}</span>
            <span class="proc-step">${esc(stepLabel)}</span>
        </div>`;
    }).join('');
}

// ── Feed ─────────────────────────────────────────────────────────────

function appendFeedEvent(ev) {
    const feed = el('eventFeed');
    if (!feed) return;

    const ts = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString() : '';
    const typeCls = 'type-' + (ev.type || 'info');
    const typeClsInner = 'ev-type-' + (ev.type || 'info');

    let detail = '';
    if (ev.data) {
        const d = ev.data;
        if (d.paper_id) detail += esc(d.paper_id) + ' ';
        if (d.title)    detail += esc(trunc(d.title, 60)) + ' ';
        if (d.step)     detail += '<span style="color:var(--green);">' + esc(d.step) + '</span> ';
        if (d.node_id)  detail += esc(d.node_id) + ' ';
        if (d.message)  detail += esc(d.message) + ' ';
        // Fallback: show raw keys
        if (!detail.trim()) {
            const keys = Object.keys(d).slice(0, 4);
            detail = keys.map(k => esc(k) + '=' + esc(trunc(String(d[k]), 30))).join(' ');
        }
    }

    const div = document.createElement('div');
    div.className = 'event ' + typeCls;
    div.innerHTML = `<span class="ev-time">${ts}</span> <span class="ev-type ${typeClsInner}">[${esc(ev.type || '?')}]</span> ${detail}`;
    feed.appendChild(div);

    // Keep max 50
    while (feed.children.length > 50) feed.removeChild(feed.firstChild);

    // Update count
    const countEl = el('feedCount');
    if (countEl) countEl.textContent = tr('common.eventsCount', { count: feed.children.length });

    // Auto-scroll if tab is active
    if (activeTab === 'feed') scrollFeedToBottom();
}

function scrollFeedToBottom() {
    const feed = el('eventFeed');
    if (feed) feed.scrollTop = feed.scrollHeight;
}

// ── Recently Discovered (Overview) ───────────────────────────────────

async function loadRecentlyDiscovered() {
    try {
        const [data, insights, discoveries] = await Promise.all([
            api('/api/recent_discoveries?limit=8'),
            api('/api/insights?limit=6'),
            api('/api/deep_insights?limit=4'),
        ]);
        renderRecentlyDiscovered(data, insights, discoveries);
    } catch (e) {
        console.error('Recent discoveries error:', e);
    }
}

function renderRecentlyDiscovered(data, insights, discoveries) {
    const grid = el('recentlyGrid');
    let items = [];

    if (discoveries && discoveries.length > 0) {
        for (const d of discoveries.filter(isDisplayableDiscovery).slice(0, 3)) {
            items.push({
                type: 'discovery',
                title: d.title || tr('label.discovery'),
                desc: d.problem_statement || d.formal_structure || d.evidence_summary || '',
                meta: `${esc(tr('label.tier', { tier: d.tier || '?' }))} | ${esc(d.novelty_status || 'unchecked')}`,
            });
        }
    }

    // Prioritize research insights over older opportunity rows.
    if (insights && insights.length > 0) {
        for (const ins of insights.slice(0, 4)) {
            items.push({
                type: 'research-insight',
                title: ins.title || tr('label.researchInsight'),
                desc: ins.hypothesis || '',
                meta: `${esc(ins.node_id)} | N:${ins.novelty_score}/5 F:${ins.feasibility_score}/5`,
                nodeId: ins.node_id,
            });
        }
    } else if (data.opportunities) {
        for (const o of data.opportunities.slice(0, 3)) {
            items.push({
                type: 'opportunity',
                title: o.title || tr('label.opportunity'),
                desc: o.description || '',
                meta: `${esc(o.node_name || o.node_id)} | ${esc(tr('common.scoreValue', { score: o.value_score || '?' }))}`,
                nodeId: o.node_id,
            });
        }
    }
    if (data.gaps) {
        for (const g of data.gaps.slice(0, 3)) {
            items.push({
                type: 'gap',
                title: tr('common.onDataset', { method: g.method_name || '', dataset: g.dataset_name || '' }),
                desc: g.gap_description || '',
                meta: `${esc(g.node_name || g.node_id)} | ${esc(tr('common.valueScore', { score: g.value_score || '?' }))}`,
                nodeId: g.node_id,
            });
        }
    }
    if (data.contradictions) {
        for (const c of data.contradictions.slice(0, 2)) {
            items.push({
                type: 'contradiction',
                title: c.description || tr('insights.type.contradiction'),
                desc: c.hypothesis || '',
                meta: `${esc(c.paper_a || '')} vs ${esc(c.paper_b || '')}`,
            });
        }
    }
    if (data.papers) {
        for (const p of data.papers.slice(0, 3)) {
            items.push({
                type: 'paper',
                title: trunc(p.title, 70),
                desc: p.plain_summary || '',
                meta: `${esc(p.id)} | ${esc(p.work_type || p.status || '')}`,
                paperId: p.id,
            });
        }
    }

    if (items.length === 0) {
        grid.innerHTML = `<p class="empty-msg">${esc(tr('overview.latestEmpty'))}</p>`;
        return;
    }

    const order = { discovery: 0, 'research-insight': 1, opportunity: 2, gap: 3, contradiction: 4, paper: 5 };
    items.sort((a, b) => (order[a.type] || 9) - (order[b.type] || 9));

    grid.innerHTML = items.slice(0, 8).map(item => {
        let click = '';
        if (item.nodeId) {
            click = `onclick="window._dg.exploreNode('${esc(item.nodeId)}')" style="cursor:pointer;"`;
        } else if (item.paperId) {
            click = `onclick="window.open('https://arxiv.org/abs/${esc(item.paperId)}', '_blank')" style="cursor:pointer;"`;
        }
        return `<div class="recently-item type-${item.type}" ${click}>
            <div class="recently-type-badge ${item.type}">${esc(item.type)}</div>
            <div class="ri-title">${esc(item.title)}</div>
            <div class="ri-desc">${esc(trunc(item.desc, 120))}</div>
            <div class="ri-meta">${item.meta}</div>
        </div>`;
    }).join('');
}

// ── Overview Graph Preview ───────────────────────────────────────────

async function loadOverviewGraph() {
    if (overviewGraphLoaded) return;
    overviewGraphLoaded = true;
    try {
        const data = await api(`/api/taxonomy/${ROOT_NODE}`);
        renderTaxonomyGraph('overviewGraph', data.node, data.children, { height: 320, isPreview: true });
    } catch (e) {
        overviewGraphLoaded = false;
        console.error('Overview graph error:', e);
    }
}

// ── Explore Tab ──────────────────────────────────────────────────────

async function navigateTo(nodeId) {
    exploreNodeId = nodeId;

    try {
        // Fetch node data + insights + patterns in parallel
        const [data, insights, patterns] = await Promise.all([
            api(`/api/taxonomy/${nodeId}`),
            api(`/api/insights?node_id=${encodeURIComponent(nodeId)}&limit=10`),
            api(`/api/patterns?node_id=${encodeURIComponent(nodeId)}&limit=8`),
        ]);
        exploreData = data;
        exploreData._insights = insights;
        exploreData._patterns = patterns;

        // Breadcrumb
        renderBreadcrumb(data.breadcrumb || []);

        // Title
        el('exploreTitle').textContent = data.node.name + ' \u2014 ' + tr('explore.title');

        // Graph
        closeAreaStory();
        renderTaxonomyGraph('exploreGraph', data.node, data.children, { height: 520, isPreview: false });

        // Summary card
        const sumCard = el('exploreSummaryCard');
        if (data.summary || data.node.description || insights.length > 0) {
            sumCard.style.display = '';
            el('exploreSummaryTitle').textContent = tr('explore.happeningTitle', { name: data.node.name });
            renderExploreSummary(data);
        } else {
            sumCard.style.display = 'none';
        }

        // Children card
        const childCard = el('exploreChildrenCard');
        if (data.children && data.children.length > 0) {
            childCard.style.display = '';
            el('exploreChildrenTitle').textContent = tr('explore.subareasTitle', { name: data.node.name, count: data.children.length });
            renderExploreChildren(data.children);
        } else {
            childCard.style.display = 'none';
        }
    } catch (e) {
        console.error('Navigate error:', e);
    }
}

function renderBreadcrumb(crumbs) {
    const bc = el('breadcrumb');
    bc.innerHTML = crumbs.map((c, i) => {
        const isLast = i === crumbs.length - 1;
        if (isLast) return `<span class="crumb active">${esc(c.name)}</span>`;
        return `<span class="crumb" onclick="window._dg.navigateTo('${esc(c.id)}')">${esc(c.name)}</span>`;
    }).join('<span class="crumb-sep">\u203A</span>');
}

function renderExploreSummary(data) {
    const body = el('exploreSummaryBody');
    const s = data.summary;
    const node = data.node;
    const children = data.children || [];
    const paperClusters = data.paper_clusters || [];

    // Chips for children
    const childChips = children.slice(0, 10).map(c =>
        `<span class="chip" onclick="window._dg.navigateTo('${esc(c.id)}')">${esc(c.name)}${c.paper_count ? ' \u00B7 ' + esc(tr('common.paperShort', { count: c.paper_count })) : ''}</span>`
    ).join('');

    let html = `<div class="summary-hero">
        <h4>${esc(node.name)}</h4>
        <p>${esc(s ? (s.overview || node.description || '') : (node.description || tr('empty.exploreSummary')))}</p>
        ${s && s.why_it_matters ? `<p>${esc(s.why_it_matters)}</p>` : ''}
        ${childChips ? `<div class="chip-row">${childChips}</div>` : ''}
    </div>`;

    if (s) {
        // Work items and gaps
        const workHtml = (s.what_people_are_building || []).map(w =>
            `<div class="summary-item"><strong>${esc(w.label || tr('explore.workstream'))}</strong><p>${esc(w.description || '')}</p>${w.paper_count ? `<div class="meta">${esc(tr('common.papersCount', { count: w.paper_count }))}</div>` : ''}</div>`
        ).join('') || `<p class="empty-msg">${esc(tr('empty.workstreams'))}</p>`;

        const gapHtml = (s.current_gaps || []).map(g => {
            const tl = g.gap_type ? `<span style="color:var(--text-dim);font-size:0.68rem;">[${esc(g.gap_type.replace(/_/g, ' '))}]</span> ` : '';
            return `<div class="summary-item"><strong>${tl}${esc(g.title || tr('explore.openGap'))}</strong><p>${esc(g.description || '')}</p>${g.why_now ? `<div class="meta">${esc(tr('explore.whyNow', { text: g.why_now }))}</div>` : ''}</div>`;
        }).join('') || `<p class="empty-msg">${esc(tr('empty.gaps'))}</p>`;

        html += `<div class="summary-grid">
            <div class="summary-card-inner"><h4>${esc(tr('explore.whatPeopleWorking'))}</h4>${workHtml}</div>
            <div class="summary-card-inner"><h4>${esc(tr('explore.whereGapsAre'))}</h4>${gapHtml}</div>
        </div>`;

        // Chips
        const patterns = (s.common_patterns || []).map(p => `<span class="chip">${esc(p)}</span>`).join('');
        const methods  = (s.common_methods || []).map(m => `<span class="chip">${esc(m)}</span>`).join('');
        const datasets = (s.common_datasets || []).map(d => `<span class="chip">${esc(d)}</span>`).join('');

        if (patterns || methods || datasets) {
            html += `<div class="summary-grid">
                <div class="summary-card-inner"><h4>${esc(tr('explore.recurringThemes'))}</h4><div class="chip-row">${patterns || `<span class="chip">${esc(tr('explore.noneYet'))}</span>`}</div></div>
                <div class="summary-card-inner"><h4>${esc(tr('explore.methodsDatasets'))}</h4>
                    ${methods ? `<div class="chip-row">${methods}</div>` : ''}
                    ${datasets ? `<div class="chip-row" style="margin-top:6px;">${datasets}</div>` : ''}
                </div>
            </div>`;
        }

        if (paperClusters.length > 0) {
            const clusterHtml = paperClusters.map(cluster => `
                <div class="summary-item">
                    <strong>${esc(cluster.label || tr('explore.paperCluster'))}</strong>
                    <p>${esc(tr('common.papersCount', { count: cluster.paper_count }))}${cluster.shared_entities?.length ? ' · ' + esc(tr('explore.sharedEntities', { entities: cluster.shared_entities.slice(0, 3).join(', ') })) : ''}</p>
                    ${cluster.sample_papers?.length ? `<div class="meta">${cluster.sample_papers.map(p => esc(trunc(p.title, 48))).join(' | ')}</div>` : ''}
                </div>
            `).join('');

            html += `<div class="summary-card-inner">
                <h4>${esc(tr('explore.paperClusters'))}</h4>
                ${clusterHtml}
            </div>`;
        } else if ((data.papers || []).length >= 10) {
            html += `<div class="summary-card-inner">
                <h4>${esc(tr('explore.paperClusters'))}</h4>
                <p class="empty-msg">${esc(tr('explore.clusterWeakSignals', { count: data.papers.length }))}</p>
            </div>`;
        }

        // Entity-relation network (rendered after innerHTML is set, below)
        const gs = data.graph_summary;
        if (gs && ((gs.top_entities && gs.top_entities.length) || (gs.top_relations && gs.top_relations.length))) {
            html += `<div class="summary-card-inner entity-graph-block">
                <h4>${esc(tr('explore.entityNetwork'))} <span class="graph-hint">${esc(tr('graph.entityHint'))}</span></h4>
                <div id="exploreEntityGraph" class="dg-graph-host dg-entity-host"></div>
            </div>`;
        }
    }

    // Research Insights for this node
    const insights = data._insights || [];
    if (insights.length > 0) {
        const typeColors = {
            contradiction_analysis: '#c4453a',
            method_transfer: '#c4704b',
            assumption_challenge: '#a8842a',
            ignored_limitation: '#7c5cbf',
            paradigm_exhaustion: '#9a9088',
            cross_domain_bridge: '#2e86ab',
        };
        const insightHtml = insights.map(ins => {
            const color = typeColors[ins.insight_type] || '#888';
            // Parse supporting papers
            let papers = [];
            try { papers = JSON.parse(ins.supporting_papers || '[]'); } catch(e) {}
            const paperLinks = papers.map(pid =>
                `<a class="paper-cite" href="https://arxiv.org/abs/${esc(pid)}" target="_blank" title="${esc(tr('common.openArxiv'))}">${esc(pid)}</a>`
            ).join(' ');
            return `<div class="insight-card" style="border-left: 3px solid ${color};">
                <div class="insight-header">
                    <span class="insight-type" style="color:${color};">${esc((ins.insight_type || '').replace(/_/g, ' '))}</span>
                    <span class="insight-scores">N:${ins.novelty_score}/5 F:${ins.feasibility_score}/5</span>
                </div>
                <div class="insight-title">${esc(ins.title)}</div>
                ${paperLinks ? `<div class="insight-papers">${paperLinks}</div>` : ''}
                <div class="insight-evidence"><span class="insight-label">${esc(tr('label.evidence'))}</span> ${esc(ins.evidence || '')}</div>
                <div class="insight-hypothesis"><span class="insight-label">${esc(tr('label.hypothesis'))}</span> ${esc(ins.hypothesis)}</div>
                <div class="insight-experiment"><span class="insight-label">${esc(tr('label.experiment'))}</span> ${esc(ins.experiment)}</div>
                ${ins.impact ? `<div class="insight-impact"><span class="insight-label">${esc(tr('label.impact'))}</span> ${esc(ins.impact)}</div>` : ''}
                <div class="insight-actions">
                    <button class="btn-preview" onclick="window._dg.previewProposal(${ins.id})">${esc(tr('label.previewProposal'))}</button>
                </div>
            </div>`;
        }).join('');

        html += `<div class="summary-card-inner insights-section">
            <h4>${esc(tr('explore.researchInsights', { count: insights.length }))}</h4>
            <div class="insights-list">${insightHtml}</div>
        </div>`;
    }

    // Cross-node patterns for this node
    const patterns = data._patterns || [];
    if (patterns.length > 0) {
        const patHtml = patterns.map(p => {
            let domains = [];
            try { domains = JSON.parse(p.domains || '[]'); } catch(e) {}
            const levelBadge = p.abstraction_level === 'universal'
                ? `<span class="pattern-level universal">${esc(tr('explore.universal'))}</span>`
                : `<span class="pattern-level cross-domain">${esc(tr('explore.crossDomain'))}</span>`;
            return `<div class="pattern-card">
                <div class="pattern-header">
                    ${levelBadge}
                    <span class="pattern-type">${esc((p.pattern_type || '').replace(/_/g, ' '))}</span>
                </div>
                <div class="pattern-text">${esc(p.pattern_text)}</div>
                ${domains.length ? `<div class="pattern-domains">${esc(tr('explore.alsoAppliesTo'))} ${domains.map(d => `<span class="pattern-domain-chip">${esc(d)}</span>`).join(' ')}</div>` : ''}
            </div>`;
        }).join('');

        html += `<div class="summary-card-inner">
            <h4>${esc(tr('explore.universalPatterns', { count: patterns.length }))}</h4>
            <div class="patterns-list">${patHtml}</div>
        </div>`;
    }

    body.innerHTML = html;

    // Render the entity-relation network into its host (now in the DOM).
    if (el('exploreEntityGraph')) {
        renderEntityGraph('exploreEntityGraph', data.graph_summary, { height: 380 });
    }
}

function renderExploreChildren(children) {
    const body = el('exploreChildrenBody');
    body.innerHTML = `<div class="children-grid">${children.map(c => `
        <div class="child-card" onclick="window._dg.navigateTo('${esc(c.id)}')">
            <div class="child-name">${esc(c.name)}</div>
            <div class="child-stats">
                <span>${esc(tr('common.papersCount', { count: c.paper_count || 0 }))}</span>
                <span>${esc(tr('common.methodsCount', { count: c.method_count || 0 }))}</span>
                ${c.gap_count ? `<span style="color:var(--green);">${esc(tr('common.gapsCount', { count: c.gap_count }))}</span>` : ''}
            </div>
        </div>
    `).join('')}</div>`;
}

// ── Knowledge Graph glue (adapter + renderer + tooltip) ──────────────
// All D3 lives in /static/js/graph/renderer.js. This layer only builds the
// model (via DGGraphAdapter), reads the theme from CSS vars, and injects the
// app's tooltip / navigation / click-through as callbacks. Swapping the
// renderer means rewriting renderer.js alone — nothing here changes.

function graphTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, fb) => (cs.getPropertyValue(name).trim() || fb);
    return {
        accent: v('--accent', '#c4704b'), green: v('--green', '#3d8b5e'),
        gold: v('--gold', '#a8842a'), purple: v('--purple', '#7c5cbf'),
        text: v('--text-primary', '#2b2520'), dim: v('--text-dim', '#9a9088'),
        muted: v('--text-muted', '#c4bdb4'), bg: v('--bg-card', '#ffffff'),
        bgElevated: v('--bg-elevated', '#f0ede6'), border: v('--border', '#e5e0d5'),
        gapLo: v('--graph-gap-lo', '#eef3ee'), gapHi: v('--graph-gap-hi', '#3d8b5e'),
        entityPalette: {
            method: v('--graph-ent-method', '#c4704b'), dataset: v('--graph-ent-dataset', '#2e86ab'),
            metric: v('--graph-ent-metric', '#a8842a'), task: v('--graph-ent-task', '#7c5cbf'),
            model: v('--graph-ent-model', '#c4453a'), artifact: v('--graph-ent-artifact', '#3d8b5e'),
            concept: v('--graph-ent-concept', '#9a9088'),
        },
    };
}

const dgTooltip = {
    show: (html, ev) => window.DGTooltip && window.DGTooltip.show(html, ev),
    move: (ev) => window.DGTooltip && window.DGTooltip.move(ev),
    hide: () => window.DGTooltip && window.DGTooltip.hide(),
};

function taxonomyLegendItems() {
    return [
        { kind: 'papers', label: tr('graph.legendPapers') },
        { kind: 'gaps', label: tr('graph.legendGaps') },
        { kind: 'methods', label: tr('graph.legendMethods') },
    ];
}

function taxonomyNodeTooltip(d) {
    if (d.role === 'parent') {
        return `<div class="tip-title">${esc(d.name)}</div>
            <div class="tip-body">${esc(trunc(d.description, 160))}</div>`;
    }
    return `<div class="tip-title">${esc(d.name)}</div>
        <div class="tip-body">${esc(trunc(d.description, 160))}</div>
        <div class="tip-stats">
            <span><b style="color:var(--accent);">${d.paper_count}</b> ${esc(tr('common.papersUnit'))}</span>
            <span><b style="color:var(--gold);">${d.method_count}</b> ${esc(tr('common.methodsUnit'))}</span>
            <span><b style="color:var(--green);">${d.gap_count}</b> ${esc(tr('common.gapsUnit'))}</span>
        </div>
        <div class="tip-hint">${esc(tr('graph.clickForStory'))}</div>`;
}

let _taxGraphHandle = null;
function renderTaxonomyGraph(containerId, parentNode, children, opts) {
    const container = el(containerId);
    if (!container || !window.DGGraphRenderer) return;
    const isPreview = !!(opts && opts.isPreview);
    const model = window.DGGraphAdapter.taxonomyToModel(parentNode, children);
    const handle = window.DGGraphRenderer.renderRadial(container, model, {
        height: (opts && opts.height) || 420,
        isPreview,
        theme: graphTheme(),
        tooltip: dgTooltip,
        nodeTooltipHtml: taxonomyNodeTooltip,
        legendItems: isPreview ? null : taxonomyLegendItems(),
        moreLabel: tr('graph.moreNodes'),
        onNodeClick: (d) => {
            if (d.role === 'parent') return;
            if (isPreview) {
                switchTab('explore');
                navigateTo(d.id);
            } else {
                openAreaStory(d);
            }
        },
    });
    if (!isPreview) _taxGraphHandle = handle;
    return handle;
}

function entityLegendItems(model) {
    const types = Array.from(new Set((model.nodes || []).map((n) => n.entity_type))).slice(0, 6);
    const t = graphTheme();
    return types.map((ty) => ({ color: t.entityPalette[ty] || t.entityPalette.concept, label: tr('entityType.' + ty, {}) !== 'entityType.' + ty ? tr('entityType.' + ty) : ty }));
}

function renderEntityGraph(containerId, graphSummary, opts) {
    const container = el(containerId);
    if (!container || !window.DGGraphRenderer) return null;
    const model = window.DGGraphAdapter.entityGraphToModel(graphSummary);
    if (!model.nodes.length) {
        container.innerHTML = `<p class="empty-msg">${esc(tr('empty.entities'))}</p>`;
        return null;
    }
    return window.DGGraphRenderer.renderNetwork(container, model, {
        height: (opts && opts.height) || 360,
        theme: graphTheme(),
        tooltip: dgTooltip,
        nodeTooltipHtml: (n) => `<div class="tip-title">${esc(n.name)}</div>
            <div class="tip-body">${esc(n.entity_type)}</div>
            <div class="tip-stats">
                <span><b>${n.paper_count}</b> ${esc(tr('common.papersUnit'))}</span>
                <span><b>${n.degree}</b> ${esc(tr('graph.links'))}</span>
            </div>
            <div class="tip-hint">${esc(tr('graph.clickEntity'))}</div>`,
        legendItems: entityLegendItems(model),
        moreLabel: tr('graph.moreNodes'),
        onNodeClick: (n) => window._dg.searchEntity(n.name),
    });
}

// In-graph "domain → gap → discovery" story panel (acceptance A2). Clicking an
// area node fetches its gaps / contradictions / discoveries and shows them
// without leaving the graph; a button navigates deeper on demand.
async function openAreaStory(node) {
    const panel = el('exploreStoryPanel');
    if (!panel) return;
    panel.hidden = false;
    panel.innerHTML = `<div class="story-loading">${esc(tr('common.loading'))}</div>`;
    try {
        const [data, insights] = await Promise.all([
            api(`/api/taxonomy/${encodeURIComponent(node.id)}`),
            api(`/api/insights?node_id=${encodeURIComponent(node.id)}&limit=12`),
        ]);
        const gaps = (data.gaps && data.gaps.length)
            ? data.gaps
            : ((data.summary && data.summary.current_gaps) || []);
        const contradictions = insights.filter((i) => i.insight_type === 'contradiction_analysis');
        const discoveries = insights.filter((i) => i.insight_type !== 'contradiction_analysis');

        const gapHtml = gaps.length ? gaps.slice(0, 5).map((g) => `
            <li class="story-item story-gap">${esc(g.title || g.gap_description || tr('explore.openGap'))}
                ${g.description ? `<span class="story-sub">${esc(trunc(g.description, 110))}</span>` : ''}</li>`).join('')
            : `<li class="story-empty">${esc(tr('empty.gaps'))}</li>`;

        const contraHtml = contradictions.length ? contradictions.slice(0, 4).map((c) => `
            <li class="story-item story-contra" onclick="window._dg.previewProposal(${c.id})">${esc(c.title)}
                <span class="story-sub">${esc(trunc(c.evidence || '', 100))}</span></li>`).join('')
            : `<li class="story-empty">${esc(tr('story.noContradictions'))}</li>`;

        const discHtml = discoveries.length ? discoveries.slice(0, 5).map((d) => `
            <li class="story-item story-disc" onclick="window._dg.previewProposal(${d.id})">${esc(d.title)}
                <span class="story-sub">N:${d.novelty_score}/5 · F:${d.feasibility_score}/5</span></li>`).join('')
            : `<li class="story-empty">${esc(tr('story.noDiscoveries'))}</li>`;

        panel.innerHTML = `
            <div class="story-head">
                <div>
                    <div class="story-kicker">${esc(tr('story.area'))}</div>
                    <h4>${esc(node.name)}</h4>
                </div>
                <button class="story-close" aria-label="close" onclick="window._dg.closeAreaStory()">×</button>
            </div>
            <div class="story-flow">
                <span class="story-flow-step">${esc(tr('story.stepArea'))}</span>
                <span class="story-flow-arrow">→</span>
                <span class="story-flow-step">${esc(tr('story.stepGap'))}</span>
                <span class="story-flow-arrow">→</span>
                <span class="story-flow-step">${esc(tr('story.stepDiscovery'))}</span>
            </div>
            <div class="story-section">
                <div class="story-section-title gap">${esc(tr('story.gaps'))} <span class="story-count">${gaps.length}</span></div>
                <ul>${gapHtml}</ul>
            </div>
            <div class="story-section">
                <div class="story-section-title contra">${esc(tr('story.contradictions'))} <span class="story-count">${contradictions.length}</span></div>
                <ul>${contraHtml}</ul>
            </div>
            <div class="story-section">
                <div class="story-section-title disc">${esc(tr('story.discoveries'))} <span class="story-count">${discoveries.length}</span></div>
                <ul>${discHtml}</ul>
            </div>
            <button class="story-enter" onclick="window._dg.navigateTo('${esc(node.id)}')">${esc(tr('story.enterArea'))}</button>`;
    } catch (e) {
        console.error('Area story error:', e);
        panel.innerHTML = `<div class="story-loading">${esc(tr('common.errorLoadingData'))}</div>
            <button class="story-close" onclick="window._dg.closeAreaStory()">×</button>`;
    }
}

function closeAreaStory() {
    const panel = el('exploreStoryPanel');
    if (panel) { panel.hidden = true; panel.innerHTML = ''; }
}

// ── Evidence Tab ─────────────────────────────────────────────────────

async function loadTaxonomyDropdown() {
    if (taxonomyFlat.length > 0) return; // already loaded
    taxonomyLoaded = true;
    try {
        taxonomyFlat = await api('/api/taxonomy');
        // Populate the <datalist> backing the evidence node typeahead. Build
        // every <option> via the DOM into one DocumentFragment and attach it
        // ONCE. Appending to innerHTML per node re-parsed the whole list on
        // each iteration \u2014 O(n^2) over ~3300 nodes, which froze the main thread
        // for seconds. A fragment is O(n); option.value carries the node id and
        // option.label the human-readable "id \u2014 name", so no manual escaping is
        // needed and the input's committed value is always a clean node id.
        const dl = el('evidenceNodeOptions');
        const frag = document.createDocumentFragment();
        for (const n of taxonomyFlat) {
            const opt = document.createElement('option');
            opt.value = n.id;
            opt.label = `${n.id} \u2014 ${n.name}`;
            frag.appendChild(opt);
        }
        dl.replaceChildren(frag);
    } catch (e) {
        taxonomyLoaded = false;
        console.error('Taxonomy dropdown error:', e);
    }
}

async function loadEvidenceForNode(nodeId) {
    if (!nodeId) {
        el('evidenceMatrixContainer').innerHTML = '';
        el('evidenceGapsCard').style.display = 'none';
        el('evidenceGraphCard').style.display = 'none';
        el('evidenceHint').textContent = tr('evidence.hint');
        return;
    }

    el('evidenceHint').textContent = tr('common.loading');

    try {
        const data = await api(`/api/taxonomy/${nodeId}`);
        const m = data.matrix;

        if (m && m.methods && m.methods.length > 0 && m.datasets && m.datasets.length > 0) {
            renderMatrix(el('evidenceMatrixContainer'), m);
            el('evidenceHint').textContent = `${tr('common.methodsCount', { count: m.methods.length })} x ${tr('common.datasetsCount', { count: m.datasets.length })}`;
        } else {
            el('evidenceMatrixContainer').innerHTML = `<p class="empty-msg">${esc(tr('empty.evidenceStructured'))}</p>`;
            el('evidenceHint').textContent = data.is_leaf ? tr('empty.evidenceBenchmark') : tr('evidence.option').replace(/-/g, '').trim();
        }

        // Gaps
        const gapsCard = el('evidenceGapsCard');
        if (data.gaps && data.gaps.length > 0) {
            gapsCard.style.display = '';
            el('evidenceGapsTitle').textContent = `${tr('evidence.gaps')} (${data.gaps.length})`;
            renderGaps(el('evidenceGapsBody'), data.gaps);
        } else {
            gapsCard.style.display = 'none';
        }

        // Entity-relation network
        const gs = data.graph_summary;
        const graphCard = el('evidenceGraphCard');
        if (gs && ((gs.top_entities && gs.top_entities.length) || (gs.top_relations && gs.top_relations.length))) {
            graphCard.style.display = '';
            renderEntityGraph('evidenceEntityGraph', gs, { height: 420 });
        } else {
            graphCard.style.display = 'none';
        }
    } catch (e) {
        console.error('Evidence load error:', e);
        el('evidenceHint').textContent = tr('common.errorLoadingData');
    }
}

// ── Heatmap colour scale ──────────────────────────────────────────────
// The benchmark matrix shades every filled (non-SOTA) cell by its value, so
// the table reads as a real heatmap instead of one flat fill. The two scale
// endpoints live in :root as CSS variables (--heat-lo / --heat-hi) so the
// palette stays themeable; we read them once and lerp in RGB space. SOTA cells
// keep their distinct green styling untouched — the only visible change is the
// gradient on ordinary filled cells.
function _hexToRgb(hex) {
    const h = String(hex).trim().replace('#', '');
    const s = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
    const n = parseInt(s, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function _lerpRgb(a, b, t) {
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)}, ${Math.round(a[1] + (b[1] - a[1]) * t)}, ${Math.round(a[2] + (b[2] - a[2]) * t)})`;
}
// Build a value→background-colour mapper for one metric. Computes min/max over
// the filled, non-SOTA cells of that metric in a single O(cells) pass so a
// large matrix never costs more than its size.
function buildMatrixHeat(matrix, metric) {
    const cs = getComputedStyle(document.documentElement);
    const lo = _hexToRgb((cs.getPropertyValue('--heat-lo').trim() || '#faf1ea'));
    const hi = _hexToRgb((cs.getPropertyValue('--heat-hi').trim() || '#e0a883'));
    let min = Infinity, max = -Infinity;
    const suffix = `|||${metric}`;
    for (const key in matrix.cells) {
        if (!key.endsWith(suffix)) continue;
        const c = matrix.cells[key];
        if (c.is_sota || c.value == null) continue;
        const v = Number(c.value);
        if (v < min) min = v;
        if (v > max) max = v;
    }
    const span = max - min;
    const map = (value) => {
        if (value == null) return '';
        const t = span > 0 ? (Number(value) - min) / span : 0.5;
        return _lerpRgb(lo, hi, Math.max(0, Math.min(1, t)));
    };
    // Expose the range so the legend can label the gradient endpoints.
    map.min = min;
    map.max = max;
    return map;
}

// The colour-scale legend that makes the heatmap self-explanatory: a low→high
// gradient bar tagged with the metric's actual min/max, plus the SOTA marker.
function matrixHeatLegendHtml(heat) {
    const hasRange = isFinite(heat.min) && isFinite(heat.max);
    return `<span class="matrix-legend">
        <span class="legend-label">${esc(tr('evidence.heatScale'))}</span>
        <span class="legend-min">${hasRange ? heat.min.toFixed(1) : ''}</span>
        <span class="legend-bar" aria-hidden="true"></span>
        <span class="legend-max">${hasRange ? heat.max.toFixed(1) : ''}</span>
        <span class="legend-sota"><span class="legend-sota-dot" aria-hidden="true"></span>${esc(tr('evidence.heatSota'))}</span>
    </span>`;
}

function renderMatrix(container, matrix) {
    if (!matrix.methods.length || !matrix.datasets.length) {
        container.innerHTML = `<p class="empty-msg">${esc(tr('empty.resultData'))}</p>`;
        return;
    }

    // Find metrics
    const metricCounts = {};
    for (const key of Object.keys(matrix.cells)) {
        const metric = key.split('|||')[2];
        metricCounts[metric] = (metricCounts[metric] || 0) + 1;
    }
    const metrics = Object.keys(metricCounts).sort((a, b) => metricCounts[b] - metricCounts[a]);
    const defaultMetric = metrics[0] || '';
    const heat = buildMatrixHeat(matrix, defaultMetric);

    let html = '<div class="matrix-controls">';
    html += `<label>${esc(tr('common.metric'))}</label>`;
    html += '<select class="matrix-metric-select" onchange="window._dg.updateMatrixMetric(this)">';
    for (const m of metrics) {
        html += `<option value="${esc(m)}"${m === defaultMetric ? ' selected' : ''}>${esc(m || tr('common.noneOption'))}</option>`;
    }
    html += '</select>';
    html += `<span class="matrix-info">${esc(tr('common.methodsCount', { count: matrix.methods.length }))} x ${esc(tr('common.datasetsCount', { count: matrix.datasets.length }))}</span>`;
    html += matrixHeatLegendHtml(heat);
    html += '</div>';

    html += '<div class="matrix-scroll"><table class="matrix-table">';
    html += `<thead><tr><th class="method-header">${esc(tr('common.methodDataset'))}</th>`;
    for (const ds of matrix.datasets) {
        html += `<th class="dataset-header" title="${esc(ds)}">${esc(trunc(ds, 16))}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const method of matrix.methods) {
        html += '<tr>';
        html += `<td class="method-cell" title="${esc(method)}">${esc(trunc(method, 22))}</td>`;
        for (const ds of matrix.datasets) {
            const key = `${method}|||${ds}|||${defaultMetric}`;
            const cell = matrix.cells[key];
            if (cell) {
                const cls = cell.is_sota ? 'cell-sota' : 'cell-filled';
                const val = cell.value != null ? Number(cell.value).toFixed(1) : '-';
                const bg = (!cell.is_sota && cell.value != null) ? heat(cell.value) : '';
                const style = bg ? ` style="background:${bg}"` : '';
                html += `<td class="matrix-cell ${cls}"${style} title="${esc(tr('common.onDataset', { method, dataset: ds }))}: ${val}${cell.paper_id ? ' (' + esc(cell.paper_id) + ')' : ''}">${val}</td>`;
            } else {
                html += `<td class="matrix-cell cell-empty" title="${esc(tr('common.noData'))}">-</td>`;
            }
        }
        html += '</tr>';
    }
    html += '</tbody></table></div>';

    container.innerHTML = html;
    container._matrixData = matrix;
}

function updateMatrixMetric(selectEl) {
    const container = selectEl.closest('.card') ? selectEl.closest('.card').querySelector('.matrix-wrap, [class*="matrix"]') : el('evidenceMatrixContainer');
    const matrix = container ? container._matrixData : null;
    if (!matrix) return;

    const metric = selectEl.value;
    const heat = buildMatrixHeat(matrix, metric);

    // Keep the legend's gradient endpoints in sync with the selected metric.
    const minEl = container.querySelector('.matrix-legend .legend-min');
    const maxEl = container.querySelector('.matrix-legend .legend-max');
    if (minEl && maxEl) {
        const hasRange = isFinite(heat.min) && isFinite(heat.max);
        minEl.textContent = hasRange ? heat.min.toFixed(1) : '';
        maxEl.textContent = hasRange ? heat.max.toFixed(1) : '';
    }

    const rows = container.querySelectorAll('tbody tr');

    rows.forEach((row, mi) => {
        const method = matrix.methods[mi];
        const cells = row.querySelectorAll('td.matrix-cell');
        cells.forEach((td, di) => {
            const ds = matrix.datasets[di];
            const key = `${method}|||${ds}|||${metric}`;
            const cell = matrix.cells[key];
            if (cell) {
                const val = cell.value != null ? Number(cell.value).toFixed(1) : '-';
                td.textContent = val;
                td.className = 'matrix-cell ' + (cell.is_sota ? 'cell-sota' : 'cell-filled');
                td.style.background = (!cell.is_sota && cell.value != null) ? heat(cell.value) : '';
                td.title = `${tr('common.onDataset', { method, dataset: ds })}: ${val}`;
            } else {
                td.textContent = '-';
                td.className = 'matrix-cell cell-empty';
                td.style.background = '';
                td.title = tr('common.noData');
            }
        });
    });
}

function renderGaps(container, gaps) {
    container.innerHTML = `<div class="gaps-list">${gaps.map(g => `
        <div class="gap-item">
            <span class="score">${g.value_score || '?'}/5</span>
            <strong>${esc(g.method_name || '')} on ${esc(g.dataset_name || '')}</strong>
            <div class="gap-desc">${esc(g.gap_description || '')}</div>
            ${g.research_proposal ? `<div class="proposal">${esc(trunc(g.research_proposal, 200))}</div>` : ''}
            ${g.why_valuable ? `<div class="gap-why">${esc(g.why_valuable)}</div>` : ''}
        </div>
    `).join('')}</div>`;
}

// ── Papers Tab ───────────────────────────────────────────────────────

async function loadPapers() {
    papersLoaded = true;
    try {
        allPapers = await api('/api/papers?limit=200');
        renderPapers();
    } catch (e) {
        console.error('Papers load error:', e);
    }
}

function renderPapers() {
    const list = el('papersList');
    const query = (el('papersSearch').value || '').toLowerCase();
    const status = el('papersStatus').value;

    let filtered = allPapers;
    if (query) {
        filtered = filtered.filter(p =>
            (p.title || '').toLowerCase().includes(query) ||
            (p.id || '').toLowerCase().includes(query)
        );
    }
    if (status) {
        filtered = filtered.filter(p => p.status === status);
    }

    if (filtered.length === 0) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.papers'))}</p>`;
        return;
    }

    renderListChunked(list, filtered, p => {
        const sc = p.status ? 's-' + p.status : '';
        return `<div class="paper-row" data-paper-id="${esc(p.id)}" onclick="window._dg.togglePaper(this)">
            <div class="paper-row-top">
                <a class="paper-link" href="https://arxiv.org/abs/${esc(p.id)}" target="_blank" onclick="event.stopPropagation();">${esc(p.id)}</a>
                <span class="paper-title">${esc(trunc(p.title, 100))}</span>
                <span class="paper-date">${esc(p.created_at || '')}</span>
                <span class="paper-status ${sc}">${esc(p.status || '')}</span>
            </div>
            <div class="paper-expanded-body">
                <div class="paper-detail-loading">${esc(tr('common.loadingDetails'))}</div>
            </div>
        </div>`;
    });
}

// ── Paper Progress Tabs ─────────────────────────────────────────────

function statusBadge(label, tone = 'dim') {
    const cls = {
        green: 'badge-green',
        red: 'badge-red',
        gold: 'badge-gold',
        accent: 'badge-accent',
        dim: 'badge-dim',
    }[tone] || 'badge-dim';
    return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function toneForPaperStage(stage) {
    const key = String(stage || '').toLowerCase();
    if (key.includes('reasoned') || key.includes('done') || key.includes('bundle_ready')) return 'green';
    if (key.includes('error') || key.includes('failed') || key.includes('blocked')) return 'red';
    if (key.includes('research') || key.includes('writing') || key.includes('verify')) return 'accent';
    if (key.includes('experiment') || key.includes('gpu') || key.includes('review')) return 'gold';
    return 'dim';
}

function paperPreviewHref(insightId, kind = 'index') {
    if (!insightId) return '';
    if (kind === 'pdf') return `/papers/${insightId}/pdf`;
    if (kind === 'tex') return `/papers/${insightId}/tex`;
    return `/papers/${insightId}`;
}

function renderMiniStatGrid(targetId, items) {
    const root = el(targetId);
    if (!root) return;
    root.innerHTML = items.map(item => `
        <div class="stat-card stat-card-mini">
            <div class="stat-number">${esc(String(item.value ?? 0))}</div>
            <div class="stat-label">${esc(item.label)}</div>
        </div>
    `).join('');
}

function renderPaperPipelineRows(rows) {
    const list = el('paperProgressPipelineList');
    const count = el('paperProgressPipelineCount');
    if (!list || !count) return;
    const papers = (rows || []).slice(0, 20);
    count.textContent = papers.length;
    if (!papers.length) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.paperProgress'))}</p>`;
        return;
    }
    renderListChunked(list, papers, item => `
        <div class="paper-flow-item">
            <div class="paper-flow-head">
                <div class="paper-flow-title">${esc(trunc(item.title || item.id || tr('common.untitledPaper'), 120))}</div>
                ${statusBadge(item.processing_stage || item.status || 'queued', toneForPaperStage(item.processing_stage || item.status))}
            </div>
            <div class="paper-flow-meta">
                <span>${esc(item.id || '-')}</span>
                <span>${esc(item.status || 'unknown')}</span>
                ${item.updated_at ? `<span>${esc(timeAgo(item.updated_at))}</span>` : ''}
            </div>
            ${item.stage_last_error ? `<div class="paper-flow-note paper-flow-error">${esc(trunc(item.stage_last_error, 240))}</div>` : ''}
        </div>
    `);
}

function renderPaperGenerationRows(jobs, manuscripts) {
    const list = el('paperProgressGenerationList');
    const count = el('paperProgressGenerationCount');
    if (!list || !count) return;

    const activeJobs = (jobs || []).filter(job => !['completed', 'failed'].includes(String(job.status || '').toLowerCase()));
    const activeManuscripts = (manuscripts || []).filter(row => {
        const status = String(row.status || '').toLowerCase();
        return status && !['stale', 'completed', 'ready'].includes(status);
    });
    const total = activeJobs.length + activeManuscripts.length;
    count.textContent = total;

    if (!total) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.paperGeneration'))}</p>`;
        return;
    }

    const jobCards = activeJobs.map(job => {
        const stage = friendlyAutomationStage(job.status, job.stage);
        const previewUrl = paperPreviewHref(job.deep_insight_id, 'index');
        return `
            <div class="paper-flow-item">
                <div class="paper-flow-head">
                <div class="paper-flow-title">${esc(trunc(job.title || tr('label.discoveryId', { id: job.deep_insight_id }), 120))}</div>
                    ${statusBadge(stage, toneForPaperStage(job.stage || job.status))}
                </div>
                <div class="paper-flow-meta">
                    <span>${esc(tr('label.discoveryId', { id: job.deep_insight_id || '-' }))}</span>
                    ${job.experiment_status ? `<span>${esc(tr('label.experimentStatus', { status: job.experiment_status }))}</span>` : ''}
                    ${job.updated_at ? `<span>${esc(timeAgo(job.updated_at))}</span>` : ''}
                </div>
                ${job.last_note ? `<div class="paper-flow-note">${esc(trunc(job.last_note, 220))}</div>` : ''}
                ${job.last_error ? `<div class="paper-flow-note paper-flow-error">${esc(trunc(job.last_error, 220))}</div>` : ''}
                <div class="paper-flow-actions">
                    <button class="btn-preview" onclick="window._dg.viewPaperGeneration(${Number(job.deep_insight_id)})">${esc(tr('common.viewDetails'))}</button>
                    ${previewUrl ? `<button class="btn-preview" onclick="window.open('${esc(previewUrl)}','_blank')">${esc(tr('label.openPaperPreview'))}</button>` : ''}
                </div>
            </div>
        `;
    });

    const manuscriptCards = activeManuscripts.map(row => `
        <div class="paper-flow-item">
            <div class="paper-flow-head">
                <div class="paper-flow-title">${esc(trunc(row.insight_title || tr('label.manuscriptId', { id: row.id }), 120))}</div>
                ${statusBadge(row.status || 'drafting', toneForPaperStage(row.status))}
            </div>
            <div class="paper-flow-meta">
                <span>${esc(tr('label.manuscriptId', { id: row.id || '-' }))}</span>
                ${row.hypothesis_verdict ? `<span>${esc(tr('label.verdict', { verdict: row.hypothesis_verdict }))}</span>` : ''}
                ${row.updated_at ? `<span>${esc(timeAgo(row.updated_at))}</span>` : ''}
            </div>
            ${row.workdir ? `<div class="paper-flow-note">${esc(trunc(row.workdir, 220))}</div>` : ''}
            <div class="paper-flow-actions">
                ${row.deep_insight_id ? `<button class="btn-preview" onclick="window.open('${esc(paperPreviewHref(row.deep_insight_id, 'index'))}','_blank')">${esc(tr('label.openPaperPreview'))}</button>` : ''}
            </div>
        </div>
    `);

    list.innerHTML = jobCards.concat(manuscriptCards).join('');
}

async function loadPaperProgressTab() {
    paperProgressLoaded = true;
    try {
        const [automation, jobs, manuscripts] = await Promise.all([
            api('/api/automation'),
            api('/api/auto_research/jobs?limit=50'),
            api('/api/manuscripts?limit=50'),
        ]);
        const current = (automation || {}).current_work || {};
        const activeManuscripts = (manuscripts || []).filter(row => {
            const status = String(row.status || '').toLowerCase();
            return status && !['stale', 'completed', 'ready'].includes(status);
        });
        renderMiniStatGrid('paperProgressStats', [
            { label: tr('paperProgress.stat.sourcePapers'), value: (current.papers || []).length },
            { label: tr('paperProgress.stat.paperJobs'), value: (jobs || []).filter(job => !['completed', 'failed'].includes(String(job.status || '').toLowerCase())).length },
            { label: tr('paperProgress.stat.activeManuscripts'), value: activeManuscripts.length },
            { label: tr('paperProgress.stat.blockedJobs'), value: ((automation || {}).auto_research || {}).blocked || 0 },
        ]);
        renderPaperPipelineRows(current.papers || []);
        renderPaperGenerationRows(jobs || [], manuscripts || []);
    } catch (e) {
        paperProgressLoaded = false;
        const listA = el('paperProgressPipelineList');
        const listB = el('paperProgressGenerationList');
        if (listA) listA.innerHTML = `<p class="empty-msg">${esc(tr('common.failedToLoad', { message: e.message }))}</p>`;
        if (listB) listB.innerHTML = `<p class="empty-msg">${esc(tr('common.failedToLoad', { message: e.message }))}</p>`;
    }
}

function manuscriptTone(status) {
    const key = String(status || '').toLowerCase();
    if (key === 'bundle_ready' || key === 'ready') return 'green';
    if (key === 'stale' || key === 'failed' || key === 'blocked') return 'red';
    if (key.includes('draft')) return 'accent';
    return 'dim';
}

function renderGeneratedPapers(manuscripts) {
    const list = el('generatedPapersList');
    const count = el('generatedPapersCount');
    if (!list || !count) return;
    const rows = (manuscripts || []).slice(0, 100);
    count.textContent = rows.length;
    if (!rows.length) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.generated'))}</p>`;
        return;
    }
    renderListChunked(list, rows, row => {
        const preview = row.deep_insight_id ? paperPreviewHref(row.deep_insight_id, 'index') : '';
        return `
            <div class="paper-flow-item">
                <div class="paper-flow-head">
                    <div class="paper-flow-title">${esc(trunc(row.insight_title || tr('label.manuscriptId', { id: row.id }), 130))}</div>
                    ${statusBadge(row.status || 'generated', manuscriptTone(row.status))}
                </div>
                <div class="paper-flow-meta">
                    <span>${esc(tr('label.manuscriptId', { id: row.id || '-' }))}</span>
                    ${row.experiment_run_id ? `<span>${esc(tr('label.experimentRun', { id: row.experiment_run_id }))}</span>` : ''}
                    ${row.hypothesis_verdict ? `<span>${esc(tr('label.verdict', { verdict: row.hypothesis_verdict }))}</span>` : ''}
                    ${row.updated_at ? `<span>${esc(timeAgo(row.updated_at))}</span>` : ''}
                </div>
                <div class="paper-flow-meta">
                    ${row.created_at ? `<span>${esc(tr('common.generated', { time: fmtDateTime(row.created_at) }))}</span>` : ''}
                    ${row.updated_at ? `<span>${esc(tr('common.updated', { time: fmtDateTime(row.updated_at) }))}</span>` : ''}
                </div>
                ${row.workdir ? `<div class="paper-flow-note">${esc(trunc(row.workdir, 220))}</div>` : ''}
                <div class="paper-flow-note">${esc(tr('label.usePaperPage'))}</div>
                <div class="paper-flow-actions">
                    ${preview ? `<button class="btn-preview" onclick="window.open('${esc(preview)}','_blank')">${esc(tr('label.openPaperPage'))}</button>` : ''}
                </div>
            </div>
        `;
    });
}

async function loadGeneratedPapersTab() {
    generatedPapersLoaded = true;
    try {
        const manuscripts = await api('/api/manuscripts?limit=100');
        const counts = (manuscripts || []).reduce((acc, row) => {
            const key = String(row.status || 'unknown').toLowerCase();
            acc.total += 1;
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, { total: 0 });
        renderMiniStatGrid('generatedPapersStats', [
            { label: tr('generated.stat.total'), value: counts.total || 0 },
            { label: tr('generated.stat.ready'), value: counts.ready || counts.bundle_ready || 0 },
            { label: tr('generated.stat.stale'), value: counts.stale || 0 },
            { label: tr('generated.stat.drafting'), value: counts.drafting || 0 },
        ]);
        renderGeneratedPapers(manuscripts || []);
    } catch (e) {
        generatedPapersLoaded = false;
        const list = el('generatedPapersList');
        if (list) list.innerHTML = `<p class="empty-msg">${esc(tr('common.failedToLoad', { message: e.message }))}</p>`;
    }
}

async function togglePaper(rowEl) {
    const isExpanded = rowEl.classList.contains('expanded');

    // Collapse all others
    $$('.paper-row.expanded').forEach(r => {
        if (r !== rowEl) r.classList.remove('expanded');
    });

    if (isExpanded) {
        rowEl.classList.remove('expanded');
        return;
    }

    rowEl.classList.add('expanded');
    const pid = rowEl.dataset.paperId;
    const body = rowEl.querySelector('.paper-expanded-body');

    try {
        const [claims, results] = await Promise.all([
            api(`/api/claims?paper_id=${encodeURIComponent(pid)}`),
            api(`/api/results?paper_id=${encodeURIComponent(pid)}&limit=20`)
        ]);

        let html = '';
        if (claims.length > 0) {
            html += `<div class="paper-claims"><strong style="color:var(--accent);font-size:0.75rem;">${esc(tr('label.claim'))}</strong>`;
            for (const c of claims.slice(0, 8)) {
                html += `<div class="paper-claim-item">${esc(c.claim_text || c.claim_type || '')}</div>`;
            }
            html += '</div>';
        }
        if (results.length > 0) {
            html += `<div class="paper-claims" style="margin-top:10px;"><strong style="color:var(--gold);font-size:0.75rem;">${esc(tr('label.results'))}</strong>`;
            for (const r of results.slice(0, 8)) {
                const val = r.metric_value != null ? Number(r.metric_value).toFixed(2) : '';
                html += `<div class="paper-claim-item">${esc(tr('common.onDataset', { method: r.method_name || '', dataset: r.dataset_name || '' }))}${val ? ': <b>' + val + '</b>' : ''} ${esc(r.metric_name || '')}</div>`;
            }
            html += '</div>';
        }
        if (!html) html = `<p class="empty-msg" style="padding:8px;">${esc(tr('empty.claimsResults'))}</p>`;
        body.innerHTML = html;
    } catch (e) {
        body.innerHTML = `<p class="empty-msg" style="padding:8px;">${esc(tr('common.failedToLoadDetails'))}</p>`;
    }
}

// ── Opportunities Tab ────────────────────────────────────────────────

async function loadInsightsTab() {
    insightsLoaded = true;
    const typeFilter = el('insightTypeFilter')?.value || '';
    const sortFilter = el('insightSortFilter')?.value || 'score';
    try {
        let url = `/api/insights?limit=200`;
        if (typeFilter) url += `&type=${typeFilter}`;
        const insights = await api(url);

        // Sort
        if (sortFilter === 'paradigm') insights.sort((a, b) => (b.paradigm_score || 0) - (a.paradigm_score || 0));
        else if (sortFilter === 'novelty') insights.sort((a, b) => b.novelty_score - a.novelty_score);
        else if (sortFilter === 'feasibility') insights.sort((a, b) => b.feasibility_score - a.feasibility_score);
        else if (sortFilter === 'recent') insights.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
        // default: by combined score (already sorted by API)

        const typeColors = {
            contradiction_analysis: '#c4453a',
            method_transfer: '#2e86ab',
            assumption_challenge: '#a8842a',
            ignored_limitation: '#7c5cbf',
            paradigm_exhaustion: '#9a9088',
            cross_domain_bridge: '#44aa88',
        };

        const list = el('insightsList');
        if (!insights.length) {
            list.innerHTML = `<p class="empty-msg">${esc(tr('empty.researchInsights'))}</p>`;
            return;
        }

        renderListChunked(list, insights, ins => {
            const color = typeColors[ins.insight_type] || '#888';
            let papers = [];
            try { papers = JSON.parse(ins.supporting_papers || '[]'); } catch(e) {}
            const paperLinks = papers.map(pid =>
                `<a class="paper-cite" href="https://arxiv.org/abs/${esc(pid)}" target="_blank">${esc(pid)}</a>`
            ).join(' ');

            const ps = ins.paradigm_score || 0;
            const psBadge = ps >= 7 ? `<span class="paradigm-badge high">P:${ps.toFixed(1)}</span>`
                          : ps >= 5 ? `<span class="paradigm-badge mid">P:${ps.toFixed(1)}</span>`
                          : ps > 0  ? `<span class="paradigm-badge low">P:${ps.toFixed(1)}</span>` : '';

            return `<div class="insight-card" style="border-left: 3px solid ${color};">
                <div class="insight-header">
                    <span class="insight-type" style="color:${color};">${esc((ins.insight_type || '').replace(/_/g, ' '))}</span>
                    ${psBadge}
                    <span class="insight-scores">N:${ins.novelty_score}/5 F:${ins.feasibility_score}/5</span>
                    <span class="insight-area" onclick="window._dg.exploreNode('${esc(ins.node_id)}')" style="cursor:pointer;color:#6a7a8a;font-size:0.68rem;">${esc(ins.node_id)}</span>
                </div>
                <div class="insight-title">${esc(ins.title)}</div>
                ${ins.rank_rationale ? `<div class="insight-rationale">${esc(ins.rank_rationale)}</div>` : ''}
                ${paperLinks ? `<div class="insight-papers">${paperLinks}</div>` : ''}
                <div class="insight-hypothesis"><span class="insight-label">${esc(tr('label.hypothesis'))}</span> ${esc(ins.hypothesis)}</div>
                <div class="insight-experiment"><span class="insight-label">${esc(tr('label.experiment'))}</span> ${esc(ins.experiment)}</div>
                ${ins.impact ? `<div class="insight-impact"><span class="insight-label">${esc(tr('label.impact'))}</span> ${esc(ins.impact)}</div>` : ''}
                <div class="insight-actions">
                    <button class="btn-preview" onclick="window._dg.previewProposal(${ins.id})">${esc(tr('label.previewProposal'))}</button>
                </div>
            </div>`;
        });
    } catch (e) {
        insightsLoaded = false;
        console.error('Insights tab error:', e);
    }
}

// ── Discoveries Tab (Tier 1 + Tier 2) ────────────────────────────────

async function loadDiscoveriesTab() {
    discoveriesLoaded = true;
    const tierFilter = el('discoveryTierFilter')?.value || '';
    try {
        let url = '/api/deep_insights?limit=50';
        if (tierFilter) url += `&tier=${tierFilter}`;
        const insights = await api(url);
        renderDiscoveries(insights);
    } catch (e) {
        discoveriesLoaded = false;
        const list = el('discoveriesList');
        if (list) list.innerHTML = `<p class="empty-msg">${esc(tr('empty.discoveries'))}</p>`;
    }
}

function isDisplayableDiscovery(d) {
    const title = String(d.title || '').trim().toLowerCase();
    if (title === 'mechanism-first insight' || title === 'mechanism first insight') return false;
    return Boolean(
        d.formal_structure || d.transformation || d.problem_statement ||
        d.proposed_method || d.experimental_plan || d.evidence_summary
    );
}

function renderDiscoveries(discoveries) {
    const list = el('discoveriesList');
    const visible = (discoveries || []).filter(isDisplayableDiscovery);
    if (visible.length === 0) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.discoveries'))}</p>`;
        return;
    }

    renderListChunked(list, visible, d => {
        const isTier1 = d.tier === 1;
        const tierColor = isTier1 ? '#c4453a' : '#2e86ab';
        const tierLabel = isTier1 ? tr('discoveries.tier1') : tr('discoveries.tier2');

        const noveltyBadge = d.novelty_status === 'novel'
            ? `<span class="paradigm-badge high">${esc(tr('status.novelty.novel'))}</span>`
            : d.novelty_status === 'partially_exists'
            ? `<span class="paradigm-badge mid">${esc(tr('status.novelty.partial'))}</span>`
            : d.novelty_status === 'exists'
            ? `<span class="paradigm-badge low">${esc(tr('status.novelty.exists'))}</span>`
            : `<span class="paradigm-badge low">${esc(tr('status.novelty.unchecked'))}</span>`;

        const scoreBadge = d.adversarial_score
            ? `<span class="insight-scores">${esc(tr('label.adversarial', { score: d.adversarial_score }))}</span>`
            : '';

        let bodyHtml = '';

        if (isTier1) {
            bodyHtml += d.formal_structure
                ? `<div class="insight-hypothesis"><span class="insight-label">${esc(tr('label.formalStructure'))}</span> ${esc(d.formal_structure)}</div>` : '';
            bodyHtml += d.transformation
                ? `<div class="insight-experiment"><span class="insight-label">${esc(tr('label.transformation'))}</span> ${esc(d.transformation)}</div>` : '';

            let fieldA = {}, fieldB = {};
            try { fieldA = JSON.parse(d.field_a || '{}'); } catch(e) {}
            try { fieldB = JSON.parse(d.field_b || '{}'); } catch(e) {}
            if (fieldA.node_id || fieldB.node_id) {
                bodyHtml += `<div class="insight-evidence">
                    <span class="insight-label">${esc(tr('label.fields'))}</span>
                    ${fieldA.node_id ? `<span class="chip" onclick="window._dg.exploreNode('${esc(fieldA.node_id)}')">${esc(fieldA.node_id)}</span>` : ''}
                    <span style="margin:0 4px;">-&gt;</span>
                    ${fieldB.node_id ? `<span class="chip" onclick="window._dg.exploreNode('${esc(fieldB.node_id)}')">${esc(fieldB.node_id)}</span>` : ''}
                </div>`;
            }

            let predictions = [];
            try { predictions = JSON.parse(d.predictions || '[]'); } catch(e) {}
            if (predictions.length) {
                bodyHtml += `<div class="insight-experiment"><span class="insight-label">${esc(tr('label.predictions'))}</span><ul style="margin:4px 0;padding-left:20px;">`;
                for (const p of predictions.slice(0, 3)) {
                    const stmt = typeof p === 'string' ? p : (p.statement || '');
                    bodyHtml += `<li>${esc(stmt)}</li>`;
                }
                bodyHtml += '</ul></div>';
            }

            if (d.adversarial_critique) {
                let critique = {};
                try { critique = JSON.parse(d.adversarial_critique); } catch(e) {}
                if (critique.strongest_attack) {
                    bodyHtml += `<div class="insight-impact"><span class="insight-label">${esc(tr('label.strongestChallenge'))}</span> ${esc(critique.strongest_attack)}</div>`;
                }
            }
        } else {
            bodyHtml += d.problem_statement
                ? `<div class="insight-hypothesis"><span class="insight-label">${esc(tr('label.problem'))}</span> ${esc(d.problem_statement)}</div>` : '';
            bodyHtml += d.existing_weakness
                ? `<div class="insight-evidence"><span class="insight-label">${esc(tr('label.weakness'))}</span> ${esc(d.existing_weakness)}</div>` : '';

            let method = {};
            try { method = JSON.parse(d.proposed_method || '{}'); } catch(e) {}
            if (method.name) {
                bodyHtml += `<div class="insight-experiment">
                    <span class="insight-label">${esc(tr('label.method'))} ${esc(method.name)}</span> (${esc(method.type || '?')})
                    <div style="margin-top:4px;">${esc(method.one_line || '')}</div>
                    ${method.definition ? `<pre style="font-size:0.72rem;margin:6px 0;white-space:pre-wrap;color:var(--text-secondary);">${esc(trunc(method.definition, 300))}</pre>` : ''}
                </div>`;
            }

            let plan = {};
            try { plan = JSON.parse(d.experimental_plan || '{}'); } catch(e) {}
            if (plan.baselines && plan.baselines.length) {
                bodyHtml += `<div class="insight-impact"><span class="insight-label">${esc(tr('label.baselines'))}</span> `;
                bodyHtml += plan.baselines.map(b => esc(b.name || b)).join(', ');
                bodyHtml += '</div>';
            }
            if (plan.datasets && plan.datasets.length) {
                bodyHtml += `<div class="insight-impact"><span class="insight-label">${esc(tr('label.datasets'))}</span> `;
                bodyHtml += plan.datasets.map(ds => esc(ds.name || ds)).join(', ');
                bodyHtml += '</div>';
            }
            if (plan.compute_budget) {
                bodyHtml += `<div class="insight-impact"><span class="insight-label">${esc(tr('label.compute'))}</span> ${esc(plan.compute_budget.total_gpu_hours || '?')} GPU-hours</div>`;
            }
        }

        return `<div class="insight-card" style="border-left: 3px solid ${tierColor};">
            <div class="insight-header">
                <span class="insight-type" style="color:${tierColor};font-weight:700;">${esc(tierLabel)}</span>
                ${noveltyBadge}
                ${scoreBadge}
            </div>
            <div class="insight-title">${esc(d.title)}</div>
            ${bodyHtml}
            ${d.evidence_summary ? `<div class="insight-evidence"><span class="insight-label">${esc(tr('label.evidence'))}</span> ${esc(trunc(d.evidence_summary, 250))}</div>` : ''}
            <div class="insight-impact"><span class="insight-label">${esc(tr('label.mode'))}</span> ${esc(tr('label.fixedAutomaticPipeline'))}</div>
        </div>`;
    });
}

// ── Experiments Tab ───────────────────────────────────────────────────

async function loadExperimentsTab() {
    experimentsLoaded = true;
    const statusFilter = el('experimentStatusFilter')?.value || '';
    try {
        const automation = await api('/api/automation');
        renderAutomationOverview(automation);
        renderAutoResearchStatus(automation.auto_research);

        const autoJobs = await api('/api/auto_research/jobs?limit=30');
        renderAutoResearchJobs(autoJobs);

        let url = '/api/experiment_groups?limit=50';
        if (statusFilter) url += `&status=${statusFilter}`;
        const groups = await api(url);
        renderExperimentGroupsV2(groups);

        const meta = await api('/api/meta_report');
        renderMetaReport(meta);
    } catch (e) {
        experimentsLoaded = false;
        const list = el('experimentsList');
        if (list) list.innerHTML = `<p class="empty-msg">Automation status failed to load: ${esc(e.message)}</p>`;
    }
}

function serviceState(name, ok, active) {
    if (ok === false) return { label: tr('service.state.missing'), color: '#c4453a' };
    if (active) return { label: tr('service.state.active'), color: '#3d8b5e' };
    return { label: tr('service.state.ready'), color: '#a8842a' };
}

function serviceCard(title, state, detail) {
    return `<div class="service-card">
        <div class="service-card-top">
            <div class="service-title">${esc(title)}</div>
            <div class="service-state" style="color:${state.color};">${esc(state.label)}</div>
        </div>
        <div class="service-detail">${detail}</div>
    </div>`;
}

function renderAutomationOverview(snapshot) {
    const grid = el('automationServicesGrid');
    const work = el('currentWorkGrid');
    if (!grid || !work || !snapshot) return;

    const paper = snapshot.paper_worker || {};
    const auto = snapshot.auto_research || {};
    const evo = snapshot.evoscientist || {};
    const po = snapshot.paperorchestra || {};
    const gpu = snapshot.gpu_scheduler || {};
    const current = snapshot.current_work || {};

    grid.innerHTML = [
        serviceCard(
            tr('experiments.service.paperPipeline'),
            serviceState('paper', true, paper.running),
            esc(tr('service.detail.paper', { batch: paper.batch_size || '?', status: paper.status || 'idle' }))
        ),
        serviceCard(
            tr('experiments.service.autoResearch'),
            serviceState('auto', true, auto.running),
            esc(tr('service.detail.auto', { jobs: auto.total || 0, experiments: auto.running_experiment || 0, blocked: auto.blocked || 0 }))
        ),
        serviceCard(
            tr('experiments.service.evoScientist'),
            serviceState('evoscientist', evo.available, (evo.active_count || 0) > 0),
            esc(tr('service.detail.evo', { count: evo.active_count || 0 }))
        ),
        serviceCard(
            tr('experiments.service.paperOrchestra'),
            serviceState('paperorchestra', po.available, (po.active_count || 0) > 0),
            esc(tr('service.detail.paperOrchestra', { bundles: (po.counts || {}).bundle_ready || 0, drafting: (po.counts || {}).drafting || 0 }))
        ),
        serviceCard(
            tr('experiments.service.gpuScheduler'),
            serviceState('gpu', true, (gpu.running_jobs || 0) > 0),
            esc(tr('service.detail.gpu', { running: gpu.running_jobs || 0, queued: gpu.queued_jobs || 0, workers: (gpu.workers || []).length }))
        ),
    ].join('');

    work.innerHTML = [
        workLane(tr('experiments.work.pipeline'), current.pipeline, item =>
            `${esc(item.status || '')} / ${esc(item.stage || '')}`, item => item.title),
        workLane(tr('experiments.work.processingPapers'), current.papers, item =>
            `${esc(item.id || '')} · ${esc(item.processing_stage || item.status || '')}`, item => item.title),
        workLane(tr('experiments.work.experimentPlans'), current.experiment_plans, item =>
            `${esc(item.status || '')} / ${esc(item.stage || '')}`, item => item.title),
        workLane(tr('experiments.work.running'), current.experiments, item =>
            `${esc(tr('label.experimentRun', { id: item.id || '' }))} · ${esc(item.status || '')} · ${esc(item.phase || '')}`, item => item.title),
        workLane(tr('experiments.work.writing'), current.manuscripts, item =>
            `${esc(tr('label.manuscriptId', { id: item.id || '' }))} · ${esc(item.status || '')}`, item => item.title),
    ].join('');
}

function workLane(title, items, metaFn, titleFn) {
    const rows = (items || []).slice(0, 4);
    const body = rows.length
        ? rows.map(item => `<div class="work-item">
            <div class="work-item-title">${esc(trunc(titleFn(item) || tr('common.untitled'), 80))}</div>
            <div>${metaFn(item)}</div>
            ${item.last_note ? `<div>${esc(trunc(item.last_note, 110))}</div>` : ''}
            ${item.last_error ? `<div style="color:#c4453a;">${esc(trunc(item.last_error, 110))}</div>` : ''}
        </div>`).join('')
        : `<div class="work-item">${esc(tr('overview.idle'))}</div>`;
    return `<div class="work-lane"><div class="work-lane-title">${esc(title)}</div>${body}</div>`;
}

function renderAutoResearchStatus(status) {
    const box = el('autoResearchStatus');
    if (!box) return;
    if (!status) {
        box.textContent = tr('status.auto.unavailable');
        return;
    }
    const running = status.running ? tr('status.auto.running') : tr('status.auto.stopped');
    const evo = status.evoscientist_available ? tr('status.evo.ready') : tr('status.evo.missing');
    box.innerHTML = `
        <strong>${running}</strong>
        <span style="margin-left:10px;">${esc(tr('status.auto.interval', { seconds: status.interval_seconds || '?' }))}</span>
        <span style="margin-left:10px;">${esc(evo)}</span>
        <span style="margin-left:10px;">${esc(tr('status.auto.jobs', { count: status.total || 0 }))}</span>
        <span style="margin-left:10px;">${esc(tr('status.auto.completed', { count: status.completed || 0 }))}</span>
        <span style="margin-left:10px;">${esc(tr('status.auto.blocked', { count: status.blocked || 0 }))}</span>
    `;
}

function friendlyAutomationStage(status, stage) {
    const key = String(stage || status || '').toLowerCase();
    if (key.includes('verification')) return tr('status.stage.checkingNovelty');
    if (key.includes('research')) return tr('status.stage.runningResearch');
    if (key.includes('review') || key.includes('formal')) return tr('status.stage.generatingPlan');
    if (key.includes('gpu')) return tr('status.stage.runningGpu');
    if (key.includes('validation') || key.includes('experiment')) return tr('status.stage.runningExperiment');
    if (key.includes('writing') || key.includes('submission') || key.includes('bundle')) return tr('status.stage.writingPaper');
    if (key.includes('blocked')) return tr('status.stage.blocked');
    if (key.includes('failed')) return tr('status.stage.failed');
    if (key.includes('complete')) return tr('status.stage.complete');
    return stage || status || tr('status.stage.queued');
}

function renderAutoResearchJobs(jobs) {
    const list = el('autoResearchList');
    if (!list) return;
    if (!jobs || !jobs.length) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.autoResearch'))}</p>`;
        return;
    }

    const colors = {
        queued: '#9a9088',
        verifying: '#2e86ab',
        researching: '#7a5ea8',
        eligible: '#a8842a',
        running_experiment: '#c4704b',
        completed: '#3d8b5e',
        blocked: '#8b5e3c',
        failed: '#c4453a',
    };

    renderListChunked(list, jobs, j => {
        const color = colors[j.status] || '#888';
        const cpu = j.cpu_eligible == null
            ? tr('status.cpu.unchecked')
            : (j.cpu_eligible ? tr('status.cpu.eligible') : tr('status.cpu.blocked'));
        const exp = j.experiment_status
            ? `<span class="insight-scores">${esc(tr('label.experimentStatus', { status: j.experiment_status }))}</span>`
            : '';
        const verdict = j.hypothesis_verdict
            ? `<span class="insight-scores">${esc(tr('label.verdict', { verdict: j.hypothesis_verdict }))}</span>`
            : '';
        const friendly = friendlyAutomationStage(j.status, j.stage);
        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">${esc(friendly).toUpperCase()}</span>
                <span class="insight-scores">${esc(cpu)}</span>
                ${exp}
                ${verdict}
            </div>
            <div class="insight-title">${esc(j.title || tr('label.discovery'))}</div>
            <div class="insight-impact"><span class="insight-label">${esc(tr('label.internalStage'))}</span> ${esc(j.stage || '')}</div>
            ${j.novelty_status ? `<div class="insight-impact"><span class="insight-label">${esc(tr('label.novelty'))}</span> ${esc(j.novelty_status)}</div>` : ''}
            ${j.cpu_reason ? `<div class="insight-evidence"><span class="insight-label">${esc(tr('label.cpuCheck'))}</span> ${esc(j.cpu_reason)}</div>` : ''}
            ${j.last_note ? `<div class="insight-experiment"><span class="insight-label">${esc(tr('common.latest'))}</span> ${esc(trunc(j.last_note, 220))}</div>` : ''}
            ${j.last_error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">${esc(tr('common.error'))}</span> ${esc(trunc(j.last_error, 220))}</div>` : ''}
        </div>`;
    });
}

function renderExperiments(runs) {
    const list = el('experimentsList');
    if (!runs || !runs.length) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.experiments'))}</p>`;
        return;
    }

    const statusColors = {
        pending: '#9a9088', scaffolding: '#a8842a', reproducing: '#2e86ab',
        testing: '#c4704b', completed: '#3d8b5e', failed: '#c4453a'
    };
    const verdictColors = {
        confirmed: '#3d8b5e', refuted: '#c4453a', inconclusive: '#a8842a'
    };
    renderListChunked(list, runs, r => {
        const color = statusColors[r.status] || '#888';
        const vColor = verdictColors[r.hypothesis_verdict] || '#888';

        const effect = r.effect_pct != null ? `${r.effect_pct >= 0 ? '+' : ''}${r.effect_pct.toFixed(2)}%` : '';
        const verdict = r.hypothesis_verdict
            ? `<span style="color:${vColor};font-weight:700;text-transform:uppercase;">${r.hypothesis_verdict}</span>`
            : '';

        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">${esc(tr('label.experimentRunStatus', { id: r.id, status: r.status }))}</span>
                ${verdict}
                ${effect ? `<span class="insight-scores">${esc(tr('label.effect', { effect }))}</span>` : ''}
                <span style="color:var(--text-dim);font-size:0.68rem;">${esc(tr('label.tier', { tier: r.insight_tier || '?' }))}</span>
            </div>
            <div class="insight-title">${esc(r.insight_title || tr('experiments.run'))}</div>
            <div style="display:flex;gap:16px;margin:6px 0;font-size:0.75rem;color:var(--text-secondary);">
                <span>${esc(tr('label.iterations', { total: r.iterations_total || 0, kept: r.iterations_kept || 0 }))}</span>
                <span>${esc(tr('label.baseline', { value: r.baseline_metric_value != null ? r.baseline_metric_value.toFixed(4) : '?' }))}</span>
                <span>${esc(tr('label.best', { value: r.best_metric_value != null ? r.best_metric_value.toFixed(4) : '?' }))}</span>
            </div>
            ${r.codebase_url && r.codebase_url !== 'scratch' ? `<div style="font-size:0.7rem;color:var(--text-dim);">${esc(tr('label.repo', { repo: r.codebase_url }))}</div>` : ''}
            <div class="insight-actions">
                <button class="btn-preview" onclick="window._dg.viewExperiment(${r.id})">${esc(tr('common.viewDetails'))}</button>
            </div>
        </div>`;
    });
}

function experimentStatusColor(status) {
    return {
        pending: '#9a9088',
        scaffolding: '#a8842a',
        reproducing: '#2e86ab',
        testing: '#c4704b',
        completed: '#3d8b5e',
        bundle_ready: '#3d8b5e',
        failed: '#c4453a',
        running_gpu: '#7a5ea8',
        running_cpu: '#7a5ea8',
    }[status] || '#888';
}

function verdictColor(verdict) {
    return {
        confirmed: '#3d8b5e',
        refuted: '#c4453a',
        inconclusive: '#a8842a',
    }[verdict] || '#888';
}

function renderTrackChips(tracks) {
    return (tracks || []).map(track => {
        const color = track.enabled ? '#3d8b5e' : '#9a9088';
        return `<span class="chip" style="border-color:${color};color:${color};">${esc(track.label)}: ${esc(track.state || (track.enabled ? 'enabled' : 'off'))}</span>`;
    }).join('');
}

function renderManuscriptBlockers(report, limit = 4) {
    const blockers = (report || {}).blockers || [];
    if (!blockers.length) return '';
    return `<div class="insight-impact" style="border-left:3px solid #c4453a;padding-left:10px;color:#c4453a;">
        <span class="insight-label">${esc(tr('label.paperBlocked'))}</span>
        ${blockers.slice(0, limit).map(x => `<div>${esc(trunc(x, 160))}</div>`).join('')}
        ${blockers.length > limit ? `<div>${esc(tr('label.moreBlockers', { count: blockers.length - limit }))}</div>` : ''}
    </div>`;
}

function renderExperimentGroupsV2(groups) {
    const list = el('experimentsList');
    if (!groups || !groups.length) {
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.experimentGroups'))}</p>`;
        return;
    }

    renderListChunked(list, groups, group => {
        const insight = group.insight || {};
        const auto = group.auto_job || {};
        const currentRun = group.canonical_run || group.latest_run || null;
        const color = experimentStatusColor((currentRun || {}).status || auto.status);
        const verdict = currentRun && currentRun.hypothesis_verdict
            ? `<span style="color:${verdictColor(currentRun.hypothesis_verdict)};font-weight:700;text-transform:uppercase;">${esc(currentRun.hypothesis_verdict)}</span>`
            : '';
        const effect = currentRun && currentRun.effect_pct != null
            ? `${currentRun.effect_pct >= 0 ? '+' : ''}${currentRun.effect_pct.toFixed(2)}%`
            : '';
        const progress = auto.stage
            ? friendlyAutomationStage(auto.status, auto.stage)
            : ((currentRun || {}).status || 'not_started');
        const currentRunLabel = currentRun ? `${tr('experiments.run')} ${currentRun.id}` : tr('empty.experimentRunCreated');
        const previewUrl = (((group || {}).paper_preview_urls || {}).index) || '';
        const plan = group.plan_snapshot || {};
        const latest = plan.latest_status || {};
        const manuscriptBlockers = plan.manuscript_blockers || {};
        const planReady = [
            plan.experiment_spec ? tr('label.planFile.experimentSpec') : '',
            plan.evidence_plan ? tr('label.planFile.evidencePlan') : '',
            plan.manuscript_input_state ? tr('label.planFile.manuscriptState') : '',
            plan.manuscript_blockers ? tr('label.planFile.manuscriptBlockers') : '',
        ].filter(Boolean).join(', ') || tr('label.waitingPlanFiles');
        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">${esc(tr('label.discoveryId', { id: insight.id }))}</span>
                <span class="insight-scores">${esc(currentRunLabel)}</span>
                ${verdict}
                ${effect ? `<span class="insight-scores">${esc(tr('label.effect', { effect }))}</span>` : ''}
                <span style="color:var(--text-dim);font-size:0.68rem;">${esc(tr('label.tier', { tier: insight.tier || '?' }))}</span>
            </div>
            <div class="insight-title">${esc(insight.title || tr('label.discovery'))}</div>
            <div class="insight-impact"><span class="insight-label">${esc(tr('experiments.currentWork'))}</span> ${esc(progress)}</div>
            ${renderManuscriptBlockers(manuscriptBlockers)}
            <details class="advanced-inline">
                <summary>${esc(tr('experiments.advancedFields'))}</summary>
                ${latest.stage ? `<div class="insight-experiment"><span class="insight-label">${esc(tr('label.latestFileStatus'))}</span> ${esc(latest.stage)} / ${esc(latest.status || '')}</div>` : ''}
                ${latest.error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">${esc(tr('label.latestError'))}</span> ${esc(trunc(latest.error, 180))}</div>` : ''}
                <div class="insight-evidence"><span class="insight-label">${esc(tr('experiments.planFiles'))}</span> ${esc(planReady)}</div>
                <div style="display:flex;gap:16px;margin:6px 0;font-size:0.75rem;color:var(--text-secondary);flex-wrap:wrap;">
                    <span>${esc(tr('label.experimentRuns', { count: group.run_count || 0 }))}</span>
                    <span>${esc(tr('label.latestExperimentRun', { status: (group.latest_run || {}).status || 'none' }))}</span>
                    <span>${esc(tr('label.submissionBundle', { status: insight.submission_status || 'not_started' }))}</span>
                </div>
                <div style="display:flex;gap:16px;margin:6px 0;font-size:0.75rem;color:var(--text-secondary);flex-wrap:wrap;">
                    <span>${esc(tr('label.experimentPath', { path: group.experiment_root || '-' }))}</span>
                    <span>${esc(tr('label.planPath', { path: group.plan_root || '-' }))}</span>
                    <span>${esc(tr('label.generatedManuscriptPath', { path: group.paper_root || '-' }))}</span>
                </div>
                <div class="chip-row" style="margin:8px 0;">${renderTrackChips(group.planned_tracks)}</div>
            </details>
            ${auto.last_note ? `<div class="insight-experiment"><span class="insight-label">${esc(tr('common.latest'))}</span> ${esc(trunc(auto.last_note, 220))}</div>` : ''}
            ${auto.last_error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">${esc(tr('common.error'))}</span> ${esc(trunc(auto.last_error, 220))}</div>` : ''}
            <div class="insight-actions">
                <button class="btn-preview" onclick="window._dg.viewExperimentGroup(${insight.id})">${esc(tr('experiments.viewHistory'))}</button>
                ${currentRun ? `<button class="btn-preview" onclick="window._dg.viewExperiment(${currentRun.id})">${esc(tr('experiments.viewRun'))}</button>` : ''}
                ${previewUrl ? `<button class="btn-preview" onclick="window.open('${esc(previewUrl)}','_blank')">${esc(tr('experiments.openManuscript'))}</button>` : ''}
            </div>
        </div>`;
    });
}

function jsonPreview(obj, emptyText = 'None') {
    if (!obj || (typeof obj === 'object' && Object.keys(obj).length === 0)) {
        return `<p class="empty-msg">${esc(emptyText)}</p>`;
    }
    return `<pre style="white-space:pre-wrap;word-break:break-word;background:var(--bg-elevated);padding:10px;border-radius:8px;font-size:0.72rem;">${esc(JSON.stringify(obj, null, 2))}</pre>`;
}

function renderPaperAssetLinks(insightId, assets) {
    if (!assets || !assets.length) {
        return `<p class="empty-msg">${esc(tr('empty.manuscriptAssets'))}</p>`;
    }
    return `<div style="display:flex;flex-direction:column;gap:6px;">${assets.slice(0, 20).map(asset => `
        <a href="/papers/${insightId}/view/${encodeURI(asset.path)}" target="_blank">${esc(asset.path)}</a>
    `).join('')}</div>`;
}

function renderMetaReport(meta) {
    const card = el('metaReportCard');
    const body = el('metaReportBody');
    if (!meta || meta.status === 'insufficient_data' || meta.total_experiments < 1) {
        card.style.display = 'none';
        return;
    }
    card.style.display = '';

    const track = meta.track_record || {};
    let html = `<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px;">
            <div class="stat-card" style="min-width:100px;">
                <div class="stat-number">${meta.total_experiments}</div>
                <div class="stat-label">${esc(tr('nav.experiments'))}</div>
            </div>
            <div class="stat-card" style="min-width:100px;">
                <div class="stat-number" style="color:#3d8b5e;">${track.total_confirmed || 0}</div>
                <div class="stat-label">${esc(tr('meta.confirmed'))}</div>
            </div>
            <div class="stat-card" style="min-width:100px;">
                <div class="stat-number" style="color:#c4453a;">${track.total_refuted || 0}</div>
                <div class="stat-label">${esc(tr('meta.refuted'))}</div>
            </div>
            <div class="stat-card" style="min-width:100px;">
                <div class="stat-number">${((track.overall_hit_rate || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">${esc(tr('meta.hitRate'))}</div>
            </div>
        </div>`;

    if (track.signal_types && track.signal_types.length) {
        html += `<h4 style="margin:12px 0 6px;">${esc(tr('meta.signalTypePerformance'))}</h4>`;
        html += `<table class="matrix-table" style="font-size:0.75rem;"><thead><tr><th>${esc(tr('meta.signal'))}</th><th>${esc(tr('meta.total'))}</th><th>${esc(tr('meta.confirmed'))}</th><th>${esc(tr('meta.refuted'))}</th><th>${esc(tr('meta.hitRate'))}</th></tr></thead><tbody>`;
        for (const s of track.signal_types) {
            html += `<tr><td>${esc(s.signal_type)}</td><td>${s.hypothesis_count}</td><td style="color:#3d8b5e;">${s.confirmed_count}</td><td style="color:#c4453a;">${s.refuted_count}</td><td><b>${((s.hit_rate || 0) * 100).toFixed(1)}%</b></td></tr>`;
        }
        html += '</tbody></table>';
    }

    const weights = meta.signal_weights || {};
    if (Object.keys(weights).length) {
        html += `<h4 style="margin:12px 0 6px;">${esc(tr('meta.learnedSignalWeights'))}</h4><div class="chip-row">`;
        for (const [k, v] of Object.entries(weights)) {
            const color = v > 1.5 ? '#3d8b5e' : v < 0.5 ? '#c4453a' : '#a8842a';
            html += `<span class="chip" style="border-color:${color};color:${color};">${esc(k)}: ${v}x</span>`;
        }
        html += '</div>';
    }

    body.innerHTML = html;
}

// ── Providers Tab ────────────────────────────────────────────────────

async function loadProviders() {
    providersLoaded = true;
    try {
        const providers = await api('/api/providers');
        renderProviders(providers);
    } catch (e) {
        console.error('Providers load error:', e);
        el('providersList').innerHTML = `<p class="empty-msg">${esc(tr('providers.failed'))}</p>`;
    }
}

function renderProviders(providers) {
    const list = el('providersList');

    if (!providers || (Array.isArray(providers) && providers.length === 0)) {
        // providers might be an object, not array
        if (typeof providers === 'object' && !Array.isArray(providers)) {
            // Convert object to array
            const arr = Object.entries(providers).map(([k, v]) => ({ name: k, ...v }));
            if (arr.length === 0) {
                list.innerHTML = `<p class="empty-msg">${esc(tr('empty.providers'))}</p>`;
                return;
            }
            renderProviderCards(arr);
            return;
        }
        list.innerHTML = `<p class="empty-msg">${esc(tr('empty.providers'))}</p>`;
        return;
    }

    if (Array.isArray(providers)) {
        renderProviderCards(providers);
    } else {
        const arr = Object.entries(providers).map(([k, v]) => ({ name: k, ...v }));
        renderProviderCards(arr);
    }
}

function renderProviderCards(providers) {
    const list = el('providersList');
    const maxCalls = Math.max(...providers.map(p => p.calls || p.total_calls || 0), 1);

    list.innerHTML = providers.map(p => {
        const calls = p.calls || p.total_calls || 0;
        const tokens = p.tokens || p.total_tokens || 0;
        const errors = p.errors || p.total_errors || 0;
        const latency = p.avg_latency || p.latency_avg || 0;
        const barPct = Math.round((calls / maxCalls) * 100);

        return `<div class="provider-card">
            <div class="provider-name">${esc(p.name || p.provider || tr('providers.unknown'))}</div>
            <div class="provider-url">${esc(p.base_url || p.url || '')}</div>
            <div class="provider-stats">
                <div class="provider-stat">
                    <span class="provider-stat-val cyan">${fmt(calls)}</span>
                    <span class="provider-stat-lbl">${esc(tr('providers.calls'))}</span>
                </div>
                <div class="provider-stat">
                    <span class="provider-stat-val gold">${fmt(tokens)}</span>
                    <span class="provider-stat-lbl">${esc(tr('providers.tokens'))}</span>
                </div>
                <div class="provider-stat">
                    <span class="provider-stat-val ${errors > 0 ? 'red' : 'green'}">${fmt(errors)}</span>
                    <span class="provider-stat-lbl">${esc(tr('providers.errors'))}</span>
                </div>
                <div class="provider-stat">
                    <span class="provider-stat-val">${latency ? latency.toFixed(1) + 's' : '-'}</span>
                    <span class="provider-stat-lbl">${esc(tr('providers.avgLatency'))}</span>
                </div>
            </div>
            <div class="provider-bar-wrap">
                <div class="provider-bar" style="width:${barPct}%"></div>
            </div>
        </div>`;
    }).join('');
}

function startProviderRefresh() {
    if (providerTimer) clearInterval(providerTimer);
    providerTimer = setInterval(() => {
        if (activeTab === 'providers') loadProviders();
    }, 10000);
}

// ── Progressive Loading ──────────────────────────────────────────────

function runWhenIdle(fn, timeout = 700) {
    if ('requestIdleCallback' in window) {
        window.requestIdleCallback(fn, { timeout });
    } else {
        setTimeout(fn, timeout);
    }
}

// Render a large list of cards without janking the main thread. A single
// `container.innerHTML = items.map(renderItem).join('')` does TWO O(n) bursts of
// work in one synchronous task: it builds every card's HTML string (often
// parsing several JSON fields per row — the production /api/deep_insights and
// /api/insights payloads are ~0.6–0.7 MB) AND parses the whole subtree into the
// DOM. On real data that single task ran into the hundreds of ms. Here we build
// AND insert one chunk at a time: the first chunk synchronously (so the tab is
// never empty), the rest across idle callbacks. Each step is O(chunk), so no
// step is a long task, and insertAdjacentHTML only parses the appended slice so
// total work stays O(n) — never the O(n²) of `innerHTML +=`. `renderItem(item,
// i)` returns the card HTML; items use inline onclick handlers, so no
// post-render event binding is needed.
function renderListChunked(container, items, renderItem, chunk = 25) {
    container.innerHTML = '';
    if (!items || !items.length) return;
    // Cancel any still-pending chunked render of an earlier call (e.g. when a
    // filter re-renders the same list before the previous run finished).
    const token = (container._chunkToken || 0) + 1;
    container._chunkToken = token;
    const buildSlice = (start) => {
        let html = '';
        const end = Math.min(start + chunk, items.length);
        for (let i = start; i < end; i++) html += renderItem(items[i], i);
        return html;
    };
    container.insertAdjacentHTML('beforeend', buildSlice(0));
    let i = chunk;
    const step = () => {
        if (container._chunkToken !== token) return; // superseded
        container.insertAdjacentHTML('beforeend', buildSlice(i));
        i += chunk;
        if (i < items.length) runWhenIdle(step, 50);
    };
    if (i < items.length) runWhenIdle(step, 50);
}

async function prefetchInactiveTabs() {
    if (inactiveTabsPrefetched) return;
    inactiveTabsPrefetched = true;

    // Only prewarm cheap things during idle. Eagerly rendering every tab here
    // built ~25k DOM nodes with zero user interaction, which is exactly the
    // "loaded then frozen" symptom: each heavy list render is a long main-thread
    // task, and the resulting giant DOM makes later layout/restyle slow too.
    // The heavy tabs already lazy-load on first activation (onTabActivated), and
    // their renders are chunked, so deferring them keeps idle responsive and the
    // DOM small until a tab is actually viewed.
    const tasks = [
        () => loadTaxonomyDropdown(), // builds a <datalist> (~30ms); keeps Evidence instant
    ];

    for (const task of tasks) {
        try {
            await task();
        } catch (e) {
            console.debug('Idle prefetch skipped:', e);
        }
    }
}

// ── Search ───────────────────────────────────────────────────────────

function initSearch() {
    const input = el('searchInput');
    const results = el('searchResults');

    input.addEventListener('input', () => {
        clearTimeout(searchTimer);
        const q = input.value.trim();
        if (q.length < 2) {
            results.classList.remove('open');
            return;
        }
        searchTimer = setTimeout(() => performSearch(q), 250);
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            results.classList.remove('open');
            input.blur();
        }
    });

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !results.contains(e.target)) {
            results.classList.remove('open');
        }
    });
}

async function performSearch(query) {
    const results = el('searchResults');
    try {
        const data = await api(`/api/search?q=${encodeURIComponent(query)}`);
        renderSearchResults(data);
    } catch (e) {
        results.innerHTML = `<div class="search-section"><p class="empty-msg">${esc(tr('search.failed'))}</p></div>`;
        results.classList.add('open');
    }
}

function renderSearchResults(data) {
    const results = el('searchResults');
    let html = '';

    if (data.nodes && data.nodes.length) {
        html += `<div class="search-section"><div class="search-section-title">${esc(tr('search.researchAreas'))}</div>`;
        for (const n of data.nodes) {
            html += `<div class="search-result-item" onclick="window._dg.searchNav('node','${esc(n.id)}')">
                <div class="sr-title">${esc(n.name)}</div>
                <div class="sr-meta">${esc(n.id)} \u00B7 ${esc(tr('common.papersCount', { count: n.paper_count || 0 }))}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.papers && data.papers.length) {
        html += `<div class="search-section"><div class="search-section-title">${esc(tr('search.sourcePapers'))}</div>`;
        for (const p of data.papers.slice(0, 8)) {
            html += `<div class="search-result-item" onclick="window.open('https://arxiv.org/abs/${esc(p.id)}','_blank')">
                <div class="sr-title">${esc(trunc(p.title, 70))}</div>
                <div class="sr-meta">${esc(p.id)}${p.work_type ? ' \u00B7 ' + esc(p.work_type) : ''}${p.published_date ? ' \u00B7 ' + esc(p.published_date) : ''}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.methods && data.methods.length) {
        html += `<div class="search-section"><div class="search-section-title">${esc(tr('search.methods'))}</div>`;
        for (const m of data.methods) {
            html += `<div class="search-result-item">
                <div class="sr-title">${esc(m.name)}</div>
                <div class="sr-meta">${esc(tr('common.papersCount', { count: m.paper_count || 0 }))} \u00B7 ${esc(tr('common.resultsCount', { count: m.result_count || 0 }))}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.opportunities && data.opportunities.length) {
        html += `<div class="search-section"><div class="search-section-title">${esc(tr('search.opportunities'))}</div>`;
        for (const o of data.opportunities) {
            html += `<div class="search-result-item" onclick="window._dg.searchNav('node','${esc(o.node_id)}')">
                <div class="sr-title">${esc(o.title)}</div>
                <div class="sr-meta">${esc(o.node_name || o.node_id)} \u00B7 ${esc(tr('common.scoreValue', { score: o.value_score || '?' }))}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.gaps && data.gaps.length) {
        html += `<div class="search-section"><div class="search-section-title">${esc(tr('search.gaps'))}</div>`;
        for (const g of data.gaps) {
            html += `<div class="search-result-item" onclick="window._dg.searchNav('node','${esc(g.node_id)}')">
                <div class="sr-title">${esc(tr('common.onDataset', { method: g.method_name, dataset: g.dataset_name }))}</div>
                <div class="sr-meta">${esc(trunc(g.gap_description, 90))}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (!html) {
        html = `<div class="search-section"><p class="empty-msg">${esc(tr('empty.search'))}</p></div>`;
    }

    results.innerHTML = html;
    results.classList.add('open');
}

function searchNav(type, id) {
    el('searchResults').classList.remove('open');
    el('searchInput').value = '';
    if (type === 'node') {
        switchTab('explore');
        navigateTo(id);
    }
}

// ── Public API (for onclick handlers in HTML strings) ────────────────

window._dg = {
    navigateTo,
    exploreNode(nodeId) {
        switchTab('explore');
        navigateTo(nodeId);
    },
    togglePaper,
    updateMatrixMetric,
    searchNav,
    closeAreaStory,
    // Entity click-through: surface the entity's papers/insights via global search.
    searchEntity(name) {
        const input = el('searchInput');
        if (input) {
            input.value = name;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.focus();
        }
    },
    viewPaperGeneration(insightId) {
        return this.viewExperimentGroup(insightId);
    },

    async viewExperimentGroup(insightId) {
        try {
            const data = await api(`/api/experiment_groups/${insightId}`);
            const insight = data.insight || {};
            const auto = data.auto_job || {};
            const runs = data.runs || [];
            const canonical = data.canonical_run || data.latest_run || null;
            const plan = data.plan_snapshot || {};
            const paperUrls = data.paper_preview_urls || {};
            const paperAssets = data.paper_assets || [];

            let html = `<div class="proposal-content" style="max-height:80vh;">
                <div class="proposal-header">
                    <h3>${esc(tr('label.discoveryId', { id: insight.id }))}: ${esc(insight.title || '')}</h3>
                    <span class="proposal-stats">${esc(tr('label.experimentRun', { id: canonical ? `${canonical.id} / ${canonical.status}` : tr('empty.experimentRunCreated') }))}</span>
                    <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">×</button>
                </div>
                <div class="proposal-body">
                <h4>${esc(tr('experiments.discoveryProgress'))}</h4>
                <p>${esc(tr('label.autoResearch', { status: `${auto.status || 'not_started'}${auto.stage ? ` / ${auto.stage}` : ''}` }))}</p>
                <p>${esc(tr('label.submissionRunCount', { status: insight.submission_status || 'not_started', count: runs.length }))}</p>
                <p>${esc(tr('label.workspace', { path: data.workspace_root || '-' }))}</p>
                <div class="chip-row" style="margin:8px 0 14px;">${renderTrackChips(data.planned_tracks)}</div>
                ${renderManuscriptBlockers(plan.manuscript_blockers || {}, 8)}
                ${auto.last_note ? `<p><b>${esc(tr('common.latest'))}</b> ${esc(auto.last_note)}</p>` : ''}
                ${auto.last_error ? `<p style="color:#c4453a;"><b>${esc(tr('common.error'))}</b> ${esc(auto.last_error)}</p>` : ''}
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:16px 0;">
                    <div style="background:var(--bg-elevated);padding:12px;border-radius:10px;">
                        <h4 style="margin-top:0;">${esc(tr('label.experimentArea'))}</h4>
                        <p><b>${esc(tr('label.experimentPath', { path: '' }))}</b> ${esc(data.experiment_root || '-')}</p>
                        <p><b>${esc(tr('label.canonicalRun', { run: '' }))}</b> ${esc(data.canonical_run_id || canonical?.id || '-')}</p>
                    </div>
                    <div style="background:var(--bg-elevated);padding:12px;border-radius:10px;">
                        <h4 style="margin-top:0;">${esc(tr('label.experimentPlanArea'))}</h4>
                        <p><b>${esc(tr('label.planPath', { path: '' }))}</b> ${esc(data.plan_root || '-')}</p>
                        ${jsonPreview(plan.latest_status, tr('common.noLatestStatus'))}
                    </div>
                    <div style="background:var(--bg-elevated);padding:12px;border-radius:10px;">
                        <h4 style="margin-top:0;">${esc(tr('label.generatedManuscriptArea'))}</h4>
                        <p><b>${esc(tr('label.generatedManuscriptPath', { path: '' }))}</b> ${esc(data.paper_root || '-')}</p>
                        <div class="insight-actions" style="margin:8px 0;">
                            ${paperUrls.index ? `<button class="btn-preview" onclick="window.open('${esc(paperUrls.index)}','_blank')">${esc(tr('label.openManuscriptPage'))}</button>` : ''}
                            ${paperUrls.pdf ? `<button class="btn-preview" onclick="window.open('${esc(paperUrls.pdf)}','_blank')">${esc(tr('common.openPdf'))}</button>` : ''}
                            ${paperUrls.tex ? `<button class="btn-preview" onclick="window.open('${esc(paperUrls.tex)}','_blank')">${esc(tr('common.openTex'))}</button>` : ''}
                        </div>
                        ${renderPaperAssetLinks(insight.id, paperAssets)}
                    </div>
                </div>`;

            html += `<h4>${esc(tr('label.planSnapshot'))}</h4>
                ${jsonPreview(plan.experiment_spec, tr('common.noExperimentSpec'))}
                ${jsonPreview(plan.manuscript_blockers, tr('common.noManuscriptBlockers'))}
                ${jsonPreview(plan.manuscript_input_state, tr('common.noManuscriptInputState'))}`;

            if (runs.length) {
                html += `<h4>${esc(tr('label.experimentHistory'))}</h4>`;
                for (const run of runs) {
                    const color = experimentStatusColor(run.status);
                    const artifactSummary = Object.entries(run.artifact_counts || {})
                        .map(([k, v]) => `${k}:${v}`).join(' · ');
                    const verdict = run.hypothesis_verdict
                        ? `<span style="color:${verdictColor(run.hypothesis_verdict)};font-weight:700;">${esc(run.hypothesis_verdict.toUpperCase())}</span>`
                        : '';
                    const badges = [];
                    if (canonical && canonical.id === run.id) badges.push(tr('label.canonical'));
                    if (run.has_plot_artifacts) badges.push(tr('label.plot'));
                    if (run.has_bundle) badges.push(tr('label.submissionBundleBadge'));
                    html += `<div style="padding:10px;margin:8px 0;border-left:3px solid ${color};background:var(--bg-elevated);border-radius:8px;">
                        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                            <strong style="color:${color};">${esc(tr('label.experimentRun', { id: run.id }))}</strong>
                            <span>${esc(run.status || 'unknown')}</span>
                            ${verdict}
                            ${badges.map(label => `<span class="chip">${esc(label)}</span>`).join('')}
                        </div>
                        <div style="margin-top:6px;font-size:0.78rem;color:var(--text-secondary);display:flex;gap:12px;flex-wrap:wrap;">
                            <span>${esc(tr('label.iterations', { total: run.iterations_total || 0, kept: run.iterations_kept || 0 }))}</span>
                            <span>${esc(tr('common.claimsCount', { count: run.claim_count || 0 }))}</span>
                            ${run.effect_pct != null ? `<span>${esc(tr('label.effect', { effect: run.effect_pct.toFixed(2) + '%' }))}</span>` : ''}
                            ${artifactSummary ? `<span>${esc(tr('label.artifacts', { artifacts: artifactSummary }))}</span>` : ''}
                        </div>
                        ${run.error_message ? `<div style="margin-top:6px;color:#c4453a;font-size:0.76rem;">${esc(trunc(run.error_message, 220))}</div>` : ''}
                        <div class="insight-actions" style="margin-top:8px;">
                            <button class="btn-preview" onclick="window._dg.viewExperiment(${run.id})">${esc(tr('common.viewRunDetails'))}</button>
                        </div>
                    </div>`;
                }
            } else {
                html += `<p>${esc(tr('empty.experimentRuns'))}</p>`;
            }

            html += '</div></div>';

            const modal = document.createElement('div');
            modal.className = 'proposal-modal';
            modal.innerHTML = `<div class="proposal-overlay" onclick="this.parentElement.remove()"></div>${html}`;
            document.body.appendChild(modal);
        } catch (e) {
            alert(tr('common.failedToLoadIdeaHistory', { message: e.message }));
        }
    },

    async viewExperiment(runId) {
        try {
            const data = await api(`/api/experiments/${runId}`);
            const run = data.run;
            const iters = data.iterations || [];
            const claims = data.claims || [];

            let html = `<div class="proposal-content" style="max-height:80vh;">
                <div class="proposal-header">
                    <h3>${esc(tr('label.experimentRun', { id: `#${run.id}` }))}: ${esc(run.insight_title || '')}</h3>
                    <span class="proposal-stats">${esc(tr('label.statusVerdict', { status: run.status, verdict: run.hypothesis_verdict || 'pending' }))}</span>
                    <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">×</button>
                </div>
                <div class="proposal-body">
                <h4>${esc(tr('label.metrics'))}</h4>
                <p>${esc(tr('label.baseline', { value: run.baseline_metric_value || '?' }))} | ${esc(tr('label.best', { value: run.best_metric_value || '?' }))} | ${esc(tr('label.effect', { effect: run.effect_pct != null ? run.effect_pct.toFixed(2) + '%' : '?' }))}</p>
                <p>${esc(tr('label.iterations', { total: run.iterations_total || 0, kept: run.iterations_kept || 0 }))}</p>
                ${run.codebase_url ? `<p>${esc(tr('label.codebase'))} <a href="${esc(run.codebase_url)}" target="_blank">${esc(run.codebase_url)}</a></p>` : ''}
                ${run.error_message ? `<p style="color:#c4453a;">${esc(tr('common.error'))} ${esc(run.error_message)}</p>` : ''}`;

            if (iters.length) {
                html += `<h4>${esc(tr('label.iterationHistory'))}</h4><table class="matrix-table" style="font-size:0.72rem;"><thead><tr><th>#</th><th>${esc(tr('label.phase'))}</th><th>${esc(tr('label.metric'))}</th><th>${esc(tr('label.status'))}</th><th>${esc(tr('label.description'))}</th></tr></thead><tbody>`;
                for (const it of iters.slice(-30)) {
                    const sColor = it.status === 'keep' ? '#3d8b5e' : it.status === 'crash' ? '#c4453a' : '#9a9088';
                    html += `<tr><td>${it.iteration_number}</td><td>${esc(it.phase)}</td><td>${it.metric_value != null ? it.metric_value.toFixed(6) : '-'}</td><td style="color:${sColor};">${esc(it.status)}</td><td>${esc(trunc(it.description || '', 60))}</td></tr>`;
                }
                html += '</tbody></table>';
            }

            if (claims.length) {
                html += `<h4>${esc(tr('label.experimentalClaims'))}</h4>`;
                for (const cl of claims) {
                    const vColor = cl.verdict === 'confirmed' ? '#3d8b5e' : cl.verdict === 'refuted' ? '#c4453a' : '#a8842a';
                    html += `<div style="padding:8px;margin:4px 0;border-left:3px solid ${vColor};background:var(--bg-elevated);">
                        <strong style="color:${vColor};">${esc(cl.verdict.toUpperCase())}</strong> (p=${cl.p_value != null ? cl.p_value.toFixed(4) : '?'})
                        <p style="margin:4px 0;font-size:0.78rem;">${esc(cl.claim_text)}</p>
                    </div>`;
                }
            }

            html += '</div></div>';

            const modal = document.createElement('div');
            modal.className = 'proposal-modal';
            modal.innerHTML = `<div class="proposal-overlay" onclick="this.parentElement.remove()"></div>${html}`;
            document.body.appendChild(modal);
        } catch (e) {
            alert(tr('common.failedToLoad', { message: e.message }));
        }
    },

    async previewProposal(insightId) {
        try {
            const res = await api(`/api/research/proposal/${insightId}`);
            // Render markdown
            function renderMd(text) {
                if (typeof marked !== 'undefined' && marked.parse) return marked.parse(text);
                // Fallback: basic markdown rendering
                return text
                    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
                    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
                    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
                    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.+?)\*/g, '<em>$1</em>')
                    .replace(/^- (.+)$/gm, '<li>$1</li>')
                    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
                    .replace(/^---$/gm, '<hr>')
                    .replace(/\n{2,}/g, '</p><p>')
                    .replace(/\n/g, '<br>')
                    .replace(/^/, '<p>').replace(/$/, '</p>');
            }
            const bodyHtml = renderMd(res.proposal);
            // Show in a modal
            const modal = document.createElement('div');
            modal.className = 'proposal-modal';
            modal.innerHTML = `<div class="proposal-overlay" onclick="this.parentElement.remove()"></div>
                <div class="proposal-content">
                    <div class="proposal-header">
                        <h3>${esc(res.title)}</h3>
                        <span class="proposal-stats">${esc(tr('common.papersCount', { count: res.paper_count }))} · ${esc(tr('common.claimsCount', { count: res.claim_count }))} · ${esc(tr('common.contradictionsCount', { count: res.contradiction_count }))}</span>
                        <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">\u00D7</button>
                    </div>
                    <div class="proposal-body">${bodyHtml}</div>
                </div>`;
            document.body.appendChild(modal);
        } catch (e) {
            alert(tr('common.failedToLoadProposal', { message: e.message }));
        }
    },
};

// ── Init ─────────────────────────────────────────────────────────────

function init() {
    // Custom tooltips for [data-i18n-title] metric cards (replaces native title).
    if (window.DGTooltip) window.DGTooltip.attachDelegated();

    if (window.dgI18n) {
        window.dgI18n.applyI18n(document);
        $$('[data-lang]').forEach(btn => {
            btn.addEventListener('click', () => window.dgI18n.setLanguage(btn.dataset.lang));
        });
        document.addEventListener('deepgraph:languagechange', () => {
            updateLiveBadge();
            if (taxonomyLoaded) loadTaxonomyDropdown();
            // Re-render graphs so legend/tooltip labels follow the new language.
            if (overviewGraphLoaded) { overviewGraphLoaded = false; loadOverviewGraph(); }
            if (exploreData) {
                closeAreaStory();
                renderTaxonomyGraph('exploreGraph', exploreData.node, exploreData.children, { height: 520, isPreview: false });
            }
        });
    }

    // Nav items
    $$('[data-tab]').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Sidebar toggle
    el('sidebarToggle').addEventListener('click', toggleSidebar);

    // "Open in Explore" button
    el('btnJumpExplore').addEventListener('click', () => {
        switchTab('explore');
        if (!exploreData) navigateTo(ROOT_NODE);
    });

    // Discovery filters + generate button
    const dtf = el('discoveryTierFilter');
    if (dtf) dtf.addEventListener('change', () => {
        discoveriesLoaded = false;
        loadDiscoveriesTab();
    });

    // Experiment filters
    const esf = el('experimentStatusFilter');
    if (esf) esf.addEventListener('change', () => {
        experimentsLoaded = false;
        loadExperimentsTab();
    });

    // Insight filters
    const itf = el('insightTypeFilter');
    const isf = el('insightSortFilter');
    if (itf) itf.addEventListener('change', () => {
        insightsLoaded = false;
        loadInsightsTab();
    });
    if (isf) isf.addEventListener('change', () => {
        insightsLoaded = false;
        loadInsightsTab();
    });

    // Evidence node select
    el('evidenceNodeSelect').addEventListener('change', (e) => {
        loadEvidenceForNode(e.target.value);
    });

    // Papers filters
    el('papersSearch').addEventListener('input', () => {
        clearTimeout(el('papersSearch')._timer);
        el('papersSearch')._timer = setTimeout(renderPapers, 200);
    });
    el('papersStatus').addEventListener('change', renderPapers);

    // Opportunities filter
    // Search
    initSearch();

    // Initial data loads
    refreshStats();
    loadRecentlyDiscovered();
    fetchEvents();
    setInterval(fetchEvents, 2000);

    // Stats refresh every 15s
    statsTimer = setInterval(refreshStats, 15000);

    // Idle work after first text content renders.
    runWhenIdle(loadOverviewGraph, 300);
    runWhenIdle(prefetchInactiveTabs, 900);

    const overviewAdvanced = el('overviewAdvanced');
    if (overviewAdvanced) {
        overviewAdvanced.addEventListener('toggle', () => {
            if (overviewAdvanced.open) loadProcessingPapers();
        });
    }

    // Periodically refresh recently discovered (every 30s)
    setInterval(loadRecentlyDiscovered, 30000);

    setInterval(() => {
        if (activeTab === 'experiments') loadExperimentsTab();
        if (activeTab === 'paper-progress') loadPaperProgressTab();
        if (activeTab === 'generated-papers') loadGeneratedPapersTab();
        if (activeTab === 'discoveries') loadDiscoveriesTab();
        if (el('overviewAdvanced')?.open) loadProcessingPapers();
    }, 10000);
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
