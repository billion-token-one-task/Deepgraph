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
let eventSource     = null;
let events          = [];        // max 50
let activePapers    = {};        // paper_id -> {title, step, startTime}
let statsCache      = null;
let allPapers       = [];
let allOpportunities = [];
let taxonomyFlat    = [];        // flat list for Evidence dropdown
let searchTimer     = null;
let statsTimer      = null;
let providerTimer   = null;
let papersLoaded    = false;
let oppsLoaded      = false;
let providersLoaded = false;
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

async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`API ${path} returned ${r.status}`);
    return r.json();
}

function timeAgo(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const s = Math.floor((Date.now() - d.getTime()) / 1000);
    if (s < 60)   return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
}

// ── Tab Navigation ───────────────────────────────────────────────────

function switchTab(tab) {
    if (tab === activeTab) return;
    activeTab = tab;

    // Update nav items
    $$('.nav-item').forEach(btn => {
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
            loadTaxonomyDropdown();
            break;
        case 'papers':
            if (!papersLoaded) loadPapers();
            break;
        case 'discoveries':
            loadDiscoveriesTab();
            break;
        case 'experiments':
            loadExperimentsTab();
            break;
        case 'insights':
            loadInsightsTab();
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

        // Top bar
        el('hdrPapers').textContent  = fmt(s.papers_processed || 0);
        el('hdrResults').textContent = fmt(s.results_total || 0);
        el('hdrInsights').textContent = fmt(s.insights_total || 0);
        el('hdrTokens').textContent  = fmt(s.tokens_consumed || 0);

        // Overview stat cards
        el('statPapers').textContent        = fmt(s.papers_processed || 0);
        el('statResults').textContent       = fmt(s.results_total || 0);
        el('statTaxonomy').textContent = fmt(s.taxonomy_nodes_total || 0);
        el('statContradictions').textContent = fmt(s.contradictions_total || 0);
        el('statInsights').textContent      = fmt(s.insights_total || 0);
        el('statTokens').textContent        = fmt(s.tokens_consumed || 0);
        el('statExperiments').textContent   = fmt(s.experiment_runs_total || 0);
        el('statDeepDiscoveries').textContent = fmt(s.deep_insights_total || 0);
        el('statCompletePapers').textContent = fmt(s.submission_bundles_total || 0);
    } catch (e) {
        console.error('Stats error:', e);
    }
}

// ── SSE Event Stream ─────────────────────────────────────────────────

let sseRetryDelay = 2000;

function startSSE() {
    if (eventSource) {
        try { eventSource.close(); } catch(e) {}
        eventSource = null;
    }
    eventSource = new EventSource('/api/events');

    eventSource.onopen = () => {
        sseRetryDelay = 2000;
    };

    eventSource.onmessage = (msg) => {
        try {
            const ev = JSON.parse(msg.data);
            events.push(ev);
            if (events.length > 50) events.shift();

            trackPaperEvent(ev);
            updateLiveBadge(ev);
            appendFeedEvent(ev);
        } catch (e) {
            console.error('SSE parse error:', e);
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        eventSource = null;
        setTimeout(startSSE, sseRetryDelay);
        sseRetryDelay = Math.min(sseRetryDelay * 1.5, 15000);
    };
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
    badge.textContent = running ? 'LIVE' : 'IDLE';
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
        listEl.innerHTML = '<p class="empty-msg">Idle</p>';
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
    if (countEl) countEl.textContent = feed.children.length + ' events';

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
        const [data, insights] = await Promise.all([
            api('/api/recent_discoveries?limit=8'),
            api('/api/insights?limit=6'),
        ]);
        renderRecentlyDiscovered(data, insights);
    } catch (e) {
        console.error('Recent discoveries error:', e);
    }
}

function renderRecentlyDiscovered(data, insights) {
    const grid = el('recentlyGrid');
    let items = [];

    // Prioritize real insights over old opportunities
    if (insights && insights.length > 0) {
        for (const ins of insights.slice(0, 4)) {
            items.push({
                type: ins.insight_type || 'insight',
                title: ins.title || 'Insight',
                desc: ins.hypothesis || '',
                meta: `${esc(ins.node_id)} | N:${ins.novelty_score}/5 F:${ins.feasibility_score}/5`,
                nodeId: ins.node_id,
            });
        }
    } else if (data.opportunities) {
        for (const o of data.opportunities.slice(0, 3)) {
            items.push({
                type: 'opportunity',
                title: o.title || 'Opportunity',
                desc: o.description || '',
                meta: `${esc(o.node_name || o.node_id)} | score ${o.value_score || '?'}/5`,
                nodeId: o.node_id,
            });
        }
    }
    if (data.gaps) {
        for (const g of data.gaps.slice(0, 3)) {
            items.push({
                type: 'gap',
                title: `${g.method_name || ''} on ${g.dataset_name || ''}`,
                desc: g.gap_description || '',
                meta: `${esc(g.node_name || g.node_id)} | value ${g.value_score || '?'}/5`,
                nodeId: g.node_id,
            });
        }
    }
    if (data.contradictions) {
        for (const c of data.contradictions.slice(0, 2)) {
            items.push({
                type: 'contradiction',
                title: c.description || 'Contradiction',
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
        grid.innerHTML = '<p class="empty-msg">Run the pipeline to discover gaps, contradictions, and opportunities.</p>';
        return;
    }

    const order = { opportunity: 0, gap: 1, contradiction: 2, paper: 3 };
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
    try {
        const data = await api(`/api/taxonomy/${ROOT_NODE}`);
        renderRadialGraph('overviewGraphSvg', data.node, data.children, 320, true);
    } catch (e) {
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
        el('exploreTitle').textContent = data.node.name + ' \u2014 Opportunity Map';

        // Graph
        renderRadialGraph('exploreGraphSvg', data.node, data.children, 520, false);

        // Summary card
        const sumCard = el('exploreSummaryCard');
        if (data.summary || data.node.description || insights.length > 0) {
            sumCard.style.display = '';
            el('exploreSummaryTitle').textContent = 'What Is Happening In ' + data.node.name + '?';
            renderExploreSummary(data);
        } else {
            sumCard.style.display = 'none';
        }

        // Children card
        const childCard = el('exploreChildrenCard');
        if (data.children && data.children.length > 0) {
            childCard.style.display = '';
            el('exploreChildrenTitle').textContent = `Sub-areas of ${data.node.name} (${data.children.length})`;
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
        `<span class="chip" onclick="window._dg.navigateTo('${esc(c.id)}')">${esc(c.name)}${c.paper_count ? ' \u00B7 ' + c.paper_count + 'p' : ''}</span>`
    ).join('');

    let html = `<div class="summary-hero">
        <h4>${esc(node.name)}</h4>
        <p>${esc(s ? (s.overview || node.description || '') : (node.description || 'No summary generated yet.'))}</p>
        ${s && s.why_it_matters ? `<p>${esc(s.why_it_matters)}</p>` : ''}
        ${childChips ? `<div class="chip-row">${childChips}</div>` : ''}
    </div>`;

    if (s) {
        // Work items and gaps
        const workHtml = (s.what_people_are_building || []).map(w =>
            `<div class="summary-item"><strong>${esc(w.label || 'Workstream')}</strong><p>${esc(w.description || '')}</p>${w.paper_count ? `<div class="meta">${w.paper_count} papers</div>` : ''}</div>`
        ).join('') || '<p class="empty-msg">No workstreams yet.</p>';

        const gapHtml = (s.current_gaps || []).map(g => {
            const tl = g.gap_type ? `<span style="color:var(--text-dim);font-size:0.68rem;">[${esc(g.gap_type.replace(/_/g, ' '))}]</span> ` : '';
            return `<div class="summary-item"><strong>${tl}${esc(g.title || 'Open gap')}</strong><p>${esc(g.description || '')}</p>${g.why_now ? `<div class="meta">Why now: ${esc(g.why_now)}</div>` : ''}</div>`;
        }).join('') || '<p class="empty-msg">No gaps yet.</p>';

        html += `<div class="summary-grid">
            <div class="summary-card-inner"><h4>What People Are Working On</h4>${workHtml}</div>
            <div class="summary-card-inner"><h4>Where The Gaps Are</h4>${gapHtml}</div>
        </div>`;

        // Chips
        const patterns = (s.common_patterns || []).map(p => `<span class="chip">${esc(p)}</span>`).join('');
        const methods  = (s.common_methods || []).map(m => `<span class="chip">${esc(m)}</span>`).join('');
        const datasets = (s.common_datasets || []).map(d => `<span class="chip">${esc(d)}</span>`).join('');

        if (patterns || methods || datasets) {
            html += `<div class="summary-grid">
                <div class="summary-card-inner"><h4>Recurring Themes</h4><div class="chip-row">${patterns || '<span class="chip">None yet</span>'}</div></div>
                <div class="summary-card-inner"><h4>Methods & Datasets</h4>
                    ${methods ? `<div class="chip-row">${methods}</div>` : ''}
                    ${datasets ? `<div class="chip-row" style="margin-top:6px;">${datasets}</div>` : ''}
                </div>
            </div>`;
        }

        if (paperClusters.length > 0) {
            const clusterHtml = paperClusters.map(cluster => `
                <div class="summary-item">
                    <strong>${esc(cluster.label || 'Paper Cluster')}</strong>
                    <p>${cluster.paper_count} papers${cluster.shared_entities?.length ? ' · shared entities: ' + esc(cluster.shared_entities.slice(0, 3).join(', ')) : ''}</p>
                    ${cluster.sample_papers?.length ? `<div class="meta">${cluster.sample_papers.map(p => esc(trunc(p.title, 48))).join(' | ')}</div>` : ''}
                </div>
            `).join('');

            html += `<div class="summary-card-inner">
                <h4>Paper Clusters</h4>
                ${clusterHtml}
            </div>`;
        } else if ((data.papers || []).length >= 10) {
            html += `<div class="summary-card-inner">
                <h4>Paper Clusters</h4>
                <p class="empty-msg">This node has ${data.papers.length} papers, but the current graph signals were not strong enough to form stable clusters yet.</p>
            </div>`;
        }

        // Graph entities
        const gs = data.graph_summary;
        if (gs && (gs.top_entities || gs.top_relations)) {
            const entHtml = (gs.top_entities || []).slice(0, 6).map(e =>
                `<div class="summary-item"><strong>${esc(e.name)}</strong><p>${esc(e.entity_type)} \u00B7 ${e.paper_count} papers \u00B7 ${e.mention_count} mentions</p></div>`
            ).join('') || '<p class="empty-msg">No entities yet.</p>';
            const relHtml = (gs.top_relations || []).slice(0, 6).map(r =>
                `<div class="summary-item"><strong>${esc(r.subject)} \u2192 ${esc(r.object)}</strong><p>${esc(r.predicate)} \u00B7 ${r.paper_count} papers</p></div>`
            ).join('') || '<p class="empty-msg">No relations yet.</p>';

            html += `<div class="summary-grid">
                <div class="summary-card-inner"><h4>Core Entities</h4>${entHtml}</div>
                <div class="summary-card-inner"><h4>Key Links</h4>${relHtml}</div>
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
                `<a class="paper-cite" href="https://arxiv.org/abs/${esc(pid)}" target="_blank" title="Open on arXiv">${esc(pid)}</a>`
            ).join(' ');
            return `<div class="insight-card" style="border-left: 3px solid ${color};">
                <div class="insight-header">
                    <span class="insight-type" style="color:${color};">${esc((ins.insight_type || '').replace(/_/g, ' '))}</span>
                    <span class="insight-scores">N:${ins.novelty_score}/5 F:${ins.feasibility_score}/5</span>
                </div>
                <div class="insight-title">${esc(ins.title)}</div>
                ${paperLinks ? `<div class="insight-papers">${paperLinks}</div>` : ''}
                <div class="insight-evidence"><span class="insight-label">Evidence:</span> ${esc(ins.evidence || '')}</div>
                <div class="insight-hypothesis"><span class="insight-label">Hypothesis:</span> ${esc(ins.hypothesis)}</div>
                <div class="insight-experiment"><span class="insight-label">Experiment:</span> ${esc(ins.experiment)}</div>
                ${ins.impact ? `<div class="insight-impact"><span class="insight-label">Impact:</span> ${esc(ins.impact)}</div>` : ''}
                <div class="insight-actions">
                    <button class="btn-research" onclick="window._dg.launchResearch(${ins.id})">Launch Research</button>
                    <button class="btn-preview" onclick="window._dg.previewProposal(${ins.id})">Preview Proposal</button>
                </div>
            </div>`;
        }).join('');

        html += `<div class="summary-card-inner insights-section">
            <h4>Research Insights (${insights.length})</h4>
            <div class="insights-list">${insightHtml}</div>
        </div>`;
    }

    // Universal patterns for this node
    const patterns = data._patterns || [];
    if (patterns.length > 0) {
        const patHtml = patterns.map(p => {
            let domains = [];
            try { domains = JSON.parse(p.domains || '[]'); } catch(e) {}
            const levelBadge = p.abstraction_level === 'universal'
                ? '<span class="pattern-level universal">Universal</span>'
                : '<span class="pattern-level cross-domain">Cross-domain</span>';
            return `<div class="pattern-card">
                <div class="pattern-header">
                    ${levelBadge}
                    <span class="pattern-type">${esc((p.pattern_type || '').replace(/_/g, ' '))}</span>
                </div>
                <div class="pattern-text">${esc(p.pattern_text)}</div>
                ${domains.length ? `<div class="pattern-domains">Also applies to: ${domains.map(d => `<span class="pattern-domain-chip">${esc(d)}</span>`).join(' ')}</div>` : ''}
            </div>`;
        }).join('');

        html += `<div class="summary-card-inner">
            <h4>Universal Patterns (${patterns.length})</h4>
            <div class="patterns-list">${patHtml}</div>
        </div>`;
    }

    body.innerHTML = html;
}

function renderExploreChildren(children) {
    const body = el('exploreChildrenBody');
    body.innerHTML = `<div class="children-grid">${children.map(c => `
        <div class="child-card" onclick="window._dg.navigateTo('${esc(c.id)}')">
            <div class="child-name">${esc(c.name)}</div>
            <div class="child-stats">
                <span>${c.paper_count || 0} papers</span>
                <span>${c.method_count || 0} methods</span>
                ${c.gap_count ? `<span style="color:var(--green);">${c.gap_count} gaps</span>` : ''}
            </div>
        </div>
    `).join('')}</div>`;
}

// ── Radial Graph (D3, static layout, no force sim) ───────────────────

function renderRadialGraph(svgId, parentNode, children, targetHeight, isPreview) {
    const svg = d3.select('#' + svgId);
    svg.selectAll('*').remove();

    const container = svg.node().parentElement;
    const width = container.clientWidth - 4;
    const height = targetHeight;
    svg.attr('width', width).attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);

    const cx = width / 2;
    const cy = height / 2;

    if (!children || children.length === 0) {
        svg.append('text')
            .attr('x', cx).attr('y', cy - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', '#9a9088').attr('font-size', '14px').attr('font-weight', '600')
            .text('Leaf domain \u2014 see the detailed analysis below');
        svg.append('text')
            .attr('x', cx).attr('y', cy + 16)
            .attr('text-anchor', 'middle')
            .attr('fill', '#b5ada4').attr('font-size', '12px')
            .text(trunc(parentNode.description || '', 80));
        return;
    }

    const maxGap = Math.max(...children.map(c => c.gap_count || 0), 1);
    const radius = Math.min(width, height) * (isPreview ? 0.32 : 0.34);
    const angleStep = (2 * Math.PI) / children.length;

    // Positions
    const nodes = [{
        id: parentNode.id, name: parentNode.name, description: parentNode.description || '',
        paper_count: 0, gap_count: 0, method_count: 0, isParent: true, x: cx, y: cy
    }];
    const links = [];

    children.forEach((child, i) => {
        const angle = angleStep * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        nodes.push({
            id: child.id, name: child.name, description: child.description || '',
            paper_count: child.paper_count || 0, gap_count: child.gap_count || 0,
            method_count: child.method_count || 0, isParent: false, x, y
        });
        links.push({ source: parentNode.id, target: child.id });
    });

    const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

    // Gradient for links
    const defs = svg.append('defs');
    const grad = defs.append('linearGradient').attr('id', 'linkGrad-' + svgId)
        .attr('gradientUnits', 'userSpaceOnUse');
    grad.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(196,112,75,0.3)');
    grad.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(196,112,75,0.06)');

    // Draw links
    svg.append('g').selectAll('line')
        .data(links).join('line')
        .attr('x1', cx).attr('y1', cy)
        .attr('x2', d => (nodeMap[d.target] || {}).x || cx)
        .attr('y2', d => (nodeMap[d.target] || {}).y || cy)
        .attr('stroke', `url(#linkGrad-${svgId})`)
        .attr('stroke-width', 1.5);

    // Draw nodes
    const nodeG = svg.append('g').selectAll('g')
        .data(nodes).join('g')
        .attr('transform', d => `translate(${d.x},${d.y})`)
        .attr('class', d => d.isParent ? 'graph-node-parent' : 'graph-node');

    // Parent node
    const parentG = nodeG.filter(d => d.isParent);
    parentG.append('circle')
        .attr('r', isPreview ? 28 : 34)
        .attr('fill', '#faf5ee')
        .attr('stroke', '#c4704b')
        .attr('stroke-width', 2.5);
    parentG.append('text')
        .attr('text-anchor', 'middle').attr('dy', 4)
        .attr('fill', '#c4704b').attr('font-size', isPreview ? '10px' : '11px').attr('font-weight', '700')
        .text(d => trunc(d.name, isPreview ? 12 : 16));

    // Child nodes
    const childG = nodeG.filter(d => !d.isParent);

    childG.append('circle')
        .attr('r', d => {
            const base = isPreview ? 18 : 22;
            const max  = isPreview ? 36 : 46;
            return Math.max(base, Math.min(max, base + (d.paper_count || 0) * 0.5));
        })
        .attr('fill', d => gapColor(d.gap_count, maxGap).fill)
        .attr('stroke', d => gapColor(d.gap_count, maxGap).stroke)
        .attr('stroke-width', d => d.gap_count > 0 ? 2 : 1.2);

    // Labels
    childG.append('text')
        .attr('text-anchor', 'middle').attr('dy', isPreview ? -2 : -4)
        .attr('fill', '#2b2520').attr('font-size', isPreview ? '9px' : '11px').attr('font-weight', '700')
        .text(d => trunc(d.name, isPreview ? 14 : 20));

    childG.append('text')
        .attr('text-anchor', 'middle').attr('dy', isPreview ? 10 : 12)
        .attr('fill', d => d.paper_count > 0 ? '#c4704b' : '#b5ada4')
        .attr('font-size', isPreview ? '8px' : '10px').attr('font-weight', '700')
        .text(d => d.paper_count > 0 ? d.paper_count + 'p' : 'empty');

    if (!isPreview) {
        childG.filter(d => d.gap_count > 0 || d.method_count > 0)
            .append('text')
            .attr('text-anchor', 'middle').attr('dy', 24)
            .attr('font-size', '9px')
            .attr('fill', d => d.gap_count > 0 ? '#3d8b5e' : '#9a9088')
            .attr('font-weight', '600')
            .text(d => {
                const parts = [];
                if (d.method_count > 0) parts.push(d.method_count + 'M');
                if (d.gap_count > 0) parts.push(d.gap_count + ' gaps');
                return parts.join(' | ');
            });
    }

    // Click handler
    childG.on('click', (e, d) => {
        if (isPreview) {
            switchTab('explore');
            navigateTo(d.id);
        } else {
            navigateTo(d.id);
        }
    });

    // Tooltip (non-preview only)
    if (!isPreview) {
        const tip = el('tooltip');

        childG.on('mouseover', (e, d) => {
            tip.innerHTML = `
                <div style="color:#c4704b;font-weight:700;margin-bottom:5px;">${esc(d.name)}</div>
                <div style="color:var(--text-secondary);margin-bottom:8px;line-height:1.5;">${esc(trunc(d.description, 160))}</div>
                <div style="display:flex;gap:12px;color:var(--text-dim);font-size:0.72rem;">
                    <span><b style="color:#c4704b;">${d.paper_count}</b> papers</span>
                    <span><b style="color:#a8842a;">${d.method_count}</b> methods</span>
                    <span><b style="color:#3d8b5e;">${d.gap_count}</b> gaps</span>
                </div>
                <div style="color:var(--text-muted);margin-top:6px;font-size:0.65rem;">Click to explore</div>
            `;
            tip.classList.add('visible');
            positionTooltip(e);
        }).on('mousemove', positionTooltip)
          .on('mouseout', () => tip.classList.remove('visible'));
    }
}

function positionTooltip(e) {
    const tip = el('tooltip');
    const pad = 14;
    let x = e.clientX + pad;
    let y = e.clientY - pad;
    // Keep in viewport
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > window.innerWidth - 10) x = e.clientX - tw - pad;
    if (y + th > window.innerHeight - 10) y = window.innerHeight - th - 10;
    if (y < 10) y = 10;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
}

function gapColor(gapCount, maxGap) {
    if (gapCount <= 0) return { fill: '#f0ede6', stroke: '#d0c9bc' };
    const t = Math.min(gapCount / Math.max(maxGap, 1), 1);
    return {
        fill: `rgb(${Math.round(250 - t * 20)},${Math.round(245 - t * 20)},${Math.round(238 - t * 20)})`,
        stroke: `rgb(${Math.round(196 - t * 40)},${Math.round(112 + t * 10)},${Math.round(75 + t * 10)})`
    };
}

// ── Evidence Tab ─────────────────────────────────────────────────────

async function loadTaxonomyDropdown() {
    if (taxonomyFlat.length > 0) return; // already loaded
    try {
        taxonomyFlat = await api('/api/taxonomy');
        const sel = el('evidenceNodeSelect');
        sel.innerHTML = '<option value="">-- Select a leaf node --</option>';
        for (const n of taxonomyFlat) {
            sel.innerHTML += `<option value="${esc(n.id)}">${esc(n.id)} \u2014 ${esc(n.name)}</option>`;
        }
    } catch (e) {
        console.error('Taxonomy dropdown error:', e);
    }
}

async function loadEvidenceForNode(nodeId) {
    if (!nodeId) {
        el('evidenceMatrixContainer').innerHTML = '';
        el('evidenceGapsCard').style.display = 'none';
        el('evidenceHint').textContent = 'Select a leaf node to view the evidence matrix.';
        return;
    }

    el('evidenceHint').textContent = 'Loading...';

    try {
        const data = await api(`/api/taxonomy/${nodeId}`);
        const m = data.matrix;

        if (m && m.methods && m.methods.length > 0 && m.datasets && m.datasets.length > 0) {
            renderMatrix(el('evidenceMatrixContainer'), m);
            el('evidenceHint').textContent = `${m.methods.length} methods x ${m.datasets.length} datasets`;
        } else {
            el('evidenceMatrixContainer').innerHTML = '<p class="empty-msg">No structured evidence for this node. Try a leaf node with papers.</p>';
            el('evidenceHint').textContent = data.is_leaf ? 'No evidence data yet.' : 'Select a leaf node.';
        }

        // Gaps
        const gapsCard = el('evidenceGapsCard');
        if (data.gaps && data.gaps.length > 0) {
            gapsCard.style.display = '';
            el('evidenceGapsTitle').textContent = `Matrix Gaps (${data.gaps.length})`;
            renderGaps(el('evidenceGapsBody'), data.gaps);
        } else {
            gapsCard.style.display = 'none';
        }
    } catch (e) {
        console.error('Evidence load error:', e);
        el('evidenceHint').textContent = 'Error loading data.';
    }
}

function renderMatrix(container, matrix) {
    if (!matrix.methods.length || !matrix.datasets.length) {
        container.innerHTML = '<p class="empty-msg">No results data yet.</p>';
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

    let html = '<div class="matrix-controls">';
    html += '<label>Metric:</label>';
    html += '<select class="matrix-metric-select" onchange="window._dg.updateMatrixMetric(this)">';
    for (const m of metrics) {
        html += `<option value="${esc(m)}"${m === defaultMetric ? ' selected' : ''}>${esc(m || '(none)')}</option>`;
    }
    html += '</select>';
    html += `<span class="matrix-info">${matrix.methods.length} methods x ${matrix.datasets.length} datasets</span>`;
    html += '</div>';

    html += '<div class="matrix-scroll"><table class="matrix-table">';
    html += '<thead><tr><th class="method-header">Method \\ Dataset</th>';
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
                html += `<td class="matrix-cell ${cls}" title="${esc(method)} on ${esc(ds)}: ${val}${cell.paper_id ? ' (' + esc(cell.paper_id) + ')' : ''}">${val}</td>`;
            } else {
                html += `<td class="matrix-cell cell-empty" title="No data">-</td>`;
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
                td.title = `${method} on ${ds}: ${val}`;
            } else {
                td.textContent = '-';
                td.className = 'matrix-cell cell-empty';
                td.title = 'No data';
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
        list.innerHTML = '<p class="empty-msg">No papers found.</p>';
        return;
    }

    list.innerHTML = filtered.map(p => {
        const sc = p.status ? 's-' + p.status : '';
        return `<div class="paper-row" data-paper-id="${esc(p.id)}" onclick="window._dg.togglePaper(this)">
            <div class="paper-row-top">
                <a class="paper-link" href="https://arxiv.org/abs/${esc(p.id)}" target="_blank" onclick="event.stopPropagation();">${esc(p.id)}</a>
                <span class="paper-title">${esc(trunc(p.title, 100))}</span>
                <span class="paper-date">${esc(p.created_at || '')}</span>
                <span class="paper-status ${sc}">${esc(p.status || '')}</span>
            </div>
            <div class="paper-expanded-body">
                <div class="paper-detail-loading">Loading details...</div>
            </div>
        </div>`;
    }).join('');
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
            html += '<div class="paper-claims"><strong style="color:var(--accent);font-size:0.75rem;">Claims & Insights</strong>';
            for (const c of claims.slice(0, 8)) {
                html += `<div class="paper-claim-item">${esc(c.claim_text || c.claim_type || '')}</div>`;
            }
            html += '</div>';
        }
        if (results.length > 0) {
            html += '<div class="paper-claims" style="margin-top:10px;"><strong style="color:var(--gold);font-size:0.75rem;">Results</strong>';
            for (const r of results.slice(0, 8)) {
                const val = r.metric_value != null ? Number(r.metric_value).toFixed(2) : '';
                html += `<div class="paper-claim-item">${esc(r.method_name || '')} on ${esc(r.dataset_name || '')}${val ? ': <b>' + val + '</b>' : ''} ${esc(r.metric_name || '')}</div>`;
            }
            html += '</div>';
        }
        if (!html) html = '<p class="empty-msg" style="padding:8px;">No claims or results extracted yet.</p>';
        body.innerHTML = html;
    } catch (e) {
        body.innerHTML = '<p class="empty-msg" style="padding:8px;">Failed to load details.</p>';
    }
}

// ── Opportunities Tab ────────────────────────────────────────────────

async function loadInsightsTab() {
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
            list.innerHTML = '<p class="empty-msg">No insights discovered yet. Run the pipeline to analyze papers.</p>';
            return;
        }

        list.innerHTML = insights.map(ins => {
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
                <div class="insight-hypothesis"><span class="insight-label">Hypothesis:</span> ${esc(ins.hypothesis)}</div>
                <div class="insight-experiment"><span class="insight-label">Experiment:</span> ${esc(ins.experiment)}</div>
                ${ins.impact ? `<div class="insight-impact"><span class="insight-label">Impact:</span> ${esc(ins.impact)}</div>` : ''}
                <div class="insight-actions">
                    <button class="btn-research" onclick="window._dg.launchResearch(${ins.id})">Generate Full Plan</button>
                    <button class="btn-preview" onclick="window._dg.previewProposal(${ins.id})">Preview Proposal</button>
                </div>
            </div>`;
        }).join('');
    } catch (e) {
        console.error('Insights tab error:', e);
    }
}

async function loadOpportunities() {
    oppsLoaded = true;
    try {
        // Load deep research insights
        const insights = await api('/api/insights?limit=100');
        allOpportunities = insights;
        allOpportunities.sort((a, b) =>
            ((b.novelty_score||0) + (b.feasibility_score||0)) - ((a.novelty_score||0) + (a.feasibility_score||0))
        );
        renderOpportunities();
    } catch (e) {
        console.error('Opportunities load error:', e);
    }
}

const insightTypeColors = {
    contradiction_analysis: { color: '#c4453a', label: 'Contradiction' },
    method_transfer:        { color: '#c4704b', label: 'Method Transfer' },
    assumption_challenge:   { color: '#a8842a', label: 'Assumption Challenge' },
    ignored_limitation:     { color: '#7c5cbf', label: 'Ignored Limitation' },
    paradigm_exhaustion:    { color: '#9a9088', label: 'Paradigm Exhaustion' },
    cross_domain_bridge:    { color: '#2e86ab', label: 'Cross-Domain Bridge' },
};

function renderOpportunities() {
    const list = el('oppList');
    const typeFilter = el('oppTypeFilter').value;

    // Rebuild filter dropdown with actual insight types
    const select = el('oppTypeFilter');
    const currentVal = select.value;
    if (select.options.length <= 1 && allOpportunities.length > 0) {
        // Clear old hardcoded options
        select.innerHTML = '<option value="">All types</option>';
        const types = [...new Set(allOpportunities.map(o => o.insight_type))].filter(Boolean).sort();
        for (const t of types) {
            const meta = insightTypeColors[t] || {};
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = meta.label || t.replace(/_/g, ' ');
            select.appendChild(opt);
        }
        if (currentVal) select.value = currentVal;
    }

    let filtered = allOpportunities;
    if (typeFilter) {
        filtered = filtered.filter(o => o.insight_type === typeFilter);
    }

    if (filtered.length === 0) {
        list.innerHTML = '<p class="empty-msg">No research insights yet. Run the pipeline to discover genuine research opportunities.</p>';
        return;
    }

    list.innerHTML = filtered.map(ins => {
        const meta = insightTypeColors[ins.insight_type] || { color: '#888', label: ins.insight_type };
        // Parse supporting papers for links
        let papers = [];
        try { papers = JSON.parse(ins.supporting_papers || '[]'); } catch(e) {}
        const paperLinks = papers.map(pid =>
            `<a class="paper-cite" href="https://arxiv.org/abs/${esc(pid)}" target="_blank">${esc(pid)}</a>`
        ).join(' ');

        return `<div class="opp-card" style="border-left: 3px solid ${meta.color};">
            <div class="opp-type-badge" style="background:${meta.color}22;color:${meta.color};">${esc(meta.label)}</div>
            <div class="opp-header">
                <div class="opp-title">${esc(ins.title)}</div>
                <div class="opp-score-group">
                    <span class="opp-score-item" title="Novelty" style="color:${(ins.novelty_score||0) >= 4 ? '#ffaa33' : '#6a7a8a'};">N:${ins.novelty_score || '?'}/5</span>
                    <span class="opp-score-item" title="Feasibility" style="color:${(ins.feasibility_score||0) >= 4 ? '#44dd88' : '#6a7a8a'};">F:${ins.feasibility_score || '?'}/5</span>
                </div>
            </div>
            ${paperLinks ? `<div class="opp-papers">${paperLinks}</div>` : ''}
            ${ins.evidence ? `<div class="opp-section">
                <div class="opp-section-label">Evidence</div>
                <div class="opp-evidence">${esc(ins.evidence)}</div>
            </div>` : ''}
            <div class="opp-section">
                <div class="opp-section-label">Hypothesis</div>
                <div class="opp-desc">${esc(ins.hypothesis)}</div>
            </div>
            ${ins.experiment ? `<div class="opp-section">
                <div class="opp-section-label">Proposed Experiment</div>
                <div class="opp-experiment">${esc(ins.experiment)}</div>
            </div>` : ''}
            ${ins.impact ? `<div class="opp-section">
                <div class="opp-section-label">Potential Impact</div>
                <div class="opp-impact">${esc(ins.impact)}</div>
            </div>` : ''}
            <div class="opp-footer">
                <span class="opp-source" onclick="switchTab('explore');navigateTo('${esc(ins.node_id)}')" style="cursor:pointer;">${esc(ins.node_id)}</span>
            </div>
        </div>`;
    }).join('');
}

// ── Discoveries Tab (Tier 1 + Tier 2 Deep Insights) ──────────────────

async function loadDiscoveriesTab() {
    const tierFilter = el('discoveryTierFilter')?.value || '';
    try {
        let url = '/api/deep_insights?limit=50';
        if (tierFilter) url += `&tier=${tierFilter}`;
        const insights = await api(url);
        renderDiscoveries(insights);
    } catch (e) {
        const list = el('discoveriesList');
        if (list) list.innerHTML = '<p class="empty-msg">No deep discoveries yet. Click Generate to run the discovery pipeline.</p>';
    }
}

function renderDiscoveries(discoveries) {
    const list = el('discoveriesList');
    if (!discoveries || discoveries.length === 0) {
        list.innerHTML = '<p class="empty-msg">No deep discoveries yet. Click Generate to run the discovery pipeline.</p>';
        return;
    }

    list.innerHTML = discoveries.map(d => {
        const isTier1 = d.tier === 1;
        const tierColor = isTier1 ? '#c4453a' : '#2e86ab';
        const tierLabel = isTier1 ? 'PARADIGM' : 'PAPER IDEA';

        const noveltyBadge = d.novelty_status === 'novel'
            ? '<span class="paradigm-badge high">NOVEL</span>'
            : d.novelty_status === 'partially_exists'
            ? '<span class="paradigm-badge mid">PARTIAL</span>'
            : d.novelty_status === 'exists'
            ? '<span class="paradigm-badge low">EXISTS</span>'
            : '<span class="paradigm-badge low">UNCHECKED</span>';

        const scoreBadge = d.adversarial_score
            ? `<span class="insight-scores">Adversarial: ${d.adversarial_score}/10</span>`
            : '';

        let bodyHtml = '';

        if (isTier1) {
            bodyHtml += d.formal_structure
                ? `<div class="insight-hypothesis"><span class="insight-label">Formal Structure:</span> ${esc(d.formal_structure)}</div>` : '';
            bodyHtml += d.transformation
                ? `<div class="insight-experiment"><span class="insight-label">Transformation:</span> ${esc(d.transformation)}</div>` : '';

            let fieldA = {}, fieldB = {};
            try { fieldA = JSON.parse(d.field_a || '{}'); } catch(e) {}
            try { fieldB = JSON.parse(d.field_b || '{}'); } catch(e) {}
            if (fieldA.node_id || fieldB.node_id) {
                bodyHtml += `<div class="insight-evidence">
                    <span class="insight-label">Fields:</span>
                    ${fieldA.node_id ? `<span class="chip" onclick="window._dg.exploreNode('${esc(fieldA.node_id)}')">${esc(fieldA.node_id)}</span>` : ''}
                    <span style="margin:0 4px;">↔</span>
                    ${fieldB.node_id ? `<span class="chip" onclick="window._dg.exploreNode('${esc(fieldB.node_id)}')">${esc(fieldB.node_id)}</span>` : ''}
                </div>`;
            }

            let predictions = [];
            try { predictions = JSON.parse(d.predictions || '[]'); } catch(e) {}
            if (predictions.length) {
                bodyHtml += '<div class="insight-experiment"><span class="insight-label">Predictions:</span><ul style="margin:4px 0;padding-left:20px;">';
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
                    bodyHtml += `<div class="insight-impact"><span class="insight-label">Strongest Challenge:</span> ${esc(critique.strongest_attack)}</div>`;
                }
            }
        } else {
            bodyHtml += d.problem_statement
                ? `<div class="insight-hypothesis"><span class="insight-label">Problem:</span> ${esc(d.problem_statement)}</div>` : '';
            bodyHtml += d.existing_weakness
                ? `<div class="insight-evidence"><span class="insight-label">Weakness:</span> ${esc(d.existing_weakness)}</div>` : '';

            let method = {};
            try { method = JSON.parse(d.proposed_method || '{}'); } catch(e) {}
            if (method.name) {
                bodyHtml += `<div class="insight-experiment">
                    <span class="insight-label">Method: ${esc(method.name)}</span> (${esc(method.type || '?')})
                    <div style="margin-top:4px;">${esc(method.one_line || '')}</div>
                    ${method.definition ? `<pre style="font-size:0.72rem;margin:6px 0;white-space:pre-wrap;color:var(--text-secondary);">${esc(trunc(method.definition, 300))}</pre>` : ''}
                </div>`;
            }

            let plan = {};
            try { plan = JSON.parse(d.experimental_plan || '{}'); } catch(e) {}
            if (plan.baselines && plan.baselines.length) {
                bodyHtml += '<div class="insight-impact"><span class="insight-label">Baselines:</span> ';
                bodyHtml += plan.baselines.map(b => esc(b.name || b)).join(', ');
                bodyHtml += '</div>';
            }
            if (plan.datasets && plan.datasets.length) {
                bodyHtml += '<div class="insight-impact"><span class="insight-label">Datasets:</span> ';
                bodyHtml += plan.datasets.map(ds => esc(ds.name || ds)).join(', ');
                bodyHtml += '</div>';
            }
            if (plan.compute_budget) {
                bodyHtml += `<div class="insight-impact"><span class="insight-label">Compute:</span> ${esc(plan.compute_budget.total_gpu_hours || '?')} GPU-hours</div>`;
            }
        }

        return `<div class="insight-card" style="border-left: 3px solid ${tierColor};">
            <div class="insight-header">
                <span class="insight-type" style="color:${tierColor};font-weight:700;">TIER ${d.tier}: ${tierLabel}</span>
                ${noveltyBadge}
                ${scoreBadge}
                <span style="color:var(--text-dim);font-size:0.68rem;">${esc(d.status || '')}</span>
            </div>
            <div class="insight-title">${esc(d.title)}</div>
            ${bodyHtml}
            ${d.evidence_summary ? `<div class="insight-evidence"><span class="insight-label">Evidence:</span> ${esc(trunc(d.evidence_summary, 250))}</div>` : ''}
            <div class="insight-actions">
                ${d.novelty_status === 'unchecked' ? `<button class="btn-preview" onclick="window._dg.verifyDiscovery(${d.id})">Verify Novelty</button>` : ''}
                <button class="btn-research" onclick="window._dg.runFullExperiment(${d.id})">SciForge Run</button>
                <button class="btn-preview" onclick="window._dg.forgeExperiment(${d.id})">Forge Only</button>
                <button class="btn-preview" onclick="window._dg.launchDeepResearch(${d.id})">Deep Research</button>
            </div>
        </div>`;
    }).join('');
}

async function generateDiscoveries(tier) {
    try {
        const body = tier ? {tier: String(tier)} : {};
        await api('/api/deep_insights/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        el('discoveriesList').innerHTML = '<p class="empty-msg">Discovery pipeline started... This may take 5-15 minutes. Refresh to see results.</p>';
    } catch (e) {
        alert('Failed to start discovery: ' + e.message);
    }
}

// ── Experiments Tab (SciForge) ────────────────────────────────────────

async function loadExperimentsTab() {
    const statusFilter = el('experimentStatusFilter')?.value || '';
    try {
        const autoStatus = await api('/api/auto_research/status');
        renderAutoResearchStatus(autoStatus);

        const autoJobs = await api('/api/auto_research/jobs?limit=30');
        renderAutoResearchJobs(autoJobs);

        let url = '/api/experiments?limit=50';
        if (statusFilter) url += `&status=${statusFilter}`;
        const runs = await api(url);
        renderExperiments(runs);

        const meta = await api('/api/meta_report');
        renderMetaReport(meta);
    } catch (e) {
        const list = el('experimentsList');
        if (list) list.innerHTML = '<p class="empty-msg">No experiments yet.</p>';
    }
}

function renderAutoResearchStatus(status) {
    const box = el('autoResearchStatus');
    if (!box) return;
    if (!status) {
        box.textContent = 'Auto Research status unavailable.';
        return;
    }
    const running = status.running ? 'RUNNING' : 'STOPPED';
    const evo = status.evoscientist_available ? 'EvoScientist ready' : 'EvoScientist missing';
    box.innerHTML = `
        <strong>${running}</strong>
        <span style="margin-left:10px;">Interval: ${status.interval_seconds || '?'}s</span>
        <span style="margin-left:10px;">${esc(evo)}</span>
        <span style="margin-left:10px;">Jobs: ${status.total || 0}</span>
        <span style="margin-left:10px;">Completed: ${status.completed || 0}</span>
        <span style="margin-left:10px;">Blocked: ${status.blocked || 0}</span>
    `;
}

function renderAutoResearchJobs(jobs) {
    const list = el('autoResearchList');
    if (!list) return;
    if (!jobs || !jobs.length) {
        list.innerHTML = '<p class="empty-msg">No auto research jobs yet.</p>';
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

    list.innerHTML = jobs.map(j => {
        const color = colors[j.status] || '#888';
        const cpu = j.cpu_eligible == null
            ? 'CPU unchecked'
            : (j.cpu_eligible ? 'CPU OK' : 'CPU blocked');
        const exp = j.experiment_status
            ? `<span class="insight-scores">Experiment: ${esc(j.experiment_status)}</span>`
            : '';
        const verdict = j.hypothesis_verdict
            ? `<span class="insight-scores">Verdict: ${esc(j.hypothesis_verdict)}</span>`
            : '';
        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">AUTO ${esc(j.status || 'queued').toUpperCase()}</span>
                <span class="insight-scores">${esc(cpu)}</span>
                ${exp}
                ${verdict}
            </div>
            <div class="insight-title">${esc(j.title || 'Deep Insight')}</div>
            <div class="insight-impact"><span class="insight-label">Stage:</span> ${esc(j.stage || '')}</div>
            ${j.novelty_status ? `<div class="insight-impact"><span class="insight-label">Novelty:</span> ${esc(j.novelty_status)}</div>` : ''}
            ${j.cpu_reason ? `<div class="insight-evidence"><span class="insight-label">CPU Check:</span> ${esc(j.cpu_reason)}</div>` : ''}
            ${j.last_note ? `<div class="insight-experiment"><span class="insight-label">Latest:</span> ${esc(trunc(j.last_note, 220))}</div>` : ''}
            ${j.last_error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">Error:</span> ${esc(trunc(j.last_error, 220))}</div>` : ''}
        </div>`;
    }).join('');
}

function renderExperiments(runs) {
    const list = el('experimentsList');
    if (!runs || !runs.length) {
        list.innerHTML = '<p class="empty-msg">No experiments yet. Forge an experiment from a deep discovery to start.</p>';
        return;
    }

    list.innerHTML = runs.map(r => {
        const statusColors = {
            pending: '#9a9088', scaffolding: '#a8842a', reproducing: '#2e86ab',
            testing: '#c4704b', completed: '#3d8b5e', failed: '#c4453a'
        };
        const verdictColors = {
            confirmed: '#3d8b5e', refuted: '#c4453a', inconclusive: '#a8842a'
        };
        const color = statusColors[r.status] || '#888';
        const vColor = verdictColors[r.hypothesis_verdict] || '#888';

        const effect = r.effect_pct != null ? `${r.effect_pct >= 0 ? '+' : ''}${r.effect_pct.toFixed(2)}%` : '';
        const verdict = r.hypothesis_verdict
            ? `<span style="color:${vColor};font-weight:700;text-transform:uppercase;">${r.hypothesis_verdict}</span>`
            : '';

        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">RUN #${r.id} [${esc(r.status)}]</span>
                ${verdict}
                ${effect ? `<span class="insight-scores">Effect: ${effect}</span>` : ''}
                <span style="color:var(--text-dim);font-size:0.68rem;">Tier ${r.insight_tier || '?'}</span>
            </div>
            <div class="insight-title">${esc(r.insight_title || 'Experiment')}</div>
            <div style="display:flex;gap:16px;margin:6px 0;font-size:0.75rem;color:var(--text-secondary);">
                <span>Iterations: ${r.iterations_total || 0} (${r.iterations_kept || 0} kept)</span>
                <span>Baseline: ${r.baseline_metric_value != null ? r.baseline_metric_value.toFixed(4) : '?'}</span>
                <span>Best: ${r.best_metric_value != null ? r.best_metric_value.toFixed(4) : '?'}</span>
            </div>
            ${r.codebase_url && r.codebase_url !== 'scratch' ? `<div style="font-size:0.7rem;color:var(--text-dim);">Repo: ${esc(r.codebase_url)}</div>` : ''}
            <div class="insight-actions">
                ${r.status === 'scaffolding' ? `<button class="btn-research" onclick="window._dg.runExperiment(${r.id})">Start Loop</button>` : ''}
                <button class="btn-preview" onclick="window._dg.viewExperiment(${r.id})">View Details</button>
            </div>
        </div>`;
    }).join('');
}

function renderMetaReport(meta) {
    const card = el('metaReportCard');
    const body = el('metaReportBody');
    if (!meta || meta.status === 'insufficient_data' || meta.total_experiments < 1) {
        card.style.display = 'none';
        return;
    }
    card.style.display = '';

    const tr = meta.track_record || {};
    let html = `<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px;">
        <div class="stat-card" style="min-width:100px;">
            <div class="stat-number">${meta.total_experiments}</div>
            <div class="stat-label">Experiments</div>
        </div>
        <div class="stat-card" style="min-width:100px;">
            <div class="stat-number" style="color:#3d8b5e;">${tr.total_confirmed || 0}</div>
            <div class="stat-label">Confirmed</div>
        </div>
        <div class="stat-card" style="min-width:100px;">
            <div class="stat-number" style="color:#c4453a;">${tr.total_refuted || 0}</div>
            <div class="stat-label">Refuted</div>
        </div>
        <div class="stat-card" style="min-width:100px;">
            <div class="stat-number">${((tr.overall_hit_rate || 0) * 100).toFixed(1)}%</div>
            <div class="stat-label">Hit Rate</div>
        </div>
    </div>`;

    if (tr.signal_types && tr.signal_types.length) {
        html += '<h4 style="margin:12px 0 6px;">Signal Type Performance</h4>';
        html += '<table class="matrix-table" style="font-size:0.75rem;"><thead><tr><th>Signal</th><th>Total</th><th>Confirmed</th><th>Refuted</th><th>Hit Rate</th></tr></thead><tbody>';
        for (const s of tr.signal_types) {
            html += `<tr><td>${esc(s.signal_type)}</td><td>${s.hypothesis_count}</td><td style="color:#3d8b5e;">${s.confirmed_count}</td><td style="color:#c4453a;">${s.refuted_count}</td><td><b>${((s.hit_rate || 0) * 100).toFixed(1)}%</b></td></tr>`;
        }
        html += '</tbody></table>';
    }

    const weights = meta.signal_weights || {};
    if (Object.keys(weights).length) {
        html += '<h4 style="margin:12px 0 6px;">Learned Signal Weights</h4><div class="chip-row">';
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
        el('providersList').innerHTML = '<p class="empty-msg">Failed to load provider data.</p>';
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
                list.innerHTML = '<p class="empty-msg">No provider data available.</p>';
                return;
            }
            renderProviderCards(arr);
            return;
        }
        list.innerHTML = '<p class="empty-msg">No provider data available.</p>';
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
            <div class="provider-name">${esc(p.name || p.provider || 'Unknown')}</div>
            <div class="provider-url">${esc(p.base_url || p.url || '')}</div>
            <div class="provider-stats">
                <div class="provider-stat">
                    <span class="provider-stat-val cyan">${fmt(calls)}</span>
                    <span class="provider-stat-lbl">Calls</span>
                </div>
                <div class="provider-stat">
                    <span class="provider-stat-val gold">${fmt(tokens)}</span>
                    <span class="provider-stat-lbl">Tokens</span>
                </div>
                <div class="provider-stat">
                    <span class="provider-stat-val ${errors > 0 ? 'red' : 'green'}">${fmt(errors)}</span>
                    <span class="provider-stat-lbl">Errors</span>
                </div>
                <div class="provider-stat">
                    <span class="provider-stat-val">${latency ? latency.toFixed(1) + 's' : '-'}</span>
                    <span class="provider-stat-lbl">Avg Latency</span>
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
        results.innerHTML = '<div class="search-section"><p class="empty-msg">Search failed.</p></div>';
        results.classList.add('open');
    }
}

function renderSearchResults(data) {
    const results = el('searchResults');
    let html = '';

    if (data.nodes && data.nodes.length) {
        html += '<div class="search-section"><div class="search-section-title">Taxonomy Nodes</div>';
        for (const n of data.nodes) {
            html += `<div class="search-result-item" onclick="window._dg.searchNav('node','${esc(n.id)}')">
                <div class="sr-title">${esc(n.name)}</div>
                <div class="sr-meta">${esc(n.id)} \u00B7 ${n.paper_count || 0} papers</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.papers && data.papers.length) {
        html += '<div class="search-section"><div class="search-section-title">Papers</div>';
        for (const p of data.papers.slice(0, 8)) {
            html += `<div class="search-result-item" onclick="window.open('https://arxiv.org/abs/${esc(p.id)}','_blank')">
                <div class="sr-title">${esc(trunc(p.title, 70))}</div>
                <div class="sr-meta">${esc(p.id)}${p.work_type ? ' \u00B7 ' + esc(p.work_type) : ''}${p.published_date ? ' \u00B7 ' + esc(p.published_date) : ''}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.methods && data.methods.length) {
        html += '<div class="search-section"><div class="search-section-title">Methods</div>';
        for (const m of data.methods) {
            html += `<div class="search-result-item">
                <div class="sr-title">${esc(m.name)}</div>
                <div class="sr-meta">${m.paper_count || 0} papers \u00B7 ${m.result_count || 0} results</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.opportunities && data.opportunities.length) {
        html += '<div class="search-section"><div class="search-section-title">Opportunities</div>';
        for (const o of data.opportunities) {
            html += `<div class="search-result-item" onclick="window._dg.searchNav('node','${esc(o.node_id)}')">
                <div class="sr-title">${esc(o.title)}</div>
                <div class="sr-meta">${esc(o.node_name || o.node_id)} \u00B7 score ${o.value_score || '?'}/5</div>
            </div>`;
        }
        html += '</div>';
    }

    if (data.gaps && data.gaps.length) {
        html += '<div class="search-section"><div class="search-section-title">Gaps</div>';
        for (const g of data.gaps) {
            html += `<div class="search-result-item" onclick="window._dg.searchNav('node','${esc(g.node_id)}')">
                <div class="sr-title">${esc(g.method_name)} on ${esc(g.dataset_name)}</div>
                <div class="sr-meta">${esc(trunc(g.gap_description, 90))}</div>
            </div>`;
        }
        html += '</div>';
    }

    if (!html) {
        html = '<div class="search-section"><p class="empty-msg">No results found.</p></div>';
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

// ── Pipeline Controls ────────────────────────────────────────────────

async function startPipeline(n) {
    try {
        await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_papers: n })
        });
    } catch (e) {
        console.error('Start pipeline error:', e);
    }
}

async function triggerTaxonomyExpansion() {
    try {
        await fetch('/api/taxonomy/expand', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ min_papers: 10 })
        });
    } catch (e) {
        console.error('Taxonomy expansion error:', e);
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

    async launchResearch(insightId) {
        if (!confirm('Launch EvoScientist research for this insight? This will start a background research session.')) return;
        try {
            const res = await api('/api/research/launch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({insight_id: insightId}),
            });
            alert(`Research launched!\n\nWorkdir: ${res.workdir}\nPID: ${res.pid}\n\nEvoScientist is now working in the background.`);
        } catch (e) {
            alert('Failed to launch: ' + e.message);
        }
    },

    async forgeExperiment(insightId) {
        if (!confirm('Forge an experiment from this discovery? This will find a codebase, generate scaffold, and prepare for the validation loop.')) return;
        try {
            await api('/api/experiments/forge', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({insight_id: insightId}),
            });
            alert('Experiment forge started! Switch to the Experiments tab to monitor.');
            setTimeout(() => { switchTab('experiments'); loadExperimentsTab(); }, 2000);
        } catch (e) {
            alert('Failed: ' + e.message);
        }
    },

    async runFullExperiment(insightId) {
        if (!confirm('Run FULL SciForge pipeline? This will:\n1. Find/clone codebase\n2. Generate scaffold\n3. Run validation loop (may take hours)\n4. Update knowledge graph\n\nContinue?')) return;
        try {
            await api('/api/experiments/run_full', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({insight_id: insightId}),
            });
            alert('Full SciForge pipeline started! Monitor in the Experiments tab.');
            setTimeout(() => { switchTab('experiments'); loadExperimentsTab(); }, 2000);
        } catch (e) {
            alert('Failed: ' + e.message);
        }
    },

    async runExperiment(runId) {
        if (!confirm('Start the validation loop for this experiment?')) return;
        try {
            await api(`/api/experiments/${runId}/run`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: '{}',
            });
            alert('Validation loop started!');
            setTimeout(loadExperimentsTab, 3000);
        } catch (e) {
            alert('Failed: ' + e.message);
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
                    <h3>Experiment #${run.id}: ${esc(run.insight_title || '')}</h3>
                    <span class="proposal-stats">Status: ${esc(run.status)} | Verdict: ${esc(run.hypothesis_verdict || 'pending')}</span>
                    <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">×</button>
                </div>
                <div class="proposal-body">
                <h4>Metrics</h4>
                <p>Baseline: ${run.baseline_metric_value || '?'} | Best: ${run.best_metric_value || '?'} | Effect: ${run.effect_pct != null ? run.effect_pct.toFixed(2) + '%' : '?'}</p>
                <p>Iterations: ${run.iterations_total || 0} total, ${run.iterations_kept || 0} kept</p>
                ${run.codebase_url ? `<p>Codebase: <a href="${esc(run.codebase_url)}" target="_blank">${esc(run.codebase_url)}</a></p>` : ''}
                ${run.error_message ? `<p style="color:#c4453a;">Error: ${esc(run.error_message)}</p>` : ''}`;

            if (iters.length) {
                html += '<h4>Iteration History</h4><table class="matrix-table" style="font-size:0.72rem;"><thead><tr><th>#</th><th>Phase</th><th>Metric</th><th>Status</th><th>Description</th></tr></thead><tbody>';
                for (const it of iters.slice(-30)) {
                    const sColor = it.status === 'keep' ? '#3d8b5e' : it.status === 'crash' ? '#c4453a' : '#9a9088';
                    html += `<tr><td>${it.iteration_number}</td><td>${esc(it.phase)}</td><td>${it.metric_value != null ? it.metric_value.toFixed(6) : '-'}</td><td style="color:${sColor};">${esc(it.status)}</td><td>${esc(trunc(it.description || '', 60))}</td></tr>`;
                }
                html += '</tbody></table>';
            }

            if (claims.length) {
                html += '<h4>Experimental Claims</h4>';
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
            alert('Failed to load: ' + e.message);
        }
    },

    async verifyDiscovery(insightId) {
        if (!confirm('Launch novelty verification for this discovery? EvoScientist will search the literature (5-15 min).')) return;
        try {
            const res = await api(`/api/deep_insights/${insightId}/verify`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: '{}',
            });
            alert(`Verification launched!\nWorkdir: ${res.workdir}\nPID: ${res.pid}`);
            setTimeout(loadDiscoveriesTab, 3000);
        } catch (e) {
            alert('Failed to launch verification: ' + e.message);
        }
    },

    async launchDeepResearch(insightId) {
        if (!confirm('Launch full EvoScientist research for this deep insight? This will start a longer background session.')) return;
        try {
            const res = await api(`/api/deep_insights/${insightId}/research`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: '{}',
            });
            alert(`Research launched!\nWorkdir: ${res.workdir}\nPID: ${res.pid}`);
        } catch (e) {
            alert('Failed to launch: ' + e.message);
        }
    },

    async startAutoResearch() {
        try {
            await api('/api/auto_research/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: '{}',
            });
            loadExperimentsTab();
        } catch (e) {
            alert('Failed to start auto research: ' + e.message);
        }
    },

    async stopAutoResearch() {
        try {
            await api('/api/auto_research/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: '{}',
            });
            loadExperimentsTab();
        } catch (e) {
            alert('Failed to stop auto research: ' + e.message);
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
                        <span class="proposal-stats">${res.paper_count} papers · ${res.claim_count} claims · ${res.contradiction_count} contradictions</span>
                        <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">\u00D7</button>
                    </div>
                    <div class="proposal-body">${bodyHtml}</div>
                    <div class="proposal-footer">
                        <button class="btn-research" onclick="window._dg.launchResearch(${insightId}); this.closest('.proposal-modal').remove();">Launch Research</button>
                    </div>
                </div>`;
            document.body.appendChild(modal);
        } catch (e) {
            alert('Failed to load proposal: ' + e.message);
        }
    },
};

// ── Init ─────────────────────────────────────────────────────────────

function init() {
    // Nav items
    $$('.nav-item').forEach(btn => {
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
    if (dtf) dtf.addEventListener('change', loadDiscoveriesTab);
    const btnGen = el('btnGenerateDiscoveries');
    if (btnGen) btnGen.addEventListener('click', () => generateDiscoveries());

    // Experiment filters
    const esf = el('experimentStatusFilter');
    if (esf) esf.addEventListener('change', loadExperimentsTab);
    const autoStart = el('btnAutoResearchStart');
    if (autoStart) autoStart.addEventListener('click', () => window._dg.startAutoResearch());
    const autoStop = el('btnAutoResearchStop');
    if (autoStop) autoStop.addEventListener('click', () => window._dg.stopAutoResearch());

    // Insight filters
    const itf = el('insightTypeFilter');
    const isf = el('insightSortFilter');
    if (itf) itf.addEventListener('change', loadInsightsTab);
    if (isf) isf.addEventListener('change', loadInsightsTab);

    // Pipeline controls
    el('btnStart20').addEventListener('click', () => startPipeline(20));
    el('btnStart100').addEventListener('click', () => startPipeline(100));
    el('btnExpand').addEventListener('click', triggerTaxonomyExpansion);

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
    loadOverviewGraph();
    loadProcessingPapers();
    startSSE();

    // Stats refresh every 15s
    statsTimer = setInterval(refreshStats, 15000);

    // Processing panel refresh every 3s (also fetches from API)
    setInterval(loadProcessingPapers, 3000);

    // Periodically refresh recently discovered (every 30s)
    setInterval(loadRecentlyDiscovered, 30000);
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
