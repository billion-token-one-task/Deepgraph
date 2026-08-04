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
let agentOfficeData = null;       // /api/agent_office snapshot
let statsCache      = null;
let allPapers       = [];
let selectedPaperId = null;
let allOpportunities = [];
let taxonomyFlat    = [];        // flat list for Evidence dropdown
let searchTimer     = null;
let statsTimer      = null;
let papersLoaded    = false;
let oppsLoaded      = false;
let sidebarCollapsed = false;
let currentAgendaId = null;      // active research agenda scope for API calls
let agendaList      = [];        // /api/v1/agendas payload
let evidenceStateMap = null;     // /api/v1/evidence_states for currentAgendaId

// ── Helpers ──────────────────────────────────────────────────────────

function fmt(n) {
    if (n == null) return '0';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
}

// i18n bridge: translate via window.t (from i18n.js) with an English
// fallback for keys the dictionary does not carry.
function tr(key, fallback) {
    if (window.t) {
        const v = window.t(key);
        if (v !== key) return v;
    }
    return fallback;
}

function esc(str) {
    if (str == null) return "";
    const d = document.createElement("div");
    d.textContent = String(str);
    return d.innerHTML;
}

function trunc(str, max) {
    if (str == null) return "";
    str = String(str);
    return str.length > max ? str.slice(0, max - 1) + "\u2026" : str;
}


function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }
function el(id) { return document.getElementById(id); }

// Endpoints that the server scopes to one research agenda. The active agenda
// id is appended automatically once initAgendaScope() has resolved it.
const AGENDA_SCOPED_PATHS = [
    '/api/deep_insights', '/api/generated_papers', '/api/experiment_groups',
    '/api/experiments', '/api/manuscripts', '/api/submission_bundles',
    '/api/meta_report', '/api/v1/evidence_states',
];

function withAgendaScope(path) {
    if (currentAgendaId == null) return path;
    if (path.includes('agenda_id=')) return path;
    if (!AGENDA_SCOPED_PATHS.some(p => path.startsWith(p))) return path;
    return path + (path.includes('?') ? '&' : '?') + 'agenda_id=' + currentAgendaId;
}

async function api(path, opts) {
    const r = await fetch(withAgendaScope(path), opts);
    if (!r.ok) throw new Error(`API ${path} returned ${r.status}`);
    return r.json();
}

async function initAgendaScope() {
    try {
        const data = await api('/api/v1/agendas');
        agendaList = data.agendas || [];
        const active = agendaList.find(a => a.is_active) || agendaList[0];
        if (active) currentAgendaId = active.id;
    } catch (e) {
        console.error('Agenda scope unavailable:', e);
    }
}

function mathJaxConfig() {
    return {
        tex: {
            inlineMath: [["\\(", "\\)"], ["$", "$"]],
            displayMath: [["\\[", "\\]"], ["$$", "$$"]],
            processEscapes: true,
            processEnvironments: true
        },
        options: {
            skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
        },
        startup: {
            typeset: false,
            ready: () => {
                window.MathJax.startup.defaultReady();
                if (window._dgTypesetMath) window._dgTypesetMath(document.body);
            }
        }
    };
}

function ensureMathJax() {
    const existingScript = document.getElementById("MathJax-script") || document.querySelector('script[src*="mathjax"]');
    if (!window.MathJax) window.MathJax = mathJaxConfig();
    if (window.MathJax.typesetPromise || existingScript || window.MathJax._dgLoading) return;
    window.MathJax._dgLoading = true;
    const script = document.createElement("script");
    script.id = "MathJax-script";
    script.defer = true;
    script.src = "/static/vendor/mathjax/tex-svg.js";
    document.head.appendChild(script);
}

const LATEX_OVERESCAPED_TOKEN = /\\\\([()\[\]]|[A-Za-z]+|[%#$&_{}])/g;

function normalizeLatexEscapes(text) {
    return String(text || "").replace(LATEX_OVERESCAPED_TOKEN, "\\$1");
}

function normalizeLatexInNode(root) {
    const skipSelector = "script,noscript,style,textarea,pre,code";
    const normalizeTextNode = node => {
        if (!node || !node.nodeValue || !node.nodeValue.includes("\\\\")) return;
        const parent = node.parentElement;
        if (parent && parent.closest(skipSelector)) return;
        const next = normalizeLatexEscapes(node.nodeValue);
        if (next !== node.nodeValue) node.nodeValue = next;
    };
    if (root.nodeType === Node.TEXT_NODE) {
        normalizeTextNode(root);
        return;
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
        normalizeTextNode(node);
        node = walker.nextNode();
    }
}

const EXISTING_MATH_DELIM = /(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]+\$)/g;
const BARE_LATEX_MARKER = /\\[A-Za-z]+|\\[%#$&_{}]/g;
const BARE_SUBSCRIPT_MARKER = /[A-Za-zΑ-Ωα-ω]+_\{?[A-Za-z0-9Α-Ωα-ω]+\}?/;
const BARE_SUBSCRIPT_RUN = /(^|[\s([{,;])((?:[A-Za-zΑ-Ωα-ω]+_\{?[A-Za-z0-9Α-Ωα-ω]+\}?)(?:\s*,\s*(?:\.\.\.,\s*)?[A-Za-zΑ-Ωα-ω]+_\{?[A-Za-z0-9Α-Ωα-ω]+\}?)*)/g;
const BARE_LATEX_BOUNDARY = /^(\s+(?:over|under|where|which|while|when|after|before|with|without|for|from|to|by|as|that|because|rather|than|and|or|is|are|be|uses|construct|decode|chosen|emphasizes|computed|obtains|remains|subject|stop|let|then|else)\b|[,;:]\s+(?:where|which|while|when|after|before|with|without|for|from|to|by|as|that|because|rather|than|and|or|construct|decode|chosen|emphasizes|computed|subject|stop|let|then|else)\b|\.\s+[A-Z])/;

function latexFragmentLeft(segment, markerIndex, floor) {
    let start = markerIndex;
    while (start > floor && !/\s/.test(segment[start - 1]) && !/[;:]/.test(segment[start - 1])) start -= 1;
    return start;
}

function latexFragmentRight(segment, markerEnd) {
    let end = markerEnd;
    while (end < segment.length) {
        const rest = segment.slice(end);
        if (BARE_LATEX_BOUNDARY.test(rest)) break;
        if (segment[end] === "\n") break;
        end += 1;
    }
    return end;
}

function wrapBareLatexSegment(segment) {
    let out = "";
    let pos = 0;
    BARE_LATEX_MARKER.lastIndex = 0;
    let match;
    while ((match = BARE_LATEX_MARKER.exec(segment)) !== null) {
        if (match.index < pos) continue;
        const start = latexFragmentLeft(segment, match.index, pos);
        const end = latexFragmentRight(segment, match.index + match[0].length);
        const fragment = segment.slice(start, end).trim();
        if (!fragment || fragment.length < 2) continue;
        out += segment.slice(pos, start);
        const leading = segment.slice(start, end).match(/^\s*/)[0];
        const trailing = segment.slice(start, end).match(/\s*$/)[0];
        const core = segment.slice(start + leading.length, end - trailing.length);
        out += leading + "\\(" + core + "\\)" + trailing;
        pos = end;
        BARE_LATEX_MARKER.lastIndex = pos;
    }
    return out + segment.slice(pos);
}

function isExistingMathPart(part) {
    EXISTING_MATH_DELIM.lastIndex = 0;
    const yes = EXISTING_MATH_DELIM.test(part);
    EXISTING_MATH_DELIM.lastIndex = 0;
    return yes;
}

function splitOutsideExistingMath(value, transform) {
    return String(value || "").split(EXISTING_MATH_DELIM).map(part => {
        if (!part || isExistingMathPart(part)) return part;
        return transform(part);
    }).join("");
}

function wrapBareSubscriptRuns(segment) {
    BARE_SUBSCRIPT_RUN.lastIndex = 0;
    return String(segment || "").replace(BARE_SUBSCRIPT_RUN, (match, prefix, core) => `${prefix}\\(${core}\\)`);
}

function wrapBareInlineLatex(text) {
    const value = String(text || "");
    if (!(/\\[A-Za-z]+|\\[%#$&_{}]/.test(value) || BARE_SUBSCRIPT_MARKER.test(value))) return value;
    const latexWrapped = splitOutsideExistingMath(value, part => wrapBareLatexSegment(part));
    return splitOutsideExistingMath(latexWrapped, part => wrapBareSubscriptRuns(part));
}

function wrapBareInlineLatexInNode(root) {
    const skipSelector = "script,noscript,style,textarea,pre,code,mjx-container";
    const mathTextSelector = ".paper-reader-section,.paper-claim-list,.insight-card,.opp-card,.proposal-body";
    const wrapTextNode = node => {
        if (!node || !node.nodeValue) return;
        if (!node.nodeValue.includes("\\") && !BARE_SUBSCRIPT_MARKER.test(node.nodeValue)) return;
        const parent = node.parentElement;
        if (parent && (parent.closest(skipSelector) || !parent.closest(mathTextSelector))) return;
        const next = wrapBareInlineLatex(node.nodeValue);
        if (next !== node.nodeValue) node.nodeValue = next;
    };
    if (root.nodeType === Node.TEXT_NODE) {
        wrapTextNode(root);
        return;
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
        wrapTextNode(node);
        node = walker.nextNode();
    }
}

function typesetMath(root) {
    if (!root) return;
    normalizeLatexInNode(root);
    wrapBareInlineLatexInNode(root);
    const mj = window.MathJax;
    if (!mj || !mj.typesetPromise) {
        ensureMathJax();
        return;
    }
    window.clearTimeout(root._dgMathTimer);
    root._dgMathTimer = window.setTimeout(() => {
        mj.typesetPromise([root]).catch(err => console.warn("MathJax typeset failed:", err));
    }, 0);
}
window._dgTypesetMath = typesetMath;
ensureMathJax();

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
        case 'overview':
            loadRecentlyDiscovered();
            break;
        case 'office':
            loadProcessingPapers();
            window.requestAnimationFrame(() => {
                if (agentOfficeRenderer) agentOfficeRenderer.rebuildForCurrentSize();
            });
            break;
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
        // Cold-start marker from the server-side stats cache: keep whatever is
        // on screen instead of overwriting real numbers with zeros.
        if (s && s.warming) return;
        statsCache = s;

        // Top bar
        el('hdrPapers').textContent  = fmt(s.papers_processed || 0);
        el('hdrResults').textContent = fmt(s.results_total || 0);
        el('hdrInsights').textContent = fmt(s.insights_total || 0);
        el('hdrTokens').textContent  = fmt(s.tokens_consumed || 0);

        // Core stat row (always visible)
        el('statPapers').textContent        = fmt(s.papers_processed || 0);
        el('statDeepDiscoveries').textContent = fmt(s.deep_insights_total || 0);
        el('statExperiments').textContent   = fmt(s.experiment_runs_total || 0);
        const decidedEl = el('statDecided');
        if (decidedEl) {
            decidedEl.textContent = fmt(s.scientific_decisions_total || 0);
            const card = decidedEl.closest('.stat-card');
            if (card && s.scientific_decisions_total > 0) {
                card.title = `Audited verdicts: ${s.decisions_supported || 0} supported, `
                    + `${s.decisions_refuted || 0} refuted, ${s.decisions_inconclusive || 0} inconclusive.`;
            }
        }
        el('statTokens').textContent        = fmt(s.tokens_consumed || 0);

        // Detail stat cards (collapsed section)
        el('statResults').textContent       = fmt(s.results_total || 0);
        el('statTaxonomy').textContent = fmt(s.taxonomy_nodes_total || 0);
        el('statContradictions').textContent = fmt(s.contradictions_total || 0);
        el('statInsights').textContent      = fmt(s.insights_total || 0);
        el('statAgendaTokens').textContent  = fmt(s.agenda_tokens_total || 0);
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
        const [data, office] = await Promise.all([api("/api/processing"), api("/api/agent_office").catch(() => null)]);
        if (office && Array.isArray(office.departments)) agentOfficeData = office;
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

const OFFICE_ASSET_BASE = "/static/vendor/pixel-agents/assets";

const OFFICE_CHARACTER_COUNT = 6;
const OFFICE_FRAME_W = 16;
const OFFICE_FRAME_H = 32;

function officeAsset(path) {
    return `${OFFICE_ASSET_BASE}/${path}`;
}

const OFFICE_FALLBACK_DEPARTMENTS = [
    { key: "paper_extraction", title: "Paper Extraction", accent: "blue", status: "working", responsibility: "Discover papers, parse PDFs, extract claims, and audit completeness.", sub_agents: [], items: [] },
    { key: "graph_construction", title: "Graph Construction", accent: "green", status: "idle", responsibility: "Maintain taxonomy, evidence graph, and opportunity signals.", sub_agents: [], items: [] },
    { key: "idea_generation", title: "Idea Generation", accent: "gold", status: "idle", responsibility: "Generate, rank, route, and verify research ideas.", sub_agents: [], items: [] },
    { key: "experiment_planning", title: "Experiment Planning", accent: "purple", status: "idle", responsibility: "Turn ideas into benchmark contracts and reviewed plans.", sub_agents: [], items: [] },
    { key: "experiment_execution", title: "Experiment Execution", accent: "red", status: "idle", responsibility: "Run validation loops, code agents, GPU jobs, and merge watchers.", sub_agents: [], items: [] },
    { key: "manuscript_generation", title: "Manuscript Generation", accent: "cyan", status: "idle", responsibility: "Draft, audit, refine, and bundle manuscripts.", sub_agents: [], items: [] },
    { key: "orchestration", title: "Orchestration", accent: "slate", status: "working", responsibility: "Coordinate workers, schedules, web service, and deployment hooks.", sub_agents: [], items: [] }
];

function officeFallbackSnapshot() {
    const activeEntries = Object.entries(activePapers).filter(([, info]) => !info.done);
    const departments = OFFICE_FALLBACK_DEPARTMENTS.map(dep => ({ ...dep, sub_agents: dep.sub_agents || [], items: [] }));
    if (activeEntries.length) {
        departments[0].items = activeEntries.slice(0, 4).map(([pid, info]) => ({
            title: info.title || pid,
            status: info.step || "processing",
            detail: pid,
            kind: "paper"
        }));
        departments[0].status = "working";
    }
    const activeCount = departments.filter(dep => dep.status === "working").length;
    return { departments, summary: { departments: departments.length, sub_agents: 0, working: activeCount, blocked: 0 } };
}

function officeSnapshot() {
    if (agentOfficeData && Array.isArray(agentOfficeData.departments)) return agentOfficeData;
    return officeFallbackSnapshot();
}

function officeStatusLabel(status) {
    if (status === "blocked") return "needs attention";
    if (status === "working") return "active";
    return "idle";
}

function officeLeadText(dep) {
    const items = Array.isArray(dep.items) ? dep.items : [];
    if (items.length) {
        const lead = items[0];
        return `${lead.status || dep.status}: ${lead.title || dep.title}`;
    }
    if (dep.status === "blocked") return "waiting on a repair path";
    if (dep.status === "working") return "coordinating background work";
    return "standing by";
}

function officeAgentAction(agent, dep, index) {
    const name = `${agent.name || agent.path || ""} ${agent.path || ""}`.toLowerCase();
    const status = dep.status || "idle";
    if (status === "blocked") return "blocked";
    if (name.includes("reader") || name.includes("extract") || name.includes("pdf") || name.includes("paper")) return "reading";
    if (name.includes("writer") || name.includes("manuscript") || name.includes("orchestra") || name.includes("figure")) return "typing";
    if (name.includes("experiment") || name.includes("forge") || name.includes("gpu") || name.includes("benchmark")) return "typing";
    if (name.includes("graph") || name.includes("taxonomy") || name.includes("map") || name.includes("route")) return "walking";
    return status === "working" ? (index % 3 === 0 ? "typing" : "walking") : "idle";
}

function officeActionVerb(action, depStatus) {
    if (depStatus === "blocked") return "needs review";
    if (action === "typing") return "drafting";
    if (action === "reading") return "reading";
    if (action === "walking") return "routing";
    return "resting";
}

function officeAccentColor(accent) {
    return {
        blue: "#3a76b8",
        green: "#3d8b5e",
        gold: "#a8842a",
        purple: "#7c5cbf",
        red: "#c4453a",
        cyan: "#2f8c99",
        slate: "#6f6256"
    }[accent || ""] || "#c4704b";
}

function officeHash(value) {
    let h = 2166136261;
    const s = String(value || "");
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}

function canvasRoundRect(ctx, x, y, w, h, r) {
    const rr = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
}

function canvasFillRoundRect(ctx, x, y, w, h, r, fill, stroke, lineWidth) {
    canvasRoundRect(ctx, x, y, w, h, r);
    if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
    }
    if (stroke) {
        ctx.lineWidth = lineWidth || 1;
        ctx.strokeStyle = stroke;
        ctx.stroke();
    }
}

function canvasTextFit(ctx, text, x, y, maxWidth) {
    let value = String(text || "");
    if (!value) return;
    if (ctx.measureText(value).width <= maxWidth) {
        ctx.fillText(value, x, y);
        return;
    }
    while (value.length > 1 && ctx.measureText(value + "...").width > maxWidth) {
        value = value.slice(0, -1);
    }
    ctx.fillText(value + "...", x, y);
}

function canvasWrapText(ctx, text, x, y, maxWidth, lineHeight, maxLines) {
    const words = String(text || "").split(/\s+/).filter(Boolean);
    let line = "";
    let lines = 0;
    for (const word of words) {
        const test = line ? `${line} ${word}` : word;
        if (ctx.measureText(test).width > maxWidth && line) {
            canvasTextFit(ctx, line, x, y + lines * lineHeight, maxWidth);
            lines += 1;
            line = word;
            if (lines >= maxLines) return;
        } else {
            line = test;
        }
    }
    if (line && lines < maxLines) canvasTextFit(ctx, line, x, y + lines * lineHeight, maxWidth);
}

function loadOfficeImage(src) {
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = src;
    });
}

function renderProcessingList() {
    const listEl = el("processingList");
    const countEl = el("processingCount");
    if (!listEl || !countEl) return;

    const data = officeSnapshot();
    const departments = Array.isArray(data.departments) ? data.departments : [];
    const summary = data.summary || {};
    const working = summary.working || departments.filter(dep => dep.status === "working").length;
    const blocked = summary.blocked || departments.filter(dep => dep.status === "blocked").length;
    const totalAgents = summary.sub_agents || departments.reduce((n, dep) => n + ((dep.sub_agents || []).length), 0);
    countEl.textContent = `${working} active / ${totalAgents} agents`;

    if (!el("agentOfficeCanvas")) {
        listEl.innerHTML = `<div class="agent-office-map">
            <div class="office-hud">
                <div><span>Departments</span><strong id="officeDeptCount">0</strong></div>
                <div><span>Sub-agents</span><strong id="officeAgentCount">0</strong></div>
                <div><span>Active</span><strong id="officeActiveCount">0</strong></div>
                <div><span>Blocked</span><strong id="officeBlockedCount">0</strong></div>
            </div>
            <div class="office-nowline" id="officeNowline"></div>
            <div class="office-stage"><canvas id="agentOfficeCanvas" aria-label="DeepGraph agent office"></canvas></div>
        </div>`;
    }

    const deptEl = el("officeDeptCount");
    const agentEl = el("officeAgentCount");
    const activeEl = el("officeActiveCount");
    const blockedEl = el("officeBlockedCount");
    if (deptEl) deptEl.textContent = departments.length;
    if (agentEl) agentEl.textContent = totalAgents;
    if (activeEl) activeEl.textContent = working;
    if (blockedEl) blockedEl.textContent = blocked;

    const nowItems = departments.flatMap(dep => (dep.items || []).slice(0, 1).map(item => ({ dep: dep.title, item }))).slice(0, 5);
    const nowLineEl = el("officeNowline");
    if (nowLineEl) {
        nowLineEl.innerHTML = nowItems.length
            ? nowItems.map(({ dep, item }) => `<span><b>${esc(dep)}</b>: ${esc(trunc(item.title || item.status || "working", 56))}</span>`).join("")
            : `<span><b>System</b>: waiting for the next scheduled job</span>`;
    }

    syncAgentOfficeCanvas(data);
}

let agentOfficeRenderer = null;

function syncAgentOfficeCanvas(data) {
    const canvas = el("agentOfficeCanvas");
    if (!canvas) return;
    if (!agentOfficeRenderer || agentOfficeRenderer.canvas !== canvas) {
        agentOfficeRenderer = new AgentOfficeCanvas(canvas);
    }
    agentOfficeRenderer.setData(data);
}

function AgentOfficeCanvas(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.assets = {};
    this.ready = false;
    this.data = null;
    this.scene = { rooms: [], agents: [], world: { w: 1800, h: 1160 } };
    this.hovered = null;
    this.mouse = null;
    this.lastLayoutWidth = 0;
    this.raf = null;
    this.t = 0;
    this.assetPromise = this.loadAssets();
    this.bind();
    this.start();
}

AgentOfficeCanvas.prototype.loadAssets = async function() {
    const paths = {
        floor: officeAsset("floors/floor_0.png"),
        wall: officeAsset("walls/wall_0.png"),
        desk: officeAsset("furniture/DESK/DESK_FRONT.png"),
        pcOn1: officeAsset("furniture/PC/PC_FRONT_ON_1.png"),
        pcOn2: officeAsset("furniture/PC/PC_FRONT_ON_2.png"),
        pcOn3: officeAsset("furniture/PC/PC_FRONT_ON_3.png"),
        pcOff: officeAsset("furniture/PC/PC_FRONT_OFF.png"),
        board: officeAsset("furniture/WHITEBOARD/WHITEBOARD.png"),
        shelf: officeAsset("furniture/BOOKSHELF/BOOKSHELF.png"),
        plant: officeAsset("furniture/PLANT/PLANT.png"),
        table: officeAsset("furniture/COFFEE_TABLE/COFFEE_TABLE.png"),
        cooler: officeAsset("furniture/WATER_COOLER/WATER_COOLER.png")
    };
    for (let i = 0; i < OFFICE_CHARACTER_COUNT; i++) paths[`char${i}`] = officeAsset(`characters/char_${i}.png`);
    const loaded = await Promise.all(Object.entries(paths).map(async ([key, src]) => [key, await loadOfficeImage(src)]));
    for (const [key, img] of loaded) this.assets[key] = img;
    this.ready = true;
    this.draw();
};

AgentOfficeCanvas.prototype.bind = function() {
    this.canvas.addEventListener("mousemove", evt => {
        const rect = this.canvas.getBoundingClientRect();
        const world = this.scene.world;
        this.mouse = {
            x: (evt.clientX - rect.left) * (world.w / Math.max(1, rect.width)),
            y: (evt.clientY - rect.top) * (world.h / Math.max(1, rect.height)),
            clientX: evt.clientX,
            clientY: evt.clientY
        };
        this.hovered = this.hitAgent(this.mouse.x, this.mouse.y);
        this.updateTooltip(evt);
    });
    this.canvas.addEventListener("mouseleave", () => {
        this.mouse = null;
        this.hovered = null;
        const tip = el("tooltip");
        if (tip) tip.classList.remove("visible");
    });
    window.addEventListener("resize", () => this.rebuildForCurrentSize());
};

AgentOfficeCanvas.prototype.setData = function(data) {
    this.data = data || officeFallbackSnapshot();
    this.scene = this.buildScene(this.data);
    this.draw();
};

AgentOfficeCanvas.prototype.availableWidth = function() {
    const parent = this.canvas.parentElement;
    const parentWidth = parent ? Math.floor(parent.clientWidth || parent.getBoundingClientRect().width || 0) : 0;
    if (parentWidth >= 360) return parentWidth;
    const map = this.canvas.closest(".agent-office-map");
    const mapWidth = map ? Math.floor(map.clientWidth || map.getBoundingClientRect().width || 0) : 0;
    if (mapWidth >= 420) return Math.max(360, mapWidth - 24);
    const content = el("mainContent");
    const contentWidth = content ? Math.floor(content.clientWidth || content.getBoundingClientRect().width || 0) : 0;
    if (contentWidth >= 520) return Math.max(360, contentWidth - 60);
    return Math.max(360, Math.min(1440, (window.innerWidth || 1200) - 360));
};

AgentOfficeCanvas.prototype.layoutFor = function(total) {
    const available = Math.max(360, this.availableWidth() - 4);
    const margin = available < 760 ? 18 : 26;
    const gap = available < 1120 ? 26 : 34;
    const minTwoColRoom = 620;
    const cols = available >= margin * 2 + gap + minTwoColRoom * 2 ? 2 : 1;
    const roomW = Math.floor((available - margin * 2 - (cols - 1) * gap) / cols);
    const roomH = Math.round(Math.max(cols === 1 ? 470 : 440, Math.min(cols === 1 ? 620 : 520, roomW * 0.62)));
    const rows = Math.ceil(Math.max(1, total) / cols);
    return { available, cols, roomW, roomH, gap, margin, rows };
};

AgentOfficeCanvas.prototype.rebuildForCurrentSize = function() {
    if (!this.data) {
        this.draw();
        return;
    }
    this.scene = this.buildScene(this.data);
    this.draw();
};

AgentOfficeCanvas.prototype.rebuildIfLayoutChanged = function() {
    if (!this.data) return;
    const width = this.availableWidth();
    if (Math.abs(width - this.lastLayoutWidth) < 18) return;
    this.scene = this.buildScene(this.data);
};

AgentOfficeCanvas.prototype.start = function() {
    if (this.raf) return;
    const tick = ts => {
        this.t = ts / 1000;
        this.draw();
        this.raf = window.requestAnimationFrame(tick);
    };
    this.raf = window.requestAnimationFrame(tick);
};

const OFFICE_PIPELINE_ORDER = [
    "paper_extraction",
    "graph_construction",
    "idea_generation",
    "experiment_planning",
    "experiment_execution",
    "manuscript_generation",
    "orchestration"
];

function officeItemText(item) {
    if (!item) return "";
    return `${item.kind || ""} ${item.status || ""} ${item.title || ""} ${item.detail || ""}`.toLowerCase();
}

function officeRankAgentsForWork(agents, dep, items) {
    const taskText = `${dep.key || ""} ${dep.title || ""} ${dep.responsibility || ""} ${(items || []).map(officeItemText).join(" ")}`.toLowerCase();
    return agents.map((agent, index) => {
        const path = `${agent.name || ""} ${agent.path || ""}`.toLowerCase();
        let score = 0;
        for (const token of path.split(/[^a-z0-9]+/).filter(t => t.length > 2)) {
            if (taskText.includes(token)) score += 3;
        }
        if (/paper|pdf|arxiv|extract|claim|reference/.test(taskText) && /paper|pdf|arxiv|extract|claim|reference|completeness/.test(path)) score += 12;
        if (/graph|taxonomy|signal|opportunity/.test(taskText) && /graph|taxonomy|signal|opportunity|summary|knowledge/.test(path)) score += 12;
        if (/idea|insight|novelty|reason|discovery/.test(taskText) && /idea|insight|novelty|reason|discovery|paradigm/.test(path)) score += 12;
        if (/review|plan|benchmark|contract|audit/.test(taskText) && /review|plan|benchmark|audit|forge|result|requirement/.test(path)) score += 12;
        if (/gpu|run|experiment|validation|scheduler|shard/.test(taskText) && /gpu|experiment|validation|executor|scheduler|tracking|ssh|benchmark/.test(path)) score += 12;
        if (/manuscript|paperorchestra|draft|bundle|figure|reference/.test(taskText) && /manuscript|orchestra|figure|literature|semantic|refinement|plotting/.test(path)) score += 12;
        if (/worker|pipeline|recovery|orchestration|service/.test(taskText) && /worker|pipeline|scheduler|watchdog|workspace|web|auto|forever/.test(path)) score += 12;
        score += Math.max(0, 6 - index) * 0.1;
        return { index, score };
    }).sort((a, b) => b.score - a.score || a.index - b.index).map(x => x.index);
}

function officeActiveSlotCount(dep, agents, items, workstationCount) {
    if ((dep.status || "idle") === "idle" || !agents.length) return 0;
    const itemCount = Math.max(1, (items || []).length);
    const baseline = dep.status === "blocked" ? 2 : 2;
    return Math.min(agents.length, workstationCount, Math.max(baseline, Math.min(workstationCount, itemCount)));
}

function officeWorkingAction(agent, dep) {
    const text = `${agent.name || ""} ${agent.path || ""} ${dep.key || ""} ${dep.title || ""}`.toLowerCase();
    if (/reader|read|pdf|extract|arxiv|reference|semantic|literature/.test(text)) return "reading";
    return "typing";
}

AgentOfficeCanvas.prototype.buildScene = function(data) {
    const departments = Array.isArray(data.departments) ? data.departments : [];
    const total = Math.max(1, departments.length);
    const layout = this.layoutFor(total);
    const { cols, roomW, roomH, gap, margin, rows } = layout;
    this.lastLayoutWidth = this.availableWidth();
    const world = {
        w: margin * 2 + cols * roomW + (cols - 1) * gap,
        h: margin * 2 + rows * roomH + (rows - 1) * gap
    };
    const positions = cols === 2 && departments.length === 7
        ? [{c:0,r:0},{c:1,r:0},{c:0,r:1},{c:1,r:1},{c:0,r:2},{c:1,r:2},{c:0.5,r:3}]
        : departments.map((_, i) => ({ c: i % cols, r: Math.floor(i / cols) }));
    const rooms = [];
    const sceneAgents = [];
    departments.forEach((dep, index) => {
        const pos = positions[index] || { c: index % cols, r: Math.floor(index / cols) };
        const room = {
            index,
            dep,
            x: margin + pos.c * (roomW + gap),
            y: margin + pos.r * (roomH + gap),
            w: roomW,
            h: roomH,
            accent: officeAccentColor(dep.accent),
            agents: [],
            items: Array.isArray(dep.items) ? dep.items : []
        };
        room.workstations = this.makeWorkstations(room);
        const agents = Array.isArray(dep.sub_agents) ? dep.sub_agents : [];
        const activeCount = officeActiveSlotCount(dep, agents, room.items, room.workstations.length);
        const activeOrder = new Set(officeRankAgentsForWork(agents, dep, room.items).slice(0, activeCount));
        const restSlots = this.makeRestSlots(room, Math.max(0, agents.length - activeCount));
        let workCursor = 0;
        let restCursor = 0;
        agents.forEach((agent, agentIndex) => {
            const seed = officeHash(`${dep.key || dep.title}:${agent.path || agent.name}:${agentIndex}`) / 4294967295;
            const isWorking = activeOrder.has(agentIndex);
            const workstation = isWorking ? room.workstations[workCursor % room.workstations.length] : null;
            const task = isWorking && room.items.length ? room.items[workCursor % room.items.length] : null;
            const restSlot = restSlots[restCursor % Math.max(1, restSlots.length)] || { x: room.x + room.w / 2, y: room.y + room.h / 2 };
            const slot = workstation ? { x: workstation.seatX, y: workstation.seatY } : restSlot;
            const action = isWorking ? officeWorkingAction(agent, dep) : "idle";
            const spriteIndex = agentIndex % OFFICE_CHARACTER_COUNT;
            const entry = {
                agent,
                dep,
                room,
                index: agentIndex,
                action,
                spriteIndex,
                seed,
                baseX: slot.x,
                baseY: slot.y,
                x: slot.x,
                y: slot.y,
                bounds: null,
                working: isWorking,
                workstation,
                task,
                roleLabel: isWorking ? officeActionVerb(action, dep.status) : "resting"
            };
            if (isWorking) workCursor += 1;
            else restCursor += 1;
            room.agents.push(entry);
            sceneAgents.push(entry);
        });
        rooms.push(room);
    });
    return { world, rooms, agents: sceneAgents };
};

AgentOfficeCanvas.prototype.makeWorkstations = function(room) {
    const count = room.w < 560 ? 2 : room.w < 720 ? 3 : 4;
    const deskW = room.w < 560 ? 100 : room.w < 720 ? 108 : 116;
    const deskH = 72;
    const sidePad = room.w < 560 ? 46 : 74;
    const gap = (room.w - sidePad * 2 - count * deskW) / Math.max(1, count - 1);
    const deskY = room.y + room.h - 118;
    const stations = [];
    for (let i = 0; i < count; i++) {
        const deskX = room.x + sidePad + i * (deskW + Math.max(18, gap));
        stations.push({
            deskX,
            deskY,
            deskW,
            deskH,
            pcX: deskX + deskW / 2 - 18,
            pcY: deskY - 36,
            seatX: deskX + deskW / 2,
            seatY: deskY - 8,
            labelX: deskX + 4,
            labelY: deskY - 58
        });
    }
    return stations;
};

AgentOfficeCanvas.prototype.makeRestSlots = function(room, count) {
    const slots = [];
    const cols = Math.min(6, Math.max(2, Math.ceil(Math.sqrt(Math.max(1, count) * 1.2))));
    const rows = Math.max(1, Math.ceil(Math.max(1, count) / cols));
    const left = room.x + 78;
    const right = room.x + room.w - 78;
    const top = room.y + 230;
    const bottom = room.y + room.h - 170;
    const usableW = Math.max(120, right - left);
    const usableH = Math.max(72, bottom - top);
    for (let i = 0; i < Math.max(1, count); i++) {
        const c = i % cols;
        const r = Math.floor(i / cols);
        slots.push({
            x: left + usableW * (c + 0.5) / cols + (((i * 31) % 15) - 7),
            y: top + usableH * (r + 0.5) / rows + (((i * 17) % 13) - 6)
        });
    }
    return slots;
};

AgentOfficeCanvas.prototype.resize = function() {
    const world = this.scene.world || { w: 1200, h: 1200 };
    const maxWidth = this.availableWidth();
    const cssW = Math.max(360, maxWidth || world.w || 1200);
    const cssH = Math.max(560, Math.round(cssW * world.h / world.w));
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(cssW * dpr) || this.canvas.height !== Math.round(cssH * dpr)) {
        this.canvas.width = Math.round(cssW * dpr);
        this.canvas.height = Math.round(cssH * dpr);
        this.canvas.style.width = `${cssW}px`;
        this.canvas.style.height = `${cssH}px`;
    }
    const scale = cssW / world.w;
    this.ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
    return { dpr, scale, cssW, cssH };
};

AgentOfficeCanvas.prototype.draw = function() {
    if (!this.ctx || !this.scene) return;
    this.rebuildIfLayoutChanged();
    const ctx = this.ctx;
    const world = this.scene.world;
    this.resize();
    ctx.clearRect(0, 0, world.w, world.h);
    this.drawOfficeBase(ctx);
    this.drawPipelineFlow(ctx, true);
    for (const room of this.scene.rooms) this.drawRoomBase(ctx, room);
    for (const room of this.scene.rooms) this.drawRoomFurniture(ctx, room);
    const sorted = [...this.scene.agents].sort((a, b) => a.y - b.y);
    for (const ag of sorted) this.drawAgent(ctx, ag);
    for (const room of this.scene.rooms) this.drawRoomOverlays(ctx, room);
    this.drawPipelineFlow(ctx, false);
    if (this.hovered) this.drawHoverLabel(ctx, this.hovered);
};

AgentOfficeCanvas.prototype.drawOfficeBase = function(ctx) {
    const { w, h } = this.scene.world;
    ctx.fillStyle = "#e7ddcc";
    ctx.fillRect(0, 0, w, h);
    this.tileImage(ctx, this.assets.floor, 0, 0, w, h, 2);
    ctx.fillStyle = "rgba(67,55,45,0.08)";
    ctx.fillRect(0, 0, w, 30);
    ctx.strokeStyle = "rgba(67,55,45,0.18)";
    ctx.lineWidth = 8;
    ctx.strokeRect(10, 10, w - 20, h - 20);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.strokeRect(18, 18, w - 36, h - 36);
};

AgentOfficeCanvas.prototype.drawPipelineFlow = function(ctx, underlay) {
    const byKey = new Map(this.scene.rooms.map(room => [room.dep.key, room]));
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = 0; i < OFFICE_PIPELINE_ORDER.length - 1; i++) {
        const from = byKey.get(OFFICE_PIPELINE_ORDER[i]);
        const to = byKey.get(OFFICE_PIPELINE_ORDER[i + 1]);
        if (!from || !to) continue;
        const active = from.dep.status !== "idle" || to.dep.status !== "idle";
        const p1 = this.roomEdgeToward(from, to);
        const p2 = this.roomEdgeToward(to, from);
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const bend = p1.y === p2.y ? 0 : (p2.x > p1.x ? 60 : -60);
        ctx.strokeStyle = underlay ? "rgba(67,55,45,0.16)" : (active ? "rgba(196,112,75,0.72)" : "rgba(67,55,45,0.25)");
        ctx.lineWidth = underlay ? 10 : 3;
        ctx.setLineDash(underlay ? [] : [12, 10]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.quadraticCurveTo(midX + bend, midY, p2.x, p2.y);
        ctx.stroke();
        if (!underlay) {
            this.drawArrowHead(ctx, p1, p2, active ? "rgba(196,112,75,0.85)" : "rgba(67,55,45,0.35)");
            if (active) {
                const phase = (this.t * 0.18 + i * 0.13) % 1;
                const packet = this.quadPoint(p1, { x: midX + bend, y: midY }, p2, phase);
                this.drawPacket(ctx, packet.x, packet.y, from.accent, `${i + 1}`);
            }
        }
    }
    ctx.restore();
};

AgentOfficeCanvas.prototype.roomEdgeToward = function(room, targetRoom) {
    const cx = room.x + room.w / 2;
    const cy = room.y + room.h / 2;
    const tx = targetRoom.x + targetRoom.w / 2;
    const ty = targetRoom.y + targetRoom.h / 2;
    const dx = tx - cx;
    const dy = ty - cy;
    if (Math.abs(dx) > Math.abs(dy)) {
        return { x: dx > 0 ? room.x + room.w + 2 : room.x - 2, y: cy };
    }
    return { x: cx, y: dy > 0 ? room.y + room.h + 2 : room.y - 2 };
};

AgentOfficeCanvas.prototype.quadPoint = function(p0, p1, p2, t) {
    const a = (1 - t) * (1 - t);
    const b = 2 * (1 - t) * t;
    const c = t * t;
    return { x: a * p0.x + b * p1.x + c * p2.x, y: a * p0.y + b * p1.y + c * p2.y };
};

AgentOfficeCanvas.prototype.drawArrowHead = function(ctx, from, to, color) {
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    const size = 12;
    ctx.save();
    ctx.fillStyle = color;
    ctx.translate(to.x, to.y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-size, -size * 0.55);
    ctx.lineTo(-size, size * 0.55);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
};

AgentOfficeCanvas.prototype.drawPacket = function(ctx, x, y, color, label) {
    canvasFillRoundRect(ctx, x - 16, y - 11, 32, 22, 5, "rgba(255,253,248,0.95)", color, 2);
    ctx.font = "800 10px Source Code Pro, monospace";
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.fillText(label, x, y + 4);
    ctx.textAlign = "left";
};

AgentOfficeCanvas.prototype.drawRoomBase = function(ctx, room) {
    const dep = room.dep;
    ctx.save();
    canvasFillRoundRect(ctx, room.x, room.y, room.w, room.h, 8, "rgba(246,239,226,0.94)", "rgba(67,55,45,0.25)", 3);
    this.tileImage(ctx, this.assets.floor, room.x + 4, room.y + 70, room.w - 8, room.h - 74, 2);
    ctx.fillStyle = "rgba(255,253,248,0.9)";
    ctx.fillRect(room.x + 4, room.y + 4, room.w - 8, 72);
    ctx.fillStyle = room.accent;
    ctx.fillRect(room.x + 18, room.y + 24, 18, 18);
    if (dep.status === "working") {
        ctx.globalAlpha = 0.55 + 0.35 * Math.sin(this.t * 4 + room.index);
        ctx.fillRect(room.x + 18, room.y + 24, 18, 18);
        ctx.globalAlpha = 1;
    }
    ctx.font = "800 28px Source Sans 3, system-ui, sans-serif";
    ctx.fillStyle = "#2b2520";
    canvasTextFit(ctx, dep.title || "Department", room.x + 48, room.y + 42, room.w - 250);
    ctx.font = "800 12px Source Code Pro, monospace";
    ctx.fillStyle = room.accent;
    const label = officeStatusLabel(dep.status || "idle").toUpperCase();
    const labelW = Math.min(180, Math.max(78, ctx.measureText(label).width + 24));
    canvasFillRoundRect(ctx, room.x + room.w - labelW - 18, room.y + 18, labelW, 32, 6, "rgba(255,255,255,0.92)", "rgba(67,55,45,0.12)", 1);
    ctx.fillText(label, room.x + room.w - labelW - 6, room.y + 39);
    ctx.font = "700 15px Source Sans 3, system-ui, sans-serif";
    ctx.fillStyle = "#8d8177";
    canvasWrapText(ctx, dep.responsibility || "", room.x + 48, room.y + 66, room.w - 96, 17, 2);
    ctx.strokeStyle = dep.status === "blocked" ? "rgba(196,69,58,0.62)" : dep.status === "working" ? `${room.accent}aa` : "rgba(67,55,45,0.16)";
    ctx.lineWidth = dep.status === "idle" ? 2 : 5;
    canvasRoundRect(ctx, room.x + 2, room.y + 2, room.w - 4, room.h - 4, 8);
    ctx.stroke();
    ctx.restore();
};

AgentOfficeCanvas.prototype.drawRoomFurniture = function(ctx, room) {
    const dep = room.dep;
    const active = dep.status !== "idle";
    this.drawImage(ctx, this.assets.shelf, room.x + 24, room.y + 156, 42, 64);
    this.drawImage(ctx, this.assets.board, room.x + room.w - 112, room.y + 114, 82, 45);
    this.drawImage(ctx, this.assets.plant, room.x + room.w - 66, room.y + room.h - 72, 36, 50);
    if (room.index % 2 === 0) this.drawImage(ctx, this.assets.cooler, room.x + 30, room.y + room.h - 92, 34, 68);
    this.drawTaskBoard(ctx, room);

    for (let i = 0; i < room.workstations.length; i++) {
        const station = room.workstations[i];
        this.drawImage(ctx, this.assets.desk, station.deskX, station.deskY, station.deskW, station.deskH);
        const occupied = room.agents.some(ag => ag.working && ag.workstation === station);
        const pc = occupied || active ? this.assets[`pcOn${(Math.floor(this.t * 2 + i) % 3) + 1}`] : this.assets.pcOff;
        this.drawImage(ctx, pc, station.pcX, station.pcY, 36, 72);
    }
    this.drawImage(ctx, this.assets.table, room.x + room.w / 2 - 56, room.y + room.h - 70, 112, 56);
};

AgentOfficeCanvas.prototype.drawTaskBoard = function(ctx, room) {
    const dep = room.dep;
    const items = Array.isArray(dep.items) ? dep.items : [];
    const bx = room.x + 88;
    const by = room.y + 100;
    const bw = room.w - 240;
    const bh = 104;
    canvasFillRoundRect(ctx, bx, by, bw, bh, 7, "rgba(43,37,32,0.84)", "rgba(255,255,255,0.28)", 1);
    ctx.font = "800 12px Source Code Pro, monospace";
    const task = items[0];
    if (task) {
        ctx.fillStyle = room.accent;
        canvasTextFit(ctx, String(task.status || "working").toUpperCase(), bx + 14, by + 22, 190);
        ctx.fillStyle = "#fffdf8";
        ctx.font = "800 18px Source Sans 3, system-ui, sans-serif";
        canvasWrapText(ctx, task.title || "Current work", bx + 14, by + 48, bw - 28, 21, 2);
        if (task.detail) {
            ctx.font = "700 12px Source Code Pro, monospace";
            ctx.fillStyle = "rgba(255,253,248,0.72)";
            canvasTextFit(ctx, task.detail, bx + 14, by + 91, bw - 28);
        }
        if (items.length > 1) {
            ctx.font = "800 11px Source Code Pro, monospace";
            ctx.fillStyle = "rgba(255,253,248,0.58)";
            ctx.fillText(`+${items.length - 1} queued`, bx + bw - 86, by + 22);
        }
    } else {
        ctx.fillStyle = "rgba(255,253,248,0.78)";
        ctx.font = "800 17px Source Sans 3, system-ui, sans-serif";
        canvasTextFit(ctx, "No active job in this department", bx + 14, by + 45, bw - 28);
        ctx.font = "700 13px Source Sans 3, system-ui, sans-serif";
        canvasTextFit(ctx, "Agents are in the lounge and will move to desks when work arrives.", bx + 14, by + 72, bw - 28);
    }
};

AgentOfficeCanvas.prototype.drawRoomOverlays = function(ctx, room) {
    const workingAgents = room.agents.filter(ag => ag.working);
    for (const ag of workingAgents) this.drawDeskBadge(ctx, ag);
    if (!workingAgents.length) return;
    const speaker = workingAgents[Math.floor(this.t / 4 + room.index) % workingAgents.length];
    const lead = speaker.task ? (speaker.task.title || speaker.task.status) : officeLeadText(room.dep);
    this.drawSpeech(ctx, speaker.x, speaker.y - 68, `${speaker.roleLabel}: ${trunc(lead, 44)}`, room);
};

AgentOfficeCanvas.prototype.drawDeskBadge = function(ctx, ag) {
    const station = ag.workstation;
    if (!station) return;
    const name = trunc(ag.agent.name || ag.agent.path || "Agent", 18);
    const status = ag.task ? trunc(ag.task.status || "working", 18) : ag.roleLabel;
    const x = station.labelX;
    const y = station.labelY;
    canvasFillRoundRect(ctx, x, y, 128, 34, 5, "rgba(255,253,248,0.92)", "rgba(67,55,45,0.14)", 1);
    ctx.font = "800 11px Source Sans 3, system-ui, sans-serif";
    ctx.fillStyle = ag.room.accent;
    canvasTextFit(ctx, status, x + 7, y + 14, 114);
    ctx.font = "800 11px Source Sans 3, system-ui, sans-serif";
    ctx.fillStyle = "#4f453d";
    canvasTextFit(ctx, name, x + 7, y + 28, 114);
};

AgentOfficeCanvas.prototype.drawAgent = function(ctx, ag) {
    const depStatus = ag.dep.status || "idle";
    const action = ag.action;
    const phase = this.t + ag.seed * 8;
    let dx = 0;
    let dy = 0;
    if (ag.working) {
        dy = Math.sin(phase * 5) * 0.8;
    } else {
        dy = Math.sin(phase * 1.1) * 1.2;
    }
    ag.x = ag.baseX + dx;
    ag.y = ag.baseY + dy;
    const scale = ag.working ? 2.05 : 1.85;
    const drawW = OFFICE_FRAME_W * scale;
    const drawH = OFFICE_FRAME_H * scale;
    const frame = this.frameFor(action, depStatus, phase);
    const img = this.assets[`char${ag.spriteIndex}`];
    const drawX = Math.round(ag.x - drawW / 2);
    const drawY = Math.round(ag.y - drawH);
    ag.bounds = { x: drawX, y: drawY, w: drawW, h: drawH + 8 };
    if (img) {
        ctx.drawImage(img, frame.col * OFFICE_FRAME_W, frame.row * OFFICE_FRAME_H, OFFICE_FRAME_W, OFFICE_FRAME_H, drawX, drawY, drawW, drawH);
    } else {
        this.drawFallbackAgent(ctx, drawX, drawY, drawW, drawH, ag.room.accent);
    }
};

AgentOfficeCanvas.prototype.frameFor = function(action, depStatus, phase) {
    if (depStatus === "blocked") return { row: 0, col: Math.floor(phase * 2) % 2 ? 5 : 6 };
    if (action === "typing") return { row: 0, col: Math.floor(phase * 3) % 2 ? 3 : 4 };
    if (action === "reading") return { row: 0, col: Math.floor(phase * 2) % 2 ? 5 : 6 };
    return { row: 0, col: 0 };
};

AgentOfficeCanvas.prototype.drawHoverLabel = function(ctx, ag) {
    const label = ag.agent.name || ag.agent.path || "Agent";
    const detail = ag.agent.path || "";
    const task = ag.task ? `${ag.task.status || "working"}: ${ag.task.title || "task"}` : ag.roleLabel;
    const x = Math.max(ag.room.x + 12, Math.min(ag.x - 100, ag.room.x + ag.room.w - 230));
    const y = Math.max(ag.room.y + 82, ag.y - 112);
    canvasFillRoundRect(ctx, x, y, 220, 62, 5, "rgba(43,37,32,0.94)", "rgba(255,255,255,0.22)", 1);
    ctx.font = "800 14px Source Sans 3, system-ui, sans-serif";
    ctx.fillStyle = "#fffdf8";
    canvasTextFit(ctx, label, x + 10, y + 19, 198);
    ctx.font = "700 10px Source Code Pro, monospace";
    ctx.fillStyle = "rgba(255,253,248,0.68)";
    canvasTextFit(ctx, detail, x + 10, y + 36, 198);
    ctx.font = "800 11px Source Sans 3, system-ui, sans-serif";
    ctx.fillStyle = ag.room.accent;
    canvasTextFit(ctx, task, x + 10, y + 53, 198);
};

AgentOfficeCanvas.prototype.drawSpeech = function(ctx, x, y, text, room) {
    ctx.font = "800 13px Source Sans 3, system-ui, sans-serif";
    const w = Math.min(320, Math.max(116, ctx.measureText(text).width + 26));
    const h = 34;
    const bx = Math.max(room.x + 16, Math.min(x - w / 2, room.x + room.w - w - 16));
    const by = Math.max(room.y + 90, y - h);
    canvasFillRoundRect(ctx, bx, by, w, h, 6, "rgba(255,255,255,0.94)", "rgba(67,55,45,0.14)", 2);
    ctx.fillStyle = room.accent;
    canvasTextFit(ctx, text, bx + 12, by + 22, w - 24);
    ctx.beginPath();
    ctx.moveTo(Math.min(Math.max(x, bx + 18), bx + w - 18), by + h);
    ctx.lineTo(Math.min(Math.max(x + 7, bx + 24), bx + w - 12), by + h + 9);
    ctx.lineTo(Math.min(Math.max(x - 7, bx + 12), bx + w - 24), by + h);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,0.94)";
    ctx.fill();
};

AgentOfficeCanvas.prototype.hitAgent = function(x, y) {
    for (let i = this.scene.agents.length - 1; i >= 0; i--) {
        const b = this.scene.agents[i].bounds;
        if (b && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) return this.scene.agents[i];
    }
    return null;
};

AgentOfficeCanvas.prototype.updateTooltip = function(evt) {
    const tip = el("tooltip");
    if (!tip) return;
    if (!this.hovered) {
        tip.classList.remove("visible");
        return;
    }
    const ag = this.hovered;
    const task = ag.task ? `${ag.task.status || "working"}: ${ag.task.title || "task"}` : ag.roleLabel;
    tip.innerHTML = `<strong>${esc(ag.agent.name || "Agent")}</strong><br>${esc(ag.agent.path || "")}<br>${esc(ag.dep.title || "Department")} - ${esc(task)}`;
    tip.style.left = `${evt.clientX + 14}px`;
    tip.style.top = `${evt.clientY + 14}px`;
    tip.classList.add("visible");
};

AgentOfficeCanvas.prototype.drawImage = function(ctx, img, x, y, w, h) {
    if (!img) return;
    ctx.drawImage(img, Math.round(x), Math.round(y), Math.round(w), Math.round(h));
};

AgentOfficeCanvas.prototype.tileImage = function(ctx, img, x, y, w, h, scale) {
    if (!img) {
        ctx.fillStyle = "rgba(255,255,255,0.2)";
        ctx.fillRect(x, y, w, h);
        return;
    }
    const tw = img.width * scale;
    const th = img.height * scale;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    for (let yy = y; yy < y + h; yy += th) {
        for (let xx = x; xx < x + w; xx += tw) {
            ctx.drawImage(img, Math.round(xx), Math.round(yy), tw, th);
        }
    }
    ctx.restore();
};

AgentOfficeCanvas.prototype.drawFallbackAgent = function(ctx, x, y, w, h, accent) {
    ctx.fillStyle = "#d8a173";
    ctx.fillRect(x + w * 0.35, y + h * 0.05, w * 0.3, h * 0.18);
    ctx.fillStyle = accent;
    ctx.fillRect(x + w * 0.25, y + h * 0.28, w * 0.5, h * 0.42);
    ctx.fillStyle = "#51453b";
    ctx.fillRect(x + w * 0.22, y + h * 0.78, w * 0.2, h * 0.16);
    ctx.fillRect(x + w * 0.58, y + h * 0.78, w * 0.2, h * 0.16);
};

// ── Feed

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
    if (!el("recentlyGrid")) return;
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

async function loadOverviewResearchMap() {
    const graph = el('overviewGraphSvg');
    if (!graph || graph.dataset.loaded === 'true') return;
    graph.dataset.loaded = 'true';
    try {
        const data = await api(`/api/taxonomy/${encodeURIComponent(ROOT_NODE)}`);
        renderRadialGraph('overviewGraphSvg', data.node, (data.children || []).slice(0, 8), 330, true);
    } catch (e) {
        graph.dataset.loaded = '';
        const card = el('overviewMapCard');
        if (card) card.style.display = 'none';
        console.error('Overview research map error:', e);
    }
}

function renderRecentlyDiscovered(data, insights) {
    const grid = el('recentlyGrid');
    if (!grid) return;
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
        renderRadialGraph('exploreGraphSvg', data.node, data.children, 600, false);

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
    const pad = isPreview ? 22 : 30;
    const cardW = Math.max(isPreview ? 132 : 170, Math.min(isPreview ? 172 : 220, width * (isPreview ? 0.22 : 0.20)));
    const cardH = isPreview ? 78 : 94;
    const rootW = Math.max(isPreview ? 180 : 230, Math.min(isPreview ? 240 : 300, width * 0.28));
    const rootH = isPreview ? 112 : 132;

    if (!children || children.length === 0) {
        drawGraphBackdrop(svg, width, height, svgId);
        const empty = svg.append('g').attr('transform', `translate(${cx - rootW / 2},${cy - rootH / 2})`);
        empty.append('rect')
            .attr('width', rootW).attr('height', rootH).attr('rx', 10)
            .attr('fill', '#fffdf8').attr('stroke', '#d9cec0').attr('stroke-width', 1.4);
        empty.append('text')
            .attr('x', rootW / 2).attr('y', 38).attr('text-anchor', 'middle')
            .attr('fill', '#2b2520').attr('font-size', isPreview ? '14px' : '16px').attr('font-weight', '800')
            .text('Leaf research area');
        empty.append('text')
            .attr('x', rootW / 2).attr('y', 62).attr('text-anchor', 'middle')
            .attr('fill', '#8d8074').attr('font-size', isPreview ? '11px' : '12px').attr('font-weight', '600')
            .text(trunc(parentNode.name || '', isPreview ? 28 : 38));
        empty.append('text')
            .attr('x', rootW / 2).attr('y', 86).attr('text-anchor', 'middle')
            .attr('fill', '#b5ada4').attr('font-size', '11px')
            .text('See detailed analysis below');
        return;
    }

    const maxGap = Math.max(...children.map(c => c.gap_count || 0), 1);
    const maxPapers = Math.max(...children.map(c => c.paper_count || 0), 1);
    const maxMethods = Math.max(...children.map(c => c.method_count || 0), 1);
    const root = {
        id: parentNode.id, name: parentNode.name, description: parentNode.description || '',
        paper_count: children.reduce((sum, c) => sum + (c.paper_count || 0), 0),
        gap_count: children.reduce((sum, c) => sum + (c.gap_count || 0), 0),
        method_count: children.reduce((sum, c) => sum + (c.method_count || 0), 0),
        x: cx - rootW / 2,
        y: cy - rootH / 2,
        w: rootW,
        h: rootH,
        isParent: true,
    };
    const childNodes = layoutTaxonomyCards(children, width, height, root, cardW, cardH, pad, isPreview)
        .map((child, i) => ({
            id: child.id,
            name: child.name,
            description: child.description || '',
            paper_count: child.paper_count || 0,
            gap_count: child.gap_count || 0,
            method_count: child.method_count || 0,
            x: child.x,
            y: child.y,
            w: cardW,
            h: cardH,
            rank: i,
            isParent: false,
        }));

    drawGraphBackdrop(svg, width, height, svgId);

    const defs = svg.append('defs');
    const shadow = defs.append('filter').attr('id', 'taxonomyShadow-' + svgId).attr('x', '-20%').attr('y', '-20%').attr('width', '140%').attr('height', '140%');
    shadow.append('feDropShadow').attr('dx', 0).attr('dy', 5).attr('stdDeviation', 7).attr('flood-color', '#4f3928').attr('flood-opacity', 0.08);

    const linkLayer = svg.append('g').attr('class', 'taxonomy-links');
    const links = linkLayer.selectAll('path')
        .data(childNodes).join('path')
        .attr('d', d => curvedLink(root, d))
        .attr('fill', 'none')
        .attr('stroke', d => d.gap_count > 0 ? 'rgba(196,112,75,0.34)' : 'rgba(151,132,112,0.24)')
        .attr('stroke-width', d => Math.max(1.2, Math.min(3.8, 1.2 + (d.paper_count / maxPapers) * 2.3)))
        .attr('stroke-linecap', 'round');

    const rootG = svg.append('g')
        .attr('class', 'graph-node-parent taxonomy-card taxonomy-root draggable')
        .attr('transform', `translate(${root.x},${root.y})`);
    drawRootCard(rootG, root, isPreview);

    const childG = svg.append('g').selectAll('g')
        .data(childNodes).join('g')
        .attr('class', 'graph-node taxonomy-card taxonomy-child draggable')
        .attr('transform', d => `translate(${d.x},${d.y})`);
    drawChildCards(childG, isPreview, maxGap, maxPapers, maxMethods);

    let dragMoved = false;
    const clampNode = d => {
        d.x = Math.max(10, Math.min(width - d.w - 10, d.x));
        d.y = Math.max(10, Math.min(height - d.h - 10, d.y));
    };
    const refreshLinks = () => {
        links.attr('d', d => curvedLink(root, d));
    };
    const dragBehavior = d3.drag()
        .on('start', function(event, d) {
            dragMoved = false;
            d._dragStartX = event.x;
            d._dragStartY = event.y;
            d3.select(this).classed('is-dragging', true).raise();
        })
        .on('drag', function(event, d) {
            if (Math.abs(event.x - d._dragStartX) + Math.abs(event.y - d._dragStartY) > 3) {
                dragMoved = true;
            }
            d.x += event.dx;
            d.y += event.dy;
            clampNode(d);
            d3.select(this).attr('transform', `translate(${d.x},${d.y})`);
            refreshLinks();
        })
        .on('end', function(event, d) {
            d3.select(this).classed('is-dragging', false);
            window.setTimeout(() => { dragMoved = false; }, 80);
            delete d._dragStartX;
            delete d._dragStartY;
        });
    rootG.datum(root).call(dragBehavior);
    childG.call(dragBehavior);

    // Click handler
    childG.on('click', (e, d) => {
        if (dragMoved) return;
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

function drawGraphBackdrop(svg, width, height, svgId) {
    const defs = svg.append('defs');
    const pattern = defs.append('pattern')
        .attr('id', 'taxonomyDots-' + svgId)
        .attr('width', 22).attr('height', 22)
        .attr('patternUnits', 'userSpaceOnUse');
    pattern.append('circle').attr('cx', 2).attr('cy', 2).attr('r', 1).attr('fill', 'rgba(196,112,75,0.10)');
    svg.append('rect')
        .attr('x', 0).attr('y', 0).attr('width', width).attr('height', height)
        .attr('rx', 8).attr('fill', '#fffdf8');
    svg.append('rect')
        .attr('x', 10).attr('y', 10).attr('width', width - 20).attr('height', height - 20)
        .attr('rx', 12).attr('fill', `url(#taxonomyDots-${svgId})`).attr('opacity', 0.55);
}

function layoutTaxonomyCards(children, width, height, root, cardW, cardH, pad, isPreview) {
    const prepared = children.map(c => ({ ...c }));
    const slots = [
        { x: pad, y: pad + 8 },
        { x: width - cardW - pad, y: pad + 8 },
        { x: pad, y: height - cardH - pad },
        { x: width - cardW - pad, y: height - cardH - pad },
        { x: cxSlot(width, cardW, -0.27), y: pad + 4 },
        { x: cxSlot(width, cardW, 0.27), y: height - cardH - pad + 2 },
        { x: pad, y: height / 2 - cardH / 2 },
        { x: width - cardW - pad, y: height / 2 - cardH / 2 },
    ];
    if (prepared.length <= slots.length) {
        return prepared.map((child, idx) => ({ ...child, ...slots[idx] }));
    }

    const cols = Math.min(4, Math.max(2, Math.ceil(prepared.length / 2)));
    const topY = pad + 8;
    const bottomY = height - cardH - pad;
    const availableW = width - pad * 2;
    return prepared.map((child, idx) => {
        const row = idx < cols ? 0 : 1;
        const col = row === 0 ? idx : idx - cols;
        const rowCount = row === 0 ? cols : prepared.length - cols;
        const gap = rowCount > 1 ? (availableW - rowCount * cardW) / (rowCount - 1) : 0;
        return {
            ...child,
            x: pad + col * (cardW + Math.max(10, gap)),
            y: row === 0 ? topY : bottomY,
        };
    });
}

function cxSlot(width, cardW, offset) {
    return width / 2 - cardW / 2 + width * offset;
}

function nodeCenter(d) {
    return { x: d.x + d.w / 2, y: d.y + d.h / 2 };
}

function curvedLink(root, child) {
    const a = nodeCenter(root);
    const b = nodeCenter(child);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const c1 = { x: a.x + dx * 0.42, y: a.y + dy * 0.08 };
    const c2 = { x: b.x - dx * 0.42, y: b.y - dy * 0.08 };
    return `M ${a.x} ${a.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${b.x} ${b.y}`;
}

function drawRootCard(g, d, isPreview) {
    g.append('rect')
        .attr('width', d.w).attr('height', d.h).attr('rx', 12)
        .attr('fill', '#fff6ed').attr('stroke', '#c4704b').attr('stroke-width', 1.8)
        .attr('filter', `url(#taxonomyShadow-${g.node().ownerSVGElement.id})`);
    g.append('rect')
        .attr('x', 12).attr('y', 12).attr('width', 34).attr('height', 34).attr('rx', 8)
        .attr('fill', '#c4704b');
    g.append('text')
        .attr('x', 29).attr('y', 34).attr('text-anchor', 'middle')
        .attr('fill', '#fffdf8').attr('font-size', '17px').attr('font-weight', '900')
        .text('DG');
    appendFittedText(g, d.name, {
        x: 58,
        y: 28,
        fill: '#2b2520',
        'font-size': isPreview ? '14px' : '17px',
        'font-weight': '850',
    }, d.w - 72);
    g.append('text')
        .attr('x', 58).attr('y', 48)
        .attr('fill', '#7a6d62').attr('font-size', isPreview ? '10px' : '12px').attr('font-weight', '600')
        .text('Current research area');
    drawMetricPills(g, [
        [`${d.paper_count}`, 'papers', '#c4704b'],
        [`${d.gap_count}`, 'gaps', '#3d8b5e'],
        [`${d.method_count}`, 'methods', '#7c5cbf'],
    ], 14, d.h - 34, d.w - 28, isPreview);
}

function drawChildCards(selection, isPreview, maxGap, maxPapers, maxMethods) {
    selection.each(function(d) {
        const g = d3.select(this);
        const color = gapColor(d.gap_count, maxGap);
        const hasPapers = d.paper_count > 0;
        g.append('rect')
            .attr('width', d.w).attr('height', d.h).attr('rx', 10)
            .attr('fill', hasPapers ? '#ffffff' : '#fbfaf7')
            .attr('stroke', hasPapers ? color.stroke : '#d9d2c7')
            .attr('stroke-width', hasPapers ? 1.6 : 1.1)
            .attr('filter', `url(#taxonomyShadow-${g.node().ownerSVGElement.id})`);
        g.append('rect')
            .attr('x', 0).attr('y', 0).attr('width', 6).attr('height', d.h).attr('rx', 3)
            .attr('fill', hasPapers ? color.stroke : '#d9d2c7');
        g.append('circle')
            .attr('cx', d.w - 22).attr('cy', 22)
            .attr('r', Math.max(8, Math.min(17, 8 + (d.paper_count / maxPapers) * 9)))
            .attr('fill', hasPapers ? color.fill : '#f0ede6')
            .attr('stroke', hasPapers ? color.stroke : '#d0c9bc')
            .attr('stroke-width', 1.2);
        g.append('text')
            .attr('x', d.w - 22).attr('y', 26).attr('text-anchor', 'middle')
            .attr('fill', hasPapers ? color.stroke : '#a79b90')
            .attr('font-size', '10px').attr('font-weight', '850')
            .text(d.paper_count || 0);
        const words = wrapSvgLabel(d.name, isPreview ? 18 : 22, 2);
        words.forEach((line, idx) => {
            appendFittedText(g, line, {
                x: 18,
                y: 24 + idx * 15,
                fill: '#2b2520',
                'font-size': isPreview ? '12px' : '14px',
                'font-weight': '800',
            }, d.w - 70);
        });
        const desc = d.description || (hasPapers ? 'Evidence-bearing sub-area' : 'Awaiting evidence');
        appendFittedText(g, desc, {
            x: 18,
            y: isPreview ? 52 : 58,
            fill: '#8d8074',
            'font-size': isPreview ? '9px' : '10.5px',
            'font-weight': '600',
        }, d.w - 36);
        drawMetricPills(g, [
            [`${d.paper_count}`, 'p', '#c4704b'],
            [`${d.gap_count}`, 'gaps', '#3d8b5e'],
            [`${d.method_count}`, 'm', '#7c5cbf'],
        ], 14, d.h - 25, d.w - 28, isPreview);
    });
}

function drawMetricPills(g, metrics, x, y, maxW, isPreview) {
    const pillH = isPreview ? 18 : 20;
    const gap = 6;
    const pillW = Math.max(38, Math.min(70, (maxW - gap * (metrics.length - 1)) / metrics.length));
    metrics.forEach(([value, label, color], idx) => {
        const px = x + idx * (pillW + gap);
        g.append('rect')
            .attr('x', px).attr('y', y)
            .attr('width', pillW).attr('height', pillH).attr('rx', 9)
            .attr('fill', color + '14').attr('stroke', color + '44').attr('stroke-width', 1);
        appendFittedText(g, `${value} ${label}`, {
            x: px + pillW / 2,
            y: y + (isPreview ? 12.5 : 14),
            'text-anchor': 'middle',
            fill: color,
            'font-size': isPreview ? '8.5px' : '9.5px',
            'font-weight': '850',
        }, pillW - 8);
    });
}

function appendFittedText(g, value, attrs, maxWidth) {
    const text = g.append('text');
    Object.entries(attrs || {}).forEach(([key, attrValue]) => text.attr(key, attrValue));
    text.text(String(value || ''));
    fitSvgText(text, maxWidth);
    return text;
}

function fitSvgText(textSelection, maxWidth) {
    const node = textSelection.node();
    if (!node || !Number.isFinite(maxWidth) || maxWidth <= 0) {
        if (node) node.textContent = '';
        return;
    }
    const original = String(node.textContent || '').trim();
    if (!original) return;
    try {
        if (node.getComputedTextLength() <= maxWidth) return;
    } catch (e) {
        node.textContent = trunc(original, Math.max(4, Math.floor(maxWidth / 7)));
        return;
    }
    const ellipsis = '…';
    let lo = 0;
    let hi = original.length;
    let best = '';
    while (lo <= hi) {
        const mid = Math.floor((lo + hi) / 2);
        const candidate = original.slice(0, mid).trimEnd() + ellipsis;
        node.textContent = candidate;
        let fits = false;
        try {
            fits = node.getComputedTextLength() <= maxWidth;
        } catch (e) {
            fits = candidate.length * 7 <= maxWidth;
        }
        if (fits) {
            best = candidate;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    node.textContent = best || ellipsis;
}

function wrapSvgLabel(text, maxChars, maxLines) {
    const words = String(text || '').split(/\s+/).filter(Boolean);
    const lines = [];
    let line = '';
    for (const word of words) {
        const next = line ? `${line} ${word}` : word;
        if (next.length > maxChars && line) {
            lines.push(line);
            line = word;
        } else {
            line = next;
        }
        if (lines.length >= maxLines) break;
    }
    if (lines.length < maxLines && line) lines.push(line);
    if (!lines.length) return ['Untitled'];
    if (lines.length === maxLines && words.join(' ').length > lines.join(' ').length) {
        lines[lines.length - 1] = trunc(lines[lines.length - 1], maxChars - 1);
    }
    return lines;
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
        // A full <select> with thousands of options forces expensive layout on
        // tab activation.  The old interface used a typeahead datalist, which
        // keeps Evidence responsive while preserving the complete taxonomy.
        const options = el('evidenceNodeOptions');
        if (!options) return;
        const fragment = document.createDocumentFragment();
        for (const n of taxonomyFlat) {
            const option = document.createElement('option');
            option.value = String(n.id);
            option.label = `${n.id} \u2014 ${n.name}`;
            fragment.appendChild(option);
        }
        options.replaceChildren(fragment);
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

    // Keep the public page interactive for unusually broad research areas.
    // The complete matrix remains in the API response; this is only a bounded
    // DOM preview, preventing tens of thousands of cells from freezing a tab.
    const MAX_RENDERED_CELLS = 1600;
    const visibleMethods = matrix.methods.slice(0, Math.min(matrix.methods.length, 40));
    const visibleDatasetCount = Math.max(1, Math.floor(MAX_RENDERED_CELLS / Math.max(visibleMethods.length, 1)));
    const visibleDatasets = matrix.datasets.slice(0, visibleDatasetCount);
    const isTruncated = visibleMethods.length < matrix.methods.length || visibleDatasets.length < matrix.datasets.length;

    let html = '<div class="matrix-controls">';
    html += '<label>Metric:</label>';
    html += '<select class="matrix-metric-select" onchange="window._dg.updateMatrixMetric(this)">';
    for (const m of metrics) {
        html += `<option value="${esc(m)}"${m === defaultMetric ? ' selected' : ''}>${esc(m || '(none)')}</option>`;
    }
    html += '</select>';
    html += `<span class="matrix-info">${matrix.methods.length} methods x ${matrix.datasets.length} datasets</span>`;
    if (isTruncated) html += `<span class="matrix-info">Showing first ${visibleMethods.length} x ${visibleDatasets.length} cells</span>`;
    html += '</div>';

    html += '<div class="matrix-scroll"><table class="matrix-table">';
    html += '<thead><tr><th class="method-header">Method \\ Dataset</th>';
    for (const ds of visibleDatasets) {
        html += `<th class="dataset-header" title="${esc(ds)}">${esc(trunc(ds, 16))}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const method of visibleMethods) {
        html += '<tr>';
        html += `<td class="method-cell" title="${esc(method)}">${esc(trunc(method, 22))}</td>`;
        for (const ds of visibleDatasets) {
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
    container._visibleMethods = visibleMethods;
    container._visibleDatasets = visibleDatasets;
}

function updateMatrixMetric(selectEl) {
    const container = selectEl.closest('.card') ? selectEl.closest('.card').querySelector('.matrix-wrap, [class*="matrix"]') : el('evidenceMatrixContainer');
    const matrix = container ? container._matrixData : null;
    if (!matrix) return;

    const metric = selectEl.value;
    const rows = container.querySelectorAll('tbody tr');

    rows.forEach((row, mi) => {
        const method = container._visibleMethods?.[mi] || matrix.methods[mi];
        const cells = row.querySelectorAll('td.matrix-cell');
        cells.forEach((td, di) => {
            const ds = container._visibleDatasets?.[di] || matrix.datasets[di];
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

// ── Manuscripts Tab ──────────────────────────────────────────────────

async function loadPapers() {
    papersLoaded = true;
    try {
        allPapers = await api('/api/generated_papers?limit=200');
        renderPapers();
    } catch (e) {
        console.error('Manuscripts load error:', e);
    }
}

function paperIsComplete(paper) {
    const status = String(paper.status || paper.manuscript_status || '').toLowerCase();
    return Boolean(
        paper.paper_complete ||
        paper.bundle_count > 0 ||
        paper.pdf_url ||
        paper.tex_url ||
        paper.main_pdf ||
        paper.main_tex ||
        ['bundle_ready', 'completed', 'paper_ready', 'ready', 'submitted', 'accepted'].includes(status)
    );
}

function renderPapers() {
    const list = el('papersList');
    const detail = el('paperDetail');
    const countBadge = el('papersCountBadge');
    const query = (el('papersSearch').value || '').toLowerCase();
    const status = el('papersStatus').value;

    let filtered = [...allPapers];
    if (query) {
        filtered = filtered.filter(p =>
            (p.title || '').toLowerCase().includes(query) ||
            (p.id || '').toLowerCase().includes(query) ||
            (p.abstract || '').toLowerCase().includes(query) ||
            (p.evidence_summary || '').toLowerCase().includes(query)
        );
    }
    if (status) {
        filtered = filtered.filter(p => p.status === status);
    }
    filtered.sort((a, b) => {
        const completeDelta = Number(paperIsComplete(b)) - Number(paperIsComplete(a));
        if (completeDelta) return completeDelta;
        return String(b.updated_at || b.created_at || '').localeCompare(String(a.updated_at || a.created_at || ''));
    });
    if (countBadge) countBadge.textContent = `${filtered.length} manuscript${filtered.length === 1 ? '' : 's'}`;

    if (filtered.length === 0) {
        list.innerHTML = '<p class="empty-msg">No generated manuscripts found.</p>';
        if (detail) {
            detail.innerHTML = `<div class="paper-reader-empty">
                <div class="paper-reader-empty-title">No matching manuscripts</div>
                <p>Adjust the search text or status filter to reopen the manuscript reader.</p>
            </div>`;
        }
        return;
    }
    if (!selectedPaperId || !filtered.some(p => p.id === selectedPaperId)) {
        selectedPaperId = filtered[0].id;
    }

    list.innerHTML = filtered.map((p, idx) => {
        const sc = p.status ? 's-' + p.status : '';
        const nodes = parseJsonArray(p.source_node_ids).slice(0, 3);
        const bundleLabel = p.bundle_count ? `${p.bundle_count} bundle${p.bundle_count === 1 ? '' : 's'}` : `${p.asset_count || 0} assets`;
        return `<div class="paper-note-card ${p.id === selectedPaperId ? 'active' : ''}" data-paper-id="${esc(p.id)}" onclick="window._dg.selectPaper('${esc(p.id)}')">
            <div class="paper-note-kicker">
                <span>Idea ${esc(String(p.insight_id || p.id))}</span>
                <span class="paper-status ${sc}">${esc(p.status || 'unknown')}</span>
            </div>
            <div class="paper-note-title">${esc(trunc(p.title, 92))}</div>
            <div class="paper-note-meta">${esc(bundleLabel)}${p.updated_at ? ` · ${esc(p.updated_at.slice(0, 10))}` : ''}</div>
            ${nodes.length ? `<div class="paper-note-tags">${nodes.map(c => `<span>${esc(c)}</span>`).join('')}</div>` : ''}
        </div>`;
    }).join('');
    renderPaperDetail(selectedPaperId);
}

async function selectPaper(pid) {
    selectedPaperId = pid;
    $$('.paper-note-card').forEach(card => card.classList.toggle('active', card.dataset.paperId === pid));
    await renderPaperDetail(pid);
}

async function renderPaperDetail(pid) {
    const detail = el('paperDetail');
    const paper = allPapers.find(p => p.id === pid);
    if (!detail || !paper) return;
    detail.innerHTML = '<div class="paper-reader-loading">Loading generated manuscript...</div>';
    try {
        const method = parseJsonObject(paper.proposed_method);
        const nodes = parseJsonArray(paper.source_node_ids);
        const assets = paper.assets || [];
        const bundles = paper.bundles || [];
        const assetRows = assets.slice(0, 18).map(a => {
            const assetPath = a.path || '';
            const url = `/papers/${encodeURIComponent(String(paper.insight_id))}/view/${encodeURI(assetPath)}`;
            return `<tr>
                <td><a href="${url}" target="_blank">${esc(assetPath)}</a></td>
                <td>${esc(a.suffix || '-')}</td>
                <td>${fmt(a.size || 0)}</td>
            </tr>`;
        }).join('');
        const bundleRows = bundles.map(b => `<tr>
            <td>${esc(b.bundle_format || 'bundle')}</td>
            <td>${esc(b.bundle_status || b.status || 'ready')}</td>
            <td>${esc(b.bundle_created_at || b.updated_at || '-')}</td>
        </tr>`).join('');

        detail.innerHTML = `
            <div class="paper-arxiv-page">
                <div class="paper-arxiv-topline">
                    <span class="paper-status ${paper.status ? 's-' + paper.status : ''}">${esc(paper.status || 'unknown')}</span>
                    <span>${esc(paper.manuscript_status || 'manuscript')}</span>
                    <span>${fmt(paper.asset_count || 0)} assets</span>
                    ${paper.bundle_count ? `<span>${fmt(paper.bundle_count)} bundle${paper.bundle_count === 1 ? '' : 's'}</span>` : ''}
                </div>
                <h1>${esc(paper.title || paper.id)}</h1>
                <div class="paper-authors">DeepGraph generated manuscript · Idea ${esc(String(paper.insight_id))}</div>
                <div class="paper-arxiv-meta">
                    <span>${esc(paper.id)}</span>
                    ${paper.updated_at ? `<span>updated ${esc(paper.updated_at)}</span>` : ''}
                    ${nodes.map(c => `<span>${esc(c)}</span>`).join('')}
                </div>
                <div class="paper-actions">
                    ${paper.preview_url ? `<a href="${esc(paper.preview_url)}" target="_blank">Preview page</a>` : ''}
                    ${paper.pdf_url ? `<a href="${esc(paper.pdf_url)}" target="_blank">Open PDF</a>` : ''}
                    ${paper.tex_url ? `<a href="${esc(paper.tex_url)}" target="_blank">Open TeX</a>` : ''}
                </div>
                <section class="paper-reader-section">
                    <h4>Generated Abstract</h4>
                    <p>${esc(paper.abstract || 'No generated abstract or problem framing is available yet.')}</p>
                </section>
                <section class="paper-reader-section">
                    <h4>Paper Notes</h4>
                    <ul class="paper-claim-list">
                        ${paper.problem_statement ? `<li><strong>Problem.</strong> ${esc(paper.problem_statement)}</li>` : ''}
                        ${method && (method.name || method.type || method.definition) ? `<li><strong>Method.</strong> ${esc([method.name, method.type, method.definition].filter(Boolean).join(' · '))}</li>` : ''}
                        ${paper.evidence_summary ? `<li><strong>Evidence.</strong> ${esc(paper.evidence_summary)}</li>` : ''}
                        ${paper.canonical_run_id ? `<li><strong>Canonical run.</strong> #${esc(String(paper.canonical_run_id))}</li>` : ''}
                    </ul>
                </section>
                <section class="paper-reader-section">
                    <h4>Submission Bundles</h4>
                    ${bundleRows ? `<table class="paper-results-table"><thead><tr><th>Format</th><th>Status</th><th>Created</th></tr></thead><tbody>${bundleRows}</tbody></table>` : '<p class="empty-msg">No submission bundle has been recorded yet.</p>'}
                </section>
                <section class="paper-reader-section">
                    <h4>Paper Assets</h4>
                    ${assetRows ? `<table class="paper-results-table"><thead><tr><th>Asset</th><th>Type</th><th>Size</th></tr></thead><tbody>${assetRows}</tbody></table>` : '<p class="empty-msg">No manuscript assets found yet.</p>'}
                </section>
            </div>`;
        typesetMath(detail);
    } catch (e) {
        detail.innerHTML = '<p class="empty-msg" style="padding:18px;">Failed to load paper details.</p>';
    }
}

function parseJsonArray(value) {
    if (Array.isArray(value)) return value;
    if (!value) return [];
    try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
        return String(value).split(',').map(x => x.trim()).filter(Boolean);
    }
}

function parseJsonObject(value) {
    if (value && typeof value === 'object' && !Array.isArray(value)) return value;
    if (!value) return {};
    try {
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    } catch (e) {
        return {};
    }
}

// ── Opportunities Tab ────────────────────────────────────────────────

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
    typesetMath(list);
}


// ── Discoveries Tab (Tier 1 + Tier 2 Deep Insights) ──────────────────

async function loadDiscoveriesTab() {
    const tierFilter = el('discoveryTierFilter')?.value || '';
    const list = el('discoveriesList');
    if (currentAgendaId == null) {
        if (list) list.innerHTML = `<p class="empty-msg">${esc(tr('ideas.emptyNoAgenda', 'No research agenda is registered yet, so there is no scope to list ideas from.'))}</p>`;
        return;
    }
    try {
        await loadEvidenceStates();
        let url = '/api/deep_insights?limit=50';
        if (tierFilter) url += `&tier=${tierFilter}`;
        const insights = await api(url);
        renderDiscoveries(insights);
    } catch (e) {
        if (list) list.innerHTML = `<p class="empty-msg">${esc(tr('ideas.emptyFiltering', 'No ready discoveries yet. Automatic discovery is still filtering candidates.'))}</p>`;
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

function renderDiscoverySources(d) {
    const sourceKinds = parseJsonArray(d.signal_mix).slice(0, 4);
    const sourceNodes = parseJsonArray(d.source_node_ids).slice(0, 4);
    const sourceRefs = parseJsonArray(d.source_signal_ids).slice(0, 3);
    let sourcePapers = parseJsonArray(d.source_paper_ids);
    if (!sourcePapers.length) sourcePapers = parseJsonArray(d.supporting_papers);
    if (!sourcePapers.length && d.evidence_packet) {
        try {
            const packet = JSON.parse(d.evidence_packet);
            sourcePapers = (packet.papers || []).map(p => p.id || p.title).filter(Boolean);
        } catch (e) {}
    }
    sourcePapers = sourcePapers.slice(0, 4);

    const chips = [];
    for (const s of sourceKinds) chips.push(`<span class="chip source-kind">${esc(String(s).replace(/_/g, ' '))}</span>`);
    for (const n of sourceNodes) chips.push(`<span class="chip" onclick="window._dg.exploreNode('${esc(n)}')">${esc(n)}</span>`);
    for (const p of sourcePapers) chips.push(`<span class="chip source-paper">${esc(p)}</span>`);
    for (const sid of sourceRefs) chips.push(`<span class="chip source-ref">source #${esc(sid)}</span>`);
    if (!chips.length) return '';
    return `<div class="insight-sources">
        <span class="insight-label">Sources:</span>
        <div class="chip-row">${chips.join('')}</div>
    </div>`;
}

function renderDiscoveries(discoveries) {
    const list = el('discoveriesList');
    const visible = (discoveries || []).filter(isDisplayableDiscovery);
    if (visible.length === 0) {
        list.innerHTML = '<p class="empty-msg">No ready discoveries yet. Automatic discovery is still filtering candidates.</p>';
        return;
    }

    list.innerHTML = visible.map(d => {
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
        const sourcesHtml = renderDiscoverySources(d);

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
                    <span style="margin:0 4px;">-&gt;</span>
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
            </div>
            <div class="insight-title">${esc(d.title)}</div>
            <div style="margin:4px 0 8px;">${scientificBadge(ideaEvidenceEntry(d.id))}</div>
            ${bodyHtml}
            ${d.evidence_summary ? `<div class="insight-evidence"><span class="insight-label">Evidence:</span> ${esc(trunc(d.evidence_summary, 250))}</div>` : ''}
            ${sourcesHtml}
            <div class="insight-impact"><span class="insight-label">Mode:</span> Fixed automatic pipeline</div>
        </div>`;
    }).join('');
    typesetMath(list);
}


// ── Experiments Tab (SciForge) ────────────────────────────────────────

async function loadExperimentsTab() {
    const statusFilter = el('experimentStatusFilter')?.value || '';
    const badge = el('timelineAgendaBadge');
    if (badge) {
        const active = agendaList.find(a => a.id === currentAgendaId);
        badge.textContent = active ? `agenda #${active.id}: ${trunc(active.name, 40)}` : 'no agenda scope';
    }
    try {
        const automation = await api('/api/automation');
        renderAutomationOverview(automation);
    } catch (e) {
        console.error('Automation snapshot failed:', e);
    }
    if (currentAgendaId != null) {
        try {
            await loadEvidenceStates();
            const timeline = await api(`/api/v1/agendas/${currentAgendaId}/timeline?limit=120`);
            renderProcessTimeline(timeline.events || []);
        } catch (e) {
            const tl = el('processTimeline');
            if (tl) tl.innerHTML = '<p class="empty-msg">Timeline unavailable. The provenance API may not be deployed yet.</p>';
        }
        try {
            const selection = await api(`/api/v1/agendas/${currentAgendaId}/selection`);
            renderSelectionRationale(selection);
        } catch (e) {
            const sr = el('selectionRationale');
            if (sr) sr.innerHTML = '<p class="empty-msg">Selection records unavailable.</p>';
        }
        try {
            let url = '/api/experiment_groups?limit=50';
            if (statusFilter) url += `&status=${statusFilter}`;
            const groups = await api(url);
            renderExperimentGroupsV2(groups);
        } catch (e) {
            const list = el('experimentsList');
            if (list) list.innerHTML = `<p class="empty-msg">Experiment history failed to load: ${esc(e.message)}</p>`;
        }
        try {
            const meta = await api('/api/meta_report');
            renderMetaReport(meta);
        } catch (e) {
            console.error('Meta report failed:', e);
        }
    } else {
        const tl = el('processTimeline');
        if (tl) tl.innerHTML = `<p class="empty-msg">${esc(tr('process.timelineNoAgenda', 'No research agenda is registered yet, so there is no process to show.'))}</p>`;
        const list = el('experimentsList');
        if (list) list.innerHTML = `<p class="empty-msg">${esc(tr('ideas.emptyNoAgenda', 'No research agenda is registered yet.'))}</p>`;
    }
}

// ── Process timeline rendering ───────────────────────────────────────

const TIMELINE_KIND_META = {
    signal:             { key: 'tl.signal',        label: 'SIGNALS',       color: '#2e86ab' },
    candidate_decision: { key: 'tl.candidate',     label: 'CANDIDATE',     color: '#a8842a' },
    authorization:      { key: 'tl.authorization', label: 'AUTHORIZATION', color: '#7a5ea8' },
    job:                { key: 'tl.run',           label: 'RUN',           color: '#4a7c9b' },
    evidence:           { key: 'tl.evidence',      label: 'EVIDENCE',      color: '#2e86ab' },
    decision:           { key: 'tl.decision',      label: 'DECISION',      color: '#3d8b5e' },
    outcome:            { key: 'tl.outcome',       label: 'OUTCOME',       color: '#3d8b5e' },
};

function timelineEventText(ev) {
    switch (ev.kind) {
        case 'signal':
            return `Frontier packet for problem #${ev.research_problem_id ?? '?'} — gate ${ev.gate_allowed ? 'allowed' : 'refused'}`
                + ((ev.gate_reason_codes || []).length ? ` [${ev.gate_reason_codes.map(esc).join(', ')}]` : '');
        case 'candidate_decision':
            return `Idea #${ev.idea_id ?? '?'} ${esc(ev.decision || 'decided')}`
                + ((ev.reason_codes || []).length ? ` [${ev.reason_codes.map(esc).join(', ')}]` : '');
        case 'authorization':
            return `Grant for idea #${ev.idea_id ?? '?'} — stage ${esc(ev.stage || '?')}, cap ${fmt(ev.token_cap || 0)} tokens`
                + (ev.max_gpu_hours ? `, ${ev.max_gpu_hours} GPU-h` : '') + ` (${esc(ev.status || 'issued')})`;
        case 'job':
            return `Job #${ev.job_id ?? '?'} [${esc(ev.backend_kind || '?')}/${esc(ev.stage || '?')}] ${esc(ev.status || '')}`
                + (ev.failure_reason ? ` — ${esc(trunc(ev.failure_reason, 160))}` : '');
        case 'evidence': {
            const blockers = (ev.context && ev.context.blockers) || [];
            return `Run #${ev.run_id ?? '?'}: ${esc(ev.from_state || '?')} -> ${esc(ev.to_state || '?')}`
                + (blockers.length ? ` — blockers: ${blockers.map(esc).join(', ')}` : '');
        }
        case 'decision':
            return `Run #${ev.run_id ?? '?'} scientifically decided: ${esc(ev.verdict || 'inconclusive')}`
                + (ev.verdict_hash ? ` (hash ${esc(String(ev.verdict_hash).slice(0, 18))}...)` : '');
        case 'outcome':
            return `Idea #${ev.idea_id ?? '?'} outcome: ${esc(ev.execution_result || '?')}`
                + (ev.effect != null && ev.baseline != null ? `, effect ${ev.effect} vs baseline ${ev.baseline}` : '')
                + (ev.verdict ? `, verdict ${esc(ev.verdict)}` : '');
        default:
            return esc(JSON.stringify(ev));
    }
}

function renderProcessTimeline(events) {
    const container = el('processTimeline');
    if (!container) return;
    if (!events.length) {
        container.innerHTML = `<p class="empty-msg">${esc(tr('process.timelineEmptyDetail', 'No process events recorded for this agenda yet. Events appear once signals, grants, or experiment runs exist.'))}</p>`;
        return;
    }
    container.innerHTML = `<div class="timeline-list">` + events.map(ev => {
        const kindMeta = TIMELINE_KIND_META[ev.kind] || { label: (ev.kind || '?').toUpperCase(), color: '#888' };
        const meta = { ...kindMeta, label: kindMeta.key ? tr(kindMeta.key, kindMeta.label) : kindMeta.label };
        const failed = (ev.kind === 'job' && /fail|timed_out|cancel/.test(ev.status || ''))
            || (ev.kind === 'signal' && !ev.gate_allowed)
            || (ev.kind === 'candidate_decision' && /reject|refuse/.test(ev.decision || ''));
        return `<div class="timeline-row${failed ? ' timeline-row-failed' : ''}">
            <span class="timeline-when">${esc(trunc(ev.at || '', 16))}</span>
            <span class="timeline-kind" style="color:${meta.color};border-color:${meta.color};">${meta.label}</span>
            <span class="timeline-text">${timelineEventText(ev)}</span>
        </div>`;
    }).join('') + `</div>`;
}

// ── Selection rationale rendering ────────────────────────────────────

function renderSelectionRationale(data) {
    const container = el('selectionRationale');
    if (!container) return;
    const selections = (data && data.selections) || [];
    const decisions = (data && data.decisions) || [];
    if (!selections.length && !decisions.length) {
        container.innerHTML = `<p class="empty-msg">${esc(tr('process.rationaleEmptyDetail', 'No selection records for this agenda yet. Rationale appears once the selector has admitted or rejected candidates.'))}</p>`;
        return;
    }
    let html = '';
    for (const sel of selections.slice(0, 5)) {
        const rejected = (sel.rejected_candidates || []).slice(0, 6).map(rc => {
            const title = typeof rc === 'string' ? rc : (rc.title || `insight #${rc.insight_id ?? rc.id ?? '?'}`);
            const why = typeof rc === 'object' ? (rc.reason || rc.why || (rc.reasons || []).join('; ') || '') : '';
            const score = typeof rc === 'object' && rc.score != null ? ` (score ${rc.score})` : '';
            return `<li><b>REJECTED</b> ${esc(trunc(title, 90))}${score}${why ? ` — ${esc(trunc(why, 140))}` : ''}</li>`;
        }).join('');
        html += `<div class="insight-card" style="border-left:3px solid #3d8b5e;">
            <div class="insight-header">
                <span class="insight-type" style="color:#3d8b5e;font-weight:700;">SELECTED insight #${sel.selected_insight_id ?? '?'}</span>
                ${sel.score != null ? `<span class="insight-scores">score ${sel.score}</span>` : ''}
                <span style="color:var(--text-dim);font-size:0.68rem;">${esc(trunc(sel.created_at || '', 16))}</span>
            </div>
            ${sel.rationale ? `<div class="insight-hypothesis"><span class="insight-label">Why:</span> ${esc(trunc(sel.rationale, 400))}</div>` : ''}
            ${rejected ? `<div class="insight-evidence"><span class="insight-label">Not chosen:</span><ul style="margin:4px 0;padding-left:18px;">${rejected}</ul></div>` : ''}
        </div>`;
    }
    if (decisions.length) {
        const rows = decisions.slice(0, 12).map(d =>
            `<div class="timeline-row${/reject|refuse/.test(d.decision || '') ? ' timeline-row-failed' : ''}">
                <span class="timeline-when">${esc(trunc(d.decided_at || '', 16))}</span>
                <span class="timeline-kind">${esc((d.decision || '?').toUpperCase())}</span>
                <span class="timeline-text">idea #${d.idea_id ?? '?'}${(d.reason_codes || []).length ? ` [${d.reason_codes.map(esc).join(', ')}]` : ''}</span>
            </div>`).join('');
        html += `<div class="timeline-list" style="margin-top:8px;">${rows}</div>`;
    }
    container.innerHTML = html;
}

function serviceState(name, ok, active) {
    if (ok === false) return { label: 'missing', color: '#c4453a' };
    if (active) return { label: 'active', color: '#3d8b5e' };
    return { label: 'ready', color: '#a8842a' };
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
            'Paper Pipeline',
            serviceState('paper', true, paper.running),
            `Batch ${esc(paper.batch_size || '?')}, ${esc(paper.status || 'idle')}`
        ),
        serviceCard(
            'Auto Research',
            serviceState('auto', true, auto.running),
            `${auto.total || 0} jobs, ${auto.running_experiment || 0} experiments, ${auto.blocked || 0} blocked`
        ),
        serviceCard(
            'EvoScientist',
            serviceState('evoscientist', evo.available, (evo.active_count || 0) > 0),
            `${evo.active_count || 0} active sessions`
        ),
        serviceCard(
            'PaperOrchestra',
            serviceState('paperorchestra', po.available, (po.active_count || 0) > 0),
            `${(po.counts || {}).bundle_ready || 0} bundles, ${(po.counts || {}).drafting || 0} drafting`
        ),
        serviceCard(
            'GPU Scheduler',
            serviceState('gpu', true, (gpu.running_jobs || 0) > 0),
            `${gpu.running_jobs || 0} running, ${gpu.queued_jobs || 0} queued, ${(gpu.workers || []).length} workers`
        ),
    ].join('');

    work.innerHTML = [
        workLane('Pipeline activity', current.pipeline, item =>
            `${esc(item.status || '')} / ${esc(item.stage || '')}`, item => item.title),
        workLane('Processing papers', current.papers, item =>
            `${esc(item.id || '')} · ${esc(item.processing_stage || item.status || '')}`, item => item.title),
        workLane('Generating experiment plans', current.experiment_plans, item =>
            `${esc(item.status || '')} / ${esc(item.stage || '')}`, item => item.title),
        workLane('Running experiments', current.experiments, item =>
            `Run #${esc(item.id || '')} · ${esc(item.status || '')} · ${esc(item.phase || '')}`, item => item.title),
        workLane('Writing papers', current.manuscripts, item =>
            `Manuscript #${esc(item.id || '')} · ${esc(item.status || '')}`, item => item.title),
    ].join('');
}

function workLane(title, items, metaFn, titleFn) {
    const rows = (items || []).slice(0, 4);
    const body = rows.length
        ? rows.map(item => `<div class="work-item">
            <div class="work-item-title">${esc(trunc(titleFn(item) || 'Untitled', 80))}</div>
            <div>${metaFn(item)}</div>
            ${item.last_note ? `<div>${esc(trunc(item.last_note, 110))}</div>` : ''}
            ${item.last_error ? `<div style="color:#c4453a;">${esc(trunc(item.last_error, 110))}</div>` : ''}
        </div>`).join('')
        : '<div class="work-item">Idle</div>';
    return `<div class="work-lane"><div class="work-lane-title">${esc(title)}</div>${body}</div>`;
}

function friendlyAutomationStage(status, stage) {
    const key = String(stage || status || '').toLowerCase();
    if (key.includes('verification')) return 'Checking novelty';
    if (key.includes('research')) return 'Running EvoScientist research';
    if (key.includes('review') || key.includes('forge') || key.includes('formal')) return 'Generating experiment plan';
    if (key.includes('gpu')) return 'Running on GPU';
    if (key.includes('validation') || key.includes('experiment')) return 'Running experiment';
    if (key.includes('writing') || key.includes('submission') || key.includes('bundle')) return 'Writing paper';
    if (key.includes('blocked')) return 'Blocked';
    if (key.includes('failed')) return 'Failed';
    if (key.includes('complete')) return 'Complete';
    return stage || status || 'Queued';
}

function renderExperiments(runs) {
    const list = el('experimentsList');
    if (!runs || !runs.length) {
        list.innerHTML = '<p class="empty-msg">No experiments are active yet. The automatic queue will start them when a discovery is ready.</p>';
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
                <button class="btn-preview" onclick="window._dg.viewExperiment(${r.id})">View Details</button>
            </div>
        </div>`;
    }).join('');
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

// ── Two-register status badges ───────────────────────────────────────
// Operational status (did the job run?) and scientific status (what does the
// evidence ladder say?) are rendered as two separate badges and never merged:
// an operationally "completed" run stays scientifically "not assessed" until
// it has actually climbed the ladder.

const SCI_STATE_LABELS = {
    planned: 'PLANNED',
    sanity_passed: 'SANITY PASSED',
    full_benchmark_complete: 'BENCHMARK DONE',
    evidence_audited: 'AUDITED',
    scientifically_decided: 'DECIDED',
    manuscript_allowed: 'MANUSCRIPT ALLOWED',
};

function operationalBadge(status) {
    const label = status || 'unknown';
    return `<span class="reg-badge reg-op" style="border-color:${experimentStatusColor(label)};color:${experimentStatusColor(label)};" title="${esc(tr('badge.run.tip', 'Operational status: whether the job ran; it makes no scientific claim'))}">${esc(tr('badge.run', 'RUN'))}: ${esc(label)}</span>`;
}

function scientificBadge(entry) {
    if (!entry || !entry.state) {
        return `<span class="reg-badge reg-sci reg-sci-none" title="${esc(tr('badge.notAssessed.tip', 'No evidence-ladder progress recorded; completion of a job is not a finding'))}">${esc(tr('badge.evidence', 'EVIDENCE'))}: ${esc(tr('badge.notAssessed', 'not assessed'))}</span>`;
    }
    if (entry.state === 'scientifically_decided' || entry.state === 'manuscript_allowed') {
        const verdict = entry.verdict || 'inconclusive';
        const cls = verdict === 'supported' ? 'reg-sci-supported'
            : verdict === 'refuted' ? 'reg-sci-refuted' : 'reg-sci-inconclusive';
        return `<span class="reg-badge reg-sci ${cls}" title="${esc(tr('badge.decided.tip', 'Scientific verdict recorded by the audited decision gate'))}">${esc(tr('badge.decided', 'DECIDED'))}: ${esc(tr('verdict.' + verdict, verdict))}</span>`;
    }
    const label = tr('sci.' + entry.state, SCI_STATE_LABELS[entry.state] || entry.state);
    return `<span class="reg-badge reg-sci reg-sci-progress" title="${esc(tr('badge.progress.tip', 'Position on the evidence ladder; not yet a scientific decision'))}">${esc(tr('badge.evidence', 'EVIDENCE'))}: ${esc(label)}</span>`;
}

async function loadEvidenceStates() {
    if (currentAgendaId == null) { evidenceStateMap = null; return null; }
    try {
        evidenceStateMap = await api('/api/v1/evidence_states');
    } catch (e) {
        evidenceStateMap = null;
    }
    return evidenceStateMap;
}

function ideaEvidenceEntry(insightId) {
    if (!evidenceStateMap || !evidenceStateMap.ideas) return null;
    return evidenceStateMap.ideas[String(insightId)] || null;
}

function runEvidenceEntry(runId) {
    if (!evidenceStateMap || !evidenceStateMap.runs) return null;
    return evidenceStateMap.runs[String(runId)] || null;
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
        <span class="insight-label">Paper blocked:</span>
        ${blockers.slice(0, limit).map(x => `<div>${esc(trunc(x, 160))}</div>`).join('')}
        ${blockers.length > limit ? `<div>${blockers.length - limit} more blocker(s)</div>` : ''}
    </div>`;
}

function renderExperimentGroupsV2(groups) {
    const list = el('experimentsList');
    if (!groups || !groups.length) {
        list.innerHTML = '<p class="empty-msg">No experiment ideas are active yet. The automatic queue will start them when ready.</p>';
        return;
    }

    list.innerHTML = groups.map(group => {
        const insight = group.insight || {};
        const auto = group.auto_job || {};
        const currentRun = group.canonical_run || group.latest_run || null;
        const color = experimentStatusColor((currentRun || {}).status || auto.status);
        const sciEntry = (currentRun && runEvidenceEntry(currentRun.id)) || ideaEvidenceEntry(insight.id);
        const badgeRow = `${operationalBadge((currentRun || {}).status || auto.status || 'not_started')}${scientificBadge(sciEntry)}`;
        const effect = currentRun && currentRun.effect_pct != null
            ? `${currentRun.effect_pct >= 0 ? '+' : ''}${currentRun.effect_pct.toFixed(2)}%`
            : '';
        const progress = auto.stage
            ? friendlyAutomationStage(auto.status, auto.stage)
            : ((currentRun || {}).status || 'not_started');
        const currentRunLabel = currentRun ? `Main run #${currentRun.id}` : 'No run created yet';
        const previewUrl = (((group || {}).paper_preview_urls || {}).index) || '';
        const plan = group.plan_snapshot || {};
        const latest = plan.latest_status || {};
        const manuscriptBlockers = plan.manuscript_blockers || {};
        const planReady = [
            plan.experiment_spec ? 'experiment spec' : '',
            plan.evidence_plan ? 'evidence plan' : '',
            plan.manuscript_input_state ? 'manuscript state' : '',
            plan.manuscript_blockers ? 'manuscript blockers' : '',
        ].filter(Boolean).join(', ') || 'waiting for plan files';
        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">IDEA #${insight.id}</span>
                <span class="insight-scores">${esc(currentRunLabel)}</span>
                ${effect ? `<span class="insight-scores">Effect: ${effect}</span>` : ''}
                <span style="color:var(--text-dim);font-size:0.68rem;">Tier ${insight.tier || '?'}</span>
            </div>
            <div class="insight-title">${esc(insight.title || 'Deep Insight')}</div>
            <div style="margin:4px 0 8px;">${badgeRow}</div>
            <div class="insight-impact"><span class="insight-label">Current work:</span> ${esc(progress)}</div>
            ${latest.stage ? `<div class="insight-experiment"><span class="insight-label">Latest file status:</span> ${esc(latest.stage)} / ${esc(latest.status || '')}</div>` : ''}
            ${latest.error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">Latest error:</span> ${esc(trunc(latest.error, 180))}</div>` : ''}
            ${renderManuscriptBlockers(manuscriptBlockers)}
            <div class="insight-evidence"><span class="insight-label">Plan files:</span> ${esc(planReady)}</div>
            <div style="display:flex;gap:16px;margin:6px 0;font-size:0.75rem;color:var(--text-secondary);flex-wrap:wrap;">
                <span>Runs: ${group.run_count || 0}</span>
                <span>Latest run: ${esc((group.latest_run || {}).status || 'none')}</span>
                <span>Bundle: ${esc(insight.submission_status || 'not_started')}</span>
            </div>
            <div class="chip-row" style="margin:8px 0;">${renderTrackChips(group.planned_tracks)}</div>
            ${auto.last_note ? `<div class="insight-experiment"><span class="insight-label">Latest:</span> ${esc(trunc(auto.last_note, 220))}</div>` : ''}
            ${auto.last_error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">Error:</span> ${esc(trunc(auto.last_error, 220))}</div>` : ''}
            <div class="insight-actions">
                <button class="btn-preview" onclick="window._dg.viewExperimentGroup(${insight.id})">View automation history</button>
                ${currentRun ? `<button class="btn-preview" onclick="window._dg.viewExperiment(${currentRun.id})">View main run</button>` : ''}
                ${previewUrl ? `<button class="btn-preview" onclick="window.open('${esc(previewUrl)}','_blank')">Open paper preview</button>` : ''}
            </div>
        </div>`;
    }).join('');
}

function renderExperimentGroups(groups) {
    const list = el('experimentsList');
    if (!groups || !groups.length) {
        list.innerHTML = '<p class="empty-msg">No experiment ideas are active yet. The automatic queue will start them when ready.</p>';
        return;
    }

    list.innerHTML = groups.map(group => {
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
            ? `${auto.status || 'queued'} / ${auto.stage}`
            : ((currentRun || {}).status || 'not_started');
        const currentRunLabel = currentRun
            ? `主实验 Run #${currentRun.id}`
            : '尚未创建 run';
        const previewUrl = (((group || {}).paper_preview_urls || {}).index) || '';
        return `<div class="insight-card" style="border-left: 3px solid ${color};">
            <div class="insight-header">
                <span class="insight-type" style="color:${color};font-weight:700;">IDEA #${insight.id}</span>
                <span class="insight-scores">${esc(currentRunLabel)}</span>
                ${verdict}
                ${effect ? `<span class="insight-scores">Effect: ${effect}</span>` : ''}
                <span style="color:var(--text-dim);font-size:0.68rem;">Tier ${insight.tier || '?'}</span>
            </div>
            <div class="insight-title">${esc(insight.title || 'Deep Insight')}</div>
            <div class="insight-impact"><span class="insight-label">当前进度:</span> ${esc(progress)}</div>
            <div style="display:flex;gap:16px;margin:6px 0;font-size:0.75rem;color:var(--text-secondary);flex-wrap:wrap;">
                <span>历史 runs: ${group.run_count || 0}</span>
                <span>最新状态: ${esc((group.latest_run || {}).status || 'none')}</span>
                <span>Bundle: ${esc(insight.submission_status || 'not_started')}</span>
            </div>
            <div class="chip-row" style="margin:8px 0;">${renderTrackChips(group.planned_tracks)}</div>
            ${auto.last_note ? `<div class="insight-experiment"><span class="insight-label">Latest:</span> ${esc(trunc(auto.last_note, 220))}</div>` : ''}
            ${auto.last_error ? `<div class="insight-impact" style="color:#c4453a;"><span class="insight-label">Error:</span> ${esc(trunc(auto.last_error, 220))}</div>` : ''}
            <div class="insight-actions">
                <button class="btn-preview" onclick="window._dg.viewExperimentGroup(${insight.id})">查看实验历史</button>
                ${currentRun ? `<button class="btn-preview" onclick="window._dg.viewExperiment(${currentRun.id})">查看主实验详情</button>` : ''}
                ${previewUrl ? `<button class="btn-preview" onclick="window.open('${esc(previewUrl)}','_blank')">打开论文预览</button>` : ''}
            </div>
        </div>`;
    }).join('');
}

function jsonPreview(obj, emptyText = '暂无') {
    if (!obj || (typeof obj === 'object' && Object.keys(obj).length === 0)) {
        return `<p class="empty-msg">${esc(emptyText)}</p>`;
    }
    return `<pre style="white-space:pre-wrap;word-break:break-word;background:var(--bg-elevated);padding:10px;border-radius:8px;font-size:0.72rem;">${esc(JSON.stringify(obj, null, 2))}</pre>`;
}

function renderPaperAssetLinks(insightId, assets) {
    if (!assets || !assets.length) {
        return '<p class="empty-msg">暂无论文资产。</p>';
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

// ── Retired public runtime panel ───────────────────────────────────────────
// The operator console is intentionally not shipped in the public dashboard.
// Keep this no-op for old callers while the dashboard is rolling out.

async function loadProviders() {
    return undefined;
}

function runtimeValue(value, suffix = '') {
    if (value === null || value === undefined || value === '') return '-';
    return `${esc(String(value))}${suffix}`;
}

function runtimeBool(value) {
    return value ? '<span class="runtime-pill good">On</span>' : '<span class="runtime-pill muted">Off</span>';
}

function renderRuntimeMetric(label, value, hint = '') {
    return `<div class="runtime-metric">
        <div class="runtime-metric-value">${value}</div>
        <div class="runtime-metric-label">${esc(label)}</div>
        ${hint ? `<div class="runtime-metric-hint">${esc(hint)}</div>` : ''}
    </div>`;
}

function percentBar(value, color = '#3d8b5e') {
    const n = Number(value);
    const pct = Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : 0;
    return `<div style="height:5px;background:var(--border);border-radius:999px;overflow:hidden;margin-top:6px;">
        <div style="width:${pct}%;height:100%;background:${color};"></div>
    </div>`;
}

function renderGpuEnvironment(gpuSnapshot) {
    if (!gpuSnapshot || !gpuSnapshot.available) {
        return '<p class="empty-msg">No NVIDIA GPU telemetry available.</p>';
    }
    const gpus = gpuSnapshot.gpus || [];
    const processes = gpuSnapshot.processes || [];
    const gpuCards = gpus.map(g => {
        const memPct = g.memory_used_pct == null ? 0 : g.memory_used_pct;
        const util = g.utilization_pct == null ? 0 : g.utilization_pct;
        return `<div class="runtime-metric gpu-runtime-metric">
            <div class="runtime-metric-label">GPU ${esc(g.index)} · ${esc(g.name || 'GPU')}</div>
            <div class="runtime-metric-value">${runtimeValue(g.memory_used_mb, ' MB')} / ${runtimeValue(g.memory_total_mb, ' MB')}</div>
            ${percentBar(memPct, memPct > 80 ? '#c4453a' : '#3d8b5e')}
            <div class="runtime-metric-hint">mem ${runtimeValue(memPct, '%')} · util ${runtimeValue(util, '%')} · ${runtimeValue(g.temperature_c, ' C')} · ${runtimeValue(g.power_w, ' W')}</div>
        </div>`;
    }).join('') || '<p class="empty-msg">No GPU devices reported.</p>';
    const processRows = processes.length
        ? `<div style="margin-top:10px;display:flex;flex-direction:column;gap:4px;">${processes.slice(0, 8).map(proc => `
            <div class="runtime-path">PID ${esc(proc.pid)} · ${esc(proc.process_name || 'process')} · ${runtimeValue(proc.used_memory_mb, ' MB')}</div>
        `).join('')}</div>`
        : '<div class="runtime-path">No active GPU compute processes reported by nvidia-smi.</div>';
    return `<div class="runtime-metric-grid">${gpuCards}</div>${processRows}`;
}

function renderRuntimeConfig(config) {
    const panel = el('runtimeConfigPanel');
    if (!panel || !config) return;

    const llm = config.llm || {};
    const primary = llm.primary || {};
    const secondary = llm.secondary || {};
    const limits = llm.limits || {};
    const runtime = config.runtime || {};
    const experiment = config.experiment || {};
    const cpu = runtime.cpu || {};
    const gpuSnapshot = experiment.gpu_snapshot || {};
    const dbInfo = runtime.database || {};
    const workspaceDisk = runtime.workspace_disk || {};
    const experimentDisk = experiment.experiment_disk || {};

    panel.innerHTML = `
        <div class="runtime-config-panel">
            <div class="runtime-section-head">
                <div>
                    <div class="runtime-kicker">Model Source</div>
                    <h4>LLM route, token source, and runtime environment</h4>
                </div>
                <div class="runtime-note">API key fields are write-only; blank keeps the current key.</div>
            </div>
            <div class="runtime-grid">
                <div class="runtime-card runtime-card-model">
                    <div class="runtime-card-head">
                        <div class="runtime-avatar">GPT</div>
                        <div>
                            <div class="runtime-card-title">Primary Provider</div>
                            <div class="runtime-card-sub">${esc(primary.api_key_configured ? `key ${primary.api_key_hint || 'configured'}` : 'no API key configured')}</div>
                        </div>
                    </div>
                    <div class="runtime-provider-model">${esc(primary.model || 'No model set')}</div>
                    <div class="runtime-provider-url">${esc(primary.base_url || 'No base URL set')}</div>
                    <div class="runtime-inline">
                        <span>${esc(primary.protocol || 'protocol?')}</span>
                        <span>${runtimeValue(primary.rpm)} rpm</span>
                        <span>${runtimeValue(limits.max_output_tokens)} max out</span>
                    </div>
                </div>
                <div class="runtime-card">
                    <div class="runtime-card-title">Experiment Runtime</div>
                    <div class="runtime-metric-grid">
                        ${renderRuntimeMetric('Auto Research', runtimeBool(experiment.auto_research_enabled))}
                        ${renderRuntimeMetric('Pipeline', runtimeBool(experiment.auto_pipeline_enabled))}
                        ${renderRuntimeMetric('Concurrency', runtimeValue(experiment.pipeline_concurrency), 'paper processing workers')}
                        ${renderRuntimeMetric('Real Benchmark', runtimeBool(experiment.require_real_benchmark), experiment.benchmark_dataset || '')}
                        ${renderRuntimeMetric('Experiment Model', runtimeValue(experiment.real_llm_model))}
                        ${renderRuntimeMetric('Synthetic Fallback', runtimeBool(experiment.allow_synthetic_fallback))}
                    </div>
                </div>
                <div class="runtime-card">
                    <div class="runtime-card-title">CPU / Memory</div>
                    <div class="runtime-metric-grid">
                        ${renderRuntimeMetric('CPU Cores', runtimeValue(cpu.count || runtime.cpu_count))}
                        ${renderRuntimeMetric('Load 1m', runtimeValue(cpu.load_1m), `${runtimeValue(cpu.load_pct_1m, '%')} of cores`)}
                        ${renderRuntimeMetric('Load 5m', runtimeValue(cpu.load_5m))}
                        ${renderRuntimeMetric('Process RAM', runtimeValue(runtime.process_rss_mb, ' MB'))}
                        ${renderRuntimeMetric('System RAM', runtimeValue(runtime.total_memory_mb, ' MB'))}
                    </div>
                    <div class="runtime-path">${esc(runtime.platform || '')}</div>
                </div>
                <div class="runtime-card runtime-card-wide">
                    <div class="runtime-card-title">GPU Environment</div>
                    <div class="runtime-inline" style="margin-bottom:8px;">
                        <span>${esc(experiment.gpu_mode || 'gpu mode?')}</span>
                        <span>${runtimeValue(experiment.gpu_worker_slots)} slots</span>
                        <span>${esc((experiment.gpu_visible_devices || []).join(',') || 'no visible devices')}</span>
                    </div>
                    ${renderGpuEnvironment(gpuSnapshot)}
                </div>
                <div class="runtime-card runtime-card-wide">
                    <div class="runtime-card-title">Storage / Database</div>
                    <div class="runtime-metric-grid">
                        ${renderRuntimeMetric('Workspace Free', runtimeValue(workspaceDisk.free_gb, ' GB'), `${runtimeValue(workspaceDisk.used_gb, ' GB')} used`)}
                        ${renderRuntimeMetric('Experiment Free', runtimeValue(experimentDisk.free_gb, ' GB'), `${runtimeValue(experimentDisk.used_gb, ' GB')} used`)}
                        ${renderRuntimeMetric('DB Backend', runtimeValue(dbInfo.backend))}
                        ${renderRuntimeMetric('PID', runtimeValue(runtime.pid))}
                    </div>
                    <div class="runtime-path">${esc(dbInfo.target || '')}</div>
                </div>
            </div>
            <div class="runtime-config-form">
                <div class="runtime-form-title">Editable Model Configuration</div>
                <div class="config-form-grid">
                    <label>Primary model
                        <input id="cfgPrimaryModel" class="config-input" value="${esc(primary.model || '')}">
                    </label>
                    <label>Primary base URL
                        <input id="cfgPrimaryBaseUrl" class="config-input" value="${esc(primary.base_url || '')}">
                    </label>
                    <label>Primary protocol
                        <select id="cfgPrimaryProtocol" class="config-input">
                            <option value="responses" ${primary.protocol === 'responses' ? 'selected' : ''}>responses</option>
                            <option value="chat_completions" ${primary.protocol === 'chat_completions' ? 'selected' : ''}>chat_completions</option>
                        </select>
                    </label>
                    <label>Primary API key
                        <input id="cfgPrimaryApiKey" class="config-input" type="password" placeholder="${esc(primary.api_key_configured ? 'Configured; leave blank to keep' : 'Paste API key')}">
                    </label>
                    <label class="config-checkbox-row">
                        <input id="cfgSecondaryEnabled" type="checkbox" ${secondary.enabled ? 'checked' : ''}>
                        <span>Enable secondary provider</span>
                    </label>
                    <label>Secondary model
                        <input id="cfgSecondaryModel" class="config-input" value="${esc(secondary.model || '')}">
                    </label>
                    <label>Secondary base URL
                        <input id="cfgSecondaryBaseUrl" class="config-input" value="${esc(secondary.base_url || '')}">
                    </label>
                    <label>Secondary API key
                        <input id="cfgSecondaryApiKey" class="config-input" type="password" placeholder="${esc(secondary.api_key_configured ? 'Configured; leave blank to keep' : 'Paste API key')}">
                    </label>
                </div>
                <div class="config-save-row">
                    <span id="providerConfigStatus">Saved changes write to .env and require restart.</span>
                    <button class="btn-preview" id="saveProviderConfig">Save provider config</button>
                </div>
            </div>
        </div>`;

    const btn = el('saveProviderConfig');
    if (btn) btn.addEventListener('click', saveRuntimeConfig);
}

async function saveRuntimeConfig() {
    const status = el('providerConfigStatus');
    const payload = {
        DEEPGRAPH_LLM_MODEL: el('cfgPrimaryModel')?.value || '',
        DEEPGRAPH_LLM_BASE_URL: el('cfgPrimaryBaseUrl')?.value || '',
        DEEPGRAPH_LLM_PROTOCOL: el('cfgPrimaryProtocol')?.value || 'responses',
        DEEPGRAPH_LLM_API_KEY: el('cfgPrimaryApiKey')?.value || '',
        DEEPGRAPH_LLM_SECONDARY_ENABLED: el('cfgSecondaryEnabled')?.checked ? 'true' : 'false',
        DEEPGRAPH_LLM_SECONDARY_MODEL: el('cfgSecondaryModel')?.value || '',
        DEEPGRAPH_LLM_SECONDARY_BASE_URL: el('cfgSecondaryBaseUrl')?.value || '',
        DEEPGRAPH_LLM_SECONDARY_API_KEY: el('cfgSecondaryApiKey')?.value || ''
    };
    try {
        if (status) status.textContent = 'Saving provider config...';
        const res = await api('/api/runtime-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (status) {
            status.textContent = res.restart_required
                ? `Saved ${res.updated.length} keys. Restart DeepGraph to apply.`
                : 'No changes were written.';
        }
    } catch (e) {
        if (status) status.textContent = `Save failed: ${e.message}`;
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
            <div class="provider-card-head">
                <div class="provider-avatar">GPT</div>
                <div>
                    <div class="provider-name">${esc(p.name || p.provider || 'Unknown')}</div>
                    <div class="provider-url">${esc(p.base_url || p.url || '')}</div>
                </div>
            </div>
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
    return undefined;
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

// ── Public API (for onclick handlers in HTML strings) ────────────────

window._dg = {
    navigateTo,
    exploreNode(nodeId) {
        switchTab('explore');
        navigateTo(nodeId);
    },
    selectPaper,
    updateMatrixMetric,
    searchNav,

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
                    <h3>Idea #${insight.id}: ${esc(insight.title || '')}</h3>
                    <span class="proposal-stats">主实验: ${esc(canonical ? `Run #${canonical.id} / ${canonical.status}` : 'not started')}</span>
                    <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">×</button>
                </div>
                <div class="proposal-body">
                <h4>Idea Progress</h4>
                <p>Auto Research: ${esc(auto.status || 'not_started')} ${auto.stage ? `/ ${esc(auto.stage)}` : ''}</p>
                <p>Submission: ${esc(insight.submission_status || 'not_started')} | Run count: ${runs.length}</p>
                <div class="chip-row" style="margin:8px 0 14px;">${renderTrackChips(data.planned_tracks)}</div>
                ${renderManuscriptBlockers(plan.manuscript_blockers || {}, 8)}
                ${auto.last_note ? `<p><b>Latest:</b> ${esc(auto.last_note)}</p>` : ''}
                ${auto.last_error ? `<p style="color:#c4453a;"><b>Error:</b> ${esc(auto.last_error)}</p>` : ''}
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:16px 0;">
                    <div style="background:var(--bg-elevated);padding:12px;border-radius:10px;">
                        <h4 style="margin-top:0;">实验区</h4>
                        <p><b>Canonical Run:</b> ${esc(data.canonical_run_id || canonical?.id || '-')}</p>
                    </div>
                    <div style="background:var(--bg-elevated);padding:12px;border-radius:10px;">
                        <h4 style="margin-top:0;">实验方案区</h4>
                        ${jsonPreview(plan.latest_status, '暂无 latest_status.json')}
                    </div>
                    <div style="background:var(--bg-elevated);padding:12px;border-radius:10px;">
                        <h4 style="margin-top:0;">论文区</h4>
                        <div class="insight-actions" style="margin:8px 0;">
                            ${paperUrls.index ? `<button class="btn-preview" onclick="window.open('${esc(paperUrls.index)}','_blank')">打开论文页</button>` : ''}
                            ${paperUrls.pdf ? `<button class="btn-preview" onclick="window.open('${esc(paperUrls.pdf)}','_blank')">打开 PDF</button>` : ''}
                            ${paperUrls.tex ? `<button class="btn-preview" onclick="window.open('${esc(paperUrls.tex)}','_blank')">打开 TeX</button>` : ''}
                        </div>
                        ${renderPaperAssetLinks(insight.id, paperAssets)}
                    </div>
                </div>`;

            html += `<h4>方案快照</h4>
                ${jsonPreview(plan.experiment_spec, '暂无 experiment_spec.json')}
                ${jsonPreview(plan.manuscript_blockers, 'No manuscript blockers')}
                ${jsonPreview(plan.manuscript_input_state, '暂无 manuscript_input_state.json')}`;

            if (runs.length) {
                html += '<h4>Experiment History</h4>';
                for (const run of runs) {
                    const color = experimentStatusColor(run.status);
                    const artifactSummary = Object.entries(run.artifact_counts || {})
                        .map(([k, v]) => `${k}:${v}`).join(' · ');
                    const verdict = run.hypothesis_verdict
                        ? `<span style="color:${verdictColor(run.hypothesis_verdict)};font-weight:700;">${esc(run.hypothesis_verdict.toUpperCase())}</span>`
                        : '';
                    const badges = [];
                    if (canonical && canonical.id === run.id) badges.push('主实验');
                    if (run.has_plot_artifacts) badges.push('可视化');
                    if (run.has_bundle) badges.push('论文包');
                    html += `<div style="padding:10px;margin:8px 0;border-left:3px solid ${color};background:var(--bg-elevated);border-radius:8px;">
                        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                            <strong style="color:${color};">Run #${run.id}</strong>
                            <span>${esc(run.status || 'unknown')}</span>
                            ${verdict}
                            ${badges.map(label => `<span class="chip">${esc(label)}</span>`).join('')}
                        </div>
                        <div style="margin-top:6px;font-size:0.78rem;color:var(--text-secondary);display:flex;gap:12px;flex-wrap:wrap;">
                            <span>Iterations: ${run.iterations_total || 0} (${run.iterations_kept || 0} kept)</span>
                            <span>Claims: ${run.claim_count || 0}</span>
                            ${run.effect_pct != null ? `<span>Effect: ${run.effect_pct.toFixed(2)}%</span>` : ''}
                            ${artifactSummary ? `<span>Artifacts: ${esc(artifactSummary)}</span>` : ''}
                        </div>
                        ${run.error_message ? `<div style="margin-top:6px;color:#c4453a;font-size:0.76rem;">${esc(trunc(run.error_message, 220))}</div>` : ''}
                        <div class="insight-actions" style="margin-top:8px;">
                            <button class="btn-preview" onclick="window._dg.viewExperiment(${run.id})">View Run Details</button>
                        </div>
                    </div>`;
                }
            } else {
                html += '<p>No runs yet for this idea.</p>';
            }

            html += '</div></div>';

            const modal = document.createElement('div');
            modal.className = 'proposal-modal';
            modal.innerHTML = `<div class="proposal-overlay" onclick="this.parentElement.remove()"></div>${html}`;
            document.body.appendChild(modal);
            typesetMath(modal);
        } catch (e) {
            alert('Failed to load idea history: ' + e.message);
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
            typesetMath(modal);
        } catch (e) {
            alert('Failed to load: ' + e.message);
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
                </div>`;
            document.body.appendChild(modal);
            typesetMath(modal);
        } catch (e) {
            alert('Failed to load proposal: ' + e.message);
        }
    },
};

// ── Init ─────────────────────────────────────────────────────────────

// ── Direction submission (operator-authorized) ───────────────────────
// Submitting a direction creates a proposal record via the token-gated
// meta-harness API. It does not start any run: compute is only authorized
// later through explicit ResourceGrants.

function openDirectionModal() {
    const existing = document.querySelector('.direction-modal');
    if (existing) { existing.remove(); }
    const modal = document.createElement('div');
    modal.className = 'proposal-modal direction-modal';
    const field = 'width:100%;background:var(--bg-elevated);color:inherit;border:1px solid var(--border);border-radius:6px;padding:7px 9px;font-size:0.82rem;';
    modal.innerHTML = `<div class="proposal-overlay" onclick="this.parentElement.remove()"></div>
        <div class="proposal-content" style="max-height:85vh;max-width:640px;">
            <div class="proposal-header">
                <h3>Propose a research direction</h3>
                <button class="btn-close" onclick="this.closest('.proposal-modal').remove()">×</button>
            </div>
            <div class="proposal-body" style="display:flex;flex-direction:column;gap:10px;">
                <label style="font-size:0.8rem;">Question or direction *
                    <textarea id="dirText" style="${field}min-height:80px;margin-top:4px;" placeholder="What should be investigated, and against what would success be measured?"></textarea>
                </label>
                <label style="font-size:0.8rem;">Contact / submitter *
                    <input id="dirContact" style="${field}margin-top:4px;" placeholder="name or handle">
                </label>
                <label style="font-size:0.8rem;">Keywords (comma-separated, optional)
                    <input id="dirKeywords" style="${field}margin-top:4px;" placeholder="e.g. latent-communication, probing">
                </label>
                <div style="display:flex;gap:10px;flex-wrap:wrap;">
                    <label style="font-size:0.8rem;flex:1;">Goal
                        <select id="dirGoal" style="${field}margin-top:4px;">
                            <option value="experiment_plan">experiment_plan</option>
                            <option value="idea_only">idea_only</option>
                            <option value="verified_evidence">verified_evidence</option>
                        </select>
                    </label>
                    <label style="font-size:0.8rem;flex:1;">Token budget (hard cap, optional)
                        <input id="dirBudget" type="number" min="1" style="${field}margin-top:4px;" placeholder="server default">
                    </label>
                </div>
                <label style="font-size:0.8rem;">Operator token *
                    <input id="dirToken" type="password" style="${field}margin-top:4px;" autocomplete="off" placeholder="X-DeepGraph-Operator-Token">
                    <span style="font-size:0.7rem;color:var(--text-dim);">Submission requires operator authorization. The token is sent once with this request and is not stored.</span>
                </label>
                <div style="font-size:0.72rem;color:var(--text-dim);line-height:1.5;">
                    Submitting registers a scoped agenda proposal. No experiment starts from this form:
                    compute is authorized separately through resource grants, and results only become
                    claims after the evidence ladder and review.
                </div>
                <div id="dirResult" style="font-size:0.78rem;"></div>
                <div style="display:flex;gap:8px;justify-content:flex-end;">
                    <button class="btn-preview" onclick="this.closest('.proposal-modal').remove()">Cancel</button>
                    <button class="btn-preview" id="dirSubmitBtn" style="border-color:#3d8b5e;color:#3d8b5e;">Submit direction</button>
                </div>
            </div>
        </div>`;
    document.body.appendChild(modal);
    el('dirSubmitBtn').addEventListener('click', submitDirection);
}

async function submitDirection() {
    const resultBox = el('dirResult');
    const direction = (el('dirText').value || '').trim();
    const contact = (el('dirContact').value || '').trim();
    const token = (el('dirToken').value || '').trim();
    const keywords = (el('dirKeywords').value || '').split(',').map(s => s.trim()).filter(Boolean);
    const goal = el('dirGoal').value;
    const budgetRaw = el('dirBudget').value;
    if (!direction || !contact || !token) {
        resultBox.innerHTML = '<span style="color:#c4453a;">Direction, contact, and operator token are required.</span>';
        return;
    }
    const agenda = { direction, contact, goal };
    if (keywords.length) agenda.keywords = keywords;
    if (budgetRaw) agenda.token_budget = parseInt(budgetRaw, 10);
    const btn = el('dirSubmitBtn');
    btn.disabled = true;
    resultBox.textContent = 'Submitting...';
    try {
        const r = await fetch('/api/meta-harness/v1/agendas', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-DeepGraph-Operator-Token': token,
            },
            body: JSON.stringify({ confirmed: true, agenda }),
        });
        const payload = await r.json().catch(() => ({}));
        if (r.ok && payload.agenda_id) {
            resultBox.innerHTML = `<span style="color:#3d8b5e;">Registered as agenda #${payload.agenda_id} (token budget ${fmt(payload.token_budget || 0)}). It now awaits the normal selection and grant process.</span>`;
            initAgendaScope();
        } else {
            resultBox.innerHTML = `<span style="color:#c4453a;">Rejected: ${esc(payload.error || `HTTP ${r.status}`)}</span>`;
        }
    } catch (e) {
        resultBox.innerHTML = `<span style="color:#c4453a;">Request failed: ${esc(e.message)}</span>`;
    } finally {
        btn.disabled = false;
    }
}

function init() {
    // Nav items
    $$('.nav-item, .advanced-nav-item').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Sidebar toggle
    el('sidebarToggle').addEventListener('click', toggleSidebar);

    // Discovery filters + generate button
    const dtf = el('discoveryTierFilter');
    if (dtf) dtf.addEventListener('change', loadDiscoveriesTab);

    // Experiment filters
    const esf = el('experimentStatusFilter');
    if (esf) esf.addEventListener('change', loadExperimentsTab);

    // Evidence node select
    el('evidenceNodeSelect').addEventListener('change', (e) => {
        loadEvidenceForNode(e.target.value);
    });

    // Manuscripts filters
    el('papersSearch').addEventListener('input', () => {
        clearTimeout(el('papersSearch')._timer);
        el('papersSearch')._timer = setTimeout(renderPapers, 200);
    });
    el('papersStatus').addEventListener('change', renderPapers);

    // Opportunities filter
    // Search
    initSearch();

    // Propose-direction form (operator token required at submit time)
    const propose = el('btnProposeDirection');
    if (propose) propose.addEventListener('click', openDirectionModal);

    // Re-render dynamic content (badges, timeline) when the language changes;
    // static chrome is re-applied by i18n.js itself.
    document.addEventListener('deepgraph:languagechange', () => {
        onTabActivated(activeTab);
    });

    // Initial data loads. Agenda scope resolves first so that scoped
    // endpoints get their agenda_id; the unscoped loads run regardless.
    initAgendaScope().finally(() => {
        refreshStats();
        loadRecentlyDiscovered();
        loadOverviewResearchMap();
        loadProcessingPapers();
        startSSE();
    });

    const openDiscoveries = el('btnOpenDiscoveries');
    if (openDiscoveries) {
        openDiscoveries.addEventListener('click', () => switchTab('discoveries'));
    }
    const openMap = el('btnJumpExplore');
    if (openMap) openMap.addEventListener('click', () => switchTab('explore'));

    // Stats refresh every 15s
    statsTimer = setInterval(refreshStats, 15000);

    // Processing panel refresh every 3s (also fetches from API)
    setInterval(loadProcessingPapers, 3000);


    setInterval(() => {
        if (activeTab === 'experiments') loadExperimentsTab();
        if (activeTab === 'discoveries') loadDiscoveriesTab();
    }, 10000);
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
