// DeepGraph Dashboard - Hierarchical Domain Tree + Method x Dataset Matrix

let eventSource = null;
let currentNodeId = null;
let currentSimulation = null;

// ── Helpers ───────────────────────────────────────────────────────

function formatNumber(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Stats ─────────────────────────────────────────────────────────

async function refreshStats() {
    try {
        const resp = await fetch('/api/stats');
        const s = await resp.json();
        document.getElementById('statPapers').textContent = s.papers_processed || 0;
        document.getElementById('statClaims').textContent = formatNumber(s.paper_insights_total || 0);
        document.getElementById('statResults').textContent = formatNumber(s.results_total || 0);
        document.getElementById('statContradictions').textContent = s.contradictions_total || 0;
        document.getElementById('statGaps').textContent = s.node_summaries_total || 0;
        document.getElementById('statTokens').textContent = formatNumber(s.tokens_consumed || 0);
    } catch (e) {
        console.error('Stats error:', e);
    }
}

// ── Processing Panel ──────────────────────────────────────────────

// Track active papers from events
let activePapers = {};  // paper_id -> {title, step, startTime}

function updateProcessingPanel() {
    const el = document.getElementById('processingList');
    const entries = Object.entries(activePapers);
    if (entries.length === 0) {
        el.innerHTML = '<p class="empty-msg">Idle</p>';
        return;
    }
    el.innerHTML = entries.map(([pid, info]) => {
        const elapsed = Math.round((Date.now() - info.startTime) / 1000);
        return `<div class="proc-item">
            <span class="proc-id">${escapeHtml(pid)}</span>
            <span class="proc-title">${escapeHtml((info.title || '').substring(0, 50))}</span>
            <span class="proc-step">${escapeHtml(info.step)} (${elapsed}s)</span>
        </div>`;
    }).join('');
}

function trackPaperEvent(event) {
    const pid = event.data.paper_id;
    if (!pid) return;
    if (event.type === 'step') {
        if (!activePapers[pid]) {
            activePapers[pid] = { title: event.data.title || pid, step: '', startTime: Date.now() };
        }
        activePapers[pid].step = event.data.step;
    } else if (event.type === 'paper_done' || event.type === 'error') {
        delete activePapers[pid];
    }
    updateProcessingPanel();
}

// ── Breadcrumb ────────────────────────────────────────────────────

function renderBreadcrumb(crumbs) {
    const el = document.getElementById('breadcrumb');
    el.innerHTML = crumbs.map((c, i) => {
        const isLast = i === crumbs.length - 1;
        if (isLast) {
            return `<span class="crumb active">${escapeHtml(c.name)}</span>`;
        }
        return `<span class="crumb" onclick="navigateTo('${c.id}')">${escapeHtml(c.name)}</span>`;
    }).join('<span class="crumb-sep">&rsaquo;</span>');
}

// ── Two-Level Graph Navigation ────────────────────────────────────

async function navigateTo(nodeId) {
    currentNodeId = nodeId;

    try {
        const resp = await fetch(`/api/taxonomy/${nodeId}`);
        if (!resp.ok) {
            console.error('Node fetch failed:', resp.status);
            return;
        }
        const data = await resp.json();

        // Update breadcrumb
        renderBreadcrumb(data.breadcrumb);

        // Update graph title
        document.getElementById('graphTitle').textContent =
            data.node.name + ' - Opportunity Map';

        // Render the two-level graph: parent in center, children around it
        renderTwoLevelGraph(data.node, data.children);

        // Show detail section with plain-language overview first
        const hasMatrix = data.matrix &&
            data.matrix.methods.length > 0 &&
            data.matrix.datasets.length > 0;
        const hasPapers = data.papers && data.papers.length > 0;
        const hasSummary = !!data.summary;

        const detailSection = document.getElementById('detailSection');
        if (hasSummary || hasMatrix || hasPapers || data.children.length > 0 || data.is_leaf) {
            detailSection.style.display = 'block';

            document.getElementById('summaryTitle').textContent =
                `What Is Happening In ${data.node.name}?`;
            renderSummary(data.node, data.summary, data.children, data.papers);

            // Render matrix
            if (hasMatrix) {
                document.getElementById('matrixTitle').textContent =
                    `Evidence Table: ${data.node.name}`;
                renderMatrix(data.matrix);
            } else {
                document.getElementById('matrixContainer').innerHTML =
                    '<p class="empty-msg">No structured benchmark evidence yet for this area.</p>';
            }

            // Render papers
            document.getElementById('papersTitle').textContent =
                `Representative Papers (${data.papers.length})`;
            renderPapers(data.papers);

            // Render gaps
            renderGaps(data.summary, data.gaps);
        } else {
            detailSection.style.display = 'none';
        }
    } catch (e) {
        console.error('Navigation error:', e);
    }
}


function renderSummary(node, summary, children, papers) {
    const el = document.getElementById('summaryContainer');

    const childHints = (children || []).slice(0, 6).map(c =>
        `<span class="chip">${escapeHtml(c.name)}${c.paper_count ? ` · ${c.paper_count} papers` : ''}</span>`
    ).join('');

    if (!summary) {
        el.innerHTML = `
            <div class="summary-layout">
                <div class="summary-hero">
                    <h3>${escapeHtml(node.name)}</h3>
                    <p>${escapeHtml(node.description || 'This area does not have a generated summary yet.')}</p>
                    ${childHints ? `<div class="chip-row">${childHints}</div>` : ''}
                </div>
                <div class="summary-empty">
                    Process more papers in this area to generate a plain-language overview and gap summary.
                </div>
            </div>
        `;
        return;
    }

    const workItems = (summary.what_people_are_building || []).map(item => `
        <div class="summary-item">
            <strong>${escapeHtml(item.label || 'Workstream')}</strong>
            <p>${escapeHtml(item.description || '')}</p>
            ${item.paper_count ? `<div class="meta">${escapeHtml(String(item.paper_count))} papers</div>` : ''}
        </div>
    `).join('') || '<p class="empty-msg">No recurring workstreams extracted yet.</p>';

    const gapItems = (summary.current_gaps || []).map(item => `
        <div class="summary-item">
            <strong>${escapeHtml(item.title || 'Open gap')}</strong>
            <p>${escapeHtml(item.description || '')}</p>
            ${item.why_now ? `<div class="meta">Why now: ${escapeHtml(item.why_now)}</div>` : ''}
        </div>
    `).join('') || '<p class="empty-msg">No opportunity themes extracted yet.</p>';

    const patternChips = (summary.common_patterns || []).map(item =>
        `<span class="chip">${escapeHtml(item)}</span>`
    ).join('');
    const methodChips = (summary.common_methods || []).map(item =>
        `<span class="chip">${escapeHtml(item)}</span>`
    ).join('');
    const datasetChips = (summary.common_datasets || []).map(item =>
        `<span class="chip">${escapeHtml(item)}</span>`
    ).join('');
    const questionChips = (summary.starter_questions || []).map(item =>
        `<span class="chip">${escapeHtml(item)}</span>`
    ).join('');

    el.innerHTML = `
        <div class="summary-layout">
            <div class="summary-hero">
                <h3>${escapeHtml(node.name)}</h3>
                <p>${escapeHtml(summary.overview || node.description || '')}</p>
                ${summary.why_it_matters ? `<p>${escapeHtml(summary.why_it_matters)}</p>` : ''}
                ${childHints ? `<div class="chip-row">${childHints}</div>` : ''}
            </div>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>What People Are Working On</h3>
                    <div class="summary-list">${workItems}</div>
                </div>
                <div class="summary-card">
                    <h3>Where The Gaps Are</h3>
                    <div class="summary-list">${gapItems}</div>
                </div>
            </div>
            ${(patternChips || methodChips || datasetChips || questionChips) ? `
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Recurring Themes</h3>
                        <div class="chip-row">${patternChips || '<span class="chip">No recurring themes yet</span>'}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Representative Signals</h3>
                        ${methodChips ? `<div class="chip-row">${methodChips}</div>` : ''}
                        ${datasetChips ? `<div class="chip-row" style="margin-top:${methodChips ? '10px' : '0'};">${datasetChips}</div>` : ''}
                    </div>
                </div>
            ` : ''}
            ${questionChips ? `
                <div class="summary-card">
                    <h3>Good Starting Questions</h3>
                    <div class="chip-row">${questionChips}</div>
                </div>
            ` : ''}
        </div>
    `;
}


function renderTwoLevelGraph(parentNode, children) {
    const svg = d3.select('#graphSvg');
    svg.selectAll('*').remove();

    const container = svg.node().parentElement;
    const width = container.clientWidth - 32;
    const height = 600;
    svg.attr('width', width).attr('height', height);

    if (!children || children.length === 0) {
        // Leaf node: show a message
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#888')
            .attr('font-size', '16px')
            .text('Leaf domain - see the matrix below');

        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2 + 30)
            .attr('text-anchor', 'middle')
            .attr('fill', '#555')
            .attr('font-size', '13px')
            .text(parentNode.description || '');
        return;
    }

    // Build nodes array: parent + children
    const nodes = [];
    const links = [];

    // Parent node (center)
    nodes.push({
        id: parentNode.id,
        name: parentNode.name,
        description: parentNode.description || '',
        paper_count: 0,
        method_count: 0,
        gap_count: 0,
        isParent: true,
        fx: width / 2,
        fy: height / 2,
    });

    // Children arranged around parent
    const angleStep = (2 * Math.PI) / children.length;
    const radius = Math.min(width, height) * 0.32;

    children.forEach((child, i) => {
        const angle = angleStep * i - Math.PI / 2;
        nodes.push({
            id: child.id,
            name: child.name,
            description: child.description || '',
            paper_count: child.paper_count || 0,
            method_count: child.method_count || 0,
            gap_count: child.gap_count || 0,
            isParent: false,
            x: width / 2 + Math.cos(angle) * radius,
            y: height / 2 + Math.sin(angle) * radius,
        });

        links.push({
            source: parentNode.id,
            target: child.id,
        });
    });

    // Stop previous simulation
    if (currentSimulation) currentSimulation.stop();

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(radius * 0.8).strength(0.3))
        .force('charge', d3.forceManyBody().strength(d => d.isParent ? -300 : -150))
        .force('center', d3.forceCenter(width / 2, height / 2).strength(0.05))
        .force('collision', d3.forceCollide().radius(50))
        .alphaDecay(0.05);
    currentSimulation = simulation;

    // Links
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', '#1e3a5f')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 2);

    // Node groups
    const node = svg.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .style('cursor', d => d.isParent ? 'default' : 'pointer')
        .on('click', (e, d) => {
            if (!d.isParent) {
                navigateTo(d.id);
            }
        })
        .call(d3.drag()
            .on('start', (e, d) => {
                if (!e.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
            })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on('end', (e, d) => {
                if (!e.active) simulation.alphaTarget(0);
                if (!d.isParent) { d.fx = null; d.fy = null; }
            }));

    // Parent: larger circle
    node.filter(d => d.isParent)
        .append('circle')
        .attr('r', 30)
        .attr('fill', '#0a2844')
        .attr('stroke', '#00d4ff')
        .attr('stroke-width', 3);

    node.filter(d => d.isParent)
        .append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', 4)
        .attr('fill', '#00d4ff')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text(d => d.name.length > 14 ? d.name.substring(0, 12) + '..' : d.name);

    // Children: sized by paper count
    const childNodes = node.filter(d => !d.isParent);

    childNodes.append('circle')
        .attr('r', d => Math.max(20, Math.min(40, 20 + (d.paper_count || 0) * 0.5)))
        .attr('fill', d => {
            if (d.gap_count > 0) return '#1a3320';
            if (d.paper_count > 0) return '#12223a';
            return '#1a1a2e';
        })
        .attr('stroke', d => {
            if (d.gap_count > 0) return '#44ff44';
            if (d.paper_count > 0) return '#3388cc';
            return '#333';
        })
        .attr('stroke-width', 2);

    // Child label
    childNodes.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', -2)
        .attr('fill', '#e0e0e0')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text(d => d.name.length > 18 ? d.name.substring(0, 16) + '..' : d.name);

    // Badges under name
    childNodes.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', 13)
        .attr('fill', '#888')
        .attr('font-size', '9px')
        .text(d => {
            const parts = [];
            if (d.paper_count > 0) parts.push(d.paper_count + 'P');
            if (d.method_count > 0) parts.push(d.method_count + 'M');
            if (d.gap_count > 0) parts.push(d.gap_count + 'G');
            return parts.join(' | ') || 'empty';
        });

    // Tooltip on hover
    const tooltip = d3.select('body').selectAll('.tooltip').data([0]).join('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', '#1a1a2e')
        .style('border', '1px solid #333')
        .style('border-radius', '6px')
        .style('padding', '10px')
        .style('font-size', '12px')
        .style('color', '#e0e0e0')
        .style('max-width', '350px')
        .style('pointer-events', 'none')
        .style('opacity', 0)
        .style('z-index', 1000);

    childNodes.on('mouseover', (e, d) => {
        tooltip.html(`
            <div style="color:#00d4ff; font-weight:bold; margin-bottom:4px;">
                ${escapeHtml(d.name)}
            </div>
            <div style="margin-bottom:6px; color:#aaa;">${escapeHtml(d.description)}</div>
            <div style="color:#888;">
                Papers: ${d.paper_count} | Methods: ${d.method_count} | Gaps: ${d.gap_count}
            </div>
            <div style="color:#555; margin-top:4px; font-style:italic;">Click to explore</div>
        `).style('opacity', 1)
          .style('left', (e.pageX + 15) + 'px')
          .style('top', (e.pageY - 10) + 'px');
    }).on('mousemove', (e) => {
        tooltip.style('left', (e.pageX + 15) + 'px')
               .style('top', (e.pageY - 10) + 'px');
    }).on('mouseout', () => {
        tooltip.style('opacity', 0);
    });

    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}


// ── Method x Dataset Matrix ───────────────────────────────────────

function renderMatrix(matrix) {
    const container = document.getElementById('matrixContainer');

    if (!matrix.methods.length || !matrix.datasets.length) {
        container.innerHTML = '<p class="empty-msg">No results data yet.</p>';
        return;
    }

    // Determine which metric to show (pick most common one, or first)
    const metricCounts = {};
    for (const key of Object.keys(matrix.cells)) {
        const metric = key.split('|||')[2];
        metricCounts[metric] = (metricCounts[metric] || 0) + 1;
    }
    const metrics = Object.keys(metricCounts).sort((a, b) => metricCounts[b] - metricCounts[a]);
    const defaultMetric = metrics[0] || '';

    // Metric selector
    let html = '<div class="matrix-controls">';
    html += '<label>Metric: </label>';
    html += '<select id="metricSelect" onchange="updateMatrixMetric()">';
    for (const m of metrics) {
        const sel = m === defaultMetric ? ' selected' : '';
        html += `<option value="${escapeHtml(m)}"${sel}>${escapeHtml(m || '(no metric)')}</option>`;
    }
    html += '</select>';
    html += `<span class="matrix-info">${matrix.methods.length} methods x ${matrix.datasets.length} datasets</span>`;
    html += '</div>';

    // Table
    html += '<div class="matrix-scroll"><table class="matrix-table">';

    // Header row
    html += '<thead><tr><th class="method-header">Method \\ Dataset</th>';
    for (const ds of matrix.datasets) {
        html += `<th class="dataset-header" title="${escapeHtml(ds)}">${escapeHtml(ds.length > 16 ? ds.substring(0, 14) + '..' : ds)}</th>`;
    }
    html += '</tr></thead>';

    // Data rows
    html += '<tbody>';
    for (const method of matrix.methods) {
        html += '<tr>';
        html += `<td class="method-cell" title="${escapeHtml(method)}">${escapeHtml(method.length > 24 ? method.substring(0, 22) + '..' : method)}</td>`;
        for (const ds of matrix.datasets) {
            const key = `${method}|||${ds}|||${defaultMetric}`;
            const cell = matrix.cells[key];
            if (cell) {
                const cls = cell.is_sota ? 'cell-sota' : 'cell-filled';
                const val = cell.value != null ? Number(cell.value).toFixed(1) : '-';
                html += `<td class="matrix-cell ${cls}" title="${escapeHtml(method)} on ${escapeHtml(ds)}: ${val} (${escapeHtml(cell.paper_id || '')})">${val}</td>`;
            } else {
                html += `<td class="matrix-cell cell-empty" title="No data: ${escapeHtml(method)} on ${escapeHtml(ds)}">-</td>`;
            }
        }
        html += '</tr>';
    }
    html += '</tbody></table></div>';

    container.innerHTML = html;

    // Store matrix data for metric switching
    container._matrixData = matrix;
}

function updateMatrixMetric() {
    const container = document.getElementById('matrixContainer');
    const matrix = container._matrixData;
    if (!matrix) return;

    const selectedMetric = document.getElementById('metricSelect').value;
    const rows = container.querySelectorAll('tbody tr');

    rows.forEach((row, mi) => {
        const method = matrix.methods[mi];
        const cells = row.querySelectorAll('td.matrix-cell');
        cells.forEach((td, di) => {
            const ds = matrix.datasets[di];
            const key = `${method}|||${ds}|||${selectedMetric}`;
            const cell = matrix.cells[key];
            if (cell) {
                const val = cell.value != null ? Number(cell.value).toFixed(1) : '-';
                td.textContent = val;
                td.className = 'matrix-cell ' + (cell.is_sota ? 'cell-sota' : 'cell-filled');
                td.title = `${method} on ${ds}: ${val} (${cell.paper_id || ''})`;
            } else {
                td.textContent = '-';
                td.className = 'matrix-cell cell-empty';
                td.title = `No data: ${method} on ${ds}`;
            }
        });
    });
}


// ── Papers List ───────────────────────────────────────────────────

function renderPapers(papers) {
    const el = document.getElementById('papersList');
    if (!papers || papers.length === 0) {
        el.innerHTML = '<p class="empty-msg">No papers classified here yet.</p>';
        return;
    }
    el.innerHTML = papers.map(p => `
        <div class="paper-item">
            <a href="https://arxiv.org/abs/${escapeHtml(p.id)}" target="_blank" class="paper-link">
                ${escapeHtml(p.id)}
            </a>
            <span class="paper-title">${escapeHtml((p.title || '').substring(0, 100))}</span>
            <span class="paper-date">${escapeHtml(p.published_date || '')}</span>
            ${(p.work_type || p.confidence) ? `<div class="paper-meta">${p.work_type ? escapeHtml(p.work_type.replaceAll('_', ' ')) : ''}${p.confidence ? `${p.work_type ? ' · ' : ''}${Number(p.confidence).toFixed(2)} confidence` : ''}</div>` : ''}
            ${p.plain_summary ? `<div class="paper-summary">${escapeHtml(p.plain_summary)}</div>` : ''}
        </div>
    `).join('');
}


// ── Gaps ──────────────────────────────────────────────────────────

function renderGaps(summary, gaps) {
    const el = document.getElementById('gapsList');
    const narrativeGaps = summary && summary.current_gaps ? summary.current_gaps : [];
    if (narrativeGaps.length > 0) {
        el.innerHTML = narrativeGaps.map(g => `
            <div class="gap-item">
                <strong>${escapeHtml(g.title || 'Open gap')}</strong>
                <div class="gap-desc">${escapeHtml(g.description || '')}</div>
                ${g.why_now ? `<div class="gap-why">Why now: ${escapeHtml(g.why_now)}</div>` : ''}
            </div>
        `).join('');
        return;
    }

    if (!gaps || gaps.length === 0) {
        el.innerHTML = '<p class="empty-msg">No opportunity themes found yet for this domain.</p>';
        return;
    }

    el.innerHTML = gaps.map(g => `
        <div class="gap-item">
            <span class="score">${g.value_score || '?'}/5</span>
            <strong>${escapeHtml(g.method_name)} on ${escapeHtml(g.dataset_name)}</strong>
            <div class="gap-desc">${escapeHtml(g.gap_description)}</div>
            ${g.research_proposal ? `<div class="proposal">${escapeHtml(g.research_proposal)}</div>` : ''}
        </div>
    `).join('');
}


// ── Contradictions ────────────────────────────────────────────────

async function refreshContradictions() {
    try {
        const resp = await fetch('/api/contradictions?limit=20');
        const contras = await resp.json();
        const el = document.getElementById('contradictionsList');
        el.innerHTML = contras.map(c => `
            <div class="contradiction-item">
                <strong>${escapeHtml(c.description || 'Contradiction')}</strong><br>
                <span class="label-a">A:</span> ${escapeHtml((c.claim_a_text || '').substring(0, 120))}<br>
                <span class="label-b">B:</span> ${escapeHtml((c.claim_b_text || '').substring(0, 120))}<br>
                ${c.hypothesis ? `<span class="hypothesis-label">Hypothesis:</span> ${escapeHtml(c.hypothesis)}` : ''}
            </div>
        `).join('') || '<p class="empty-msg">No contradictions yet.</p>';
    } catch (e) {
        console.error('Contradictions error:', e);
    }
}


// ── Live Feed ─────────────────────────────────────────────────────

function addEventToFeed(event) {
    const feed = document.getElementById('eventFeed');
    const div = document.createElement('div');
    div.className = `event ${event.type}`;

    const time = event.timestamp ? event.timestamp.substring(11, 19) : '';
    let text = '';

    switch (event.type) {
        case 'paper_done':
            const nodes = (event.data.taxonomy_nodes || []).join(', ');
            text = `Paper <b>${escapeHtml(event.data.paper_id)}</b>: ` +
                   `${event.data.claims} claims, ${event.data.results || 0} results`;
            if (nodes) text += ` [${escapeHtml(nodes)}]`;
            if (event.data.contradictions > 0) {
                text += `, <span class="text-red">${event.data.contradictions} contradictions!</span>`;
            }
            break;
        case 'contradiction':
            text = `<span class="text-red">CONTRADICTION:</span> ${escapeHtml(event.data.description)}`;
            break;
        case 'gap':
            text = `<span class="text-green">GAP:</span> ${escapeHtml(event.data.method || '')} on ${escapeHtml(event.data.dataset || '')} - ${escapeHtml(event.data.description)}`;
            break;
        case 'summary':
            text = `<span class="text-cyan">SUMMARY:</span> ${escapeHtml(event.data.node_id)} updated`;
            break;
        case 'error':
            text = `<span class="text-orange">ERROR:</span> ${escapeHtml(event.data.paper_id)}: ${escapeHtml(event.data.error)}`;
            break;
        case 'step':
            text = `<span class="text-dim">${escapeHtml(event.data.paper_id || event.data.node_id || '')} &rarr; ${escapeHtml(event.data.step)}</span>`;
            break;
        case 'pipeline_start':
            text = `<span class="text-cyan">Analysis started (${event.data.max_papers} papers)</span>`;
            break;
        case 'pipeline_done':
            text = `<span class="text-cyan">Analysis complete. ${event.data.papers_processed} papers processed.</span>`;
            break;
        default:
            text = `${event.type}: ${JSON.stringify(event.data).substring(0, 100)}`;
    }

    div.innerHTML = `<span class="time">[${time}]</span> ${text}`;
    feed.insertBefore(div, feed.firstChild);
    while (feed.children.length > 200) feed.removeChild(feed.lastChild);
}

function connectSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource('/api/events');
    eventSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        addEventToFeed(event);
        trackPaperEvent(event);
        if (['paper_done', 'contradiction', 'gap', 'summary', 'pipeline_done'].includes(event.type)) {
            refreshStats();
        }
        if (event.type === 'paper_done') {
            // Refresh graph when papers are classified
            navigateTo(currentNodeId);
        }
        if (event.type === 'summary') navigateTo(currentNodeId);
        if (event.type === 'contradiction') refreshContradictions();
        if (event.type === 'pipeline_done') {
            navigateTo(currentNodeId);
            activePapers = {};
            updateProcessingPanel();
            document.getElementById('liveBadge').textContent = 'DONE';
            document.getElementById('liveBadge').className = 'live-badge';
        }
    };
}


// ── Pipeline Control ──────────────────────────────────────────────

async function startPipeline(n = 20) {
    document.getElementById('liveBadge').textContent = 'LIVE';
    document.getElementById('liveBadge').className = 'live-badge running';
    await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_papers: n })
    });
}


// ── Refresh All ───────────────────────────────────────────────────

function refreshAll() {
    refreshStats();
    navigateTo(currentNodeId);
    refreshContradictions();
}


// ── Init ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    currentNodeId = document.body.dataset.rootNode || 'ml';
    refreshAll();
    connectSSE();
    setInterval(refreshStats, 10000);
});
