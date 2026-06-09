/* ═══════════════════════════════════════════════════════════════════
   DeepGraph — Graph Renderer (D3, the ONLY module that touches D3)

   Swap-in contract: to replace the rendering tech (e.g. canvas, WebGL, a
   different lib) rewrite THIS FILE only. The public signatures must not change:

     DGGraphRenderer.renderRadial(container, model, options) -> handle
     DGGraphRenderer.renderNetwork(container, model, options) -> handle

   `container` is a DOM element the renderer owns (it clears and fills it).
   `model`     is the adapter's { kind, nodes, links } — never a raw API payload.
   `options`   injects EVERYTHING environment-specific so this file stays pure of
               app globals (no #tooltip, no navigateTo/switchTab, no window._dg):
       height           number, px
       isPreview        bool — compact, fewer affordances, no zoom by default
       enableZoom       bool — defaults to !isPreview
       theme            { accent, green, gold, purple, text, dim, muted, bg,
                          border, gapLo, gapHi, entityPalette:{type:hex} }
       onNodeClick(node, event)
       tooltip          { show(html, ev), move(ev), hide() }   (optional)
       nodeTooltipHtml(node) -> html string                    (optional)
       legendItems      [{ kind|color, label }]                (optional)
       emptyText        string for the empty/leaf state        (optional)
       labelMaxChars    override label budget

   handle = { destroy(), svg }.
   ═══════════════════════════════════════════════════════════════════ */
(function (root, factory) {
  const d3 = (root && root.d3) || (typeof require !== 'undefined' ? require('d3') : undefined);
  const api = factory(d3);
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.DGGraphRenderer = api;
})(typeof window !== 'undefined' ? window : undefined, function (d3) {
  'use strict';

  const DEFAULT_THEME = {
    accent: '#c4704b',
    green: '#3d8b5e',
    gold: '#a8842a',
    purple: '#7c5cbf',
    blue: '#2e86ab',
    red: '#c4453a',
    text: '#2b2520',
    dim: '#9a9088',
    muted: '#c4bdb4',
    bg: '#ffffff',
    bgElevated: '#f0ede6',
    border: '#e5e0d5',
    gapLo: '#eef3ee',   // no/low gaps
    gapHi: '#3d8b5e',   // many gaps (salient)
    entityPalette: {
      method: '#c4704b',
      dataset: '#2e86ab',
      metric: '#a8842a',
      task: '#7c5cbf',
      model: '#c4453a',
      artifact: '#3d8b5e',
      concept: '#9a9088',
    },
  };

  function theme(opts) {
    const t = Object.assign({}, DEFAULT_THEME, opts.theme || {});
    t.entityPalette = Object.assign({}, DEFAULT_THEME.entityPalette, (opts.theme && opts.theme.entityPalette) || {});
    return t;
  }

  function truncLabel(s, max) {
    if (!s) return '';
    return s.length > max ? s.slice(0, max - 1) + '…' : s;
  }

  /* Mount a fresh, sized SVG inside the container and (optionally) a zoom layer.
     Returns { svg, layer, width, height, zoom }. */
  function mountSvg(container, opts) {
    const sel = d3.select(container);
    sel.selectAll('svg.dg-graph-svg').remove();
    const width = Math.max(240, (container.clientWidth || 640) - 4);
    const height = opts.height || 420;
    const svg = sel.append('svg')
      .attr('class', 'dg-graph-svg')
      .attr('width', '100%')
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const layer = svg.append('g').attr('class', 'dg-zoom-layer');
    let zoom = null;
    const enableZoom = opts.enableZoom != null ? opts.enableZoom : !opts.isPreview;
    if (enableZoom) {
      zoom = d3.zoom().scaleExtent([0.3, 6]).on('zoom', (ev) => {
        layer.attr('transform', ev.transform);
      });
      svg.call(zoom);
      svg.on('dblclick.zoom', null);
    }
    return { svg, layer, width, height, zoom };
  }

  /* Wire injected tooltip + click + neighbour-highlight onto a node selection.
     adjacency: Map(nodeId -> Set(neighbourId)). */
  function wireInteractions(nodeSel, linkSel, allNodeSel, adjacency, opts) {
    const tip = opts.tooltip;
    const htmlFor = opts.nodeTooltipHtml;
    nodeSel
      .style('cursor', 'pointer')
      .on('click', (ev, d) => { if (opts.onNodeClick) opts.onNodeClick(d, ev); })
      .on('mouseover', function (ev, d) {
        if (tip && htmlFor) { tip.show(htmlFor(d), ev); }
        const nbrs = adjacency.get(d.id) || new Set();
        allNodeSel.style('opacity', (o) => (o.id === d.id || nbrs.has(o.id) ? 1 : 0.18));
        if (linkSel) {
          linkSel.style('opacity', (l) => {
            const s = l.source.id != null ? l.source.id : l.source;
            const tg = l.target.id != null ? l.target.id : l.target;
            return s === d.id || tg === d.id ? 0.95 : 0.06;
          });
        }
      })
      .on('mousemove', (ev) => { if (tip) tip.move(ev); })
      .on('mouseout', function () {
        if (tip) tip.hide();
        allNodeSel.style('opacity', 1);
        if (linkSel) linkSel.style('opacity', null);
      });
  }

  function drawLegend(svg, items, t) {
    if (!items || !items.length) return;
    const g = svg.append('g').attr('class', 'dg-legend').attr('transform', 'translate(14,14)');
    let y = 0;
    items.forEach((it) => {
      const row = g.append('g').attr('transform', `translate(0,${y})`).attr('data-legend', it.kind || 'item');
      if (it.kind === 'papers') {
        row.append('circle').attr('cx', 7).attr('cy', 0).attr('r', 7).attr('fill', 'none').attr('stroke', t.dim).attr('stroke-width', 1.4);
        row.append('circle').attr('cx', 7).attr('cy', 0).attr('r', 3).attr('fill', t.dim);
      } else if (it.kind === 'gaps') {
        row.append('rect').attr('x', 0).attr('y', -6).attr('width', 14).attr('height', 12).attr('rx', 3)
          .attr('fill', t.gapHi).attr('opacity', 0.85);
      } else if (it.kind === 'methods') {
        row.append('circle').attr('cx', 7).attr('cy', 0).attr('r', 6).attr('fill', t.bg).attr('stroke', t.gold).attr('stroke-width', 2);
      } else {
        row.append('circle').attr('cx', 7).attr('cy', 0).attr('r', 6).attr('fill', it.color || t.accent);
      }
      row.append('text').attr('x', 20).attr('y', 4).attr('class', 'dg-legend-label')
        .attr('font-size', '10.5px').attr('fill', t.dim).text(it.label);
      y += 20;
    });
  }

  function emptyState(container, opts, msg) {
    const m = mountSvg(container, Object.assign({}, opts, { enableZoom: false }));
    m.layer.append('text')
      .attr('x', m.width / 2).attr('y', m.height / 2)
      .attr('text-anchor', 'middle').attr('fill', '#9a9088')
      .attr('font-size', '14px').attr('font-weight', '600')
      .text(msg || '');
    return { destroy() { d3.select(container).selectAll('svg.dg-graph-svg').remove(); }, svg: m.svg.node() };
  }

  // ── Radial (taxonomy) ──────────────────────────────────────────────
  function renderRadial(container, model, options) {
    const opts = options || {};
    const t = theme(opts);
    const nodes = model.nodes || [];
    const parent = nodes.find((n) => n.role === 'parent') || nodes[0] || {};
    let children = nodes.filter((n) => n.role === 'child');

    if (!children.length) {
      return emptyState(container, opts, opts.emptyText || truncLabel(parent.description || parent.name || '', 80));
    }

    // Defensive cap: a radial star with hundreds of labelled nodes is both
    // unreadable and a DOM-cost hazard (this is the freeze the perf gate guards).
    // Real taxonomy nodes have ~12 children, so this never fires in practice.
    const maxNodes = opts.maxNodes || 60;
    let overflow = 0;
    if (children.length > maxNodes) {
      overflow = children.length - maxNodes;
      children = children.slice().sort((a, b) => (b.paper_count - a.paper_count) || (b.gap_count - a.gap_count)).slice(0, maxNodes);
      if (typeof console !== 'undefined') console.info(`[DGGraphRenderer] radial capped to ${maxNodes} nodes (+${overflow} hidden)`);
    }

    const m = mountSvg(container, opts);
    const cx = m.width / 2;
    const cy = m.height / 2;
    const isPreview = !!opts.isPreview;
    const labelMax = opts.labelMaxChars || (isPreview ? 16 : 28);

    const maxPapers = Math.max(1, d3.max(children, (c) => c.paper_count) || 1);
    const maxGap = Math.max(1, d3.max(children, (c) => c.gap_count) || 1);
    const rNode = d3.scaleSqrt().domain([0, maxPapers]).range(isPreview ? [10, 26] : [14, 36]);
    const gapColor = (g) => (g > 0 ? d3.interpolateRgb(t.gapLo, t.gapHi)(Math.min(1, g / maxGap)) : t.bgElevated);

    const radius = Math.min(m.width, m.height) * (isPreview ? 0.30 : 0.34);
    const step = (2 * Math.PI) / children.length;
    parent.x = cx; parent.y = cy;
    children.forEach((c, i) => {
      const a = step * i - Math.PI / 2;
      c._angle = a;
      c.x = cx + Math.cos(a) * radius;
      c.y = cy + Math.sin(a) * radius;
    });

    const adjacency = new Map();
    adjacency.set(parent.id, new Set(children.map((c) => c.id)));
    children.forEach((c) => adjacency.set(c.id, new Set([parent.id])));

    // links (parent -> child); carry {source,target} so highlight logic works
    const radialLinks = children.map((c) => ({ source: parent.id, target: c.id, _c: c }));
    const linkSel = m.layer.append('g').attr('class', 'dg-links').selectAll('line')
      .data(radialLinks).join('line')
      .attr('class', 'dg-link')
      .attr('x1', cx).attr('y1', cy)
      .attr('x2', (d) => d._c.x).attr('y2', (d) => d._c.y)
      .attr('stroke', t.accent).attr('stroke-opacity', 0.22)
      .attr('stroke-width', (d) => 1 + Math.min(3, (d._c.paper_count || 0) / Math.max(1, maxPapers) * 3));

    const allG = m.layer.append('g').attr('class', 'dg-nodes');

    // parent
    const pg = allG.append('g').datum(parent).attr('class', 'dg-node dg-node-parent')
      .attr('transform', `translate(${cx},${cy})`);
    pg.append('circle').attr('r', isPreview ? 26 : 32).attr('fill', t.bg)
      .attr('stroke', t.accent).attr('stroke-width', 2.5);
    pg.append('text').attr('text-anchor', 'middle').attr('dy', 4)
      .attr('fill', t.accent).attr('font-size', isPreview ? '10px' : '11.5px').attr('font-weight', 700)
      .text(truncLabel(parent.name, isPreview ? 14 : 20));

    // children
    const cg = allG.selectAll('g.dg-node-child').data(children).join('g')
      .attr('class', 'dg-node dg-node-child')
      .attr('transform', (d) => `translate(${d.x},${d.y})`);

    cg.append('circle')
      .attr('r', (d) => rNode(d.paper_count))
      .attr('fill', (d) => gapColor(d.gap_count))
      .attr('stroke', (d) => (d.gap_count > 0 ? t.gapHi : t.border))
      .attr('stroke-width', (d) => (d.gap_count > 0 ? 2.2 : 1.2));

    // method badge (corner): small ringed circle, count inside (non-preview)
    const badged = cg.filter((d) => d.method_count > 0);
    badged.append('circle')
      .attr('class', 'dg-method-badge')
      .attr('cx', (d) => rNode(d.paper_count) * 0.72)
      .attr('cy', (d) => -rNode(d.paper_count) * 0.72)
      .attr('r', isPreview ? 5 : 7)
      .attr('fill', t.bg).attr('stroke', t.gold).attr('stroke-width', 2);
    if (!isPreview) {
      badged.append('text')
        .attr('x', (d) => rNode(d.paper_count) * 0.72)
        .attr('y', (d) => -rNode(d.paper_count) * 0.72 + 3)
        .attr('text-anchor', 'middle').attr('font-size', '8px').attr('font-weight', 700)
        .attr('fill', t.gold).text((d) => d.method_count);
    }

    // labels: placed OUTSIDE the node along the radial direction, anchored by
    // side so 10+ children stay readable instead of overlapping a centred trunc.
    cg.append('text')
      .attr('class', 'dg-node-label')
      .attr('text-anchor', (d) => (Math.cos(d._angle) < -0.2 ? 'end' : (Math.cos(d._angle) > 0.2 ? 'start' : 'middle')))
      .attr('dx', (d) => {
        const off = rNode(d.paper_count) + 5;
        return Math.cos(d._angle) < -0.2 ? -off : (Math.cos(d._angle) > 0.2 ? off : 0);
      })
      .attr('dy', (d) => {
        const off = rNode(d.paper_count) + (Math.sin(d._angle) > 0 ? 14 : -8);
        return Math.abs(Math.cos(d._angle)) <= 0.2 ? off : 4;
      })
      .attr('fill', t.text).attr('font-size', isPreview ? '9.5px' : '11px').attr('font-weight', 600)
      .text((d) => truncLabel(d.name, labelMax));

    // paper count sub-label
    cg.append('text')
      .attr('text-anchor', 'middle').attr('dy', 3)
      .attr('fill', (d) => (d.paper_count > 0 ? t.accent : t.muted))
      .attr('font-size', isPreview ? '8px' : '9.5px').attr('font-weight', 700)
      .attr('pointer-events', 'none')
      .text((d) => (d.paper_count > 0 ? d.paper_count : ''));

    const allNodeSel = allG.selectAll('g.dg-node');
    wireInteractions(cg, linkSel, allNodeSel, adjacency, opts);
    // parent hover highlights everything (no dim)
    pg.on('mouseover', () => allNodeSel.style('opacity', 1));

    drawLegend(m.svg, opts.legendItems, t);
    if (overflow > 0) {
      m.svg.append('text').attr('class', 'dg-overflow')
        .attr('x', m.width - 12).attr('y', m.height - 12).attr('text-anchor', 'end')
        .attr('fill', t.dim).attr('font-size', '11px')
        .text((opts.moreLabel || '+%d more').replace('%d', overflow));
    }

    return {
      svg: m.svg.node(),
      destroy() { d3.select(container).selectAll('svg.dg-graph-svg').remove(); },
    };
  }

  // ── Network (entity-relation) ──────────────────────────────────────
  function renderNetwork(container, model, options) {
    const opts = options || {};
    const t = theme(opts);
    let nodes = (model.nodes || []).map((n) => Object.assign({}, n)); // copy: sim mutates x/y
    let rawLinks = model.links || [];

    if (!nodes.length) {
      return emptyState(container, opts, opts.emptyText || '');
    }

    // Defensive cap (the adapter usually caps already; backend caps to ~12).
    const maxNodes = opts.maxNodes || 80;
    let overflow = 0;
    if (nodes.length > maxNodes) {
      overflow = nodes.length - maxNodes;
      nodes = nodes.slice().sort((a, b) => (b.degree - a.degree) || (b.paper_count - a.paper_count)).slice(0, maxNodes);
      const keep = new Set(nodes.map((n) => n.id));
      rawLinks = rawLinks.filter((l) => keep.has(l.source) && keep.has(l.target));
      if (typeof console !== 'undefined') console.info(`[DGGraphRenderer] network capped to ${maxNodes} nodes (+${overflow} hidden)`);
    }

    const m = mountSvg(container, opts);
    const cx = m.width / 2;
    const cy = m.height / 2;
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const links = [];
    const adjacency = new Map(nodes.map((n) => [n.id, new Set()]));
    for (let i = 0; i < rawLinks.length; i += 1) {
      const l = rawLinks[i];
      if (byId.has(l.source) && byId.has(l.target)) {
        links.push(Object.assign({}, l));
        adjacency.get(l.source).add(l.target);
        adjacency.get(l.target).add(l.source);
      }
    }

    const maxDeg = Math.max(1, d3.max(nodes, (n) => n.degree) || 1);
    const maxPapers = Math.max(1, d3.max(nodes, (n) => n.paper_count) || 1);
    const rNode = d3.scaleSqrt().domain([0, maxDeg + maxPapers]).range([6, 22]);
    const sizeOf = (n) => rNode((n.degree || 0) + (n.paper_count || 0));
    const colorOf = (n) => t.entityPalette[n.entity_type] || t.entityPalette.concept;

    // Deterministic initial layout on a ring (no Math.random -> reproducible).
    const ringR = Math.min(m.width, m.height) * 0.36;
    nodes.forEach((n, i) => {
      const a = (2 * Math.PI * i) / nodes.length;
      n.x = cx + Math.cos(a) * ringR;
      n.y = cy + Math.sin(a) * ringR;
    });

    // Pretty force layout only for realistic sizes; large N falls back to the
    // static ring so the renderer stays O(n) (the N=5000 perf gate). The backend
    // caps to ~12 nodes, so force is the normal path.
    const forceMax = opts.forceMaxNodes || 140;
    if (nodes.length <= forceMax && links.length) {
      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id((d) => d.id).distance(70).strength(0.6))
        .force('charge', d3.forceManyBody().strength(-160))
        .force('center', d3.forceCenter(cx, cy))
        .force('collide', d3.forceCollide().radius((n) => sizeOf(n) + 6))
        .stop();
      const ticks = Math.min(200, 80 + nodes.length);
      for (let i = 0; i < ticks; i += 1) sim.tick();
    }
    // clamp into viewport
    const pad = 30;
    nodes.forEach((n) => {
      n.x = Math.max(pad, Math.min(m.width - pad, n.x));
      n.y = Math.max(pad, Math.min(m.height - pad, n.y));
    });

    // d3.forceLink replaces link.source/target ids with node objects; resolve either form.
    const endp = (e) => (e && e.x != null ? e : byId.get(e));
    const maxLinkPapers = Math.max(1, d3.max(links, (l) => l.paper_count) || 1);
    const linkSel = m.layer.append('g').attr('class', 'dg-links').selectAll('line')
      .data(links).join('line')
      .attr('class', 'dg-link')
      .attr('x1', (l) => endp(l.source).x).attr('y1', (l) => endp(l.source).y)
      .attr('x2', (l) => endp(l.target).x).attr('y2', (l) => endp(l.target).y)
      .attr('stroke', t.border).attr('stroke-opacity', 0.5)
      .attr('stroke-width', (l) => 1 + (l.paper_count || 0) / maxLinkPapers * 2.5);

    const g = m.layer.append('g').attr('class', 'dg-nodes').selectAll('g')
      .data(nodes).join('g')
      .attr('class', 'dg-node dg-node-entity')
      .attr('transform', (n) => `translate(${n.x},${n.y})`);

    g.append('circle')
      .attr('r', sizeOf).attr('fill', colorOf).attr('fill-opacity', 0.9)
      .attr('stroke', t.bg).attr('stroke-width', 1.5);

    // labels for the most connected nodes (avoid clutter on large graphs)
    const labelCut = nodes.length <= 30 ? 0 : (d3.quantile(nodes.map((n) => n.degree).sort(d3.ascending), 0.6) || 0);
    g.filter((n) => n.degree >= labelCut)
      .append('text')
      .attr('class', 'dg-node-label')
      .attr('text-anchor', 'middle')
      .attr('dy', (n) => sizeOf(n) + 11)
      .attr('fill', t.text).attr('font-size', '10px').attr('font-weight', 600)
      .attr('pointer-events', 'none')
      .text((n) => truncLabel(n.name, 22));

    wireInteractions(g, linkSel, g, adjacency, opts);
    drawLegend(m.svg, opts.legendItems, t);
    if (overflow > 0) {
      m.svg.append('text').attr('class', 'dg-overflow')
        .attr('x', m.width - 12).attr('y', m.height - 12).attr('text-anchor', 'end')
        .attr('fill', t.dim).attr('font-size', '11px')
        .text((opts.moreLabel || '+%d more').replace('%d', overflow));
    }

    return {
      svg: m.svg.node(),
      destroy() { d3.select(container).selectAll('svg.dg-graph-svg').remove(); },
    };
  }

  return { renderRadial, renderNetwork };
});
