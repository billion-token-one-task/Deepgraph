// Manuscript Routing dashboard card controller (issue #15 / D4 UI).
// Calls the /api/manuscript/* endpoints exposed by web/manuscript_routes.py.

(function () {
  const $ = (id) => document.getElementById(id);

  function fmtJSON(obj) {
    try { return JSON.stringify(obj, null, 2); } catch (e) { return String(obj); }
  }

  async function jget(url) {
    const r = await fetch(url);
    return { status: r.status, body: await r.json().catch(() => null) };
  }
  async function jpost(url, payload) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    return { status: r.status, body: await r.json().catch(() => null) };
  }

  function setResult(label, body) {
    const el = $("mrResultView");
    if (!el) return;
    el.textContent = label + ":\n" + fmtJSON(body);
  }

  async function loadVenues() {
    const { status, body } = await jget("/api/manuscript/venues");
    if (status >= 300) { setResult("ERROR (venues)", body); return; }
    const venues = (body && body.venues) || [];
    const badge = $("venueCountBadge");
    if (badge) badge.textContent = `${venues.length} venues`;
    setResult("VENUES", venues);
  }

  function parseState() {
    const raw = ($("mrStateInput") && $("mrStateInput").value || "").trim();
    if (!raw) return {};
    try { return JSON.parse(raw); }
    catch (e) { alert("State JSON parse error: " + e.message); return null; }
  }

  async function previewRoute() {
    const state = parseState();
    if (state === null) return;
    const includeTiebreak = !!($("mrIncludeTiebreak") && $("mrIncludeTiebreak").checked);
    const payload = Object.assign({}, state, { include_tiebreak: includeTiebreak });
    const { status, body } = await jpost("/api/manuscript/route", payload);
    setResult(status >= 300 ? "ERROR (route)" : "ROUTE PREVIEW", body);
  }

  async function previewLint() {
    const templateId = ($("mrTemplateIdInput") && $("mrTemplateIdInput").value || "").trim();
    const source = ($("mrSourceInput") && $("mrSourceInput").value || "");
    const pageRaw = ($("mrPageCountInput") && $("mrPageCountInput").value || "").trim();
    const payload = { template_id: templateId, source: source };
    if (pageRaw) {
      const n = parseInt(pageRaw, 10);
      if (!Number.isNaN(n)) payload.page_count = n;
    }
    const { status, body } = await jpost("/api/manuscript/lint", payload);
    setResult(status >= 300 ? "ERROR (lint)" : "LINT PREVIEW", body);
  }

  function init() {
    const tabBtn = document.querySelector('[data-tab="agenda"]');
    if (tabBtn) {
      let bootstrapped = false;
      tabBtn.addEventListener("click", () => {
        if (!bootstrapped) { bootstrapped = true; loadVenues(); }
      });
    }
    const bindings = [
      ["mrLoadVenuesBtn", loadVenues],
      ["mrPreviewRouteBtn", previewRoute],
      ["mrPreviewLintBtn", previewLint],
    ];
    bindings.forEach(([id, fn]) => {
      const el = $(id);
      if (el) el.addEventListener("click", fn);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
