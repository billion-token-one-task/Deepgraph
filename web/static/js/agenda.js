// Research Agenda tab controller (issue #9 step 6 UI).
// Calls the /api/research_agenda/* endpoints exposed by web/agenda_routes.py.

(function () {
  const $ = (id) => document.getElementById(id);
  let currentAgenda = null;
  let latestSelection = null;
  let latestReviewId = null;

  function tr(key, vars) {
    return window.t ? window.t(key, vars) : key;
  }

  function fmtJSON(obj) {
    try { return JSON.stringify(obj, null, 2); } catch (e) { return String(obj); }
  }

  async function jget(url) {
    const r = await fetch(url);
    return { status: r.status, body: await r.json().catch(() => null) };
  }
  async function jpost(url, payload, opts) {
    opts = opts || {};
    const init = { method: "POST" };
    if (opts.raw) {
      init.body = payload;
      init.headers = { "Content-Type": opts.contentType || "text/plain" };
    } else {
      init.body = JSON.stringify(payload || {});
      init.headers = { "Content-Type": "application/json" };
    }
    const r = await fetch(url, init);
    return { status: r.status, body: await r.json().catch(() => null) };
  }

  function renderAgenda(agenda) {
    const badge = $("agendaActiveBadge");
    const info = $("agendaCurrentInfo");
    if (!agenda) {
      badge.textContent = tr("agenda.noAgenda");
      info.textContent = tr("agenda.noActive");
      return;
    }
    badge.textContent = agenda.is_active
      ? tr("agenda.active", { id: agenda.id, name: agenda.name })
      : `#${agenda.id} ${agenda.name}`;
    info.innerHTML = `
      <div><strong>${tr("agenda.focus")}</strong>: ${(agenda.focus || []).join(", ") || "—"}</div>
      <div><strong>${tr("agenda.preferKeywords")}</strong>: ${((agenda.prefer || {}).keywords || []).join(", ") || "—"}</div>
      <div><strong>${tr("agenda.preferTiers")}</strong>: ${((agenda.prefer || {}).tiers || []).join(", ") || "—"}</div>
      <div><strong>${tr("agenda.reject")}</strong>: ${fmtJSON(agenda.reject || {})}</div>
    `;
  }

  function renderSelection(sel) {
    const el = $("agendaSelectionView");
    if (!sel) {
      el.textContent = tr("agenda.noSelection");
      return;
    }
    el.innerHTML = `
      <div><strong>${tr("agenda.selection", { id: sel.id })}</strong> · ${tr("agenda.status")}=<code>${sel.status}</code></div>
      <div>insight_id: ${sel.selected_insight_id ?? "—"} · score: ${(sel.score ?? 0).toFixed ? sel.score.toFixed(3) : sel.score}</div>
      <div>experiment_run_id: ${sel.experiment_run_id ?? "—"} · manuscript_run_id: ${sel.manuscript_run_id ?? "—"} · bundle_id: ${sel.submission_bundle_id ?? "—"}</div>
      <details style="margin-top:6px;">
        <summary>${tr("agenda.candidatesJson")}</summary>
        <pre style="font-size:0.72rem;max-height:200px;overflow:auto;">${fmtJSON(sel.candidates_json || sel.candidates || [])}</pre>
      </details>
    `;
  }

  async function loadCurrent() {
    const { status, body } = await jget("/api/research_agenda/current");
    currentAgenda = (status === 200 && body) ? body.agenda : null;
    renderAgenda(currentAgenda);

    if (currentAgenda) {
      const r = await jget(`/api/research_agenda/selection/latest?agenda_id=${currentAgenda.id}`);
      latestSelection = (r.status === 200 && r.body) ? r.body.selection : null;
      renderSelection(latestSelection);
    } else {
      renderSelection(null);
    }
  }

  async function uploadAgenda() {
    const text = $("agendaYamlInput").value.trim();
    if (!text) { alert(tr("agenda.pasteYamlFirst")); return; }
    const { status, body } = await jpost("/api/research_agenda", text, {
      raw: true, contentType: "application/x-yaml",
    });
    if (status >= 300) { alert(tr("agenda.uploadFailed", { detail: fmtJSON(body) })); return; }
    await loadCurrent();
  }

  async function runSelect() {
    const mode = $("agendaDispatchMode").value;
    const { status, body } = await jpost("/api/research_agenda/select", { dispatch_mode: mode });
    if (status >= 300) { alert(tr("agenda.selectFailed", { detail: fmtJSON(body) })); return; }
    latestSelection = body.selection;
    renderSelection(latestSelection);
  }

  async function runReview() {
    if (!latestSelection) { alert(tr("agenda.noSelectionAlert")); return; }
    const { status, body } = await jpost(
      `/api/research_agenda/selection/${latestSelection.id}/review`, {}
    );
    if (status >= 300) { alert(tr("agenda.reviewFailed", { detail: fmtJSON(body) })); return; }
    latestReviewId = body.review && body.review.id;
    $("agendaLoopView").textContent = tr("agenda.reviewLabel") + ":\n" + fmtJSON(body.review);
  }

  async function runPlan() {
    if (!latestSelection) { alert(tr("agenda.noSelectionAlert")); return; }
    const payload = latestReviewId ? { review_id: latestReviewId } : {};
    const { status, body } = await jpost(
      `/api/research_agenda/selection/${latestSelection.id}/plan`, payload
    );
    if (status >= 300) { alert(tr("agenda.planFailed", { detail: fmtJSON(body) })); return; }
    $("agendaLoopView").textContent = tr("agenda.revisionPlan") + ":\n" + fmtJSON(body.plan);
  }

  async function inspectLoop() {
    if (!latestSelection) { alert(tr("agenda.noSelectionAlert")); return; }
    const { status, body } = await jget(`/api/research_agenda/loop/${latestSelection.id}`);
    if (status >= 300) { $("agendaLoopView").textContent = tr("agenda.error", { detail: fmtJSON(body) }); return; }
    $("agendaLoopView").textContent = fmtJSON(body.loop);
  }

  function init() {
    const tabBtn = document.querySelector('[data-tab="agenda"]');
    if (!tabBtn) return;
    let bootstrapped = false;
    tabBtn.addEventListener("click", () => {
      if (!bootstrapped) { bootstrapped = true; loadCurrent(); }
    });
    const bindings = [
      ["agendaLoadBtn", loadCurrent],
      ["agendaUploadBtn", uploadAgenda],
      ["agendaSelectBtn", runSelect],
      ["agendaReviewBtn", runReview],
      ["agendaPlanBtn", runPlan],
      ["agendaLoopBtn", inspectLoop],
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
