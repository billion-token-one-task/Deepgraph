/* The evidence ladder behind one verdict, as a drill-down.
 *
 * Why this is its own file: the same component is mounted by the temporary
 * review page and, once approved, by the Evidence tab. Writing it twice would
 * guarantee the reviewed version and the shipped version drift.
 *
 * What it renders is what the ledger holds. Two things are removed upstream by
 * web/judge_demo_routes.py and neither is evidence for any claim: the compute
 * vendor baked into some actor names, and the hardware class a grant asked
 * for. Everything a reader would need to check the verdict -- the frozen
 * holdout and its hash, the permutation test, the independent evaluator and
 * its hash, the artifact and claim-ledger hashes, the signed verdict -- is
 * here in full, because a drill-down nobody can check is decoration.
 *
 * Each rung expands to the evidence attached to THAT rung, rather than piling
 * every field at the bottom. A reader who wants to know what "holdout" means
 * opens the audit rung and finds the holdout; that is the whole interaction.
 */
(function () {
  "use strict";

  var LANG_KEY = "deepgraph.lang";

  function lang() {
    try {
      var v = localStorage.getItem(LANG_KEY);
      return v === "en" || v === "zh" ? v : "zh";
    } catch (e) { return "zh"; }
  }

  /* i18n.js owns the dashboard's strings and defines window.t. This component
   * also runs on the standalone review page where i18n.js is not loaded, so
   * every key carries its own pair and window.t is used only when it actually
   * knows the key. Falling through to window.t unconditionally would render
   * raw key names on the review page. */
  var COPY = {
    "ladder.heading":        ["证据阶梯", "Evidence ladder"],
    "ladder.loading":        ["正在从账本读取…", "Reading the ledger…"],
    "ladder.failed":        ["读取失败, 账本没有返回这条 run。", "Could not read this run from the ledger."],
    "ladder.utc":            ["时间为 UTC", "times are UTC"],
    "ladder.reached":        ["已通过", "reached"],
    "ladder.notReached":     ["未通过", "not reached"],
    "ladder.operatorEntered":["人工录入", "operator-entered"],
    "ladder.operatorFrozen": ["候选方法由人手从论文抽取并冻结, 这是一次受审计的复现, 不是系统自主发现",
                              "The candidate method was transcribed from a paper by a human and frozen. This is a reproduction under audit, not an autonomous discovery."],
    "ladder.preregistered":  ["预注册内容", "What was pre-registered"],
    "ladder.metric":         ["指标", "Metric"],
    "ladder.baseline":       ["基线", "baseline"],
    "ladder.grant":          ["预算授权", "Budget grant"],
    "ladder.grantNote":      ["每一级台阶单独申请预算, 上一笔结清才发下一笔",
                              "Each rung asks for its own budget; the previous grant must settle before the next is issued"],
    "ladder.holdout":        ["留出集", "Holdout"],
    "ladder.evaluator":      ["独立评审方", "Independent evaluator"],
    "ladder.permutation":    ["置换检验", "Permutation test"],
    "ladder.significant":    ["显著", "significant"],
    "ladder.notSignificant": ["不显著", "not significant"],
    "ladder.blockers":       ["阻断原因", "Blocked by"],
    "ladder.rawArtifacts":   ["原始产物", "Raw artifacts"],
    "ladder.claimLedger":    ["主张账本", "Claim ledger"],
    "ladder.benchmark":      ["基准契约", "Benchmark contract"],
    "ladder.verdictHash":    ["判决签名", "Verdict signature"],
    "ladder.record":         ["账本记录", "Ledger record"],
    "ladder.cost":           ["实际开销", "What it cost"],
    "ladder.wall":           ["秒挂钟", "s wall clock"],
    "ladder.noEvidence":     ["这一级还没有留下记录。", "Nothing recorded at this rung yet."]
  };

  function tr(key) {
    var pair = COPY[key];
    if (pair) return lang() === "zh" ? pair[0] : pair[1];
    return (typeof window.t === "function") ? window.t(key) : key;
  }

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function hash(value) {
    if (!value) return "";
    return '<code class="dg-hash" title="' + esc(value) + '">' + esc(value) + "</code>";
  }

  function field(label, valueHtml) {
    if (!valueHtml) return "";
    return '<div class="dg-field"><span class="dg-field-label">' + esc(label)
      + '</span><span class="dg-field-value">' + valueHtml + "</span></div>";
  }

  function num(value, digits) {
    if (value == null || value === "") return "";
    var n = Number(value);
    return isFinite(n) ? n.toFixed(digits) : String(value);
  }

  /* Which evidence belongs to which rung. The point of the drill-down is that
   * a rung's own record is behind that rung: opening "evidence audit" is how a
   * reader finds the holdout, rather than scrolling to a footer that lists
   * everything the run ever produced. */
  function rungEvidence(state, data) {
    var s = data.statistics || {};
    var a = data.audit || {};
    var d = data.decision || {};
    var c = data.cost || {};
    var grants = data.grants || [];
    var out = [];

    function grantsFor(stage) {
      var picked = grants.filter(function (g) { return g.stage === stage; });
      if (!picked.length) return "";
      return picked.map(function (g) {
        return "grant " + esc(g.id) + " &middot; " + esc(g.token_cap) + " token &middot; "
          + esc(g.max_gpu_hours) + " GPU-h &middot; " + esc(g.status);
      }).join("<br>");
    }

    if (state === "planned") {
      out.push(field(tr("ladder.metric"), s.metric_name ? esc(s.metric_name) : ""));
      out.push(field(tr("ladder.grant"), grantsFor("proposal") || grantsFor("pilot")));
      out.push('<p class="dg-note">' + esc(tr("ladder.grantNote")) + "</p>");
    } else if (state === "sanity_passed") {
      out.push(field(tr("ladder.grant"), grantsFor("pilot")));
    } else if (state === "full_benchmark_complete") {
      if (s.metric_value != null) {
        out.push(field(tr("ladder.metric"),
          esc(s.metric_name || "") + " = " + esc(s.metric_value)
          + " &nbsp;vs&nbsp; " + esc(tr("ladder.baseline")) + " " + esc(s.baseline_value)
          + (s.effect_pct == null ? ""
             : " (" + (Number(s.effect_pct) > 0 ? "+" : "") + num(s.effect_pct, 1) + "%)")));
      }
      out.push(field(tr("ladder.grant"), grantsFor("full_benchmark")));
    } else if (state === "evidence_audited") {
      out.push(field(tr("ladder.holdout"),
        a.holdout_ref ? esc(a.holdout_ref) + "<br>" + hash(a.holdout_hash) : ""));
      out.push(field(tr("ladder.evaluator"),
        a.evaluator_ref ? esc(a.evaluator_ref) + "<br>" + hash(a.evaluator_hash) : ""));
      if (s.p_value != null) {
        out.push(field(tr("ladder.permutation"),
          "p = " + num(s.p_value, 6) + " &nbsp;(alpha = " + esc(s.alpha) + ", "
          + esc(s.significant ? tr("ladder.significant") : tr("ladder.notSignificant")) + ")"));
      }
      out.push(field(tr("ladder.rawArtifacts"), hash(a.raw_artifacts_hash)));
      out.push(field(tr("ladder.claimLedger"), hash(a.claim_ledger_hash)));
      out.push(field(tr("ladder.benchmark"), hash(a.benchmark_contract_hash)));
      out.push(field(tr("ladder.grant"), grantsFor("evidence_audit")));
    } else if (state === "scientifically_decided") {
      if ((s.blockers || []).length) {
        out.push(field(tr("ladder.blockers"), esc(s.blockers.join(", "))));
      }
      out.push(field(tr("ladder.verdictHash"), hash(d.verdict_hash)));
      out.push(field(tr("ladder.record"),
        d.id ? "scientific_decision_records #" + esc(d.id)
               + (d.created_at ? " &middot; " + esc(d.created_at) : "") : ""));
    } else if (state === "manuscript_allowed") {
      var cost = [];
      if (c.tokens != null) cost.push(esc(c.tokens) + " token");
      if (c.gpu_hours != null) cost.push(num(c.gpu_hours, 3) + " GPU-h");
      if (c.wall_seconds != null) cost.push(num(c.wall_seconds, 0) + " " + tr("ladder.wall"));
      out.push(field(tr("ladder.cost"), cost.join(" &middot; ")));
    }

    var body = out.filter(Boolean).join("");
    return body || '<p class="dg-note">' + esc(tr("ladder.noEvidence")) + "</p>";
  }

  function renderLadder(data) {
    var zh = lang() === "zh";
    var rungs = (data.ladder || []).map(function (r) {
      var label = zh ? r.label_zh : r.label_en;
      var why = zh ? r.why_zh : r.why_en;
      var actor = zh ? r.actor_zh : r.actor_en;
      var meta = [];
      if (r.at) meta.push(esc(r.at));
      if (actor) meta.push(esc(actor));
      return '<details class="dg-rung' + (r.reached ? " is-reached" : " is-pending") + '">'
        + '<summary class="dg-rung-head">'
        + '<span class="dg-rung-mark" aria-label="'
        + esc(r.reached ? tr("ladder.reached") : tr("ladder.notReached")) + '">'
        + (r.reached ? "[x]" : "[ ]") + "</span>"
        + '<span class="dg-rung-name">' + esc(label) + "</span>"
        + '<span class="dg-rung-meta">' + meta.join(" &middot; ") + "</span>"
        + (r.operator_entered
            ? '<span class="dg-flag">' + esc(tr("ladder.operatorEntered")) + "</span>" : "")
        + "</summary>"
        + '<div class="dg-rung-body">'
        + '<p class="dg-rung-why">' + esc(why) + "</p>"
        + rungEvidence(r.state, data)
        + "</div></details>";
    }).join("");

    return '<div class="dg-ladder">'
      + '<div class="dg-ladder-head">' + esc(tr("ladder.heading"))
      + ' <span class="dg-note-inline">' + esc(tr("ladder.utc")) + "</span></div>"
      + (data.operator_frozen
          ? '<p class="dg-caveat">' + esc(tr("ladder.operatorFrozen")) + "</p>" : "")
      + rungs
      + "</div>";
  }

  var cache = {};

  /* Fetch once per run and keep it: a reader opening and closing a row should
   * not re-query, and the ledger row cannot change while they look at it. */
  function load(runId) {
    if (cache[runId]) return Promise.resolve(cache[runId]);
    return fetch("/api/v1/judge/ladder/" + encodeURIComponent(runId), {
      headers: { Accept: "application/json" }
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    }).then(function (data) {
      cache[runId] = data;
      return data;
    });
  }

  function mount(container, runId) {
    if (!container || runId == null) return Promise.resolve();
    if (container.dataset.dgLadderLoaded === String(runId)) return Promise.resolve();
    container.innerHTML = '<p class="dg-note">' + esc(tr("ladder.loading")) + "</p>";
    return load(runId).then(function (data) {
      container.innerHTML = renderLadder(data);
      container.dataset.dgLadderLoaded = String(runId);
    }).catch(function (error) {
      console.error("evidence ladder unavailable:", error);
      container.innerHTML = '<p class="dg-note">' + esc(tr("ladder.failed")) + "</p>";
      delete container.dataset.dgLadderLoaded;
    });
  }

  /* Re-render what is already on screen when the language changes, so the
   * toggle does not leave half the page in the other language. */
  function relabel(root) {
    var scope = root || document;
    var mounted = scope.querySelectorAll("[data-dg-ladder-loaded]");
    for (var i = 0; i < mounted.length; i++) {
      var node = mounted[i];
      var runId = node.dataset.dgLadderLoaded;
      if (cache[runId]) node.innerHTML = renderLadder(cache[runId]);
    }
  }

  document.addEventListener("deepgraph:languagechange", function () { relabel(); });

  window.dgEvidenceLadder = { mount: mount, relabel: relabel, verdictPhrase: function (data) {
    if (!data || !data.verdict_phrase) return "";
    return lang() === "zh" ? data.verdict_phrase.zh : data.verdict_phrase.en;
  } };
})();
