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
    "ladder.noEvidence":     ["这一级还没有留下记录。", "Nothing recorded at this rung yet."],
    "ladder.claim":          ["事先声明要验证的效果", "The effect it said in advance it would produce"],
    "ladder.thresholds":     ["事先划的达标线", "Thresholds set in advance"],
    "ladder.thExciting":     ["很好", "strong"],
    "ladder.thSolid":        ["算数", "counts"],
    "ladder.thDisappointing":["不理想", "weak"],
    "ladder.direction":      ["越高越好", "higher is better"],
    "ladder.directionLower": ["越低越好", "lower is better"],
    "ladder.confidence":     ["事先给自己的把握", "Confidence it gave itself in advance"],
    "ladder.reference":      ["预测所依据的文献", "Prior work the prediction was taken from"],
    "ladder.noReference":    ["未给出文献依据", "no prior work cited"],
    "ladder.falsification":  ["事先写明的证伪条件", "What it said in advance would falsify it"],
    "ladder.fPrimary":       ["主判据", "primary"],
    "ladder.fRobustness":    ["稳健性", "robustness"],
    "ladder.fIntegrity":     ["完整性", "integrity"],
    "ladder.program":        ["研究方案全文", "The full research program"],
    "ladder.problem":        ["要解决的问题", "The problem it set out to solve"],
    "ladder.measured":       ["实测", "measured"],
    "ladder.control":        ["对照", "control"],
    "ladder.files":          ["可下载的原始记录", "Downloadable records"],
    "ladder.filesNote":      ["每个文件旁边是它的 sha256, 下载后可自行校验",
                              "Each file is listed with its sha256 so it can be checked after download"],
    "ladder.download":       ["下载", "download"],
    "ladder.holdoutNote":    ["这一趟用的是全新样本, 与前面测过的不重叠",
                              "This pass used a fresh sample with no overlap with anything measured earlier"]
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

  function measurementFor(data, stage) {
    var found = (data.measurements || []).filter(function (m) { return m.stage === stage; })[0];
    if (!found || found.metric_value == null) return "";
    var base = (data.statistics || {}).baseline_value;
    var line = esc(found.metric_key || "") + " = <b>" + esc(found.metric_value) + "</b>";
    if (base != null) {
      line += " &nbsp;vs&nbsp; " + esc(tr("ladder.control")) + " " + esc(base);
    }
    return line;
  }

  function filesFor(data, stage) {
    var files = (data.artifacts || []).filter(function (a) { return a.stage === stage; });
    if (!files.length) return "";
    var rows = files.map(function (a) {
      return '<div class="dg-file"><a class="dg-file-link" href="' + esc(a.url)
        + '" download>' + esc(a.kind) + "</a>"
        + '<code class="dg-hash">' + esc(a.sha256) + "</code></div>";
    }).join("");
    return field(tr("ladder.files"), rows
      + '<p class="dg-note">' + esc(tr("ladder.filesNote")) + "</p>");
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
    var x = data.expectation || {};
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
      /* Everything here was written down before any measurement existed. It is
       * the answer to the question a verdict always provokes -- "worked
       * against what?" -- and burying it was the single biggest gap in the
       * first version of this page. */
      out.push(field(tr("ladder.claim"), x.claim ? esc(x.claim) : ""));
      if (data.problem) out.push(field(tr("ladder.problem"), esc(data.problem)));
      var th = x.thresholds || {};
      var thParts = [];
      if (th.solid != null) thParts.push(esc(tr("ladder.thSolid")) + " &ge; " + esc(th.solid));
      if (th.exciting != null) thParts.push(esc(tr("ladder.thExciting")) + " &ge; " + esc(th.exciting));
      if (th.disappointing != null) thParts.push(esc(tr("ladder.thDisappointing")) + " &lt; " + esc(th.disappointing));
      if (thParts.length) {
        out.push(field(tr("ladder.thresholds"),
          esc(x.metric_name || "") + " ("
          + esc(x.metric_direction === "lower" ? tr("ladder.directionLower") : tr("ladder.direction"))
          + ") &middot; " + thParts.join(" &middot; ")));
      }
      if (x.confidence != null) out.push(field(tr("ladder.confidence"), esc(x.confidence)));
      out.push(field(tr("ladder.reference"),
        x.effect_reference ? esc(x.effect_reference)
                           : '<span class="dg-muted">' + esc(tr("ladder.noReference")) + "</span>"));
      var f = x.falsification || {};
      var fParts = [];
      if (f.primary) fParts.push("<b>" + esc(tr("ladder.fPrimary")) + "</b> " + esc(f.primary));
      if (f.robustness) fParts.push("<b>" + esc(tr("ladder.fRobustness")) + "</b> " + esc(f.robustness));
      if (f.integrity) fParts.push("<b>" + esc(tr("ladder.fIntegrity")) + "</b> " + esc(f.integrity));
      if (fParts.length) out.push(field(tr("ladder.falsification"), fParts.join("<br>")));
      if (data.program_md) {
        out.push('<details class="dg-sub"><summary>' + esc(tr("ladder.program"))
          + '</summary><pre class="dg-program">' + esc(data.program_md) + "</pre></details>");
      }
      out.push(field(tr("ladder.grant"), grantsFor("proposal") || grantsFor("pilot")));
      out.push('<p class="dg-note">' + esc(tr("ladder.grantNote")) + "</p>");
    } else if (state === "sanity_passed") {
      out.push(field(tr("ladder.measured"), measurementFor(data, "pilot")));
      out.push(field(tr("ladder.grant"), grantsFor("pilot")));
      out.push(filesFor(data, "pilot"));
    } else if (state === "full_benchmark_complete") {
      out.push(field(tr("ladder.measured"), measurementFor(data, "full_benchmark")
        || (s.metric_value != null
            ? esc(s.metric_name || "") + " = <b>" + esc(s.metric_value) + "</b> &nbsp;vs&nbsp; "
              + esc(tr("ladder.control")) + " " + esc(s.baseline_value)
            : "")));
      out.push(field(tr("ladder.grant"), grantsFor("full_benchmark")));
      out.push(filesFor(data, "full_benchmark"));
    } else if (state === "evidence_audited") {
      /* The held-out number belongs here and nowhere else. Showing only the
       * full-benchmark figure, which is usually the larger of the two, is the
       * selective reporting this whole ladder exists to prevent. */
      out.push(field(tr("ladder.measured"), measurementFor(data, "evidence_audit")));
      out.push(field(tr("ladder.holdout"),
        a.holdout_ref ? esc(a.holdout_ref) + "<br>" + hash(a.holdout_hash)
                        + '<p class="dg-note">' + esc(tr("ladder.holdoutNote")) + "</p>" : ""));
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
      out.push(filesFor(data, "evidence_audit"));
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
