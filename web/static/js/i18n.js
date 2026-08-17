/* DeepGraph dashboard i18n. Keep en/zh key sets identical.
   Ported from the pre-upgrade frontend (prod-snapshot-20260621) and re-keyed
   for the current template. Static chrome uses data-i18n / data-i18n-placeholder
   attributes; dynamic strings go through window.t(). */
(function () {
  "use strict";

  const I18N = {
    en: {
      "lang.en": "EN",
      "lang.zh": "ZH",
      "app.live.idle": "IDLE",
      "app.live.live": "LIVE",
      "topbar.search.placeholder": "Search papers, methods, insights...",
      "topbar.papers": "Corpus Papers",
      "topbar.results": "Structured Results",
      "topbar.insights": "Extracted Insights",
      "topbar.tokens": "Analysis Tokens",

      "nav.research": "Research",
      "nav.map": "Research Map",
      "nav.evidence": "Evidence",
      "nav.manuscripts": "Manuscripts",
      "nav.ideas": "Ideas",
      "nav.process": "Process",
      "nav.more": "More",
      "nav.activity": "Activity",
      "nav.propose": "Propose direction",
      "footer.scoped": "Research work runs only through approved agendas and scoped grants.",

      "overview.outcomes": "Research outcomes",
      "overview.outcomesNote": "Confirmed findings, open questions, and promising directions appear here first.",
      "overview.browseIdeas": "Browse ideas",
      "overview.map": "Research map",
      "overview.mapNote": "Explore the active research landscape and its evidence-bearing areas.",
      "overview.openMap": "Open map",
      "overview.portfolio": "Research portfolio and process",
      "overview.portfolioNote": "Aggregate counts and in-progress work are available here when you need context.",
      "overview.noOutcomes": "No published research outcomes yet.",
      "stat.papersAnalyzed": "Source Papers Analyzed",
      "stat.paperIdeas": "Paper Ideas Generated",
      "stat.experimentRuns": "Experiment Runs",
      "stat.experimentBreakdown": "{completed} completed · {failed} failed",
      "stat.decidedFindings": "Decided Findings",
      "stat.decidedPending": "{n} candidates awaiting adjudication",
      "stat.analysisTokens": "Analysis Tokens",
      "stat.corpusPapers": "Corpus Papers",
      "stat.pendingPapers": "Awaiting Analysis",
      "stat.errorPapers": "Processing Errors",
      "stat.results": "Structured Results",
      "stat.taxonomyNodes": "Taxonomy Nodes",
      "stat.contradictions": "Contradictions",
      "stat.insights": "Extracted Insights",
      "stat.graphEntities": "Graph Entities",
      "stat.graphRelations": "Graph Relations",
      "stat.agendaTokens": "Agenda Tokens",
      "stat.submissionBundles": "Submission Bundles",

      "office.title": "Agent Office",
      "office.subtitle": "Live pixel workspace for DeepGraph departments and sub-agents.",
      "office.idle": "office idle",
      "office.mapping": "Mapping DeepGraph departments...",

      "explore.title": "Research Area Explorer",
      "explore.summary": "Node Summary",
      "explore.children": "Sub-areas",

      "evidence.title": "Method x Dataset Matrix",
      "evidence.select": "Select taxonomy node:",
      "evidence.option": "Search a research area...",
      "evidence.hint": "Select a leaf node to view the evidence matrix.",
      "evidence.gaps": "Matrix Gaps",

      "papers.title": "Manuscript Library",
      "papers.subtitle": "Notebook-style reading desk for DeepGraph-generated manuscripts.",
      "papers.filter": "Filter manuscripts...",
      "papers.allStatuses": "All statuses",
      "papers.bundleReady": "Bundle ready",
      "papers.drafting": "Drafting",
      "papers.failed": "Failed",
      "papers.notStarted": "Not started",
      "papers.listHeading": "Manuscripts",
      "papers.selectOne": "Select a manuscript",
      "papers.selectHint": "Choose a generated manuscript from the notebook list to open a submission-style reading page.",

      "ideas.title": "Generated Paper Ideas",
      "ideas.allTiers": "All Tiers",
      "ideas.tier1": "Tier 1: Paradigm",
      "ideas.tier2": "Tier 2: Paper Ideas",
      "ideas.empty": "No deep discoveries yet. Discovery runs in fixed automatic mode.",
      "ideas.emptyNoAgenda": "No research agenda is registered yet, so there is no scope to list ideas from.",
      "ideas.emptyFiltering": "No ready discoveries yet. Automatic discovery is still filtering candidates.",

      "process.timeline": "Process Timeline",
      "process.noScope": "no agenda scope",
      "process.timelineNote": "Chronological record for the active agenda: signals, candidate decisions, resource authorizations, jobs, evidence-ladder transitions, and verdicts. Failures and refused gate transitions are shown alongside successes.",
      "process.timelineEmpty": "No process events recorded yet.",
      "process.timelineEmptyDetail": "No process events recorded for this agenda yet. Events appear once signals, grants, or experiment runs exist.",
      "process.timelineNoAgenda": "No research agenda is registered yet, so there is no process to show.",
      "process.rationale": "Selection Rationale",
      "process.rationaleBadge": "why work was chosen",
      "process.rationaleEmpty": "No selection records yet.",
      "process.rationaleEmptyDetail": "No selection records for this agenda yet. Rationale appears once the selector has admitted or rejected candidates.",
      "process.services": "Automation Services",
      "process.readOnly": "Read-only automatic mode",
      "process.ideaExperiments": "Idea Experiments",
      "process.metaReport": "Meta-Learning Report",
      "process.filter.all": "All",
      "process.filter.pending": "Pending",
      "process.filter.scaffolding": "Scaffolding",
      "process.filter.reproducing": "Reproducing",
      "process.filter.testing": "Testing",
      "process.filter.completed": "Completed",
      "process.filter.failed": "Failed",

      "badge.run": "RUN",
      "badge.evidence": "EVIDENCE",
      "badge.decided": "DECIDED",
      "badge.notAssessed": "not assessed",
      "badge.notAssessed.tip": "No evidence-ladder progress recorded; completion of a job is not a finding",
      "badge.run.tip": "Operational status: whether the job ran; it makes no scientific claim",
      "badge.decided.tip": "Scientific verdict recorded by the audited decision gate",
      "badge.progress.tip": "Position on the evidence ladder; not yet a scientific decision",
      "sci.planned": "PLANNED",
      "sci.sanity_passed": "SANITY PASSED",
      "sci.full_benchmark_complete": "BENCHMARK DONE",
      "sci.evidence_audited": "AUDITED",
      "sci.scientifically_decided": "DECIDED",
      "sci.manuscript_allowed": "MANUSCRIPT ALLOWED",
      "verdict.supported": "supported",
      "verdict.refuted": "refuted",
      "verdict.inconclusive": "inconclusive",

      "tl.legacy": "LEGACY IMPORT",
      "tl.signal": "SIGNALS",
      "tl.candidate": "CANDIDATE",
      "tl.authorization": "AUTHORIZATION",
      "tl.run": "RUN",
      "tl.evidence": "EVIDENCE",
      "tl.decision": "DECISION",
      "tl.outcome": "OUTCOME",
    },
    zh: {
      "lang.en": "EN",
      "lang.zh": "中文",
      "app.live.idle": "空闲",
      "app.live.live": "运行中",
      "topbar.search.placeholder": "搜索论文、方法、洞见...",
      "topbar.papers": "收录论文",
      "topbar.results": "结构化结果",
      "topbar.insights": "文献抽取洞见",
      "topbar.tokens": "分析 Token",

      "nav.research": "研究",
      "nav.map": "研究地图",
      "nav.evidence": "证据",
      "nav.manuscripts": "论文稿",
      "nav.ideas": "想法",
      "nav.process": "过程",
      "nav.more": "更多",
      "nav.activity": "活动",
      "nav.propose": "提交研究方向",
      "footer.scoped": "所有研究工作仅通过已批准的议程与限定范围的资源授权运行。",

      "overview.outcomes": "研究成果",
      "overview.outcomesNote": "已确认的发现、开放问题与有希望的方向会优先显示在这里。",
      "overview.browseIdeas": "浏览想法",
      "overview.map": "研究地图",
      "overview.mapNote": "探索当前活跃的研究版图及其证据分布。",
      "overview.openMap": "打开地图",
      "overview.portfolio": "研究组合与过程",
      "overview.portfolioNote": "需要背景信息时，可在此查看汇总计数与进行中的工作。",
      "overview.noOutcomes": "尚无已发布的研究成果。",
      "stat.papersAnalyzed": "已分析源论文",
      "stat.paperIdeas": "已生成论文想法",
      "stat.experimentRuns": "实验运行",
      "stat.experimentBreakdown": "{completed} 已完成 · {failed} 失败",
      "stat.decidedFindings": "已判定发现",
      "stat.decidedPending": "{n} 项候选待裁决",
      "stat.analysisTokens": "分析 Token",
      "stat.corpusPapers": "收录论文",
      "stat.pendingPapers": "待继续分析",
      "stat.errorPapers": "处理错误",
      "stat.results": "结构化结果",
      "stat.taxonomyNodes": "分类节点",
      "stat.contradictions": "矛盾",
      "stat.insights": "文献抽取洞见",
      "stat.graphEntities": "图谱实体",
      "stat.graphRelations": "图谱关系",
      "stat.agendaTokens": "议程 Token",
      "stat.submissionBundles": "投稿包",

      "office.title": "智能体办公室",
      "office.subtitle": "DeepGraph 各部门与子智能体的实时像素工作区。",
      "office.idle": "办公室空闲",
      "office.mapping": "正在映射 DeepGraph 部门...",

      "explore.title": "研究领域浏览器",
      "explore.summary": "节点摘要",
      "explore.children": "子领域",

      "evidence.title": "方法 x 数据集矩阵",
      "evidence.select": "选择分类节点:",
      "evidence.option": "搜索研究领域...",
      "evidence.hint": "选择叶子节点以查看证据矩阵。",
      "evidence.gaps": "矩阵空白",

      "papers.title": "论文稿库",
      "papers.subtitle": "以笔记本方式阅读 DeepGraph 生成的论文稿。",
      "papers.filter": "筛选论文稿...",
      "papers.allStatuses": "全部状态",
      "papers.bundleReady": "投稿包就绪",
      "papers.drafting": "撰写中",
      "papers.failed": "失败",
      "papers.notStarted": "未开始",
      "papers.listHeading": "论文稿",
      "papers.selectOne": "选择一篇论文稿",
      "papers.selectHint": "从左侧列表选择一篇生成的论文稿，打开投稿样式的阅读页。",

      "ideas.title": "已生成论文想法",
      "ideas.allTiers": "全部层级",
      "ideas.tier1": "层级 1: 范式",
      "ideas.tier2": "层级 2: 论文想法",
      "ideas.empty": "尚无深度发现。发现流程以固定自动模式运行。",
      "ideas.emptyNoAgenda": "尚未注册任何研究议程，因此没有可列出想法的范围。",
      "ideas.emptyFiltering": "尚无可展示的发现。自动发现仍在筛选候选。",

      "process.timeline": "过程时间线",
      "process.noScope": "无议程范围",
      "process.timelineNote": "当前议程的按时间记录：信号、候选决策、资源授权、任务、证据阶梯迁移与判定。失败与被拒绝的门禁迁移与成功同样展示。",
      "process.timelineEmpty": "尚无过程事件记录。",
      "process.timelineEmptyDetail": "该议程尚无过程事件。信号、授权或实验运行产生后，事件将出现在这里。",
      "process.timelineNoAgenda": "尚未注册任何研究议程，因此没有可展示的过程。",
      "process.rationale": "选择理由",
      "process.rationaleBadge": "为什么选择这些工作",
      "process.rationaleEmpty": "尚无选择记录。",
      "process.rationaleEmptyDetail": "该议程尚无选择记录。选择器接纳或拒绝候选后，理由将显示在这里。",
      "process.services": "自动化服务",
      "process.readOnly": "只读自动模式",
      "process.ideaExperiments": "想法实验",
      "process.metaReport": "元学习报告",
      "process.filter.all": "全部",
      "process.filter.pending": "等待中",
      "process.filter.scaffolding": "搭建中",
      "process.filter.reproducing": "复现中",
      "process.filter.testing": "测试中",
      "process.filter.completed": "已完成",
      "process.filter.failed": "失败",

      "badge.run": "运行",
      "badge.evidence": "证据",
      "badge.decided": "判定",
      "badge.notAssessed": "未评估",
      "badge.notAssessed.tip": "尚无证据阶梯进展；任务完成本身不构成任何发现",
      "badge.run.tip": "运营状态：任务是否运行；不构成任何科学论断",
      "badge.decided.tip": "由经审计的决策门禁记录的科学判定",
      "badge.progress.tip": "证据阶梯上的位置；尚未形成科学判定",
      "sci.planned": "已规划",
      "sci.sanity_passed": "健全性通过",
      "sci.full_benchmark_complete": "基准完成",
      "sci.evidence_audited": "已审计",
      "sci.scientifically_decided": "已判定",
      "sci.manuscript_allowed": "允许成稿",
      "verdict.supported": "支持",
      "verdict.refuted": "反驳",
      "verdict.inconclusive": "不确定",

      "tl.legacy": "历史导入",
      "tl.signal": "信号",
      "tl.candidate": "候选",
      "tl.authorization": "授权",
      "tl.run": "运行",
      "tl.evidence": "证据",
      "tl.decision": "判定",
      "tl.outcome": "结局",
    },
  };

  function preferredLanguage() {
    const saved = localStorage.getItem("deepgraph.lang");
    if (saved === "en" || saved === "zh") return saved;
    return (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  let currentLanguage = preferredLanguage();
  // Declare the document language once, up front. We deliberately do NOT
  // rewrite documentElement.lang on every toggle: changing this inherited,
  // style-affecting attribute forces a full-document style recalc on this
  // dashboard's large DOM, and no CSS here keys off :lang().
  try { document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en"; } catch (_) {}

  function t(key, vars) {
    const table = I18N[currentLanguage] || I18N.en;
    let text = table[key] || I18N.en[key] || key;
    if (vars) {
      text = text.replace(/\{([a-zA-Z0-9_]+)\}/g, (_, name) => (
        vars[name] == null ? "" : String(vars[name])
      ));
    }
    return text;
  }

  function applyI18n(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    scope.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
    });
    document.querySelectorAll("[data-lang]").forEach((node) => {
      node.classList.toggle("active", node.dataset.lang === currentLanguage);
    });
  }

  function setLanguage(lang) {
    currentLanguage = lang === "zh" ? "zh" : "en";
    localStorage.setItem("deepgraph.lang", currentLanguage);
    applyI18n(document);
    document.dispatchEvent(new CustomEvent("deepgraph:languagechange", { detail: { lang: currentLanguage } }));
  }

  function wire() {
    applyI18n(document);
    document.querySelectorAll("[data-lang]").forEach((node) => {
      node.addEventListener("click", () => setLanguage(node.dataset.lang));
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }

  window.dgI18n = {
    I18N,
    t,
    applyI18n,
    setLanguage,
    getLanguage: () => currentLanguage,
  };
  window.t = t;
})();
