/* DeepGraph dashboard i18n. Keep en/zh key sets identical. */
(function () {
  "use strict";

  const I18N = {
    en: {
      "app.live.idle": "IDLE",
      "app.live.live": "LIVE",
      "topbar.search.placeholder": "Search source papers, methods, research insights...",
      "lang.en": "EN",
      "lang.zh": "ZH",
      "nav.overview": "Overview",
      "nav.explore": "Explore",
      "nav.evidence": "Evidence",
      "nav.generated": "Generated",
      "nav.insights": "Insights",
      "nav.papers": "Papers",
      "nav.advanced": "Advanced",
      "nav.paperProgress": "Paper Progress",
      "nav.discoveries": "Discoveries",
      "nav.experiments": "Experiments",
      "nav.feed": "Feed",
      "nav.providers": "Providers",
      "nav.agenda": "Agenda",
      "footer.fixedFlow": "Fixed-flow mode: pipeline, discovery, experiment runs, and generated manuscripts run automatically.",
      "overview.sourcePapers": "Source Paper",
      "overview.results": "Result",
      "overview.researchAreas": "Research Area",
      "overview.contradictions": "Contradiction",
      "overview.researchInsights": "Research Insight",
      "overview.tokens": "Token",
      "overview.experimentRuns": "Experiment Run",
      "overview.discoveries": "Discovery",
      "overview.submissionBundles": "Submission Bundle",
      "overview.latest": "Latest Activity",
      "overview.latestEmpty": "Run the pipeline to discover gaps, contradictions, opportunities, and discoveries.",
      "overview.processing": "Current Processing",
      "overview.idle": "Idle",
      "overview.graph": "Research Area Explorer",
      "overview.openExplore": "Open in Explore",
      "overview.advanced": "Advanced Process Metrics",
      "explore.title": "Research Area Explorer",
      "explore.summary": "Research Area Summary",
      "explore.children": "Sub-area",
      "evidence.title": "Benchmark Matrix",
      "evidence.select": "Select research area:",
      "evidence.option": "-- Select a leaf research area --",
      "evidence.hint": "Select a leaf research area to view the benchmark matrix.",
      "evidence.gaps": "Matrix Gaps",
      "papers.title": "Source Paper",
      "papers.filter": "Filter source papers...",
      "papers.allStatuses": "All statuses",
      "paperProgress.pipeline": "Source Papers In Progress",
      "paperProgress.generation": "Generated Manuscript Progress",
      "generated.title": "Generated Manuscript",
      "discoveries.title": "Discovery",
      "discoveries.allTiers": "All tiers",
      "discoveries.tier1": "Tier 1: Paradigm",
      "discoveries.tier2": "Tier 2: Discovery",
      "experiments.services": "Automation Services",
      "experiments.readOnly": "Read-only automatic mode",
      "experiments.autoResearch": "Auto Research",
      "experiments.fixedMode": "Fixed automatic mode",
      "experiments.ideaRuns": "Discovery Experiment Runs",
      "experiments.all": "All",
      "experiments.pending": "Pending",
      "experiments.scaffolding": "Scaffolding",
      "experiments.reproducing": "Reproducing",
      "experiments.testing": "Testing",
      "experiments.completed": "Completed",
      "experiments.failed": "Failed",
      "experiments.meta": "Meta-Learning Report",
      "insights.title": "Research Insight",
      "insights.allTypes": "All types",
      "insights.paradigm": "Paradigm-Breaking",
      "insights.score": "By N+F Score",
      "insights.novelty": "By Novelty",
      "insights.feasibility": "By Feasibility",
      "insights.recent": "Most Recent",
      "feed.title": "Pipeline Event Feed",
      "providers.title": "LLM Providers",
      "providers.refresh": "Auto-refresh 10s",
      "agenda.title": "Research Agenda",
      "agenda.noAgenda": "no agenda",
      "agenda.refresh": "Refresh",
      "agenda.select": "Run Selector + Dispatch",
      "agenda.upload": "Upload agenda (YAML)",
      "agenda.uploadButton": "Upload",
      "agenda.latestSelection": "Latest Selection",
      "agenda.review": "Run Review",
      "agenda.plan": "Build Revision Plan",
      "agenda.inspect": "Inspect Full Loop",
      "agenda.loop": "Loop Inspection",
      "manuscript.routing": "Manuscript Routing & Format Lint",
      "manuscript.refreshVenues": "Refresh Venues",
      "manuscript.previewRoute": "Preview Route",
      "manuscript.previewLint": "Preview Lint",
      "manuscript.includeTiebreak": "include tiebreak",
      "manuscript.stateJson": "State JSON (for route preview)",
      "manuscript.lintPayload": "Lint payload (template_id + source)"
    },
    zh: {
      "app.live.idle": "空闲",
      "app.live.live": "运行中",
      "topbar.search.placeholder": "搜索文献、方法、研究洞见...",
      "lang.en": "英",
      "lang.zh": "中",
      "nav.overview": "概览",
      "nav.explore": "探索",
      "nav.evidence": "证据",
      "nav.generated": "生成",
      "nav.insights": "洞见",
      "nav.papers": "文献",
      "nav.advanced": "高级",
      "nav.paperProgress": "文献进度",
      "nav.discoveries": "深度发现",
      "nav.experiments": "实验",
      "nav.feed": "事件流",
      "nav.providers": "模型服务",
      "nav.agenda": "研究议程",
      "footer.fixedFlow": "固定流程模式：流水线、深度发现、实验运行和生成稿件会自动执行。",
      "overview.sourcePapers": "文献",
      "overview.results": "基准结果",
      "overview.researchAreas": "研究领域",
      "overview.contradictions": "矛盾",
      "overview.researchInsights": "研究洞见",
      "overview.tokens": "Token",
      "overview.experimentRuns": "实验运行",
      "overview.discoveries": "深度发现",
      "overview.submissionBundles": "投稿包",
      "overview.latest": "最新动态",
      "overview.latestEmpty": "运行流水线以发现空白、矛盾、机会点和深度发现。",
      "overview.processing": "当前处理",
      "overview.idle": "空闲",
      "overview.graph": "研究领域探索",
      "overview.openExplore": "在探索中打开",
      "overview.advanced": "高级过程指标",
      "explore.title": "研究领域探索",
      "explore.summary": "研究领域摘要",
      "explore.children": "子领域",
      "evidence.title": "基准矩阵",
      "evidence.select": "选择研究领域：",
      "evidence.option": "-- 选择叶子研究领域 --",
      "evidence.hint": "选择叶子研究领域以查看基准矩阵。",
      "evidence.gaps": "矩阵空白",
      "papers.title": "文献",
      "papers.filter": "筛选文献...",
      "papers.allStatuses": "全部状态",
      "paperProgress.pipeline": "处理中（文献）",
      "paperProgress.generation": "生成稿件进度",
      "generated.title": "生成稿件",
      "discoveries.title": "深度发现",
      "discoveries.allTiers": "全部层级",
      "discoveries.tier1": "层级 1：范式",
      "discoveries.tier2": "层级 2：深度发现",
      "experiments.services": "自动化服务",
      "experiments.readOnly": "只读自动模式",
      "experiments.autoResearch": "自动研究",
      "experiments.fixedMode": "固定自动模式",
      "experiments.ideaRuns": "深度发现实验运行",
      "experiments.all": "全部",
      "experiments.pending": "待处理",
      "experiments.scaffolding": "脚手架",
      "experiments.reproducing": "复现中",
      "experiments.testing": "测试中",
      "experiments.completed": "已完成",
      "experiments.failed": "失败",
      "experiments.meta": "元学习报告",
      "insights.title": "研究洞见",
      "insights.allTypes": "全部类型",
      "insights.paradigm": "按范式突破",
      "insights.score": "按 N+F 分数",
      "insights.novelty": "按新颖性",
      "insights.feasibility": "按可行性",
      "insights.recent": "最近",
      "feed.title": "流水线事件流",
      "providers.title": "模型服务商",
      "providers.refresh": "10 秒自动刷新",
      "agenda.title": "研究议程",
      "agenda.noAgenda": "无议程",
      "agenda.refresh": "刷新",
      "agenda.select": "运行选择器并分发",
      "agenda.upload": "上传议程（YAML）",
      "agenda.uploadButton": "上传",
      "agenda.latestSelection": "最新选择",
      "agenda.review": "运行评审",
      "agenda.plan": "生成修订计划",
      "agenda.inspect": "检查完整循环",
      "agenda.loop": "循环检查",
      "manuscript.routing": "稿件路由与格式检查",
      "manuscript.refreshVenues": "刷新会议模板",
      "manuscript.previewRoute": "预览路由",
      "manuscript.previewLint": "预览格式检查",
      "manuscript.includeTiebreak": "包含平局规则",
      "manuscript.stateJson": "状态 JSON（用于路由预览）",
      "manuscript.lintPayload": "格式检查载荷（template_id + source）"
    },
  };

  function preferredLanguage() {
    const saved = localStorage.getItem("deepgraph.lang");
    if (saved === "en" || saved === "zh") return saved;
    return (navigator.language || "").toLowerCase().startsWith('zh') ? "zh" : "en";
  }

  let currentLanguage = preferredLanguage();

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
    scope.querySelectorAll("[data-i18n-title]").forEach((node) => {
      node.setAttribute("title", t(node.dataset.i18nTitle));
    });
    document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en";
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

  window.dgI18n = {
    I18N,
    t,
    applyI18n,
    setLanguage,
    getLanguage: () => currentLanguage,
  };
  window.t = t;
})();
