/* ═══════════════════════════════════════════════════════════════════
   DeepGraph — Custom Tooltip controller (acceptance D)

   Replaces native `title=` popups with a styled, instant tooltip that matches
   the dashboard. One shared element (#tooltip) serves two callers:

     1. The graph renderer — via the injected { show, move, hide } handle, so the
        renderer never touches #tooltip itself (keeps it swap-clean).
     2. Any element carrying data-i18n-title (the 9 metric cards) — wired here by
        document-level delegation. Text is resolved through window.t at hover
        time, so it always follows the current language (zh/en) and reuses the
        existing i18n keys; nothing sets the native `title` attribute anymore.
   ═══════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  function tipEl() { return document.getElementById('tooltip'); }

  function position(ev) {
    const tip = tipEl();
    if (!tip) return;
    const pad = 14;
    let x = ev.clientX + pad;
    let y = ev.clientY - pad;
    const tw = tip.offsetWidth || 240;
    const th = tip.offsetHeight || 80;
    if (x + tw > window.innerWidth - 10) x = ev.clientX - tw - pad;
    if (y + th > window.innerHeight - 10) y = window.innerHeight - th - 10;
    if (y < 10) y = 10;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }

  function show(html, ev) {
    const tip = tipEl();
    if (!tip) return;
    tip.innerHTML = html;
    tip.classList.add('visible');
    if (ev) position(ev);
  }

  function move(ev) {
    const tip = tipEl();
    if (tip && tip.classList.contains('visible')) position(ev);
  }

  function hide() {
    const tip = tipEl();
    if (tip) tip.classList.remove('visible');
  }

  /* Delegated tooltips for [data-i18n-title] elements (metric cards etc.). */
  function attachDelegated() {
    const t = (key) => (window.t ? window.t(key) : key);
    document.addEventListener('mouseover', (ev) => {
      const host = ev.target.closest && ev.target.closest('[data-i18n-title]');
      if (!host) return;
      const text = t(host.dataset.i18nTitle);
      if (!text) return;
      show(`<div class="tip-body">${text}</div>`, ev);
    });
    document.addEventListener('mousemove', (ev) => {
      const host = ev.target.closest && ev.target.closest('[data-i18n-title]');
      if (host) move(ev);
    });
    document.addEventListener('mouseout', (ev) => {
      const host = ev.target.closest && ev.target.closest('[data-i18n-title]');
      if (!host) return;
      // only hide when leaving the host entirely
      if (!ev.relatedTarget || !host.contains(ev.relatedTarget)) hide();
    });
  }

  window.DGTooltip = { show, move, hide, position, attachDelegated };
})();
