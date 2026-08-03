"""Process-internal TTL cache for the heavy /api/stats query (issue #34).

The dashboard's first paint blocks on ``/api/stats``, which runs ~30
``COUNT(*)`` queries — several of them full-table scans over large tables
(``graph_relations`` ~500k rows, ``entity_resolutions`` ~180k, ``results``
~110k …). Under concurrency this balloons to ~18s and the nine metric cards
stay blank for ten-plus seconds.

This cache serves ``/api/stats`` from an in-process snapshot. The heavy query
runs only in the warm-up and in a single long-lived background refresher
thread — never in the request-serving thread. A single-process waitress
deployment makes an in-process cache sufficient (issue #34 non-goals: no Redis
/ multi-process).

Guarantees relevant to the issue's red lines:
- ``get()`` is a pure read: it never runs the heavy query in the caller
  thread, so per-request COUNT(*) scans are gone.
- Numbers are never fabricated: before the first warm-up completes ``get()``
  returns ``None`` (the route renders a "warming" marker), never fake values.
- The cache is lazy: constructing it does not run the query, so importing the
  web app (as tests and startup do) stays cheap.
"""
import logging
import threading
import time

logger = logging.getLogger(__name__)


class StatsCache:
    def __init__(self, compute, ttl: float = 30.0, time_func=time.monotonic):
        self._compute = compute
        self._ttl = float(ttl)
        self._time = time_func
        self._lock = threading.Lock()
        self._value = None
        self._stamp = None        # monotonic timestamp of the last successful compute
        self._refresher = None    # the single background refresher thread
        self.compute_count = 0    # successful heavy recomputes (observability / tests)

    def get(self):
        """Return the current cached stats snapshot.

        Pure read — never runs the heavy query in the caller thread. Returns
        the last computed dict (a snapshot at most ``ttl``-ish seconds stale),
        or ``None`` before the first warm-up completes.
        """
        with self._lock:
            return self._value

    def _recompute(self):
        value = self._compute()
        with self._lock:
            self._value = value
            self._stamp = self._time()
            self.compute_count += 1
        return value

    def prewarm(self):
        """Synchronously compute once and populate the cache.

        Called at startup (in a background thread so neither import nor server
        start blocks) and by tests for determinism.
        """
        return self._recompute()

    def start_background_refresh(self):
        """Start the single daemon thread that refreshes the snapshot every
        ``ttl`` seconds. Idempotent; intended to be called once at startup."""
        with self._lock:
            if self._refresher is not None and self._refresher.is_alive():
                return
            self._refresher = threading.Thread(
                target=self._refresh_loop, name="stats-cache-refresh", daemon=True
            )
            self._refresher.start()

    def _refresh_loop(self):
        while True:
            time.sleep(self._ttl)
            try:
                self._recompute()
            except Exception:  # pragma: no cover - defensive, logged not fatal
                logger.exception("stats cache background refresh failed")

    def invalidate(self):
        with self._lock:
            self._value = None
            self._stamp = None
