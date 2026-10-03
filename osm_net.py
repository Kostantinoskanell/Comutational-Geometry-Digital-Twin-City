"""Resilient Overpass access for osmnx calls (W0: Overpass resilience).

The public overpass-api.de instance times out or refuses connections often
enough to have aborted scene startup. with_overpass_fallback() runs an
osmnx call against the main instance and, on network failure, the standard
public mirrors, each with a bounded timeout; callers decide how to degrade
when all fail. Only public OSM map queries (a bounding box / tags) are sent.
"""
from __future__ import annotations

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api",
    "https://lz4.overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
)
PER_ENDPOINT_TIMEOUT_S = 60
_preferred = 0      # circuit breaker: start from the endpoint that last worked ...
_down: set = set()  # ... and try endpoints that failed at the network level last


def _is_network_error(exc: BaseException) -> bool:
    try:
        import requests
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            code = getattr(getattr(exc, "response", None), "status_code", None)
            return code in (429, 502, 503, 504)
    except Exception:
        pass
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "ResponseStatusCode"))


def with_overpass_fallback(fn, label: str = "overpass"):
    """Run fn() (an osmnx call) across Overpass endpoints; re-raise the last
    network error if all fail, or any non-network error immediately (e.g.
    osmnx's 'no data in area', which a mirror would not change)."""
    import osmnx as ox
    global _preferred
    orig_url, orig_timeout = ox.settings.overpass_url, ox.settings.requests_timeout
    last = None
    n = len(OVERPASS_ENDPOINTS)
    order = [(_preferred + k) % n for k in range(n)]
    order = [i for i in order if i not in _down] + [i for i in order if i in _down]
    try:
        for i in order:
            url = OVERPASS_ENDPOINTS[i]
            ox.settings.overpass_url = url
            ox.settings.requests_timeout = PER_ENDPOINT_TIMEOUT_S
            try:
                out = fn()
                if i:
                    print(f"[{label}] served by mirror {url}")
                _preferred = i
                _down.discard(i)
                return out
            except Exception as exc:
                if not _is_network_error(exc):
                    raise
                last = exc
                _down.add(i)
                print(f"[{label}] {url} unavailable ({type(exc).__name__}); trying next endpoint")
    finally:
        ox.settings.overpass_url, ox.settings.requests_timeout = orig_url, orig_timeout
    raise last
