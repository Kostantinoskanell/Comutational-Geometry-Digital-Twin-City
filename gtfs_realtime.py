"""gtfs_realtime.py — live GTFS-Realtime VehiclePositions ingestion.

Fetches and parses a GTFS-Realtime VehiclePositions feed (the standard protobuf
format most transit agencies publish) and hands back a clean list of live
vehicle dicts.  A background poller thread keeps the latest snapshot fresh
without ever blocking the render loop.

Graceful degradation
---------------------
Every external dependency is optional and probed at call time:
  • protobuf bindings  (google.transit.gtfs_realtime_pb2  OR
                        gtfs_realtime_pb2 from the gtfs-realtime-bindings pkg)
  • requests           (falls back to urllib from the stdlib)
If a feed can't be fetched or parsed, the poller logs once and the rest of the
app keeps running on the simulated schedule.

Public API
----------
parse_vehicle_positions(raw_bytes) -> list[dict]
    Decode a raw protobuf payload into vehicle dicts.

fetch_feed(url, api_key=None, api_key_param=None, timeout=10.0) -> bytes | None
    HTTP GET the feed (requests if available, else urllib).

GTFSRealtimePoller
    Daemon thread that polls `url` every `interval` seconds and exposes the
    latest parsed snapshot via .snapshot()  (thread-safe).

Vehicle dict schema
-------------------
{
  "vehicle_id": str,
  "trip_id":    str | None,
  "route_id":   str | None,
  "lat":        float,
  "lon":        float,
  "bearing":    float | None,   # degrees, 0=N clockwise (None if absent)
  "speed":      float | None,   # m/s if the agency supplies it
  "timestamp":  int | None,     # POSIX seconds
}
"""
from __future__ import annotations

import threading
import time
import urllib.request


# ── Protobuf binding probe (done once, lazily) ──────────────────────────────

def _load_pb2():
    """Return the gtfs_realtime_pb2 module or None if no bindings are installed."""
    try:
        from google.transit import gtfs_realtime_pb2  # gtfs-realtime-bindings
        return gtfs_realtime_pb2
    except Exception:
        pass
    try:
        import gtfs_realtime_pb2  # some installs expose it top-level
        return gtfs_realtime_pb2
    except Exception:
        return None


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_vehicle_positions(raw_bytes: bytes) -> list[dict]:
    """Decode a GTFS-RT protobuf payload into a list of vehicle dicts.

    Returns [] on any failure (missing bindings, malformed payload, no
    vehicle entities present).
    """
    pb2 = _load_pb2()
    if pb2 is None:
        return []

    try:
        feed = pb2.FeedMessage()
        feed.ParseFromString(raw_bytes)
    except Exception as exc:
        print(f"[gtfs-rt] parse error: {exc}")
        return []

    out: list[dict] = []
    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        pos = v.position
        if pos is None or not (pos.latitude or pos.longitude):
            continue

        trip_id  = v.trip.trip_id   if v.HasField("trip")    else None
        route_id = v.trip.route_id  if v.HasField("trip")    else None
        vid      = v.vehicle.id     if v.HasField("vehicle") else (ent.id or "?")
        bearing  = float(pos.bearing) if pos.HasField("bearing") else None
        speed    = float(pos.speed)   if pos.HasField("speed")   else None
        ts       = int(v.timestamp)   if v.HasField("timestamp") else None

        out.append({
            "vehicle_id": str(vid),
            "trip_id":    str(trip_id) if trip_id else None,
            "route_id":   str(route_id) if route_id else None,
            "lat":        float(pos.latitude),
            "lon":        float(pos.longitude),
            "bearing":    bearing,
            "speed":      speed,
            "timestamp":  ts,
        })
    return out


# ── Fetching ─────────────────────────────────────────────────────────────────

def fetch_feed(
    url: str,
    api_key: str | None = None,
    api_key_param: str | None = None,
    timeout: float = 10.0,
) -> bytes | None:
    """HTTP GET the feed and return raw bytes (None on any error).

    If `api_key` is given it is sent as an `Authorization` / `api_key` HTTP
    header by default, or appended as the query parameter named `api_key_param`
    when that is supplied (some agencies key the feed via the query string).
    """
    if not url:
        return None

    full_url = url
    headers: dict[str, str] = {"User-Agent": "city-digital-twin/1.0"}
    if api_key:
        if api_key_param:
            sep = "&" if "?" in url else "?"
            full_url = f"{url}{sep}{api_key_param}={api_key}"
        else:
            headers["Authorization"] = api_key
            headers["api_key"] = api_key

    # Prefer requests if present (better proxy / TLS handling); else urllib.
    try:
        import requests
        resp = requests.get(full_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except ImportError:
        pass
    except Exception as exc:
        print(f"[gtfs-rt] fetch failed: {exc}")
        return None

    try:
        req = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as exc:
        print(f"[gtfs-rt] fetch failed: {exc}")
        return None


# ── Background poller ─────────────────────────────────────────────────────────

class GTFSRealtimePoller:
    """Daemon thread that refreshes a VehiclePositions snapshot on an interval.

    The render loop reads .snapshot() — a cheap, thread-safe copy of the most
    recent parse — so polling latency never stalls rendering.
    """

    def __init__(
        self,
        url: str,
        interval: float = 15.0,
        api_key: str | None = None,
        api_key_param: str | None = None,
    ) -> None:
        self.url = url
        self.interval = max(2.0, float(interval))
        self.api_key = api_key
        self.api_key_param = api_key_param

        self._lock = threading.Lock()
        self._snapshot: list[dict] = []
        self._last_ok_t: float = 0.0
        self._fetch_count: int = 0
        self._fail_count: int = 0
        self._consec_fail: int = 0
        self._stale_warned: bool = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = _load_pb2() is not None

    @property
    def available(self) -> bool:
        """True if protobuf bindings are present (feed can actually be parsed)."""
        return self._available

    def start(self) -> bool:
        """Spawn the poller thread.  Returns False if bindings are missing."""
        if not self._available:
            print(
                "[gtfs-rt] protobuf bindings not found — install "
                "'gtfs-realtime-bindings' to enable live vehicles. "
                "Falling back to simulated schedule."
            )
            return False
        if self._thread is not None:
            return True
        self._thread = threading.Thread(
            target=self._run, name="gtfs-rt-poller", daemon=True
        )
        self._thread.start()
        print(f"[gtfs-rt] poller started — {self.url} every {self.interval:.0f}s")
        return True

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> list[dict]:
        """Thread-safe shallow copy of the latest parsed vehicle list."""
        with self._lock:
            return list(self._snapshot)

    def age_seconds(self) -> float:
        """Seconds since the last successful fetch (inf if never)."""
        if self._last_ok_t <= 0.0:
            return float("inf")
        return time.time() - self._last_ok_t

    def _run(self) -> None:
        while not self._stop.is_set():
            raw = fetch_feed(self.url, self.api_key, self.api_key_param)
            self._fetch_count += 1
            if raw is not None:
                vehicles = parse_vehicle_positions(raw)
                if vehicles:
                    with self._lock:
                        self._snapshot = vehicles
                        self._last_ok_t = time.time()
                    self._consec_fail = 0
                    self._stale_warned = False
                    if self._fetch_count == 1 or self._fetch_count % 20 == 0:
                        print(f"[gtfs-rt] {len(vehicles)} live vehicles")
                else:
                    self._fail_count += 1
                    self._consec_fail += 1
                    print(f"[gtfs-rt] successful fetch but no vehicles parsed (fetch #{self._fetch_count})")
            else:
                self._fail_count += 1
                self._consec_fail += 1
                if self._consec_fail >= 2 and self._consec_fail % 5 == 0:
                    print(f"[gtfs-rt] {self._consec_fail} consecutive fetch failures")

            # Stale-data warning when feed hasn't updated in >3 intervals
            if self.age_seconds() > self.interval * 3 and not self._stale_warned:
                print(
                    f"[gtfs-rt] WARNING: feed data is stale "
                    f"({self.age_seconds():.0f}s since last successful update)"
                )
                self._stale_warned = True

            # Wait `interval`, but wake promptly on stop().
            self._stop.wait(self.interval)
