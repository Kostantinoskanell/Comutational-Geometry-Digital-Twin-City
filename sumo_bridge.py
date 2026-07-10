"""sumo_bridge.py — TraCI bridge for SUMO co-simulation.

Runs Eclipse SUMO as a child process and drives it step-by-step through TraCI,
extracting live vehicle state each step so the PyVista app can render an
industrial-grade traffic microsimulation in the same scene.

Coordinate alignment
---------------------
SUMO works in its own planar network coordinates (usually UTM with a netOffset).
We never try to reconcile that frame directly with the app's local tmerc CRS.
Instead each vehicle's (x, y) is converted to (lon, lat) **locally** via the
sumolib network object — pure Python, no TraCI round-trip — and the caller then
projects (lon, lat) into the app's local metres with the same pyproj transformer
used for GTFS-Realtime.  Two cheap local conversions, perfect alignment.

Graceful degradation
---------------------
SUMO and traci are both optional and probed at start time.  If either is
missing, `check_sumo()` explains how to install them and the app keeps running
on its own IDM car layer.

Public API
----------
check_sumo() -> (ok: bool, message: str)
SumoConnection(sumo_cfg, binary, step_length, use_gui, net_file, port)
    .start() -> bool
    .step()
    .vehicles() -> list[dict]
    .sim_time() -> float
    .close()

Vehicle dict schema
-------------------
{ "id": str, "lon": float, "lat": float, "angle": float,   # compass deg, 0=N CW
  "speed": float, "type": str, "vclass": str, "length": float }
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Availability probe ───────────────────────────────────────────────────────

def resolve_sumo_home() -> str | None:
    """Return SUMO_HOME from the env var, or fall back to the eclipse-sumo pip
    package's bundled location.  Sets os.environ['SUMO_HOME'] when found so
    child tools (netconvert, randomTrips.py) inherit it."""
    home = os.environ.get("SUMO_HOME")
    if home and os.path.isdir(home):
        return home
    try:
        import sumo  # eclipse-sumo wheel exposes SUMO_HOME
        pkg_home = getattr(sumo, "SUMO_HOME", None)
        if pkg_home and os.path.isdir(pkg_home):
            os.environ["SUMO_HOME"] = pkg_home   # propagate to subprocesses
            return pkg_home
    except Exception:
        pass
    return home or None


def _sumo_home_tools_on_path() -> None:
    """Add $SUMO_HOME/tools to sys.path so `import traci`/`sumolib` works."""
    import sys
    home = resolve_sumo_home()
    if home:
        tools = os.path.join(home, "tools")
        if os.path.isdir(tools) and tools not in sys.path:
            sys.path.append(tools)


def check_sumo() -> tuple[bool, str]:
    """Return (ok, message).  ok=True means SUMO + traci are usable."""
    _sumo_home_tools_on_path()
    have_traci = False
    try:
        import traci  # noqa: F401
        have_traci = True
    except Exception:
        pass

    have_binary = (
        shutil.which("sumo") is not None
        or shutil.which("sumo-gui") is not None
        or (os.environ.get("SUMO_HOME")
            and os.path.isdir(os.path.join(os.environ["SUMO_HOME"], "bin")))
    )

    if have_traci and have_binary:
        return True, "SUMO + traci available"

    msg = "SUMO co-simulation unavailable — "
    missing = []
    if not have_binary:
        missing.append("SUMO not found (install from https://eclipse.dev/sumo "
                       "and set $SUMO_HOME)")
    if not have_traci:
        missing.append("traci not importable (pip install traci, or set $SUMO_HOME)")
    return False, msg + "; ".join(missing)


def _net_file_from_cfg(cfg_path: str) -> str | None:
    """Parse a .sumocfg and return the absolute path of its <net-file>."""
    try:
        tree = ET.parse(cfg_path)
        node = tree.find(".//net-file")
        if node is None:
            return None
        val = node.get("value", "").split()[0]
        if not val:
            return None
        p = Path(val)
        if not p.is_absolute():
            p = Path(cfg_path).parent / p
        return str(p) if p.exists() else None
    except Exception:
        return None


# ── Connection ────────────────────────────────────────────────────────────────

class SumoConnection:

    def __init__(
        self,
        sumo_cfg: str,
        binary: str = "sumo",
        step_length: float = 0.1,
        use_gui: bool = False,
        net_file: str = "",
        port: int | None = None,
    ) -> None:
        self.sumo_cfg    = sumo_cfg
        self.binary      = binary
        self.step_length = max(0.01, float(step_length))
        self.use_gui     = use_gui
        self.net_file    = net_file
        self.port        = port

        self._traci = None
        self._net   = None        # sumolib net for XY→lonLat
        self._ready = False
        self._closed = False

    @property
    def ready(self) -> bool:
        return self._ready and not self._closed

    def _resolve_binary(self) -> str | None:
        name = "sumo-gui" if self.use_gui else (self.binary or "sumo")
        # explicit path?
        if os.path.sep in name and os.path.exists(name):
            return name
        found = shutil.which(name)
        if found:
            return found
        home = os.environ.get("SUMO_HOME")
        if home:
            cand = os.path.join(home, "bin", name)
            if os.path.exists(cand) or os.path.exists(cand + ".exe"):
                return cand
        return None

    def start(self) -> bool:
        ok, msg = check_sumo()
        if not ok:
            print(f"[sumo] {msg}")
            return False

        if not self.sumo_cfg or not os.path.exists(self.sumo_cfg):
            print(f"[sumo] config not found: {self.sumo_cfg}")
            return False

        binary = self._resolve_binary()
        if binary is None:
            print("[sumo] could not locate the SUMO binary on PATH or $SUMO_HOME/bin")
            return False

        # Load the network for local XY→lon/lat conversion (no TraCI round-trips).
        net_file = self.net_file or _net_file_from_cfg(self.sumo_cfg)
        if net_file:
            try:
                import sumolib
                self._net = sumolib.net.readNet(net_file)
            except Exception as exc:
                print(f"[sumo] could not load net for geo-conversion ({exc}) — "
                      "vehicle alignment requires a geo-referenced net; disabling")
                return False
        else:
            print("[sumo] no <net-file> found in cfg; pass --sumo-net explicitly. "
                  "Geo-conversion is required for alignment — disabling")
            return False

        if not self._net.hasGeoProj():
            print("[sumo] network has no geo-projection (build it from OSM with "
                  "netconvert so SUMO knows lon/lat) — disabling")
            return False

        try:
            import traci
            cmd = [binary, "-c", self.sumo_cfg,
                   "--step-length", str(self.step_length),
                   "--start", "--quit-on-end"]
            if self.port is not None:
                traci.start(cmd, port=int(self.port))
            else:
                traci.start(cmd)
            self._traci = traci
            self._ready = True
            print(f"[sumo] connected via TraCI ({os.path.basename(binary)}, "
                  f"step={self.step_length}s)")
            return True
        except Exception as exc:
            print(f"[sumo] failed to start: {exc}")
            return False

    def step(self) -> None:
        if not self.ready:
            return
        try:
            self._traci.simulationStep()
        except Exception as exc:
            print(f"[sumo] step error: {exc}")
            self._closed = True

    def sim_time(self) -> float:
        if not self.ready:
            return 0.0
        try:
            return float(self._traci.simulation.getTime())
        except Exception:
            return 0.0

    def vehicles(self) -> list[dict]:
        """Extract current vehicle state.  (x,y)→(lon,lat) done locally."""
        if not self.ready:
            return []
        tr = self._traci
        out: list[dict] = []
        try:
            ids = tr.vehicle.getIDList()
        except Exception:
            return []
        for vid in ids:
            try:
                x, y     = tr.vehicle.getPosition(vid)
                lon, lat = self._net.convertXY2LonLat(x, y)
                vclass   = str(tr.vehicle.getVehicleClass(vid))
                # SUMO's internal default type should render as a passenger car
                if vclass in ("DEFAULT_VEHTYPE", "ignoring", ""):
                    vclass = "passenger"
                out.append({
                    "id":     str(vid),
                    "lon":    float(lon),
                    "lat":    float(lat),
                    "angle":  float(tr.vehicle.getAngle(vid)),
                    "speed":  float(tr.vehicle.getSpeed(vid)),
                    "type":   str(tr.vehicle.getTypeID(vid)),
                    "vclass": vclass,
                    "length": float(tr.vehicle.getLength(vid)),
                })
            except Exception:
                continue
        return out

    def persons(self) -> list[dict]:
        """Extract pedestrian positions from SUMO (empty list if none simulated)."""
        if not self.ready:
            return []
        tr = self._traci
        out: list[dict] = []
        try:
            ids = tr.person.getIDList()
        except Exception:
            return []
        for pid in ids:
            try:
                x, y     = tr.person.getPosition(pid)
                lon, lat = self._net.convertXY2LonLat(x, y)
                # Stage type: 0=waiting-for-depart 1=waiting 2=walking 3=riding
                try:
                    stage = int(tr.person.getStage(pid).type)
                except Exception:
                    stage = 2
                out.append({
                    "id":    str(pid),
                    "lon":   float(lon),
                    "lat":   float(lat),
                    "angle": float(tr.person.getAngle(pid)),
                    "speed": float(tr.person.getSpeed(pid)),
                    "stage": stage,
                })
            except Exception:
                continue
        return out

    def restart(self, new_cfg: str | None = None) -> bool:
        """Close and re-open the SUMO connection, optionally with a new .sumocfg."""
        self.close()
        self._ready = False
        self._closed = False   # reset so start() can proceed
        if new_cfg:
            self.sumo_cfg = new_cfg
            resolved = _net_file_from_cfg(new_cfg)
            if resolved:
                self.net_file = resolved
        return self.start()

    # ── TraCI control helpers ──────────────────────────────────────────────────

    def tl_ids(self) -> list[str]:
        """Return the list of traffic-light junction IDs known to SUMO."""
        if not self.ready:
            return []
        try:
            return list(self._traci.trafficlight.getIDList())
        except Exception:
            return []

    def tl_set_program(self, tl_id: str, program: str) -> bool:
        """Switch a TL junction to an existing program ('0', 'off', …)."""
        if not self.ready:
            return False
        try:
            self._traci.trafficlight.setProgram(tl_id, program)
            return True
        except Exception as exc:
            print(f"[sumo] tl_set_program({tl_id!r}, {program!r}): {exc}")
            return False

    def tl_add_program(self, tl_id: str, phases: list[dict]) -> bool:
        """Define a new TL program via TraCI.

        phases: [{"state": "GGrrGGrr", "duration": 30}, …]
        """
        if not self.ready:
            return False
        try:
            logic = self._traci.trafficlight.Logic(
                programID="editor",
                type=0,          # static (non-actuated)
                currentPhaseIndex=0,
                phases=[
                    self._traci.trafficlight.Phase(
                        duration=float(p["duration"]),
                        state=str(p["state"]),
                    )
                    for p in phases
                ],
            )
            self._traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, logic)
            return True
        except Exception as exc:
            print(f"[sumo] tl_add_program({tl_id!r}): {exc}")
            return False

    # ── Routing helpers ────────────────────────────────────────────────────────

    def nearest_edge_id(self, lon: float, lat: float, radius: float = 60.0) -> str | None:
        """Nearest drivable non-internal edge id to a WGS84 point, or None."""
        if self._net is None:
            return None
        try:
            x, y = self._net.convertLonLat2XY(lon, lat)
            cands = self._net.getNeighboringEdges(x, y, radius)
            best_id: str | None = None
            best_dist = float("inf")
            for edge, dist in cands:
                eid = edge.getID()
                if eid.startswith(":"):
                    continue
                # Prefer edges that allow passenger cars (sumolib API varies)
                try:
                    if hasattr(edge, "allows") and not edge.allows("passenger"):
                        continue
                except Exception:
                    pass
                if dist < best_dist:
                    best_dist = dist
                    best_id = eid
            return best_id
        except Exception as exc:
            print(f"[sumo] nearest_edge_id({lon:.6f}, {lat:.6f}): {exc}")
            return None

    def vehicle_lonlat(self, vid: str) -> tuple[float, float] | None:
        """Current (lon, lat) of a vehicle in WGS84, or None."""
        if not self.ready or self._net is None:
            return None
        try:
            x, y = self._traci.vehicle.getPosition(vid)
            lon, lat = self._net.convertXY2LonLat(x, y)
            return float(lon), float(lat)
        except Exception as exc:
            print(f"[sumo] vehicle_lonlat({vid!r}): {exc}")
            return None

    def vehicle_set_target(self, vid: str, edge_id: str) -> bool:
        """Reroute a vehicle to end at edge_id (traci changeTarget). True on success."""
        if not self.ready:
            return False
        try:
            self._traci.vehicle.changeTarget(vid, edge_id)
            print(f"[sumo] vehicle {vid} retargeted to edge {edge_id}")
            return True
        except Exception as exc:
            print(f"[sumo] vehicle_set_target({vid!r}, {edge_id!r}): {exc}")
            return False

    def vehicle_route_shape_lonlat(self, vid: str) -> list[tuple[float, float]] | None:
        """Polyline (lon,lat) of the vehicle's REMAINING route, or None.

        Concatenates edge shapes from the current route index onward.
        """
        if not self.ready or self._net is None:
            return None
        try:
            route = self._traci.vehicle.getRoute(vid)
            idx = int(self._traci.vehicle.getRouteIndex(vid))
            pts: list[tuple[float, float]] = []
            for eid in route[max(idx, 0):]:
                edge = self._net.getEdge(eid)
                for x, y in edge.getShape():
                    lon, lat = self._net.convertXY2LonLat(x, y)
                    pt = (float(lon), float(lat))
                    if pts and pts[-1] == pt:
                        continue   # skip consecutive duplicates
                    pts.append(pt)
            return pts if pts else None
        except Exception as exc:
            print(f"[sumo] vehicle_route_shape_lonlat({vid!r}): {exc}")
            return None

    def tl_link_states(self) -> list[tuple[float, float, str]]:
        """Per-signal-link light states from SUMO's ACTUAL controllers.

        Returns [(lon, lat, state_char), ...] — one entry per controlled link,
        positioned at the end of its incoming lane.  state_char is SUMO's
        letter code: G/g=green, y/Y=yellow, r/R=red (o/O=off treated as red).
        Link positions are static and cached; only states are re-read.
        """
        if not self.ready or self._net is None:
            return []
        tr = self._traci
        if not hasattr(self, "_tl_link_pos_cache"):
            # tl_id → [ (lon, lat) per link index ]
            self._tl_link_pos_cache: dict[str, list[tuple[float, float] | None]] = {}
        out: list[tuple[float, float, str]] = []
        try:
            for tl_id in tr.trafficlight.getIDList():
                pos_list = self._tl_link_pos_cache.get(tl_id)
                if pos_list is None:
                    pos_list = []
                    try:
                        links = tr.trafficlight.getControlledLinks(tl_id)
                        for link in links:
                            if not link:
                                pos_list.append(None)
                                continue
                            in_lane = link[0][0]     # (inLane, outLane, viaLane)
                            try:
                                shape = self._net.getLane(in_lane).getShape()
                                x, y = shape[-1]
                                pos_list.append(
                                    tuple(map(float, self._net.convertXY2LonLat(x, y))))
                            except Exception:
                                pos_list.append(None)
                    except Exception:
                        pos_list = []
                    self._tl_link_pos_cache[tl_id] = pos_list

                if not pos_list:
                    continue
                state = str(tr.trafficlight.getRedYellowGreenState(tl_id))
                for i, ch in enumerate(state):
                    if i < len(pos_list) and pos_list[i] is not None:
                        lon, lat = pos_list[i]
                        out.append((lon, lat, ch))
        except Exception as exc:
            print(f"[sumo] tl_link_states: {exc}")
        return out

    def find_route_time(self, from_edge: str, to_edge: str) -> float | None:
        """Estimated travel time (s) between two edges via SUMO's router —
        no vehicle needs to be spawned."""
        if not self.ready:
            return None
        try:
            stage = self._traci.simulation.findRoute(from_edge, to_edge)
            tt = float(stage.travelTime)
            return tt if tt > 0 else None
        except Exception as exc:
            print(f"[sumo] find_route_time({from_edge!r}, {to_edge!r}): {exc}")
            return None

    # ── Analytics helpers (emissions / congestion / KPIs / incidents) ─────────

    def vehicle_emission_snapshot(self, pollutant: str = "nox") -> list[dict]:
        """Per-vehicle NATIVE emission rates from SUMO's HBEFA model.

        Returns [{"lon", "lat", "rate"}] with rate in g/s (SUMO reports mg/s).
        Unlike the speed-based EEA approximation, HBEFA accounts for
        acceleration — a car accelerating from a red light emits far more
        than one cruising at the same speed.
        """
        if not self.ready or self._net is None:
            return []
        tr = self._traci
        getter = {
            "co2": tr.vehicle.getCO2Emission,
            "nox": tr.vehicle.getNOxEmission,
            "pm":  tr.vehicle.getPMxEmission,
        }.get(pollutant)
        if getter is None:
            return []
        out: list[dict] = []
        try:
            for vid in tr.vehicle.getIDList():
                try:
                    x, y = tr.vehicle.getPosition(vid)
                    lon, lat = self._net.convertXY2LonLat(x, y)
                    out.append({
                        "lon":  float(lon),
                        "lat":  float(lat),
                        "rate": float(getter(vid)) / 1000.0,   # mg/s → g/s
                    })
                except Exception:
                    continue
        except Exception as exc:
            print(f"[sumo] vehicle_emission_snapshot: {exc}")
        return out

    def edge_congestion(self) -> dict[str, tuple[float, float, int]]:
        """Live per-edge congestion: edge_id → (mean_speed, freeflow_speed, n_veh).

        Only edges that currently carry vehicles are returned (cheap).
        Free-flow speeds are read from the net file once and cached.
        """
        if not self.ready or self._net is None:
            return {}
        tr = self._traci
        if not hasattr(self, "_ff_speed_cache"):
            self._ff_speed_cache: dict[str, float] = {}
        out: dict[str, tuple[float, float, int]] = {}
        try:
            for eid in tr.edge.getIDList():
                if eid.startswith(":"):
                    continue
                n = int(tr.edge.getLastStepVehicleNumber(eid))
                if n == 0:
                    continue
                ff = self._ff_speed_cache.get(eid)
                if ff is None:
                    try:
                        ff = float(self._net.getEdge(eid).getSpeed())
                    except Exception:
                        ff = 13.9
                    self._ff_speed_cache[eid] = ff
                out[eid] = (float(tr.edge.getLastStepMeanSpeed(eid)), ff, n)
        except Exception as exc:
            print(f"[sumo] edge_congestion: {exc}")
        return out

    def edge_shape_lonlat(self, eid: str) -> list[tuple[float, float]] | None:
        """Edge centerline as lon/lat points (cached — net geometry is static)."""
        if self._net is None:
            return None
        if not hasattr(self, "_edge_shape_cache"):
            self._edge_shape_cache: dict[str, list | None] = {}
        if eid in self._edge_shape_cache:
            return self._edge_shape_cache[eid]
        try:
            shape = self._net.getEdge(eid).getShape()
            pts = [tuple(map(float, self._net.convertXY2LonLat(x, y))) for x, y in shape]
            self._edge_shape_cache[eid] = pts if len(pts) >= 2 else None
        except Exception:
            self._edge_shape_cache[eid] = None
        return self._edge_shape_cache[eid]

    def kpi_snapshot(self) -> dict:
        """City-health KPIs for the HUD.

        mean_timeloss_s : mean per-vehicle delay vs free-flow driving (s)
        mean_waiting_s  : mean accumulated standstill time (s)
        running/arrived : current + total completed vehicle counts
        collisions      : vehicles currently colliding
        """
        if not self.ready:
            return {}
        tr = self._traci
        try:
            ids = list(tr.vehicle.getIDList())
            timeloss = waiting = 0.0
            for vid in ids:
                try:
                    timeloss += float(tr.vehicle.getTimeLoss(vid))
                    waiting  += float(tr.vehicle.getAccumulatedWaitingTime(vid))
                except Exception:
                    continue
            n = max(1, len(ids))
            self._kpi_arrived = getattr(self, "_kpi_arrived", 0) + int(
                tr.simulation.getArrivedNumber())
            return {
                "running":         len(ids),
                "arrived_total":   int(self._kpi_arrived),
                "mean_timeloss_s": timeloss / n,
                "mean_waiting_s":  waiting / n,
                "collisions":      len(tr.simulation.getCollidingVehiclesIDList()),
            }
        except Exception as exc:
            print(f"[sumo] kpi_snapshot: {exc}")
            return {}

    def incident_break_random_vehicle(self, duration_s: float = 60.0) -> str | None:
        """Force a random vehicle to a standstill (simulated breakdown).

        Returns the vehicle id, or None.  Call incident_release(vid) to let it
        drive again (setSpeed(-1) returns control to the car-following model).
        """
        if not self.ready:
            return None
        tr = self._traci
        try:
            ids = list(tr.vehicle.getIDList())
            if not ids:
                return None
            import random as _random
            vid = _random.choice(ids)
            tr.vehicle.setSpeed(vid, 0.0)
            print(f"[sumo] INCIDENT: vehicle {vid} broken down for {duration_s:.0f}s")
            return str(vid)
        except Exception as exc:
            print(f"[sumo] incident: {exc}")
            return None

    def incident_release(self, vid: str) -> bool:
        """End a breakdown — the vehicle resumes normal car-following."""
        if not self.ready:
            return False
        try:
            self._traci.vehicle.setSpeed(vid, -1.0)
            print(f"[sumo] incident cleared: vehicle {vid} moving again")
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self._traci is not None and not self._closed:
            try:
                self._traci.close()
            except Exception:
                pass
        self._closed = True
        self._ready = False
