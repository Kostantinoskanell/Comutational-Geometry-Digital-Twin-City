"""scenario_mixin.py — UI wiring for the scenario-comparison planning tool.

Keys (registered in main_ast6.py):
    7  save current state as scenario A
    8  save current state as scenario B
    9  run headless A-vs-B comparison in SUMO
    0  toggle the 3-D travel-time diff overlay

Threading contract: SUMO builds/runs happen in daemon background threads;
plotter/VTK is touched ONLY from the main thread, via _scenario_poll() which
the main animation loop calls every ~250 ms.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pyvista as pv

_OVERLAY_NAME = "_scenario_overlay"
_SUMMARY_NAME = "scenario_summary"


class ScenarioMixin:

    # ── state ─────────────────────────────────────────────────────────────────

    def _init_scenarios(self) -> None:
        self.scenario_state = {"A": None, "B": None, "running": False,
                               "diff_actor": None, "diff_visible": False,
                               "results": None}

    def _scenario_dir(self, slot: str) -> Path:
        return Path(__file__).parent / "scenarios" / slot

    # ── overlay helpers (MAIN THREAD ONLY) ────────────────────────────────────

    def _scenario_show_overlay(self, msg: str) -> None:
        try:
            self.plotter.add_text(
                msg, name=_OVERLAY_NAME, position=(0.35, 0.5),
                viewport=True, font_size=12, color="#ffd54f")
        except Exception as exc:
            print(f"[scenario] overlay error: {exc}")

    def _scenario_clear_overlay(self) -> None:
        try:
            self.plotter.add_text(
                "", name=_OVERLAY_NAME, position=(0.35, 0.5),
                viewport=True, font_size=12, color="#ffd54f")
        except Exception as exc:
            print(f"[scenario] overlay error: {exc}")

    # ── save scenario (key 7 / 8) ─────────────────────────────────────────────

    def _scenario_save(self, slot: str) -> None:
        st = self.scenario_state
        if st.get("running"):
            print("[scenario] a scenario task is already running — please wait")
            return
        st["running"] = True
        st[f"building_{slot}"] = True
        self._scenario_show_overlay(f"Saving scenario {slot}…")

        def _worker():
            ok = False
            try:
                from scenario_compare import save_scenario
                ok = save_scenario(self, slot)
            except Exception as exc:
                print(f"[scenario] save {slot} failed: {exc}")
            finally:
                st[slot] = str(self._scenario_dir(slot)) if ok else None
                st[f"building_{slot}"] = False
                st["running"] = False
                st["_msg_done"] = True   # picked up by _scenario_poll (main thread)

        threading.Thread(target=_worker, daemon=True,
                         name=f"scenario-save-{slot}").start()

    # ── run comparison (key 9) ────────────────────────────────────────────────

    def _scenario_run_comparison(self) -> None:
        st = self.scenario_state
        if st.get("running"):
            print("[scenario] a scenario task is already running — please wait")
            return

        cfg_a = self._scenario_dir("A") / "scn.sumocfg"
        cfg_b = self._scenario_dir("B") / "scn.sumocfg"
        missing = [s for s, c in (("A", cfg_a), ("B", cfg_b)) if not c.exists()]
        if missing:
            print(f"[scenario] scenario(s) {', '.join(missing)} not built — "
                  f"press 7/8 to save them first")
            return

        st["running"] = True
        self._scenario_show_overlay("Running scenario comparison A vs B…")

        def _worker():
            try:
                import json
                from scenario_compare import run_headless, compare

                seed, end = 42, 900.0
                try:
                    with open(self._scenario_dir("A") / "scenario.json") as fh:
                        d = json.load(fh).get("demand", {})
                    seed = int(d.get("seed", seed))
                    end = float(d.get("end", end))
                except Exception:
                    pass

                res_a = run_headless(str(cfg_a), duration_s=end, seed=seed)
                if res_a is None:
                    print("[scenario] run A failed — comparison aborted")
                    return
                res_b = run_headless(str(cfg_b), duration_s=end, seed=seed)
                if res_b is None:
                    print("[scenario] run B failed — comparison aborted")
                    return

                st["results"] = compare(res_a, res_b)
                st["_render_diff_pending"] = True   # main thread renders
            except Exception as exc:
                print(f"[scenario] comparison failed: {exc}")
            finally:
                st["running"] = False
                st["_msg_done"] = True

        threading.Thread(target=_worker, daemon=True,
                         name="scenario-compare").start()

    # ── poll (called from the main animation loop) ────────────────────────────

    def _scenario_poll(self) -> None:
        st = getattr(self, "scenario_state", None)
        if not st:
            return
        if st.pop("_msg_done", False):
            self._scenario_clear_overlay()
        if st.pop("_render_diff_pending", False):
            self._scenario_render_diff()

    # ── 3-D diff overlay ──────────────────────────────────────────────────────

    @staticmethod
    def _diff_color(delta: float, dmax: float) -> np.ndarray:
        """Diverging blue → light grey → red, linear RGB interpolation."""
        blue = np.array([0x21, 0x66, 0xac], dtype=float)
        grey = np.array([0xf7, 0xf7, 0xf7], dtype=float)
        red = np.array([0xb2, 0x18, 0x2b], dtype=float)
        t = float(np.clip(delta, -dmax, dmax)) / dmax   # [-1, 1]
        if t < 0.0:
            c = blue + (grey - blue) * (t + 1.0)
        else:
            c = grey + (red - grey) * t
        return c.astype(np.uint8)

    def _scenario_render_diff(self) -> None:
        report = self.scenario_state.get("results")
        if not report:
            print("[scenario] no comparison results to render")
            return
        deltas = report.get("edge_tt_delta_s", {})
        if not deltas:
            print("[scenario] no per-edge deltas in report")
            return

        try:
            from scenario_compare import edge_ids_for_graph
            graph_edges = edge_ids_for_graph(self.street_graph)
        except Exception as exc:
            print(f"[scenario] edge-id mapping failed: {exc}")
            return

        # Match SUMO result edge ids to graph edge ids (netconvert may split
        # edges: "123#0" → "123#0.25" etc.) — exact match, else longest
        # graph id that prefixes the result id. Aggregate mean delta per edge.
        graph_ids = [ge[0] for ge in graph_edges]
        gid_set = set(graph_ids)
        gid_sorted = sorted(graph_ids, key=len, reverse=True)
        per_edge: dict[str, list[float]] = {}
        for rid, dv in deltas.items():
            if rid in gid_set:
                per_edge.setdefault(rid, []).append(float(dv))
                continue
            for gid in gid_sorted:
                if rid.startswith(gid):
                    per_edge.setdefault(gid, []).append(float(dv))
                    break
        if not per_edge:
            print("[scenario] no SUMO edges matched the graph — diff not drawn")
            return
        edge_delta = {gid: float(np.mean(vs)) for gid, vs in per_edge.items()}

        abs_d = np.abs(np.array(list(edge_delta.values()), dtype=float))
        dmax = max(float(np.percentile(abs_d, 95)), 1.0)

        # Build one PolyData with per-point RGB
        z = 1.2
        all_pts: list[np.ndarray] = []
        all_lines: list[int] = []
        all_rgb: list[np.ndarray] = []
        offset = 0
        for edge_id, u, v, key, data in graph_edges:
            dv = edge_delta.get(edge_id)
            if dv is None:
                continue
            geom = data.get("geometry")
            if geom is not None and hasattr(geom, "coords"):
                xy = np.asarray([[c[0], c[1]] for c in geom.coords], dtype=float)
            else:
                try:
                    nu, nv = self.street_graph.nodes[u], self.street_graph.nodes[v]
                    xy = np.array([[nu["x"], nu["y"]], [nv["x"], nv["y"]]],
                                  dtype=float)
                except Exception:
                    continue
            if xy.shape[0] < 2:
                continue
            n = xy.shape[0]
            pts = np.column_stack([xy, np.full(n, z)])
            all_pts.append(pts)
            all_lines.extend([n] + list(range(offset, offset + n)))
            all_rgb.append(np.tile(self._diff_color(dv, dmax), (n, 1)))
            offset += n

        if not all_pts:
            print("[scenario] no drawable edges in diff")
            return

        try:
            poly = pv.PolyData()
            poly.points = np.vstack(all_pts)
            poly.lines = np.asarray(all_lines, dtype=np.int64)
            poly["rgb"] = np.vstack(all_rgb).astype(np.uint8)

            # Replace any previous diff overlay
            try:
                self.plotter.remove_actor("scenario_diff")
            except Exception:
                pass
            actor = self.plotter.add_mesh(
                poly, scalars="rgb", rgb=True, line_width=6,
                name="scenario_diff", lighting=False)
            self.scenario_state["diff_actor"] = actor
            self.scenario_state["diff_visible"] = True
        except Exception as exc:
            print(f"[scenario] diff overlay render failed: {exc}")
            return

        # Summary panel
        try:
            s = report.get("summary", {})

            def _pctf(v):
                return f"{v:+.1f}%" if v is not None else "n/a"

            pct = s.get("emissions_delta_pct", {})
            txt = (
                "Scenario diff (B - A)\n"
                f"mean edge tt: {s.get('mean_edge_tt_delta_s', 0.0):+.2f} s\n"
                f"CO2 {_pctf(pct.get('co2'))}   NOx {_pctf(pct.get('nox'))}   "
                f"PM {_pctf(pct.get('pm'))}\n"
                f"noise >65dB: {s.get('noise_exceed_delta_cell_s', 0.0):+.0f} cell·s\n"
                f"arrived: A={s.get('vehicles_arrived_a', 0)}  "
                f"B={s.get('vehicles_arrived_b', 0)}\n"
                "blue = faster in B, red = slower in B   [0] toggle"
            )
            self.scenario_state["_summary_text"] = txt
            self.plotter.add_text(
                txt, name=_SUMMARY_NAME, position=(0.02, 0.70),
                viewport=True, font_size=9, color="#ffffff")
        except Exception as exc:
            print(f"[scenario] summary panel error: {exc}")

        print(f"[scenario] diff overlay rendered "
              f"({len(edge_delta)} edges, ±{dmax:.1f} s scale)")

    # ── toggle (key 0) ────────────────────────────────────────────────────────

    def _scenario_toggle_diff(self) -> None:
        st = self.scenario_state
        actor = st.get("diff_actor")
        if actor is None:
            print("[scenario] no diff overlay yet — run a comparison first (key 9)")
            return
        visible = not st.get("diff_visible", False)
        st["diff_visible"] = visible
        try:
            actor.SetVisibility(visible)
            self.plotter.add_text(
                st.get("_summary_text", "") if visible else "",
                name=_SUMMARY_NAME, position=(0.02, 0.70),
                viewport=True, font_size=9, color="#ffffff")
            self.plotter.render()
        except Exception as exc:
            print(f"[scenario] toggle error: {exc}")
        print(f"[scenario] diff overlay {'shown' if visible else 'hidden'}")
