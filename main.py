from __future__ import annotations

import argparse
import math

import numpy as np
import pyvista as pv

from osm_3d_buildings import build_3d_buildings_and_street_graph
from shadow_engine import compute_shadows
from spatial_trees import OctreeNode
from streetlight_ga import optimize_streetlights

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run building, shadow, and streetlight GA tests.")
    parser.add_argument(
        "--address",
        default="Patras, Greece",
        help="Address to geocode and query in OpenStreetMap.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=150.0,
        help="Search radius in meters.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=10.0,
        help="Extrusion height in meters.",
    )
    parser.add_argument(
        "--mode",
        choices=["view", "shadow", "ga", "all"],
        default="view",
        help="Execution mode.",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Skip opening the PyVista interactive window.",
    )
    parser.add_argument(
        "--sun-dir",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 2.0],
        metavar=("SX", "SY", "SZ"),
        help="Sun direction vector for shadow mode.",
    )
    parser.add_argument("--n-lights", type=int, default=12, help="Number of streetlights for GA mode.")
    parser.add_argument("--light-radius", type=float, default=40.0, help="Streetlight illumination radius.")
    parser.add_argument("--w1", type=float, default=1.0, help="Weight for dark area term.")
    parser.add_argument("--w2", type=float, default=0.5, help="Weight for double-lit area term.")
    parser.add_argument("--grid-step", type=float, default=10.0, help="Grid spacing for candidate light positions.")
    parser.add_argument("--population", type=int, default=48, help="GA population size.")
    parser.add_argument("--generations", type=int, default=40, help="GA generation count.")
    parser.add_argument("--mutation", type=float, default=0.15, help="GA mutation rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible GA runs.")
    parser.add_argument(
        "--optimize-on-open",
        action="store_true",
        help="Run GA optimization before opening the interactive viewer.",
    )
    return parser.parse_args()


def _mesh_triangles(mesh: pv.PolyData) -> np.ndarray:
    tri = mesh.triangulate()
    if tri.n_cells == 0:
        return np.empty((0, 3, 3), dtype=float)

    faces = tri.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangle-only faces after triangulation.")

    pts = np.asarray(tri.points, dtype=float)
    return pts[faces[:, 1:4]]


def _build_octree_from_buildings(buildings_mesh: pv.PolyData) -> OctreeNode:
    triangles = _mesh_triangles(buildings_mesh)
    if triangles.shape[0] == 0:
        raise ValueError("Building mesh has no triangles to insert in octree.")

    tri_min = np.min(triangles.reshape(-1, 3), axis=0)
    tri_max = np.max(triangles.reshape(-1, 3), axis=0)

    # Expand bounds slightly to avoid numeric edge cases at box boundaries.
    pad = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    root = OctreeNode(aabb_min=tri_min - pad, aabb_max=tri_max + pad)
    root.insert_many(triangles)
    return root


def _build_ground_mesh_for_tests(buildings_mesh: pv.PolyData) -> pv.PolyData:
    if buildings_mesh.n_points == 0:
        raise ValueError("Cannot build ground test mesh without building points.")

    xmin, xmax, ymin, ymax, _, _ = buildings_mesh.bounds
    dx = xmax - xmin
    dy = ymax - ymin
    margin = max(5.0, 0.15 * max(dx, dy, 1.0))

    center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 0.0)
    plane = pv.Plane(
        center=center,
        i_size=(xmax - xmin) + 2.0 * margin,
        j_size=(ymax - ymin) + 2.0 * margin,
        i_resolution=80,
        j_resolution=80,
    )
    return plane.triangulate()


def _sun_dir_from_hour(hour: float) -> np.ndarray:
    """Map local hour [0, 24] to a sun direction vector."""
    hour = float(np.clip(hour, 0.0, 24.0))

    # Elevation is positive between roughly 06:00 and 18:00.
    elevation = math.sin(math.pi * (hour - 6.0) / 12.0)
    if elevation <= 0.0:
        return np.array([0.0, 0.0, -1.0], dtype=float)

    azimuth = 2.0 * math.pi * (hour - 6.0) / 24.0
    xy_mag = math.sqrt(max(0.0, 1.0 - elevation * elevation))

    return np.array(
        [
            math.cos(azimuth) * xy_mag,
            math.sin(azimuth) * xy_mag,
            elevation,
        ],
        dtype=float,
    )


def _build_spotlight_discs(positions_xy: np.ndarray, radius: float, z: float = 0.08) -> pv.PolyData | None:
    if positions_xy.size == 0:
        return None

    merged: pv.PolyData | None = None
    for xy in positions_xy:
        disc = pv.Disc(center=(float(xy[0]), float(xy[1]), z), inner=0.0, outer=float(radius), c_res=48)
        merged = disc if merged is None else merged.merge(disc)

    return merged


def _initial_light_positions(ground_mesh: pv.PolyData, n_lights: int, seed: int) -> np.ndarray:
    pts = np.asarray(ground_mesh.points, dtype=float)
    xy = np.unique(pts[:, :2], axis=0)
    if xy.shape[0] == 0:
        raise ValueError("Ground mesh has no points for initial light placement.")

    rng = np.random.default_rng(seed)
    count = max(1, int(n_lights))
    if xy.shape[0] >= count:
        idx = rng.choice(xy.shape[0], size=count, replace=False)
    else:
        idx = rng.choice(xy.shape[0], size=count, replace=True)

    return np.asarray(xy[idx], dtype=float)


def main() -> None:
    args = parse_args()
    ground_mesh: pv.PolyData | None = None
    shadow_mask: np.ndarray | None = None
    best_positions: np.ndarray | None = None
    octree_root: OctreeNode | None = None

    print(f"Downloading OSM data for: {args.address}")
    try:
        buildings_mesh, street_graph = build_3d_buildings_and_street_graph(
            address=args.address,
            radius=args.radius,
            extrusion_height=args.height,
        )
    except Exception as exc:
        print(f"Failed to fetch/build geometry: {exc}")
        return

    print(
        "Street graph loaded: "
        f"{street_graph.number_of_nodes()} nodes, {street_graph.number_of_edges()} edges"
    )

    if buildings_mesh.n_points == 0:
        print("No buildings found in this area. Try a larger radius or a denser location.")
        return

    run_precompute = args.no_view or args.optimize_on_open

    if (args.mode == "shadow" or args.mode == "all") and run_precompute:
        try:
            ground_mesh = _build_ground_mesh_for_tests(buildings_mesh)
            octree_root = _build_octree_from_buildings(buildings_mesh)
            shadow_mask, lit_ratio = compute_shadows(
                ground_mesh=ground_mesh,
                octree_root=octree_root,
                sun_dir=np.asarray(args.sun_dir, dtype=float),
            )
            shadowed = int(np.count_nonzero(shadow_mask))
            total = int(shadow_mask.size)
            print(f"Shadow test complete: {shadowed}/{total} ground triangles shadowed")
            print(f"Lit area ratio: {lit_ratio:.6f}")
        except Exception as exc:
            print(f"Shadow test failed: {exc}")

    if (args.mode == "ga" or args.mode == "all") and run_precompute:
        try:
            if ground_mesh is None:
                ground_mesh = _build_ground_mesh_for_tests(buildings_mesh)
            ga_result = optimize_streetlights(
                ground_mesh=ground_mesh,
                n_lights=args.n_lights,
                light_radius=args.light_radius,
                w1=args.w1,
                w2=args.w2,
                grid_step=args.grid_step,
                population_size=args.population,
                generations=args.generations,
                mutation_rate=args.mutation,
                seed=args.seed,
            )
            best_positions = np.asarray(ga_result["best_positions"], dtype=float)
            print(f"GA test complete: best_cost={ga_result['best_cost']:.6f}")
            print(f"GA lit ratio: {ga_result['lit_ratio']:.6f}")
            print("Best light coordinates (x, y):")
            for row in best_positions:
                print(f"  {row[0]:.3f}, {row[1]:.3f}")
        except Exception as exc:
            print(f"GA test failed: {exc}")

    if args.mode != "view" and args.mode != "all":
        return

    if args.no_view:
        print(
            "Mesh generated successfully: "
            f"{buildings_mesh.n_points} points, {buildings_mesh.n_cells} cells"
        )
        return

    print("Opening 3D viewer...")
    try:
        if ground_mesh is None:
            ground_mesh = _build_ground_mesh_for_tests(buildings_mesh)
        if octree_root is None:
            octree_root = _build_octree_from_buildings(buildings_mesh)
        if best_positions is None:
            best_positions = _initial_light_positions(ground_mesh, args.n_lights, args.seed)

        plotter = pv.Plotter()
        plotter.add_mesh(
            buildings_mesh,
            color="white",
            show_edges=True,
            edge_color="black",
            opacity=0.9,
            label="Buildings",
        )

        scene_state: dict[str, object] = {
            "hour": 12.0,
            "spot_radius": float(args.light_radius),
            "ground_actor": None,
            "spotlight_actor": None,
            "is_night": False,
        }

        light_points = np.column_stack(
            (best_positions[:, 0], best_positions[:, 1], np.full(best_positions.shape[0], 0.5))
        )
        plotter.add_points(
            light_points,
            color="#f4d35e",
            point_size=16,
            render_points_as_spheres=True,
        )

        # Fast initial render so the window appears immediately.
        scene_state["ground_actor"] = plotter.add_mesh(
            ground_mesh,
            color="#f4f1de",
            show_edges=False,
            opacity=0.72,
        )
        initial_discs = _build_spotlight_discs(best_positions, float(args.light_radius))
        if initial_discs is not None:
            scene_state["spotlight_actor"] = plotter.add_mesh(
                initial_discs,
                color="#f6bd60",
                opacity=0.22,
                show_edges=False,
            )
        plotter.add_text("Move 'Hour' slider to compute shadows", name="status", font_size=10)

        def _render_ground(hour: float, spot_radius: float) -> None:
            prev_ground = scene_state.get("ground_actor")
            prev_spot = scene_state.get("spotlight_actor")
            if prev_ground is not None:
                plotter.remove_actor(prev_ground, reset_camera=False)
            if prev_spot is not None:
                plotter.remove_actor(prev_spot, reset_camera=False)

            sun_dir = _sun_dir_from_hour(hour)
            is_night = bool(sun_dir[2] <= 0.0)
            scene_state["is_night"] = is_night

            if is_night:
                night_ground = ground_mesh.copy()
                night_ground.cell_data["night"] = np.ones(night_ground.n_cells, dtype=np.uint8)
                scene_state["ground_actor"] = plotter.add_mesh(
                    night_ground,
                    color="#1c2230",
                    show_edges=False,
                    opacity=0.9,
                )
                plotter.set_background("#05070f")
            else:
                mask, lit_ratio = compute_shadows(
                    ground_mesh=ground_mesh,
                    octree_root=octree_root,
                    sun_dir=sun_dir,
                )
                shaded = ground_mesh.copy()
                shaded.cell_data["shadow"] = mask.astype(np.uint8)
                scene_state["ground_actor"] = plotter.add_mesh(
                    shaded,
                    scalars="shadow",
                    clim=[0, 1],
                    cmap=["#f4f1de", "#3d405b"],
                    show_edges=False,
                    opacity=0.72,
                )
                plotter.set_background("#e8ebf0")
                plotter.add_text(f"Hour {hour:04.1f}  |  Lit Ratio {lit_ratio:.3f}", name="status", font_size=10)

            discs = _build_spotlight_discs(best_positions, spot_radius)
            if discs is not None:
                scene_state["spotlight_actor"] = plotter.add_mesh(
                    discs,
                    color="#f6bd60",
                    opacity=0.55 if is_night else 0.22,
                    show_edges=False,
                )

            if is_night:
                plotter.add_text(f"Hour {hour:04.1f}  |  Night Mode", name="status", font_size=10)

            plotter.render()

        def _on_time_change(value: float) -> None:
            scene_state["hour"] = float(value)
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        def _on_radius_change(value: float) -> None:
            scene_state["spot_radius"] = float(value)
            _render_ground(float(scene_state["hour"]), float(scene_state["spot_radius"]))

        plotter.add_slider_widget(
            _on_time_change,
            rng=[0.0, 24.0],
            value=12.0,
            title="Hour",
            pointa=(0.025, 0.10),
            pointb=(0.31, 0.10),
            style="modern",
            interaction_event="end",
        )
        plotter.add_slider_widget(
            _on_radius_change,
            rng=[max(5.0, args.light_radius * 0.4), args.light_radius * 3.0],
            value=float(args.light_radius),
            title="Spotlight Radius",
            pointa=(0.36, 0.10),
            pointb=(0.68, 0.10),
            style="modern",
            interaction_event="end",
        )

        if args.optimize_on_open:
            ga_result = optimize_streetlights(
                ground_mesh=ground_mesh,
                n_lights=args.n_lights,
                light_radius=args.light_radius,
                w1=args.w1,
                w2=args.w2,
                grid_step=args.grid_step,
                population_size=args.population,
                generations=args.generations,
                mutation_rate=args.mutation,
                seed=args.seed,
            )
            best_positions = np.asarray(ga_result["best_positions"], dtype=float)
            _render_ground(hour=12.0, spot_radius=float(args.light_radius))

        plotter.camera.ParallelProjectionOn()
        plotter.view_isometric()
        plotter.show()
    except Exception as exc:
        print(f"Visualization failed: {exc}")

if __name__ == "__main__":
    main()