from __future__ import annotations

import argparse
from typing import Any


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
    parser.add_argument("--pole-height", type=float, default=3.0, help="Streetlight pole height in meters.")
    parser.add_argument("--ground-resolution", type=int, default=80, help="Ground mesh resolution for test plane.")
    parser.add_argument(
        "--coverage-jobs",
        type=int,
        default=8,
        help="Workers for coverage matrix build (default: 8; set 0 for auto).",
    )
    parser.add_argument(
        "--ga-jobs",
        type=int,
        default=8,
        help="Worker threads for GA fitness evaluation (default: 8).",
    )
    parser.add_argument(
        "--ga-progress-every",
        type=int,
        default=5,
        help="Print GA progress every N generations.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory to store fast-startup cache artifacts.",
    )
    parser.add_argument(
        "--fast-startup",
        dest="fast_startup",
        action="store_true",
        help="Use cached OSM + cached coverage matrix + lower default ground resolution.",
    )
    parser.add_argument(
        "--no-fast-startup",
        dest="fast_startup",
        action="store_false",
        help="Disable fast startup caches and run full uncached setup.",
    )
    parser.set_defaults(gui=True, fast_startup=True)
    parser.add_argument(
        "--gui",
        dest="gui",
        action="store_true",
        help="Open a parameter input window before running (default: enabled).",
    )
    parser.add_argument(
        "--no-gui",
        dest="gui",
        action="store_false",
        help="Disable the startup GUI and use CLI args only.",
    )
    parser.add_argument(
        "--hide-roads",
        action="store_true",
        help="Do not render street graph overlays.",
    )
    parser.add_argument(
        "--optimize-on-open",
        action="store_true",
        help="Run GA optimization before opening the interactive viewer.",
    )
    return parser.parse_args()


def apply_gui_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Optional Tkinter form for parameter input before execution."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        print("GUI input unavailable (tkinter not found). Continuing with CLI args.")
        return args

    root = tk.Tk()
    root.title("Mini Motorways Lighting Setup")
    root.geometry("520x460")

    address_var = tk.StringVar(value=str(args.address))
    radius_var = tk.StringVar(value=str(args.radius))
    lights_var = tk.StringVar(value=str(args.n_lights))
    light_radius_var = tk.StringVar(value=str(args.light_radius))
    coverage_jobs_var = tk.StringVar(value=str(args.coverage_jobs))
    ga_jobs_var = tk.StringVar(value=str(args.ga_jobs))
    mode_var = tk.StringVar(value=str(args.mode))
    fast_var = tk.BooleanVar(value=bool(args.fast_startup))
    roads_var = tk.BooleanVar(value=not bool(args.hide_roads))

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    def _row(lbl: str, widget: Any, r: int) -> None:
        ttk.Label(frame, text=lbl).grid(row=r, column=0, sticky="w", padx=4, pady=5)
        widget.grid(row=r, column=1, sticky="ew", padx=4, pady=5)

    frame.columnconfigure(1, weight=1)
    _row("Address", ttk.Entry(frame, textvariable=address_var), 0)
    _row("Radius (m)", ttk.Entry(frame, textvariable=radius_var), 1)
    _row("Mode", ttk.Combobox(frame, textvariable=mode_var, values=["view", "shadow", "ga", "all"], state="readonly"), 2)
    _row("Streetlights (N)", ttk.Entry(frame, textvariable=lights_var), 3)
    _row("Spotlight Radius", ttk.Entry(frame, textvariable=light_radius_var), 4)
    _row("Coverage Jobs", ttk.Entry(frame, textvariable=coverage_jobs_var), 5)
    _row("GA Jobs", ttk.Entry(frame, textvariable=ga_jobs_var), 6)

    ttk.Checkbutton(frame, text="Fast Startup (cache + lower res)", variable=fast_var).grid(
        row=7, column=0, columnspan=2, sticky="w", padx=4, pady=4
    )
    ttk.Checkbutton(frame, text="Show Roads", variable=roads_var).grid(
        row=8, column=0, columnspan=2, sticky="w", padx=4, pady=4
    )

    result = {"ok": False}

    def _run() -> None:
        result["ok"] = True
        root.destroy()

    def _cancel() -> None:
        root.destroy()

    btns = ttk.Frame(frame)
    btns.grid(row=9, column=0, columnspan=2, sticky="e", pady=8)
    ttk.Button(btns, text="Cancel", command=_cancel).pack(side="right", padx=6)
    ttk.Button(btns, text="Run", command=_run).pack(side="right")

    root.mainloop()
    if not result["ok"]:
        return args

    try:
        args.address = address_var.get().strip() or args.address
        args.radius = float(radius_var.get())
        args.mode = mode_var.get().strip() or args.mode
        args.n_lights = int(lights_var.get())
        args.light_radius = float(light_radius_var.get())
        args.coverage_jobs = int(coverage_jobs_var.get())
        args.ga_jobs = int(ga_jobs_var.get())
        args.fast_startup = bool(fast_var.get())
        args.hide_roads = not bool(roads_var.get())
    except Exception as exc:
        print(f"Invalid GUI input ({exc}). Using previous arguments.")

    return args
