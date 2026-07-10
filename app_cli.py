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
        default=250.0,
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
    parser.add_argument("--n-cars", type=int, default=-1, help="Number of animated cars (-1 for auto-density).")
    parser.add_argument("--n-peds", type=int, default=-1, help="Number of pedestrian agents (-1 for auto, 50–200).")
    parser.add_argument("--n-cyclists", type=int, default=-1, help="Number of cyclist agents (-1 for auto, 10–60).")
    parser.add_argument("--n-parked-cars", type=int, default=-1,
                        help="Parked cars placed in OSM parking lots (-1 = auto, ≈ buildings × 0.25).")
    parser.add_argument("--gtfs", type=str, default="", help="Path to GTFS directory or .zip archive for bus simulation.")
    parser.add_argument("--n-buses", type=int, default=-1, help="Max bus agents from GTFS (-1 = up to 30).")
    parser.add_argument("--gtfs-rt-url", type=str, default="", help="GTFS-Realtime VehiclePositions feed URL for LIVE bus positions.")
    parser.add_argument("--gtfs-rt-key", type=str, default="", help="API key for the GTFS-RT feed (sent as Authorization/api_key header).")
    parser.add_argument("--gtfs-rt-key-param", type=str, default="", help="If set, send the API key as this query parameter instead of a header.")
    parser.add_argument("--gtfs-rt-interval", type=float, default=15.0, help="Seconds between GTFS-RT feed polls (min 2).")
    parser.add_argument("--sumo-cfg", type=str, default="", help="Path to a .sumocfg to enable SUMO co-simulation (SUMO drives vehicles, app renders).")
    parser.add_argument("--sumo-net", type=str, default="", help="Explicit SUMO .net.xml (else read from the .sumocfg). Must be geo-referenced.")
    parser.add_argument("--sumo-binary", type=str, default="sumo", help="SUMO binary name or path (default 'sumo').")
    parser.add_argument("--sumo-gui", action="store_true", help="Launch sumo-gui alongside (shows SUMO's own window too).")
    parser.add_argument("--sumo-step", type=float, default=0.1, help="SUMO simulation step length in seconds (default 0.1).")
    parser.add_argument("--sumo-port", type=int, default=0, help="TraCI port (0 = let traci pick one).")
    parser.add_argument(
        "--engine",
        choices=["idm", "sumo"],
        default="idm",
        help=(
            "Traffic engine: 'idm' (built-in Intelligent Driver Model, default) or 'sumo' "
            "(SUMO co-simulation as the primary traffic source). In sumo mode, all overlays "
            "(heatmap, noise, AQ) read from the SUMO snapshot. If --sumo-cfg is not given, "
            "the scene is built automatically from --address/--radius and cached for reuse. "
            "Degrades gracefully to idm with a warning if SUMO deps are missing."
        ),
    )
    parser.add_argument("--validate-od", type=int, default=0, help="Validate model travel times against a routing engine over N random O-D pairs (0 = off).")
    parser.add_argument("--validate-engine", type=str, default="osrm", help="Reference routing engine for --validate-od (currently 'osrm', no API key).")
    parser.add_argument("--validate-osrm-host", type=str, default="https://router.project-osrm.org", help="OSRM host for travel-time validation.")
    parser.add_argument("--validate-congested", metavar="N", type=int, default=0,
                       help="Compare simulated congested travel times vs OSRM over N O-D pairs (requires active traffic)")
    parser.add_argument("--validate-engines", metavar="N", type=int, default=0,
                       help="Compare IDM vs SUMO travel times over N shared O-D pairs (requires --engine sumo)")
    parser.add_argument("--solo", action="store_true", help="Run with a single solar car and display live energy telemetry.")
    parser.add_argument(
        "--car-detail",
        choices=["ultra", "low"],
        default="ultra",
        help="Car rendering detail: ultra (point sprites) or low (cuboids).",
    )
    parser.add_argument(
        "--traffic-speed",
        type=float,
        default=1.0,
        help="Traffic speed multiplier for animated cars (1.0 = default).",
    )
    parser.add_argument(
        "--debug-cars",
        action="store_true",
        help="Print car loading/runtime debug logs to terminal.",
    )
    parser.add_argument(
        "--solar-fleet",
        action="store_true",
        help="Use only solarcar.obj for all traffic; enable solar routing UI.",
    )
    parser.add_argument("--light-radius", type=float, default=40.0, help="Streetlight illumination radius.")
    parser.add_argument(
        "--light-strategy",
        choices=["smart", "ga"],
        default="smart",
        help="Streetlight placement strategy: smart greedy spacing or genetic algorithm.",
    )
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
    parser.add_argument(
        "--show-cars",
        dest="show_cars",
        action="store_true",
        default=True,
        help="Show animated car traffic in the viewer (default: enabled).",
    )
    parser.add_argument(
        "--hide-cars",
        dest="show_cars",
        action="store_false",
        help="Hide animated car traffic in the viewer.",
    )
    parser.add_argument(
        "--data-source",
        choices=["overture", "osm"],
        default="overture",
        help="Data source for buildings and street graph (default: overture).",
    )
    parser.add_argument(
        "--no-dem",
        dest="use_dem",
        action="store_false",
        default=True,
        help="Skip Copernicus DEM terrain fetch (faster startup, flat ground).",
    )
    parser.add_argument(
        "--no-ms-buildings",
        dest="use_ms_buildings",
        action="store_false",
        default=True,
        help="Skip Microsoft Building Footprints height fetch (faster startup, OSM heights only).",
    )
    # ── Profiling ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Enable profiling mode: run for --profile-duration seconds, record per-frame "
            "timings split into idm_sim / vtk_actors / render / overlay / event_pump, "
            "print a summary table, and save profile_report.json. "
            "Use --n-cars to set the car count (default 200 when --profile is active)."
        ),
    )
    parser.add_argument(
        "--profile-duration",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Seconds to run before auto-exiting in --profile mode (default: 60).",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="profile_report.json",
        metavar="PATH",
        help="Output path for the JSON profile report (default: profile_report.json).",
    )
    return parser.parse_args()


def apply_gui_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Styled Tkinter parameter form for City Digital Twin."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        print("GUI input unavailable (tkinter not found). Continuing with CLI args.")
        return args

    # ── Palette ───────────────────────────────────────────────────────────────
    BG         = "#1e2130"
    PANEL      = "#262c3f"
    ACCENT     = "#4fc3f7"
    TEXT       = "#e8eaf6"
    SUBTEXT    = "#90caf9"
    BTN_RUN    = "#00bfa5"
    BTN_CANCEL = "#546e7a"
    FIELD_BG   = "#2e3450"
    TROUGH     = "#3a4060"

    root = tk.Tk()
    root.title("City Digital Twin")
    root.geometry("600x580")
    root.minsize(540, 520)
    root.configure(bg=BG)

    # ── Theme & Styles ────────────────────────────────────────────────────────
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("BG.TFrame",          background=BG)
    style.configure("TFrame",             background=PANEL)
    style.configure("TLabel",             background=PANEL, foreground=TEXT,    font=("Helvetica", 10))
    style.configure("Sub.TLabel",         background=PANEL, foreground=SUBTEXT, font=("Helvetica", 9))
    style.configure("Info.TLabel",        background=BG,    foreground=SUBTEXT, font=("Helvetica", 9))
    style.configure("TEntry",             fieldbackground=FIELD_BG, foreground=TEXT, insertcolor=TEXT,
                                          bordercolor=ACCENT, lightcolor=PANEL, darkcolor=PANEL)
    style.configure("TCombobox",          fieldbackground=FIELD_BG, foreground=TEXT,
                                          selectbackground=ACCENT,  selectforeground=BG,
                                          bordercolor=ACCENT, arrowcolor=ACCENT)
    style.map("TCombobox",                fieldbackground=[("readonly", FIELD_BG)],
                                          foreground=[("readonly", TEXT)])
    style.configure("TCheckbutton",       background=PANEL, foreground=TEXT)
    style.map("TCheckbutton",             background=[("active", PANEL)], foreground=[("active", ACCENT)])
    style.configure("TRadiobutton",       background=PANEL, foreground=TEXT)
    style.map("TRadiobutton",             background=[("active", PANEL)], foreground=[("active", ACCENT)])
    style.configure("Horizontal.TScale",  background=PANEL, troughcolor=TROUGH,
                                          sliderlength=18,   sliderrelief="flat")
    style.map("Horizontal.TScale",        background=[("active", ACCENT)])
    style.configure("TNotebook",          background=BG, tabmargins=[2, 4, 2, 0])
    style.configure("TNotebook.Tab",      background=PANEL, foreground=SUBTEXT,
                                          padding=[10, 4], font=("Helvetica", 10, "bold"))
    style.map("TNotebook.Tab",            background=[("selected", ACCENT)],
                                          foreground=[("selected", BG)])
    style.configure("TLabelframe",        background=PANEL, foreground=ACCENT, bordercolor=ACCENT)
    style.configure("TLabelframe.Label",  background=PANEL, foreground=ACCENT,
                                          font=("Helvetica", 9, "bold"))
    style.configure("Run.TButton",        background=BTN_RUN,    foreground=BG,
                                          font=("Helvetica", 10, "bold"), borderwidth=0, padding=[14, 6])
    style.map("Run.TButton",              background=[("active", "#26a69a")])
    style.configure("Cancel.TButton",     background=BTN_CANCEL, foreground=TEXT,
                                          font=("Helvetica", 10),        borderwidth=0, padding=[14, 6])
    style.map("Cancel.TButton",           background=[("active", "#607d8b")])

    # ── Variables ─────────────────────────────────────────────────────────────
    address_var      = tk.StringVar(value=str(args.address))
    radius_var       = tk.StringVar(value=str(int(float(args.radius))))
    mode_var         = tk.StringVar(value=str(args.mode))
    data_source_var  = tk.StringVar(value=str(getattr(args, "data_source", "overture")))
    fast_var         = tk.BooleanVar(value=bool(args.fast_startup))
    optimize_var     = tk.BooleanVar(value=bool(getattr(args, "optimize_on_open", False)))

    cars_var         = tk.StringVar(value=str(int(args.n_cars)))
    car_detail_var   = tk.StringVar(value=str(args.car_detail))
    traffic_var      = tk.DoubleVar(value=float(args.traffic_speed))
    roads_var        = tk.BooleanVar(value=not bool(args.hide_roads))
    show_cars_var    = tk.BooleanVar(value=bool(getattr(args, "show_cars", True)))
    solar_fleet_var  = tk.BooleanVar(value=bool(getattr(args, "solar_fleet", False)))

    lights_var       = tk.StringVar(value=str(int(args.n_lights)))
    light_radius_var = tk.StringVar(value=str(int(float(args.light_radius))))
    light_strategy_var = tk.StringVar(value=str(getattr(args, "light_strategy", "smart")))
    cov_jobs_var     = tk.StringVar(value=str(int(args.coverage_jobs)))
    ga_jobs_var      = tk.StringVar(value=str(int(args.ga_jobs)))

    # ── Entry row helper ──────────────────────────────────────────────────────
    def _entry_row(
        parent: "ttk.Frame",
        label: str,
        var: "tk.Variable",
        hint: str,
        row: int,
    ) -> None:
        """Render a label | entry | hint-label row."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 4), pady=5)
        ttk.Entry(parent, textvariable=var, font=("Helvetica", 10), width=10).grid(
            row=row, column=1, sticky="ew", padx=4, pady=5
        )
        ttk.Label(parent, text=hint, style="Sub.TLabel").grid(
            row=row, column=2, sticky="w", padx=(2, 8), pady=5
        )

    # ── Outer shell ───────────────────────────────────────────────────────────
    outer = ttk.Frame(root, style="BG.TFrame", padding=10)
    outer.pack(fill="both", expand=True)

    notebook = ttk.Notebook(outer)
    notebook.pack(fill="both", expand=True, pady=(0, 6))

    # ── TAB 1: Location & Source ──────────────────────────────────────────────
    tab1 = ttk.Frame(notebook, padding=10)
    tab1.columnconfigure(1, weight=1)
    notebook.add(tab1, text="\U0001f4cd Location & Source")

    ttk.Label(tab1, text="Address").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=5)
    ttk.Entry(tab1, textvariable=address_var, font=("Helvetica", 10)).grid(
        row=0, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=5
    )
    ttk.Label(tab1, text="Radius (m)").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=5)
    ttk.Entry(tab1, textvariable=radius_var, font=("Helvetica", 10)).grid(
        row=1, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=5
    )
    ttk.Label(tab1, text="Mode").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=5)
    ttk.Combobox(
        tab1, textvariable=mode_var, values=["view", "shadow", "ga", "all"], state="readonly"
    ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=5)

    src_frame = ttk.LabelFrame(tab1, text="Map data source", padding=8)
    src_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
    ttk.Radiobutton(src_frame, text="\u25c9  Overture Maps",    variable=data_source_var, value="overture").pack(side="left", padx=14)
    ttk.Radiobutton(src_frame, text="\u25ef  OpenStreetMap",    variable=data_source_var, value="osm").pack(side="left", padx=14)

    ttk.Checkbutton(tab1, text="Fast Startup  (cache + lower res)",         variable=fast_var).grid(
        row=4, column=0, columnspan=3, sticky="w", padx=10, pady=3)
    ttk.Checkbutton(tab1, text="Optimize on Open  (run GA before viewer)",  variable=optimize_var).grid(
        row=5, column=0, columnspan=3, sticky="w", padx=10, pady=3)

    # ── TAB 2: Traffic ────────────────────────────────────────────────────────
    tab2 = ttk.Frame(notebook, padding=10)
    tab2.columnconfigure(1, weight=1)
    notebook.add(tab2, text="\U0001f697 Traffic")

    _entry_row(tab2, "Cars (N)",          cars_var,    "0 – 60",   row=0)
    ttk.Label(tab2, text="Car Detail").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=5)
    ttk.Combobox(
        tab2, textvariable=car_detail_var, values=["ultra", "low"], state="readonly"
    ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=5)

    solar_frame = ttk.LabelFrame(tab2, text="Solar car fleet", padding=(10, 8))
    solar_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 10))
    solar_frame.columnconfigure(0, weight=1)
    ttk.Checkbutton(
        solar_frame,
        text="All cars use solarcar.obj",
        variable=solar_fleet_var,
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(
        solar_frame,
        text="Turns on solar routing, harvest, and energy sliders in the 3D viewer.",
        style="Sub.TLabel",
        wraplength=480,
    ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    ttk.Label(tab2, text="Traffic Speed (\u00d7)").grid(
        row=3, column=0, sticky="w", padx=(8, 4), pady=(8, 2))
    traffic_val_label = ttk.Label(tab2, text=f"{traffic_var.get():.1f}", style="Sub.TLabel")
    traffic_val_label.grid(row=3, column=2, sticky="e", padx=(2, 8), pady=(8, 2))

    def _on_traffic_scale(*_: object) -> None:
        traffic_val_label.config(text=f"{traffic_var.get():.1f}")

    traffic_scale = ttk.Scale(
        tab2,
        from_=0.0,
        to=3.0,
        orient="horizontal",
        variable=traffic_var,
        command=lambda _v: _on_traffic_scale(),
    )
    traffic_scale.grid(row=4, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 8))
    ttk.Label(tab2, text="0 = paused", style="Sub.TLabel").grid(
        row=5, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 4))

    ttk.Checkbutton(tab2, text="Show Roads", variable=roads_var).grid(
        row=6, column=0, columnspan=3, sticky="w", padx=10, pady=3)
    ttk.Checkbutton(tab2, text="Show Cars",  variable=show_cars_var).grid(
        row=7, column=0, columnspan=3, sticky="w", padx=10, pady=3)

    # ── TAB 3: Lighting ───────────────────────────────────────────────────────
    tab3 = ttk.Frame(notebook, padding=10)
    tab3.columnconfigure(1, weight=1)
    notebook.add(tab3, text="\U0001f4a1 Lighting")

    _entry_row(tab3, "Streetlights (N)",  lights_var,       "2 – 40",    row=0)
    _entry_row(tab3, "Spotlight Radius",   light_radius_var, "10 – 100 m", row=1)
    ttk.Label(tab3, text="Placement").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=5)
    ttk.Combobox(
        tab3,
        textvariable=light_strategy_var,
        values=["smart", "ga"],
        state="readonly",
    ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=5)
    _entry_row(tab3, "Coverage Jobs",      cov_jobs_var,     "1 – 16",     row=3)
    _entry_row(tab3, "GA Jobs",            ga_jobs_var,      "1 – 16",     row=4)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    bar = ttk.Frame(outer, style="BG.TFrame", padding=(8, 4))
    bar.pack(fill="x", side="bottom")
    bar.columnconfigure(0, weight=1)

    info_label = ttk.Label(bar, style="Info.TLabel")
    info_label.grid(row=0, column=0, sticky="w")

    def _on_source_change(*_: object) -> None:
        if data_source_var.get() == "overture":
            info_label.config(text="\u2139  Overture: validated buildings + streets from cloud")
        else:
            info_label.config(text="\u2139  OSM: community data via OSMnx (requires internet)")

    data_source_var.trace_add("write", _on_source_change)
    _on_source_change()

    result = {"ok": False}

    def _run() -> None:
        result["ok"] = True
        root.destroy()

    def _cancel() -> None:
        root.destroy()

    btn_frame = ttk.Frame(bar, style="BG.TFrame")
    btn_frame.grid(row=0, column=1, sticky="e")
    ttk.Button(btn_frame, text="Cancel",    style="Cancel.TButton", command=_cancel).pack(side="left", padx=(0, 6))
    ttk.Button(btn_frame, text="\u25b6  Run", style="Run.TButton",    command=_run).pack(side="left")

    root.mainloop()
    if not result["ok"]:
        return args

    try:
        args.address          = address_var.get().strip() or args.address
        args.radius           = float(radius_var.get())
        args.mode             = mode_var.get().strip() or args.mode
        args.data_source      = data_source_var.get()
        args.fast_startup     = bool(fast_var.get())
        args.optimize_on_open = bool(optimize_var.get())
        args.n_cars           = int(cars_var.get())
        args.car_detail       = car_detail_var.get().strip() or args.car_detail
        args.traffic_speed    = round(float(traffic_var.get()), 2)
        args.hide_roads       = not bool(roads_var.get())
        args.show_cars        = bool(show_cars_var.get())
        args.solar_fleet      = bool(solar_fleet_var.get())
        args.n_lights         = int(lights_var.get())
        args.light_radius     = float(light_radius_var.get())
        args.light_strategy   = light_strategy_var.get().strip() or args.light_strategy
        args.coverage_jobs    = int(cov_jobs_var.get())
        args.ga_jobs          = int(ga_jobs_var.get())
    except Exception as exc:
        print(f"Invalid GUI input ({exc}). Using previous arguments.")

    return args
