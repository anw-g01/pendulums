from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DPSystemParams:
    """Physical parameters for the Double Pendulum (DP) system."""
    g: float = 9.81
    m1: float = 0.2
    m2: float = 0.2
    r1: float = 0.4
    r2: float = 0.4
    theta1_0: float = 0
    theta2_0: float = 0
    w1_0: float = 0
    w2_0: float = 0
    T_secs: float = 10.0
    steps: int = 1000
    ode_method: str = "RK45"
    rtol: float = 1e-3
    atol: float = 1e-6


@dataclass
class DPConfig:
    """All configuration parameters for the Double Pendulum."""
    # figure properties:
    figure_size: Tuple[int, int] = (8, 8)
    figure_title: str = None
    dashboard_figure_size: Tuple[int, int] = (16, 9)
    dashboard_figure_title: str = None
    grid_alpha: float = 0.2
    dashed_line_alpha: float = 0.15
    dashed_line_width: float = 0.8
    display_legend: bool = True
    legend_location: str = "upper right"
    # axis configuration:
    n_max_ticks: int = 3
    x_axis_limits: Optional[Tuple[float, float]] = None
    y_axis_limits: Optional[Tuple[float, float]] = None
    max_axis_extent: float = 1.2
    in_degrees: bool = True         # angles in degrees or radians
    xlim_timespan: bool = True      # whether to limit x-axis to the time span (for the graphs plot)
    # markers, colours, and linestyles:
    m1_colour: str = "tab:green"
    m2_colour: str = "tab:red"
    m1_markersize: int = 300
    m2_markersize: int = 300
    link1_colour: str = "black"
    link2_colour: str = "black"
    link_linewidth: float = 0.8
    origin: Tuple[float, float] = (0, 0)
    origin_colour: str = "black"
    origin_markersize: int = 12
    draw_n_frames: int = 5    # for the static plot
    # energy plots:
    ke_colour: str = "tab:cyan"
    pe_colour: str = "tab:purple"
    te_colour: str = "tab:olive"
    # time series plots parameters:
    display_mode: list = field(default_factory=lambda: ["start", "end"])
    show_trail: bool = True
    num_frames: int = 5
    view_trail_length_pct: float = 100     # view on the static plot only
    # animation parameters:
    fps: int = 60
    bitrate: int = 50_000
    trail_length_pct: float = 5
    trail_linewidth: float = 0.75
    trail_alpha: float = 0.75
    video_duration: float = 10.0