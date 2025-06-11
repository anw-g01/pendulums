from dataclasses import dataclass
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
    """Configuration for plotting the Double Pendulum within a figure."""
    # figure, axes, and grids:
    figure_size: Tuple[int, int] = (10, 10)
    figure_title: str = "Double Pendulum Simulation"
    grid_alpha: float = 0.2
    n_max_ticks: int = 5
    origin: Tuple[float, float] = (0, 0)
    x_axis_limits: Optional[Tuple[float, float]] = (None, None)
    y_axis_limits: Optional[Tuple[float, float]] = (None, None)
    max_axis_extent: float = 1.1
    display_legend: bool = True
    in_degrees: bool = True     # angles in degrees or radians
    # markers, colours, and linestyles:
    m1_colour: str = "tab:green"
    m2_colour: str = "tab:red"
    m1_markersize: int = 200
    m2_markersize: int = 200
    link1_colour: str = "black"
    link2_colour: str = "black"
    link_linewidth: float = 0.75
    draw_n_frames: int = 5    # for the static plot
    # energy plots:
    ke_colour: str = "tab:blue"
    pe_colour: str = "tab:orange"
    te_colour: str = "tab:purple"
    # animation
    trail_length_pct: float = 5
    trail_linewidth: float = 0.8