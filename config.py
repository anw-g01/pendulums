from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SystemParams:
    """Physical parameters for the Double Pendulum (DP) system."""
    g: float = 9.81,
    m1: float = 0.3,
    m2: float = 0.3,
    r1: float = 0.5,
    r2: float = 0.5,
    theta1_0: float = 0,
    theta2_0: float = 0,
    T: float = 10.0,
    steps: int = 1000,


@dataclass
class PlotConfig:
    """Configuration for plotting the Double Pendulum within a figure."""
    # figure, axes, and grids:
    figure_size: Tuple[int, int] = (10, 10)
    figure_title: str = "Double Pendulum Simulation"
    grid_alpha: float = 0.1
    n_max_ticks: int = 5
    origin: Tuple[float, float] = (0, 0)
    x_axis_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    y_axis_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    max_axis_extent: float = 1.1,
    display_legend: bool = True
    # markers, colours, and linestyles:
    m1_colour: str = "tab:green"
    m2_colour: str = "tab:red"
    m1_markersize: int = 150
    m2_markersize: int = 150
    link1_colour: str = "black"
    link2_colour: str = "black"
    link_linewidth: float = 0.75
    trail_linewidth: float = 0.8
    trail_length_pct: float = 5,
    draw_n_frames: int = 5