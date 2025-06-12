import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter
import datetime
from tqdm_pbar import tqdmFA
from config import DPSystemParams, DPConfig
from typing import Tuple, Optional
from ode_systems import dp_ode
# configure matplotlib defaults
plt.rcParams.update({
    "font.size": 8,
    "font.family": "monospace",
    "lines.linewidth": 0.8
})


class DoublePendulum:

    def __init__(self, params: Optional[DPSystemParams] = None, config: Optional[DPConfig] = None) -> None:
        self.params = params if params else DPSystemParams()
        self.config = config if config else DPConfig()
        self.t = None
        self.Z = None
        self._solve()    # automatically solve the system upon initialisation

    def _solve(self) -> None:
        """Private method to solve the ODE system for the double pendulum."""
        func = dp_ode(self.params)    # get the ODE system function f(t, Z)
        # --- INITIAL CONDITIONS --- #
        p = self.params    # shorthand alias for quick reference
        theta1_0, theta2_0 = np.radians(p.theta1_0), np.radians(p.theta2_0)
        w1_0, w2_0 = np.radians(p.w1_0), np.radians(p.w2_0)
        Z0 = [theta1_0, theta2_0, w1_0, w2_0]    # initial conditions (state vector)
        # --- EVALUATION AND SOLVE --- #
        T, steps = p.T_secs, p.steps
        method, rtol, atol = p.ode_method, p.rtol, p.atol
        t_span = (0, T)
        t_eval = np.linspace(0, T, steps)
        dt = (t_span[1] - t_span[0]) / (steps - 1)    # rough estimate of time step (not used in solve_ivp)
        print(f"\nrunning ODE solver ({method=})...")
        print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]}), {steps=:,}, dt≈{dt:.2f}")
        sol = solve_ivp(func, t_span, Z0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
        # --- EXTRACT OUTPUTS --- #
        t, Z = sol.t, sol.y
        success = sol.success
        print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
        print(f"t.shape: {t.shape}, Z.shape: {Z.shape}\n")
        self.t, self.Z = t, Z

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Public interface to solve or re-solve the system and return the solution."""
        if self.t is None or self.Z is None:
            self._solve()
        return self.t, self.Z   # return existing cached solution if available
    
    def _check_solved(self) -> None:
        """Check if the system has been solved, raise an error if not."""
        if self.t is None or self.Z is None:
            print(f"\nEmpty solutions, re-running ODE solver...")
            self.t, self.Z = self.solve()    # call the solve method to ensure t and Z are populated

    def plot_dynamics(self, xlim_timespan: bool = True) -> None:
        """Plot θ(t), ω(t), and E(t) for a Double Pendulum (DP) system."""
        
        self._check_solved()    # check if the system has been solved
        t, Z = self.t, self.Z
        # unpack parameters and config
        p, cf = self.params, self.config
        r1, r2, m1, m2, g = p.r1, p.r2, p.m1, p.m2, p.g
        theta1, theta2, w1, w2 = Z[0], Z[1], Z[2], Z[3]    # state variables
        if cf.in_degrees:    # convert angles to degrees if required
            # don't modify the original arrays, as they will be used for energy calculations (in radians)
            theta1_plot, theta2_plot = np.degrees(theta1), np.degrees(theta2)   
            w1_plot, w2_plot = np.degrees(w1), np.degrees(w2)
        
        # ----- FIGURE SETUP ----- #
        fig, ax = plt.subplots(nrows=3, figsize=cf.figure_size, sharex=True)

        # ----- ANGULAR DISPLACEMENT, θ(t) ----- #
        ax[0].set_ylabel("angular displacement, " + (r"$\theta$ ($^{\circ}$)" if cf.in_degrees else r"$\theta$ (rad)"))
        ax[0].plot(t, theta1_plot, color=cf.m1_colour, label=r"$\theta_1$")
        ax[0].plot(t, theta2_plot, color=cf.m2_colour, label=r"$\theta_2$")

        # ----- ANGULAR VELOCITY, ω(t) ----- #
        ax[1].set_ylabel("angular velocity, " + (r"$\omega$ ($^{\circ}/s$)" if cf.in_degrees else r"$\omega$ (rad/s)"))
        ax[1].plot(t, w1_plot, color=cf.m1_colour, label=r"$\omega_1$")
        ax[1].plot(t, w2_plot, color=cf.m2_colour, label=r"$\omega_2$")

        # ----- SYSTEM ENERGY, E(t) ----- #
        # cartesian velocity components
        x1_dot = r1*w1*np.cos(theta1)
        x2_dot = x1_dot + r2*w2*np.cos(theta2)
        y1_dot = r1*w1*np.sin(theta1)
        y2_dot = y1_dot + r2*w2*np.sin(theta2)
        # kinetic energy, KE
        T = 0.5*m1*(x1_dot**2 + y1_dot**2) + 0.5*m2*(x2_dot**2 + y2_dot**2)
        # potential energy, PE
        y1 = -r1*np.cos(theta1)
        y2 = y1 - r2*np.cos(theta2)
        V = m1*g*y1 + m2*g*y2
        # total energy
        E = T + V
        ax[2].set_xlabel(r"time, $t$ ($s$)")
        ax[2].set_ylabel(r"system energy, $E$ ($J$)")
        ax[2].plot(t, T, color=cf.ke_colour, label=r"$E_k$")
        ax[2].plot(t, V, color=cf.pe_colour, label=r"$E_p$")
        ax[2].plot(t, E, color=cf.te_colour, label=r"$E_T$")

        # configure for each subfigure:
        for axis in ax:
            axis.grid(True, alpha=cf.grid_alpha)
            axis.yaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
            if cf.display_legend:
                axis.legend(loc="upper right")
            if xlim_timespan:
                axis.set_xlim(t[0], t[-1])    

        fig.tight_layout()
        plt.show()

    def _setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """Returns the base figure and axes used for the Double Pendulum plot and animation."""
        cf = self.config
        # setup main figure and axes
        fig, ax = plt.subplots(figsize=cf.figure_size)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)")
        ax.xaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
        # grids and dashed lines
        ax.grid(True, alpha=cf.grid_alpha)
        hline = ax.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        # origin pivot point marker
        x0, y0 = cf.origin      # origin coordinates (pivot point)
        ax.scatter(x0, y0, marker="o", color=cf.origin_colour, s=cf.origin_markersize, zorder=1)  # origin pivot
        # set axis limits
        # independent overrides (if only one set of limits is provided):
        L = self.params.r1 + self.params.r2     # maximum oustretched length of the pendulum links
        axis_extent = cf.max_axis_extent * L    # same on all sides
        x_extent = cf.x_axis_limits if cf.x_axis_limits is not None else (-axis_extent, axis_extent)
        y_extent = cf.y_axis_limits if cf.y_axis_limits is not None else (-axis_extent, axis_extent)
        ax.set_xlim(x_extent), ax.set_ylim(y_extent)
        return fig, ax

    def plot_frames(
        self,
        display_mode: list[str] = ["start", "end"],
        show_trail: bool = True,
        trail_length_pct: float = 100,
        num_frames: int = 5
    ) -> None:
        """Plot STATIC Double Pendulum frames using the configured parameters."""
        p, cf = self.params, self.config
        t, Z = self.t, self.Z
        r1, r2, theta1, theta2 = p.r1, p.r2, Z[0], Z[1]
        # array of coordinates of masses (bobs):
        x0, y0 = cf.origin
        x1, y1 = x0 + r1*np.sin(theta1), y0 - r1*np.cos(theta1)
        x2, y2 = x1 + r2*np.sin(theta2), y1 - r2*np.cos(theta2)
        # setup base figure and axes
        fig, ax = self._setup_figure()
        trail_length = int((trail_length_pct / 100) * len(t)) if trail_length_pct > 0 else 0
        def draw(ax: plt.Axes, i: int, show_trail: bool = False, alpha: float = 1.0) -> None:
            """Draws the double pendulum at step index `i` with adjustable transparency."""
            i = len(t) - 1 if i == -1 else i
            # plot links
            ax.plot([x0, x1[i]], [y0, y1[i]], color=cf.link1_colour, linewidth=cf.link_linewidth, alpha=alpha, zorder=1)    
            ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color=cf.link2_colour, linewidth=cf.link_linewidth, alpha=alpha, zorder=2)
            # plot masses (bobs):
            ax.scatter(x1[i], y1[i], marker="o", color=cf.m1_colour, s=cf.m1_markersize, edgecolors="none", alpha=alpha, zorder=3)               
            ax.scatter(x2[i], y2[i], marker="o", color=cf.m2_colour, s=cf.m2_markersize, edgecolors="none", alpha=alpha, zorder=4)
            if show_trail:
                i0 = max(0, i - trail_length)
                ax.plot(x1[i0:i + 1], y1[i0:i + 1], color=cf.m1_colour, linewidth=cf.trail_linewidth, alpha=alpha, zorder=0)
                ax.plot(x2[i0:i + 1], y2[i0:i + 1], color=cf.m2_colour, linewidth=cf.trail_linewidth, alpha=alpha, zorder=0)
        # draw first frame with low opacity:
        if "start" in display_mode: draw(ax, i=0, alpha=0.4)
        # draw last frame (default: with full trail):
        if "end" in display_mode: draw(ax, i=-1, show_trail=show_trail)  
        # draw all frames with increasing opacity:
        if "fade" in display_mode:
            step = max(1, len(t) // num_frames)   # equally spaced frames (to draw)
            indices = list(range(0, len(t), step))
            n = len(indices)
            for idx, i in enumerate(indices):
                # opacity fades from 0.1 to 1.0 linearly across frames
                alpha = 0.1 + 0.9 * (idx / (n - 1)) if n > 1 else 1.0
                show_trail_this_frame = (i == indices[-1]) and show_trail    # bool: only show trail on the last frame
                draw(ax, i=i, alpha=alpha, show_trail=show_trail_this_frame)
        plt.show()


def animate_dp(
        t, Z, p,
        frames_per_second: int = 60,
        dots_per_inch: int = 200,
        m1_colour: str = "tab:green",
        m2_colour: str = "tab:red",
        link1_colour: str = "black",
        link2_colour: str = "black",
        m1_markersize: float = 150,
        m2_markersize: float = 150,
        line_width: float = 0.65,
        trail_linewidth: float = 1,
        trail_length_pct: float = 5,
        origin: tuple = (0, 0),
        figure_size: tuple = (8, 7),
        grid_alpha: float = 0.1,
        n_max_ticks: int = 4,
        x_axis_limits: tuple = None,
        y_axis_limits: tuple = None,
        max_axis_extent: float = 1.2,
        display_legend: bool = False
    ) -> None:
    """
    Animate the double pendulum using matplotlib's FuncAnimation.
    """
    # check first and last frames + figure size before animatino writing process:
    print(f"<CLOSE FIGURE WINDOW TO START ANIMATION WRITING>")
    # plot_frames(
    #     t, Z, p,
    #     m1_colour=m1_colour, m2_colour=m2_colour,
    #     link1_colour=link1_colour, link2_colour=link2_colour,
    #     m1_markersize=m1_markersize, m2_markersize=m2_markersize,
    #     line_width=line_width, figure_size=figure_size,
    #     grid_alpha=grid_alpha, n_max_ticks=n_max_ticks,
    #     x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits,
    #     max_axis_extent=max_axis_extent, display_legend=display_legend,
    #     show_trail=True,    
    #     trail_length_pct=100,   # show full trails from start to finish
    #     draw_start=True,        # draw first frame
    #     draw_end=True,          # draw last frame
    # )

    interval = int(1000 / frames_per_second)    # convert FPS to milliseconds
    steps = len(t)    # total number of time steps
    trail_length = int((trail_length_pct / 100) * steps) 
    print(f"\n{steps:,} steps @ {frames_per_second} fps (~{interval * 1e-3:.3f} sec/frame)")
    print(f"writing {steps} frames to MP4...\n")

    r1, r2, m1, m2, g, T, *_ = p
    theta1, theta2, w1, w2 = Z[0], Z[1], Z[2], Z[3]

    # array of coordinates of masses (bobs):
    x0, y0 = origin
    x1 = r1 * np.sin(theta1) + x0
    x2 = x1 + r2 * np.sin(theta2)
    y1 = - r1 * np.cos(theta1) + y0
    y2 = y1 - r2 * np.cos(theta2)

    # ----- FIGURE ----- #
    fig, ax = plt.subplots(figsize=figure_size)
    ax.xaxis.set_major_locator(MaxNLocator(n_max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(n_max_ticks))
    ax.set_xlabel(r"$x$ ($m$)")
    ax.set_ylabel(r"$y$ ($m$)")
    ax.set_aspect("equal")
    # grids and dashed lines
    ax.grid(True, alpha=grid_alpha)
    dashed_alpha, dashed_linewidth = 0.1, 1
    hline = ax.axhline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth)
    vline = ax.axvline(0, linestyle="--", color="black", alpha=dashed_alpha, linewidth=dashed_linewidth)
    hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
    ax.scatter(x0, y0, marker="o", color="black", s=20)    # origin pivot

    # plot elements to be updated
    link1, = ax.plot([], [], color=link1_colour, linewidth=line_width, zorder=1)
    link2, = ax.plot([], [], color=link2_colour, linewidth=line_width, zorder=2)
    mass1 = ax.scatter([], [], marker="o", color=m1_colour, s=m1_markersize, zorder=3)
    mass2 = ax.scatter([], [], marker="o", color=m2_colour, s=m2_markersize, zorder=3)
    if trail_length_pct > 0:    # show trails for masses
        m1_trail, = ax.plot([], [], color=m1_colour, linewidth=trail_linewidth, zorder=0)
        m2_trail, = ax.plot([], [], color=m2_colour, linewidth=trail_linewidth, zorder=0)

    # --- AXIS LIMITS & LEGEND --- #
    # independent overrides (if only one set of limits is provided):
    if x_axis_limits:
        ax.set_xlim(x_axis_limits)
    else:
        x_extent = max_axis_extent * np.max(np.abs(x2))
        ax.set_xlim(-x_extent, x_extent)
    if y_axis_limits:
        ax.set_ylim(y_axis_limits)
    else:
        y_extent = max_axis_extent * np.max(np.abs(y2))
        ax.set_ylim(-y_extent, y_extent)
    if display_legend:
        ax.legend(loc="best")

    # ----- PROGRESS BAR ----- #
    pbar = tqdmFA(total=len(t))
    
    # --- ANIMATION FUNCTION SETUP --- #
    def init() -> tuple:
        link1.set_data([], []), link2.set_data([], [])
        mass1.set_offsets([x1[0], y1[0]]), mass2.set_offsets([x2[0], y2[0]])
        if trail_length_pct > 0:
            m1_trail.set_data([], []), m2_trail.set_data([], [])
            return link1, link2, mass1, mass2, m1_trail, m2_trail
        return link1, link2, mass1, mass2

    def update(frame) -> tuple:
        # update link and mass positions:
        link1.set_data([x0, x1[frame]], [y0, y1[frame]])
        link2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        mass1.set_offsets([x1[frame], y1[frame]])
        mass2.set_offsets([x2[frame], y2[frame]])
        # update trails if applicable:
        if trail_length_pct > 0:
            i0 = max(0, frame - trail_length)   # start index for the trail
            m1_trail.set_data(x1[i0: frame + 1], y1[i0: frame + 1])
            m2_trail.set_data(x2[i0: frame + 1], y2[i0: frame + 1])
            pbar.update(1)
            return link1, link2, mass1, mass2, m1_trail, m2_trail
        pbar.update(1)    # else
        return link1, link2, mass1, mass2

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=range(len(t)),
        init_func=init,
        interval=interval,
        repeat=False,
        blit=True,
    )

    writer = FFMpegWriter(
        fps=frames_per_second,
        bitrate=30_000,
    )

    angle1, angle2 = np.degrees(theta1[0]), np.degrees(theta2[1])
    file_name = (
        "dp_sim_"
        f"θ1={angle1:.0f}_θ2={angle2:.0f}_"
        f"{steps=:,}({T=})_@{trail_length_pct}%_"
        f"dpi={dots_per_inch}_"
        f"{m1=:.1f}_{m2=:.1f}_"
        f"{r1=:.1f}_{r2=:.1f}_"
        ".mp4"
    )

    ani.save(filename=file_name, writer=writer, dpi=dots_per_inch)
    pbar.close()    # close progress bar
    print(f"\nanimation saved as '{file_name}.mp4'")

    # --- REPORT --- #
    elapsed = int(pbar.format_dict["elapsed"])
    t = datetime.timedelta(seconds=elapsed)
    print(f"\ntotal elapsed time: {t}")

    avg_iter_per_sec = steps / t.total_seconds()
    if 1 / avg_iter_per_sec < 1:
        avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
    else:
        avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
    print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")

    return None


def dp1() -> None:
    """Example usage of the DoublePendulum class to plot dynamics."""
    dp = DoublePendulum(
        params=DPSystemParams(
            theta1_0=55,  
            theta2_0=130,  
            T_secs=10,              # simulation time in seconds
            steps=600,              # number of steps in the simulation
            ode_method="RK45",      # ODE solver method
            rtol=1e-6,              # relative tolerance for the ODE solver
        ),
        config=DPConfig(
            figure_size=(10, 10),       # size of the figure
            x_axis_limits=(-0.9, 0.9),  # x-axis limits
            y_axis_limits=(-0.9, 0.2),  # y-axis limits
            trail_length_pct=5,         # length of the trail as a percentage of the total steps
        )
    )

    # dp.plot_dynamics()    # Plot θ(t), ω(t), and E(t)

    dp.plot_frames()
    dp.plot_frames(display_mode=["end"], trail_length_pct=5)