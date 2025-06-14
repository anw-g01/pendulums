import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
import datetime
from tqdm_pbar import tqdmFA
from config import DPSystemParams, DPConfig
from typing import Tuple, Optional
from ode_systems import dp_ode
# configure matplotlib defaults
plt.rcParams.update({
    "font.size": 10,
    "font.family": "Latin Modern Roman",
    "mathtext.fontset": "cm",
    "lines.linewidth": 1,
    "xtick.labelsize": 8,    # smaller ticker markers
    "ytick.labelsize": 8,
})


class DP:
    """Class to visualise and simulate a Double Pendulum (DP) system."""

    def __init__(
        self,
        params: Optional[DPSystemParams] = None, 
        config: Optional[DPConfig] = None
    ) -> None:
        self.params = params if params else DPSystemParams()
        self.config = config if config else DPConfig()
        self.file_prefix = "dp_sim_"    # prefix for the output files
        self.t = None       # time vector
        self.theta1 = None  # angle array of the first mass (bob) in radians
        self.theta2 = None  # angle array of the second mass (bob) in radians
        self.w1 = None      # angular velocity array of the first mass (bob)
        self.w2 = None      # angular velocity array of the second mass (bob)
        self.x1 = None      # x-coordinate array of the first mass (bob)
        self.y1 = None      # y-coordinate array of the first mass (bob)
        self.x2 = None      # x-coordinate array of the second mass (bob)
        self.y2 = None      # y-coordinate array of the second mass (bob)
        self.T = None       # kinetic energy array
        self.V = None       # potential energy array
        self.E = None       # total energy array
        # automatically solve the system upon initialisation and populate all attributes:
        self._solve()       
        self.steps = len(self.t)    # number of time steps
        # attributes for animation writing - see _animation_params():
        self.used_steps = None      # number of frames to write to MP4
        self.trail_length = None    # trail length of masses in frames
        self.interval = None        # seconds per frame (1000/fps milliseconds)

    def _solve(self) -> None:
        """Private method to solve the ODE system for the double pendulum."""
        # get the ODE system function f(t, Z):
        func = dp_ode(self.params)    
        
        # --- INITIAL CONDITIONS --- #
        p = self.params    # shorthand alias for quick reference
        theta1_0, theta2_0 = np.radians(p.theta1_0), np.radians(p.theta2_0)
        w1_0, w2_0 = np.radians(p.w1_0), np.radians(p.w2_0)
        Z0 = [theta1_0, theta2_0, w1_0, w2_0]    # initial conditions (state vector)
        
        # --- EVALUATION AND SOLVE --- #
        T, time_steps = p.T_secs, p.steps
        method, rtol, atol = p.ode_method, p.rtol, p.atol
        t_span = (0, T)
        t_eval = np.linspace(0, T, time_steps)
        dt = (t_span[1] - t_span[0]) / (time_steps - 1)    # rough estimate of time step (not used in solve_ivp)
        print(f"\nrunning ODE solver ({method=})...")
        print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]}), {time_steps=:,}, dt≈{dt:.2f}")
        sol = solve_ivp(func, t_span, Z0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
        
        # --- EXTRACT OUTPUTS --- #
        t, Z = sol.t, sol.y
        success = sol.success
        print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
        print(f"t.shape: {t.shape}, Z.shape: {Z.shape}")
        self.t = t
        # state vector Z contains angles and angular velocities:
        self.theta1, self.theta2 = Z[0], Z[1]    # angles in radians
        self.w1, self.w2 = Z[2], Z[3]            # angular velocities in radians/sec

        # calculate coordinates of the double pendulum masses (bobs):
        self._coordinates() 

        # calculate energies after solving the system:
        self._system_energies()    
    
    def _coordinates(self) -> None:
        """Calculate the coordinates of the double pendulum masses (bobs) over the computed time span based on the angles and lengths."""
        p = self.params
        r1, r2 = p.r1, p.r2
        x0, y0 = self.config.origin    # origin coordinates (pivot point)
        self.x1 = x0 + r1*np.sin(self.theta1)
        self.y1 = y0 - r1*np.cos(self.theta1)
        self.x2 = self.x1 + r2*np.sin(self.theta2)
        self.y2 = self.y1 - r2*np.cos(self.theta2)

    def _system_energies(self) -> None:
        """Calculate the kinetic, potential, and total energy of the double pendulum system over the time span."""
        print(f"\ncalculating system energies (E_k, E_p, E_T)...")
        p = self.params
        r1, r2, m1, m2, g = p.r1, p.r2, p.m1, p.m2, p.g

        # cartesian velocity components:
        x1_dot = r1*self.w1*np.cos(self.theta1)
        x2_dot = x1_dot + r2*self.w2*np.cos(self.theta2)
        y1_dot = r1*self.w1*np.sin(self.theta1)
        y2_dot = y1_dot + r2*self.w2*np.sin(self.theta2)

        # kinetic energy, KE or E_k:
        self.T = 0.5*m1*(x1_dot**2 + y1_dot**2) + 0.5*m2*(x2_dot**2 + y2_dot**2)

        # potential energy, PE or E_p:
        self.V = m1*g*self.y1 + m2*g*self.y2

        # total system energy, E or E_T:
        self.E = self.T + self.V
        print(f"T.shape={self.T.shape}, V.shape={self.V.shape}, E.shape={self.E.shape}")

    def _wrap_angles(self, angles: Tuple[np.ndarray, np.ndarray], verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Wrap angles to the range [-π, π] and handle discontinuities (large angle jumps due to full arm link rotations) for better visualisation."""
        # wrap angles to [-π, π] range:
        theta1 = np.mod(angles[0] + np.pi, 2*np.pi) - np.pi    # to wrap [0, 2π] use np.mod(θ, 2*np.pi)
        theta2 = np.mod(angles[1] + np.pi, 2*np.pi) - np.pi
        # find indices just before any angle jumps (only likely for the second angle):
        diff1, diff2 = np.diff(theta1), np.diff(theta2)
        threshold = np.pi
        jumps1 = np.where(np.abs(diff1) > threshold)[0]    # first index to access the array
        jumps2 = np.where(np.abs(diff2) > threshold)[0]
        if verbose:
            print(f"\nlarge angle jumps: θ1: {len(jumps1)}, θ2: {len(jumps2)}")
        if len(jumps1) == 0 and len(jumps2) == 0:
            return theta1, theta2
        # make copies for modification:
        theta1_plot, theta2_plot = theta1.copy(), theta2.copy()
        # handle theta1 and theta2 jumps:
        if len(jumps1) > 0:
            theta1_plot[jumps1 + 1] = np.nan    # set the point AFTER each jump to NaN
        if len(jumps2) > 0:
            theta2_plot[jumps2 + 1] = np.nan    
        return theta1_plot, theta2_plot
   
    def get_y_extent(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[float, float]:
        """Calculate y-axis limits of timeseries plots based on the maximum absolute values of two arrays."""
        y_min = np.nanmin(np.concatenate((arr1, arr2))) * 1.1
        y_max = np.nanmax(np.concatenate((arr1, arr2))) * 1.1
        return (y_min, y_max)
    
    def _plot_graphs(self, fig: plt.Figure, gs: GridSpec) -> Tuple[plt.Axes, ...]:
        """Helper method to populate the time series graphs of θ(t), ω(t), and E(t) into the input figure and GridSpec."""
        cf = self.config

        # wrap angles to handle discontinuities and prepare for plotting:
        theta1_plot, theta2_plot = self._wrap_angles((self.theta1, self.theta2), verbose=True)
        if cf.in_degrees:    
            theta1_plot, theta2_plot = np.degrees(theta1_plot), np.degrees(theta2_plot)   
            w1_plot, w2_plot = np.degrees(self.w1), np.degrees(self.w2)

        # ----- FIGURE SETUP ----- #
        ax0 = fig.add_subplot(gs[0, 0])                 # angular displacement
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)     # angular velocity
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)     # system energy
        # remove ticks for the first two subplots:
        ax0.tick_params(labelbottom=False)    
        ax1.tick_params(labelbottom=False) 

        # ----- ANGULAR DISPLACEMENT, θ(t) ----- #
        ax0.set_ylabel(r"angular displacement, " + (r"$\theta$ ($^{\circ}$)" if cf.in_degrees else r"$\theta$ ($\text{rad}$)"))
        ax0.plot(self.t, theta1_plot, color=cf.m1_colour, label=r"$\theta_1$")
        ax0.plot(self.t, theta2_plot, color=cf.m2_colour, label=r"$\theta_2$")
        ax0.set_ylim(self.get_y_extent(theta1_plot, theta2_plot))    # set y-axis limits based on the maximum absolute values

        # ----- ANGULAR VELOCITY, ω(t) ----- #
        ax1.set_ylabel(r"angular velocity, " + (r"$\omega$ ($^{\circ}/s$)" if cf.in_degrees else r"$\omega$ ($\text{rad}$/s"))
        ax1.plot(self.t, w1_plot, color=cf.m1_colour, label=r"$\omega_1$")
        ax1.plot(self.t, w2_plot, color=cf.m2_colour, label=r"$\omega_2$")
        ax1.set_ylim(self.get_y_extent(w1_plot, w2_plot))    # set y-axis limits based on the maximum absolute values
        
        # ----- SYSTEM ENERGY, E(t) ----- #
        ax2.set_xlabel(r"time, $t$ ($s$)")
        ax2.set_ylabel(r"system energy, $E$ ($J$)")
        ax2.plot(self.t, self.T, color=cf.ke_colour, label=r"$E_k$")
        ax2.plot(self.t, self.V, color=cf.pe_colour, label=r"$E_p$")
        ax2.plot(self.t, self.E, color=cf.te_colour, label=r"$E_T$", linewidth=plt.rcParams["lines.linewidth"] + 0.5)
        ax2.set_ylim(self.get_y_extent(self.T, self.V))    # set y-axis limits based on the maximum absolute value of E

        # configurations for each subfigure:
        for axis in (ax0, ax1, ax2):
            axis.grid(True, alpha=cf.grid_alpha)
            hline = axis.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width, zorder=0)
            hline.set_dashes([10, 10])    # dashed horizontal line at y=0
            axis.yaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
            if cf.display_legend:
                axis.legend(loc=cf.legend_location)
            if cf.xlim_timespan:
                axis.set_xlim(self.t[0], self.t[-1])

        return ax0, ax1, ax2

    def plot_graphs(self) -> Tuple[plt.Figure, GridSpec, Tuple[plt.Axes, ...]]:
        """Plot θ(t), ω(t), and E(t) for a Double Pendulum (DP) system."""
        fig = plt.figure(figsize=self.config.figure_size)
        gs = GridSpec(nrows=3, ncols=1, figure=fig)
        ax0, ax1, ax2 = self._plot_graphs(fig, gs)    # call helper method to populate axes
        fig.tight_layout()
        plt.show()
        return fig, gs, (ax0, ax1, ax2)

    def _setup_figure(self, fig: plt.Figure, gs: GridSpec) -> plt.Axes:
        """Helper function to populate the base figure and axes used for the Double Pendulum plot and animation."""
        cf = self.config
        ax = fig.add_subplot(gs[0, 0])
        ax.set_xlabel(r"$x$ ($m$)")
        ax.set_ylabel(r"$y$ ($m$)", labelpad=-16)
        ax.xaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
        ax.grid(True, alpha=cf.grid_alpha)
        
        # dashed cross lines:
        hline = ax.axhline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width)
        vline = ax.axvline(0, linestyle="--", color="black", alpha=cf.dashed_line_alpha, linewidth=cf.dashed_line_width)
        hline.set_dashes([10, 10]), vline.set_dashes([10, 10])
        
        # origin pivot point marker
        x0, y0 = cf.origin      # origin coordinates (pivot point)
        ax.scatter(x0, y0, marker="o", color=cf.origin_colour, s=cf.origin_markersize, zorder=1)  # origin pivot
        # set axis limits and aspect ratio:
        L = self.params.r1 + self.params.r2         # maximum oustretched length of the pendulum links
        axis_extent = cf.max_axis_extent * L        # same on all sides
        x_extent = cf.x_axis_limits if cf.x_axis_limits is not None else (-axis_extent, axis_extent)
        y_extent = cf.y_axis_limits if cf.y_axis_limits is not None else (-axis_extent, axis_extent)
        ax.set_xlim(x_extent), ax.set_ylim(y_extent)
        return ax
    
    def _plot_frames(self, fig: plt.Figure, gs: GridSpec) -> plt.Axes:
        """Helper method to populate the static Double Pendulum frames into the input figure and GridSpec."""
        cf = self.config
        x0, y0 = cf.origin
        ax = self._setup_figure(fig, gs)    # call helper method to setup the base figure and axes
        trail_length = int((cf.view_trail_length_pct / 100) * self.steps) if cf.view_trail_length_pct > 0 else 0
        def draw(ax: plt.Axes, i: int, show_trail: bool = False, alpha: float = 1.0) -> None:
            """Draws the double pendulum at step index `i` with adjustable transparency."""
            i = len(self.t) - 1 if i == -1 else i
            # plot links
            ax.plot([x0, self.x1[i]], [y0, self.y1[i]], color=cf.link1_colour, linewidth=cf.link_linewidth, alpha=alpha, zorder=1)    
            ax.plot([self.x1[i], self.x2[i]], [self.y1[i], self.y2[i]], color=cf.link2_colour, linewidth=cf.link_linewidth, alpha=alpha, zorder=2)
            # plot masses (bobs):
            ax.scatter(self.x1[i], self.y1[i], marker="o", color=cf.m1_colour, s=cf.m1_markersize, edgecolors="none", alpha=alpha, zorder=3)               
            ax.scatter(self.x2[i], self.y2[i], marker="o", color=cf.m2_colour, s=cf.m2_markersize, edgecolors="none", alpha=alpha, zorder=4)
            if cf.show_trail:
                i0 = max(0, i - trail_length)
                ax.plot(self.x1[i0:i + 1], self.y1[i0:i + 1], color=cf.m1_colour, linewidth=cf.trail_linewidth, alpha=cf.trail_alpha, zorder=0)
                ax.plot(self.x2[i0:i + 1], self.y2[i0:i + 1], color=cf.m2_colour, linewidth=cf.trail_linewidth, alpha=cf.trail_alpha, zorder=0)
        # draw first frame with low opacity:
        if "start" in cf.display_mode: draw(ax, i=0, alpha=0.4)
        # draw last frame (default: with full trail):
        if "end" in cf.display_mode: draw(ax, i=-1, show_trail=cf.show_trail)  
        # draw all frames with increasing opacity:
        if "fade" in cf.display_mode:
            step = max(1, self.steps // cf.num_frames)   # equally spaced frames (to draw)
            indices = list(range(0, self.steps, step))
            n = len(indices)
            for idx, i in enumerate(indices):
                # opacity fades from 0.1 to 1.0 linearly across frames
                alpha = 0.1 + 0.9 * (idx / (n - 1)) if n > 1 else 1.0
                show_trail_this_frame = (i == indices[-1]) and cf.show_trail    # bool: only show trail on the last frame
                draw(ax, i=i, alpha=alpha, show_trail=show_trail_this_frame)
        return ax

    def plot_frames(self) -> Tuple[plt.Figure, plt.GridSpec, plt.Axes]:
        """Plot STATIC Double Pendulum frames using the configured parameters."""
        fig = plt.figure(figsize=self.config.figure_size)
        gs = GridSpec(nrows=1, ncols=1, figure=fig)
        ax = self._plot_frames(fig, gs)
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.show()
        return fig, gs, ax

    def plot_dashboard(self) -> plt.Figure:
        """Displays both the graph plots and static DP frames in a single figure using subfigures."""
        cf = self.config
        fig = plt.figure(figsize=cf.dashboard_figure_size, constrained_layout=True)
        if cf.dashboard_figure_title:
            fig.suptitle(cf.dashboard_figure_title)
        outer = GridSpec(nrows=1, ncols=2, wspace=0.08, figure=fig)
        gs1 = outer[0].subgridspec(nrows=3, ncols=1)
        gs2 = outer[1].subgridspec(nrows=1, ncols=1)
        # build each plot from existing methods:
        self._plot_graphs(fig, gs1)    # plot θ(t), ω(t), and E(t)
        ax2 = self._plot_frames(fig, gs2)    # plot static frames
        ax2.set_aspect("equal", adjustable="datalim")    # data limits (xlim and ylim) will be changed to achieve the aspect ratio
        plt.show()
        return fig

    def _get_filename(self, dpi: int) -> str:
        """Generate a filename for the animation based on the parameters."""
        p, cf = self.params, self.config
        m1, m2, r1, r2 = p.m1, p.m2, p.r1, p.r2
        return (
            f"{self.file_prefix}_"
            f"dpi={dpi}_"
            f"θ1={p.theta1_0:.0f}_θ2={p.theta2_0:.0f}_"
            f"steps={self.steps:,}(T={p.T_secs:.0f})_"
            f"@{cf.trail_length_pct}%trail_"
            f"vid_dur={cf.video_duration:.1f}s_"
            f"{m1=:.1f}_{m2=:.1f}_"
            f"{r1=:.1f}_{r2=:.1f}"
            ".mp4"
        )
    
    def _animation_params(self, title: str, dpi: int) -> None:
        """Set animation parameters as attributes based on the configuration."""
        cf = self.config

        # calculate key parameters for the animation:
        self.used_steps = int(cf.fps * cf.video_duration)    # number of steps for the animation
        self.trail_length = int((cf.trail_length_pct / 100) * self.used_steps)    # trail length in frames
        self.interval = int(1000 / cf.fps)              # seconds per frame (converted to milliseconds)
        
        # check if the number of steps is sufficient for the video duration:
        if self.used_steps > self.steps:
            raise ValueError(f"{self.used_steps:,} frames required for a {cf.video_duration:.1f} sec video at {cf.fps} FPS, but only {self.steps:,.0f} frames ({self.steps / cf.fps:.1f} sec) available. Increase `steps` or reduce `video_duration`.")
        
        # print animation parameters and stats:
        print(f"\n{title}")
        print(f"NOTE: using {self.used_steps}/{len(self.t)} time steps => {cf.video_duration:.1f} sec video duration")
        print(f"{self.used_steps:,} frames @ {cf.fps} fps (~{self.interval * 1e-3:.3f} sec/frame), dpi={dpi}")
        print(f"writing {self.used_steps} frames to MP4...\n")

    def _write_to_mp4(self, ani: FuncAnimation, pbar: tqdmFA, dpi: int) -> None:
        """Write the animation to an MP4 file using the configured parameters."""
        cf = self.config
        
        # start animation:
        writer = FFMpegWriter(fps=cf.fps, bitrate=cf.bitrate)
        file_name = self._get_filename(dpi)    # generate filename for the animation
        ani.save(filename=file_name, writer=writer, dpi=dpi)
        pbar.close()    # close progress bar
        print(f"\nanimation saved as '{file_name}'")

        # quick report:
        elapsed = int(pbar.format_dict["elapsed"])
        time = datetime.timedelta(seconds=elapsed)
        print(f"\ntotal elapsed time: {time}")
        avg_iter_per_sec = self.used_steps / time.total_seconds()
        if 1 / avg_iter_per_sec < 1:
            avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
        else:
            avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
        print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")

    def animate(self, show_plot_first: bool = True, dpi: int = 200) -> None:
        """Animate the double pendulum using matplotlib's FuncAnimation."""
        cf = self.config
        x0, y0 = cf.origin    # origin coordinates (main pivot point)
        if show_plot_first:
            print(f"\nshowing first and last frames of animation...")
            print(f"<CLOSE FIGURE WINDOW TO START WRITING TO MP4>")
            # show the first and last frames, and trail length:
            self.plot_frames()

        # set animation parameters and print key stats:
        self._animation_params("# ---------- DP ANIMATION ---------- #", dpi)    
        
        # ----- FIGURE SETUP ----- #
        fig = plt.figure(figsize=cf.figure_size)
        gs = GridSpec(nrows=1, ncols=1, figure=fig)    # single subplot for the animation
        ax = self._setup_figure(fig, gs)    # call helper method to setup the base figure and axes
        ax.set_aspect("equal")    # set equal aspect ratio for the axes

        # ------ INITIALISE PLOT ELEMENTS ----- #
        link1,          = ax.plot([], [], color=cf.link1_colour, linewidth=cf.link_linewidth, zorder=1)
        link2,          = ax.plot([], [], color=cf.link2_colour, linewidth=cf.link_linewidth, zorder=2)
        mass1           = ax.scatter([], [], marker="o", color=cf.m1_colour, s=cf.m1_markersize, zorder=3)
        mass2           = ax.scatter([], [], marker="o", color=cf.m2_colour, s=cf.m2_markersize, zorder=3)
        if cf.trail_length_pct > 0:    # show trails for masses
            m1_trail,   = ax.plot([], [], color=cf.m1_colour, linewidth=cf.trail_linewidth, alpha=cf.trail_alpha, zorder=4)
            m2_trail,   = ax.plot([], [], color=cf.m2_colour, linewidth=cf.trail_linewidth, alpha=cf.trail_alpha, zorder=4)

        # ----- ANIMATION FUNCTIONS ----- #

        def init() -> tuple:
            link1.set_data([], [])
            link2.set_data([], [])
            mass1.set_offsets([self.x1[0], self.y1[0]])
            mass2.set_offsets([self.x2[0], self.y2[0]])
            artists = [link1, link2, mass1, mass2]    # list of artists to initialise
            if cf.trail_length_pct > 0:
                m1_trail.set_data([], []), m2_trail.set_data([], [])
                artists.extend([m1_trail, m2_trail])
            return tuple(artists)
        
        def update(frame: int) -> tuple:
            # update link and mass positions:
            link1.set_data([x0, self.x1[frame]], [y0, self.y1[frame]])
            link2.set_data([self.x1[frame], self.x2[frame]], [self.y1[frame], self.y2[frame]])
            mass1.set_offsets([self.x1[frame], self.y1[frame]])
            mass2.set_offsets([self.x2[frame], self.y2[frame]])
            artists = [link1, link2, mass1, mass2]    # list of artists to be updated
            # update trails:
            if cf.trail_length_pct > 0:
                i0 = max(0, frame - self.trail_length)    # start index for the trail
                m1_trail.set_data(self.x1[i0: frame + 1], self.y1[i0: frame + 1])
                m2_trail.set_data(self.x2[i0: frame + 1], self.y2[i0: frame + 1])
                artists.extend([m1_trail, m2_trail])
            pbar.update(1)    # update progress bar
            return tuple(artists)

        # ----- START ANIMATION ----- #
        pbar = tqdmFA(total=self.used_steps, fps=cf.fps)    # initialise custom tqdm progress bar
        ani = FuncAnimation(fig, update, frames=range(self.used_steps), init_func=init, blit=True)
        self.file_prefix = "dp_sim_"
        self._write_to_mp4(ani, pbar, dpi)

    def animate_dashboard(
        self,
        dpi: int = 200,
    ) -> None:
        """Animate the double pendulum alongside the time series graphs as a full dashboard layout animation."""

        print(f"\nshowing first and last frames of animation...")
        print(f"<CLOSE FIGURE WINDOW TO START WRITING TO MP4>")
        fig_dash = self.plot_dashboard()    # show the first and last frames, and trail length

        cf = self.config
        x0, y0 = cf.origin    # origin coordinates (main pivot point)
        theta1_plot, theta2_plot = self._wrap_angles((self.theta1, self.theta2))    # wrap angles to handle discontinuities
        if cf.in_degrees:    
            theta1_plot, theta2_plot = np.degrees(theta1_plot), np.degrees(theta2_plot)   
            w1_plot, w2_plot = np.degrees(self.w1), np.degrees(self.w2)

        # set animation parameters and print key stats:
        self._animation_params("# ---------- DP DASHBOARD ANIMATION ---------- #", dpi)

        # ------ FIGURE SETUP ----- #
        fig = plt.figure(figsize=cf.dashboard_figure_size, constrained_layout=True)
        if cf.dashboard_figure_title:
            fig.suptitle(cf.dashboard_figure_title)
        outer = GridSpec(nrows=1, ncols=2, wspace=0.08, figure=fig)
        gs1 = outer[0].subgridspec(nrows=3, ncols=1)    # time series graphs
        gs2 = outer[1].subgridspec(nrows=1, ncols=1)    # double pendulum 

        # ------ LEFT SIDE - time series graphs ----- #
        ax0 = fig.add_subplot(gs1[0, 0])                # θ(t)
        ax1 = fig.add_subplot(gs1[1, 0], sharex=ax0)    # ω(t)
        ax2 = fig.add_subplot(gs1[2, 0], sharex=ax0)    # E(t)
        ax0.tick_params(labelbottom=False)              # remove x-ticks 
        ax1.tick_params(labelbottom=False)    
        # Labels and grid
        ax2.set_xlabel(r"time, $t$ ($s$)")
        ax0.set_ylabel(r"angular displacement, " + (r"$\theta$ ($^{\circ}$)" if cf.in_degrees else r"$\theta$ ($\text{rad}$)"))
        ax1.set_ylabel(r"angular velocity, " + (r"$\omega$ ($^{\circ}/s$)" if cf.in_degrees else r"$\omega$ ($\text{rad}$/s"))
        ax2.set_ylabel(r"system energy, $E$ ($J$)")

        # ------ RIGHT SIDE - double pendulum ----- #
        ax3 = self._setup_figure(fig, gs2)              # call helper method to setup the base figure and axes for the DP

        # ------ INITIALISE PLOT ELEMENTS ----- #
        theta1_line,    = ax0.plot([], [], color=cf.m1_colour, label=r"$\theta_1$")
        theta2_line,    = ax0.plot([], [], color=cf.m2_colour, label=r"$\theta_2$")
        w1_line,        = ax1.plot([], [], color=cf.m1_colour, label=r"$\omega_1$")
        w2_line,        = ax1.plot([], [], color=cf.m2_colour, label=r"$\omega_2$")
        T_line,         = ax2.plot([], [], color=cf.ke_colour, label=r"$E_k$")
        V_line,         = ax2.plot([], [], color=cf.pe_colour, label=r"$E_p$")
        E_line,         = ax2.plot([], [], color=cf.te_colour, label=r"$E_T$", linewidth=plt.rcParams["lines.linewidth"] + 0.5)
        link1,          = ax3.plot([], [], color=cf.link1_colour, linewidth=cf.link_linewidth, zorder=1)
        link2,          = ax3.plot([], [], color=cf.link2_colour, linewidth=cf.link_linewidth, zorder=2)
        mass1           = ax3.scatter([], [], marker="o", color=cf.m1_colour, s=cf.m1_markersize, zorder=3)
        mass2           = ax3.scatter([], [], marker="o", color=cf.m2_colour, s=cf.m2_markersize, zorder=3)
        # plot trails for the pendulum masses:
        if cf.trail_length_pct > 0:    
            m1_trail,   = ax3.plot([], [], color=cf.m1_colour, linewidth=cf.trail_linewidth, alpha=cf.trail_alpha, zorder=4)
            m2_trail,   = ax3.plot([], [], color=cf.m2_colour, linewidth=cf.trail_linewidth, alpha=cf.trail_alpha, zorder=4)
        
        # configure time series graphs:
        for axis in (ax0, ax1, ax2):
            axis.grid(True, alpha=cf.grid_alpha)
            axis.yaxis.set_major_locator(MaxNLocator(cf.n_max_ticks))
            if cf.display_legend:
                axis.legend(loc=cf.legend_location)
            if cf.xlim_timespan:
                axis.set_xlim(self.t[0], self.t[self.used_steps - 1])    # set x-axis limits to the time span
        # set y-axis limits for the time series graphs:
        ax0.set_ylim(self.get_y_extent(theta1_plot, theta2_plot))   # set y-axis limits based on the min/max values
        ax1.set_ylim(self.get_y_extent(w1_plot, w2_plot))           
        ax2.set_ylim(self.get_y_extent(self.T, self.V))          
        # set axis limits and aspect ratio for DP plot:
        ax_dp = fig_dash.get_axes()[-1]
        xlim, ylim = ax_dp.get_xlim(), ax_dp.get_ylim() # get x and y limits from the static DP plot
        print(xlim, ylim)
        ax3.set_xlim(xlim), ax3.set_ylim(ylim)          
        ax3.set_aspect("equal", adjustable="datalim") # set equal aspect ratio for the axes   

        # ----- ANIMATION FUNCTIONS ----- #

        def init() -> tuple:
            theta1_line.set_data([], [])
            theta2_line.set_data([], [])
            w1_line.set_data([], [])
            w2_line.set_data([], [])
            T_line.set_data([], [])
            V_line.set_data([], [])
            E_line.set_data([], [])
            link1.set_data([], [])
            link2.set_data([], [])
            mass1.set_offsets([self.x1[0], self.y1[0]])
            mass2.set_offsets([self.x2[0], self.y2[0]])
            artists = [theta1_line, theta2_line, w1_line, w2_line, T_line, V_line, E_line, link1, link2, mass1, mass2]
            if cf.trail_length_pct > 0:    # show trails for masses
                m1_trail.set_data([], [])
                m2_trail.set_data([], [])
                artists.extend([m1_trail, m2_trail])
            return tuple(artists)
        
        def update(frame: int) -> tuple:
            # update time series data for the current frame:
            theta1_line.set_data(self.t[:frame], theta1_plot[:frame])
            theta2_line.set_data(self.t[:frame], theta2_plot[:frame])
            w1_line.set_data(self.t[:frame], w1_plot[:frame])
            w2_line.set_data(self.t[:frame], w2_plot[:frame])
            T_line.set_data(self.t[:frame], self.T[:frame])
            V_line.set_data(self.t[:frame], self.V[:frame])
            E_line.set_data(self.t[:frame], self.E[:frame])
            # update double pendulum link and mass positions:
            link1.set_data([x0, self.x1[frame]], [y0, self.y1[frame]])
            link2.set_data([self.x1[frame], self.x2[frame]], [self.y1[frame], self.y2[frame]])
            mass1.set_offsets([self.x1[frame], self.y1[frame]])
            mass2.set_offsets([self.x2[frame], self.y2[frame]])
            artists = [theta1_line, theta2_line, w1_line, w2_line, T_line, V_line, E_line, link1, link2, mass1, mass2]
            if cf.trail_length_pct > 0:
                i0 = max(0, frame - self.trail_length)    # start index for the trail
                m1_trail.set_data(self.x1[i0: frame], self.y1[i0: frame])
                m2_trail.set_data(self.x2[i0: frame], self.y2[i0: frame])
                artists.extend([m1_trail, m2_trail])
            pbar.update(1)
            return tuple(artists)
        
        # ----- START ANIMATION ----- #
        pbar = tqdmFA(total=self.used_steps, fps=cf.fps)    # initialise custom tqdm progress bar
        ani = FuncAnimation(fig, update, frames=range(self.used_steps), init_func=init, blit=True)
        self.file_prefix = "dp_dash_"
        self._write_to_mp4(ani, pbar, dpi)    # write the animation to MP4 file


def dp1() -> None:
    """Example usage of the DoublePendulum class to plot dynamics."""
    dp = DP(
        params=DPSystemParams(
            theta1_0=75,  
            theta2_0=165,  
            T_secs=10,              # simulation time in seconds
            steps=1000,              # number of steps in the simulation
            ode_method="RK45",      # ODE solver method
            rtol=1e-9,              # relative tolerance for the ODE solver
        ),
        config=DPConfig(
            x_axis_limits=(-0.9, 0.9),      # x-axis limits
            y_axis_limits=(-0.9, 0.4),      # y-axis limits
            trail_length_pct=4,             # length of the trail as a percentage of the total steps
            video_duration=10.0,            # duration of MP4 video in seconds
            # static plot configurations:
            display_mode=["start", "end"],  
            num_frames=5,               
            show_trail=True,            
            view_trail_length_pct=100,      # view trail length in the static plot only
        )
    )

    # dp.plot_graphs()    # Plot θ(t), ω(t), and E(t)

    # dp.plot_frames()

    # dp.plot_dashboard()

    # dp.animate(dpi=200)

    dp.animate_dashboard(dpi=250) 