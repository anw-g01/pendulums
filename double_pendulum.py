import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "monospace"
plt.rcParams["lines.linewidth"] = 0.8


def dp_sim(
        r1: float = 0.4,
        r2: float = 0.45,
        m1: float = 0.5,
        m2: float = 0.5,
        theta1_0: float = 20,
        theta2_0: float = 55,
        w1_0: float = 0,
        w2_0: float = 0,
        g: float = 9.81,
        T: float = 10,
        steps: int = 1000,
        method: str = "RK45"
    ) -> tuple:

    a1 = lambda theta1, theta2: r2/r1 * (m2 / (m1 + m2)) * np.cos(theta1 - theta2)
    a2 = lambda theta1, theta2: r1/r2 * np.cos(theta1 - theta2)
    f1 = lambda theta1, theta2, w2: - r2/r1 * (m2 / (m1 + m2)) * w2**2 * np.sin(theta1 - theta2) - g/r1 * np.sin(theta1)    # 'w' is omega (or dtheta)
    f2 = lambda theta1, theta2, w1: r1/r2 * w1**2 * np.sin(theta1 - theta2) - g/r2 * np.sin(theta2)

    def g1(z1, z2, z3, z4):
        """State variables: z1 = θ1, z2 = θ2, z3 = ω1, z4 = ω2."""
        alpha1, alpha2 = a1(z1, z2), a2(z1, z2)
        func1, func2 = f1(z1, z2, z4), f2(z1, z2, z3)
        return (func1 - alpha1 * func2) / (1 - alpha1 * alpha2)

    def g2(z1, z2, z3, z4):
        """State variables: z1 = θ1, z2 = θ2, z3 = ω1, z4 = ω2."""
        alpha1, alpha2 = a1(z1, z2), a2(z1, z2)
        func1, func2 = f1(z1, z2, z4), f2(z1, z2, z3)
        return (- alpha2 * func1 + func2) / (1 - alpha1 * alpha2)

    def func_dp(t, Z):
        """
        Numerically integrate a system of ODEs given an initial value:
        dy/dt = f(t, Z) and Z(t0) = Z0, where Z is the state vector.
        """
        z1, z2, z3, z4 = Z              # unpack state variables
        dz1 = z3                        # dθ1
        dz2 = z4                        # dθ2
        dz3 = g1(z1, z2, z3, z4)        # ddθ1
        dz4 = g2(z1, z2, z3, z4)        # ddθ2
        return [dz1, dz2, dz3, dz4]     # return state derivatives

    # ----- INITICAL CONDITIONS ----- #
    theta1_0, theta2_0 = np.radians(theta1_0), np.radians(theta2_0)
    w1_0, w2_0 = np.radians(w1_0), np.radians(w2_0)
    Z0 = [theta1_0, theta2_0, w1_0, w2_0]   # initial conditions (state vector)

    # ----- EVALUATION & SOLVE ----- #
    t_span = (0, T)
    t_eval = np.linspace(0, T, steps)
    dt = (t_span[1] - t_span[0]) / (steps - 1)

    print(f"\nrunning ODE solver ({method=})...")
    print(f"t_eval=({t_eval[0]:.0f}, {t_eval[-1]}), {steps=:,}, dt≈{dt:.2f}")#
    sol = solve_ivp(func_dp, t_span, Z0, method="RK45", t_eval=t_eval)

    # ----- EXTRACT OUTPUTS ----- #
    t, Z = sol.t, sol.y
    success = sol.success
    print(f"\nsolver success: {success} ({sol.nfev:,} nfev)")
    print(f"t.shape: {t.shape}, Z.shape: {Z.shape}\n")
    p = [r1, r2, m1, m2, g, theta1_0, theta2_0, w1_0, w2_0] # also return vector of essential parameters

    return t, Z, p


def plot_graphs(
        t, Z, p,
        in_degrees: bool = True,
        m1_colour: str = "tab:green",
        m2_colour: str = "tab:red",
        ke_colour: str = "tab:blue",
        pe_colour: str = "tab:orange",
        te_colour: str = "tab:purple",
        figure_size: tuple = (8, 7),
        grid_alpha: float = 0.15,
        y_axis_max_ticks: int = 5,
        show_legend: bool = True
    ) -> None:
    """
    Plot θ(t), ω(t), and E(t) over time for a double pendulum.

    params:
        `t`: array of time values
        `Z`: state matrix [θ1, θ2, ω1, ω2]
        `p`: parameter vector [r1, r2, m1, m2, g, θ1_0, θ2_0, ω1_0, ω2_0]
        `in_degrees`: flag between degrees and radians
        ... : plotting customisation parameters
    """
    r1, r2, m1, m2, g, *_ = p
    theta1, theta2, w1, w2 = Z[0], Z[1], Z[2], Z[3]

    if in_degrees:
        theta1_plot, theta2_plot = np.degrees(theta1), np.degrees(theta2)
        w1_plot, w2_plot = np.degrees(w1), np.degrees(w2)
    else:
        theta1_plot, theta2_plot = theta1, theta2
        w1_plot, w2_plot = w1, w2

    fig, ax = plt.subplots(nrows=3, figsize=figure_size, sharex=True)

    # ----- ANGULAR DISPLACEMENT ----- #
    ax[0].grid(True, alpha=grid_alpha)
    ax[0].yaxis.set_major_locator(MaxNLocator(y_axis_max_ticks))
    ax[0].set_ylabel("angular displacement, " + (r"$\theta$ ($^{\circ}$)" if in_degrees else r"$\theta$ (rad)"))
    ax[0].plot(t, theta1_plot, color=m1_colour, label=r"$\theta_1$")
    ax[0].plot(t, theta2_plot, color=m2_colour, label=r"$\theta_2$")
    if show_legend:
        ax[0].legend(loc="best")

    # ----- ANGULAR VELOCITY ----- #
    ax[1].grid(True, alpha=grid_alpha)
    ax[1].yaxis.set_major_locator(MaxNLocator(y_axis_max_ticks))
    ax[1].set_ylabel("angular velocity, " + (r"$\omega$ ($^{\circ}/s$)" if in_degrees else r"$\omega$ (rad/s)"))
    ax[1].plot(t, w1_plot, color=m1_colour, label=r"$\omega_1$")
    ax[1].plot(t, w2_plot, color=m2_colour, label=r"$\omega_2$")
    if show_legend:
        ax[1].legend(loc="best")

    # ----- ENERGY PLOTS ----- #
    # cartesian velocity components
    x1_dot = r1 * w1 * np.cos(theta1)
    y1_dot = r1 * w1 * np.sin(theta1)
    x2_dot = x1_dot + r2 * w2 * np.cos(theta2)
    y2_dot = y1_dot + r2 * w2 * np.sin(theta2)
    # kinetic energy, KE
    T = 0.5 * m1 * (x1_dot**2 + y1_dot**2) + 0.5 * m2 * (x2_dot**2 + y2_dot**2)
    # potential energy, PE
    y1 = -r1 * np.cos(theta1)
    y2 = y1 - r2 * np.cos(theta2)
    V = m1 * g * y1 + m2 * g * y2
    # total energy
    E = T + V

    ax[2].grid(True, alpha=grid_alpha)
    ax[2].set_xlabel(r"time, $t$ ($s$)")
    ax[2].set_ylabel(r"system energy, $E$ ($J$)")
    ax[2].yaxis.set_major_locator(MaxNLocator(y_axis_max_ticks))
    ax[2].plot(t, T, color=ke_colour, label=r"KE ($T$)")
    ax[2].plot(t, V, color=pe_colour, label=r"PE ($V$)")
    ax[2].plot(t, E, color=te_colour, label=r"Total ($E$)")
    if show_legend:
        ax[2].legend(loc="best")

    plt.tight_layout()
    plt.show()
