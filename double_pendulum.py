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