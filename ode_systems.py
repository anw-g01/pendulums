import numpy as np
from typing import Callable
from config import DPSystemParams


def dp_ode(p: DPSystemParams) -> Callable[[float, np.ndarray], np.ndarray]:
    """ODE system of equations for the Double Pendulum (DP)."""
    # unpack required system parameters from dataclass:
    r1, r2, m1, m2, g = p.r1, p.r2, p.m1, p.m2, p.g
    
    # define the ODE system:
    a1 = lambda theta1, theta2: r2/r1 * (m2 / (m1 + m2)) * np.cos(theta1 - theta2)
    a2 = lambda theta1, theta2: r1/r2 * np.cos(theta1 - theta2)
    f1 = lambda theta1, theta2, w2: - r2/r1 * (m2 / (m1 + m2)) * w2**2 * np.sin(theta1 - theta2) - g/r1 * np.sin(theta1)    # 'w' is omega (or dtheta)
    f2 = lambda theta1, theta2, w1: r1/r2 * w1**2 * np.sin(theta1 - theta2) - g/r2 * np.sin(theta2)

    def g1(z1: float, z2: float, z3: float, z4: float) -> float:
        """State variables: z1 = θ1, z2 = θ2, z3 = ω1, z4 = ω2."""
        alpha1, alpha2 = a1(z1, z2), a2(z1, z2)
        func1, func2 = f1(z1, z2, z4), f2(z1, z2, z3)
        return (func1 - alpha1 * func2) / (1 - alpha1 * alpha2)

    def g2(z1: float, z2: float, z3: float, z4: float) -> float:
        """State variables: z1 = θ1, z2 = θ2, z3 = ω1, z4 = ω2."""
        alpha1, alpha2 = a1(z1, z2), a2(z1, z2)
        func1, func2 = f1(z1, z2, z4), f2(z1, z2, z3)
        return (- alpha2 * func1 + func2) / (1 - alpha1 * alpha2)

    def func(t: float, Z: np.ndarray) -> np.ndarray:
        """
        ODE system of equations for the Double Pendulum (DP).
        """
        z1, z2, z3, z4 = Z
        dz1 = z3
        dz2 = z4
        dz3 = g1(z1, z2, z3, z4)
        dz4 = g2(z1, z2, z3, z4)
        return np.array([dz1, dz2, dz3, dz4])
    
    return func
