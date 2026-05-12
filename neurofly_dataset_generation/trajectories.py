"""Trajectory definitions used for dataset generation."""

import numpy as np


class SpiralTrajectory:
    """A fixed-radius 3D spiral trajectory for RotorPy."""

    def __init__(self, radius=2.0, omega=1.0, v_z=0.5):
        self.r = radius
        self.w = omega
        self.v_z = v_z

    def update(self, t):
        r = self.r
        w = self.w

        x = np.array([r * np.cos(w * t), r * np.sin(w * t), self.v_z * t])
        v = np.array([-r * w * np.sin(w * t), r * w * np.cos(w * t), self.v_z])
        a = np.array([-r * w**2 * np.cos(w * t), -r * w**2 * np.sin(w * t), 0.0])
        j = np.array([r * w**3 * np.sin(w * t), -r * w**3 * np.cos(w * t), 0.0])
        s = np.array([r * w**4 * np.cos(w * t), r * w**4 * np.sin(w * t), 0.0])
        yaw = np.arctan2(v[1], v[0])

        return {
            "x": x,
            "x_dot": v,
            "x_ddot": a,
            "x_dddot": j,
            "x_ddddot": s,
            "yaw": yaw,
            "yaw_dot": w,
            "yaw_ddot": 0.0,
        }


class ConicalSpiralTrajectory:
    """A 3D spiral whose radius shrinks linearly over time."""

    def __init__(self, radius_start=2.0, radius_rate=0.15, omega=1.0, v_z=0.5):
        self.r0 = radius_start
        self.c = radius_rate
        self.w = omega
        self.v_z = v_z

    def update(self, t):
        w = self.w
        radius = max(0.0, self.r0 - self.c * t)
        effective_rate = self.c if radius > 0.0 else 0.0

        cos_wt = np.cos(w * t)
        sin_wt = np.sin(w * t)
        x = np.array([radius * cos_wt, radius * sin_wt, self.v_z * t])
        v = np.array(
            [
                -effective_rate * cos_wt - radius * w * sin_wt,
                -effective_rate * sin_wt + radius * w * cos_wt,
                self.v_z,
            ]
        )
        a = np.array(
            [
                2 * effective_rate * w * sin_wt - radius * w**2 * cos_wt,
                -2 * effective_rate * w * cos_wt - radius * w**2 * sin_wt,
                0.0,
            ]
        )
        j = np.array(
            [
                3 * effective_rate * w**2 * cos_wt + radius * w**3 * sin_wt,
                3 * effective_rate * w**2 * sin_wt - radius * w**3 * cos_wt,
                0.0,
            ]
        )
        s = np.array(
            [
                -4 * effective_rate * w**3 * sin_wt + radius * w**4 * cos_wt,
                -4 * effective_rate * w**3 * cos_wt + radius * w**4 * sin_wt,
                0.0,
            ]
        )
        yaw = np.arctan2(v[1], v[0])
        yaw_dot, yaw_ddot = _conical_spiral_yaw_derivatives(
            radius, effective_rate, w
        )

        return {
            "x": x,
            "x_dot": v,
            "x_ddot": a,
            "x_dddot": j,
            "x_ddddot": s,
            "yaw": yaw,
            "yaw_dot": yaw_dot,
            "yaw_ddot": yaw_ddot,
        }


def _conical_spiral_yaw_derivatives(radius, radius_rate, omega):
    if radius <= 0.0:
        return omega, 0.0

    c_sq = radius_rate**2
    rw_sq = (radius * omega) ** 2
    denom = c_sq + rw_sq
    yaw_dot = omega * (2 * c_sq + rw_sq) / denom
    yaw_ddot = 2 * (radius_rate**3) * radius * (omega**3) / (denom**2)
    return yaw_dot, yaw_ddot


def _conical_spiral_flat_output(radius_start, radius_rate, omega, v_z, t):
    radius = max(0.0, radius_start - radius_rate * t)
    effective_rate = radius_rate if radius > 0.0 else 0.0

    cos_wt = np.cos(omega * t)
    sin_wt = np.sin(omega * t)
    x = np.array([radius * cos_wt, radius * sin_wt, v_z * t])
    v = np.array(
        [
            -effective_rate * cos_wt - radius * omega * sin_wt,
            -effective_rate * sin_wt + radius * omega * cos_wt,
            v_z,
        ]
    )
    a = np.array(
        [
            2 * effective_rate * omega * sin_wt - radius * omega**2 * cos_wt,
            -2 * effective_rate * omega * cos_wt - radius * omega**2 * sin_wt,
            0.0,
        ]
    )
    yaw = np.arctan2(v[1], v[0])
    yaw_dot, yaw_ddot = _conical_spiral_yaw_derivatives(
        radius, effective_rate, omega
    )
    return x, v, a, yaw, yaw_dot, yaw_ddot


class RampedConicalSpiralTrajectory:
    """A conical spiral with a smooth start from hover."""

    def __init__(
        self,
        radius_start=2.0,
        radius_rate=0.15,
        omega=1.0,
        v_z=0.5,
        ramp_time=8.0,
    ):
        self.r0 = radius_start
        self.c = radius_rate
        self.w = omega
        self.v_z = v_z
        self.ramp_time = ramp_time

    def _progress(self, t):
        if self.ramp_time <= 0.0:
            return t, 1.0, 0.0

        if t <= 0.0:
            return 0.0, 0.0, 0.0

        if t < self.ramp_time:
            u = t / self.ramp_time
            tau = self.ramp_time * (2.5 * u**4 - 3.0 * u**5 + u**6)
            tau_dot = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            tau_ddot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.ramp_time
            return tau, tau_dot, tau_ddot

        return t - 0.5 * self.ramp_time, 1.0, 0.0

    def update(self, t):
        tau, tau_dot, tau_ddot = self._progress(t)
        x_base, v_base, a_base, yaw, yaw_dot_base, yaw_ddot_base = (
            _conical_spiral_flat_output(self.r0, self.c, self.w, self.v_z, tau)
        )

        return {
            "x": x_base,
            "x_dot": v_base * tau_dot,
            "x_ddot": a_base * tau_dot**2 + v_base * tau_ddot,
            "x_dddot": np.zeros(3),
            "x_ddddot": np.zeros(3),
            "yaw": yaw,
            "yaw_dot": yaw_dot_base * tau_dot,
            "yaw_ddot": yaw_ddot_base * tau_dot**2 + yaw_dot_base * tau_ddot,
        }
