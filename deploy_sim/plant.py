#!/usr/bin/env python3
"""A deployment-realistic quadrotor plant behind the pycrazyswarm command API.

Kinematic simulators (including the stock Crazyswarm one) treat `cmdFullState`
as "teleport the vehicle to this setpoint", so they have zero tracking error,
zero overshoot, zero noise and a perfect clock. Every failure we hit on real
hardware lives precisely in what that removes. This module models those effects
instead, so an offline run predicts the real flight:

  * closed-loop position control with the REAL onboard Mellinger gains from
    crazyflieTypes.yaml (kp_xy=0.4, kd_xy=0.2, kp_z=1.25, kd_z=0.4). The z loop
    has damping ratio zeta = kd/(2*sqrt(kp)) = 0.18 -> ~56% overshoot, which is
    exactly what makes the real drone climb far above the planned altitude;
  * finite thrust: bounded vertical / horizontal acceleration;
  * state-estimator lag (first-order), the other half of the tracking error;
  * motion-capture measurement noise (1.7 mm rms, measured);
  * control-loop timing that runs at ~90 Hz with jitter, not a perfect 100 Hz.

It deliberately does NOT expose velocity(), so the bridge falls back to
finite-difference velocity exactly as it does on hardware.

It exposes the same method names as pycrazyswarm (`position`, `cmdFullState`,
`takeoff`, `goTo`, `land`, ...), so the SAME control code runs here and on real
hardware. It deliberately does NOT expose `velocity()`, because the real backend
does not either -- forcing the controller to finite-difference position exactly
as it must in the field.

CALIBRATION / VALIDITY. Gains are the stock Crazyflie Mellinger position gains;
`est_tau` and the noise/rate figures were fitted to one real flight, where this
plant reproduced a geofence exit to within 8 mm and matched the altitude
step-response prediction (57%/2.84 s modelled vs 56%/2.86 s theoretical). It
still UNDER-predicts peak altitude overshoot. Treat it as a good predictor, not
as ground truth, and keep real safety margins.
"""
from __future__ import annotations

import numpy as np

# Onboard Mellinger position gains (crazyflieTypes.yaml, type "default").
KP = np.array([0.4, 0.4, 1.25])
KD = np.array([0.2, 0.2, 0.40])
# Thrust envelope: a Crazyflie hovers at ~0.5 throttle; usable accel beyond g.
A_MAX_XY = 5.5      # m/s^2 (~30 deg tilt)
A_MAX_Z_UP = 9.0    # m/s^2
A_MAX_Z_DOWN = 5.0  # m/s^2 (limited by falling, not thrust)


class Quadrotor:
    """Quadrotor plant + onboard position loop behind the pycrazyswarm cf API."""

    # est_tau lumps EKF response + mocap/radio latency. Calibrated so the
    # altitude step response overshoots ~65%, bracketing the 56% ideal-gain
    # prediction and the 77% measured on flight mppi_20260724_153603.
    def __init__(self, pos, est_tau=0.08, noise=0.0017, rng=None):
        self.p = np.asarray(pos, float).copy()      # true position
        self.v = np.zeros(3)                        # true velocity
        self.p_est = self.p.copy()                  # estimator output (lagged)
        self.v_est = np.zeros(3)
        self.est_tau = float(est_tau)
        self.noise = float(noise)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # active setpoint
        self.p_ref = self.p.copy()
        self.v_ref = np.zeros(3)
        self.a_ff = np.zeros(3)
        self.mode = "idle"
        self._goto = None
        self.ledRGB = (0, 0, 1)

    # ---- pycrazyswarm-compatible API (NOTE: no velocity(), like real hardware)
    def position(self):
        return self.p_est + self.rng.normal(0, self.noise, 3)

    def cmdFullState(self, pos, vel, acc, yaw, omega):
        self.mode = "fullstate"
        self.p_ref = np.asarray(pos, float)
        self.v_ref = np.asarray(vel, float)
        self.a_ff = np.asarray(acc, float)

    def takeoff(self, targetHeight, duration, groupMask=0):
        self._start_goto(np.array([self.p[0], self.p[1], float(targetHeight)]), duration)

    def land(self, targetHeight, duration, groupMask=0):
        self._start_goto(np.array([self.p[0], self.p[1], float(targetHeight)]), duration)

    def goTo(self, goal, yaw, duration, relative=False, groupMask=0):
        self._start_goto(np.asarray(goal, float), duration)

    def notifySetpointsStop(self, remainValidMillisecs=100):
        pass

    def cmdStop(self):
        self.mode = "idle"

    def emergency(self):
        self.mode = "idle"

    def setLEDColor(self, r, g, b):
        self.ledRGB = (r, g, b)

    # ---- internals
    def _start_goto(self, target, duration):
        self.mode = "goto"
        self._goto = (self.p_est.copy(), np.asarray(target, float), max(1e-3, float(duration)), 0.0)

    def integrate(self, dt):
        if self.mode == "goto" and self._goto is not None:
            p0, p1, dur, el = self._goto
            el = min(dur, el + dt)
            s = el / dur
            s = 3 * s ** 2 - 2 * s ** 3                      # smoothstep, like a poly traj
            self.p_ref = p0 + (p1 - p0) * s
            self.v_ref = (p1 - p0) * (6 * (el / dur) * (1 - el / dur)) / dur
            self.a_ff = np.zeros(3)
            self._goto = (p0, p1, dur, el)
        if self.mode == "idle":
            self.v *= max(0.0, 1.0 - 5.0 * dt)
            self.p = self.p + self.v * dt
            self._estimator(dt)
            return
        # Onboard position loop on the ESTIMATED state (that lag is real).
        a = KP * (self.p_ref - self.p_est) + KD * (self.v_ref - self.v_est) + self.a_ff
        a[:2] = np.clip(a[:2], -A_MAX_XY, A_MAX_XY)
        a[2] = np.clip(a[2], -A_MAX_Z_DOWN, A_MAX_Z_UP)
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        if self.p[2] < 0.0:                                  # ground
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])
        self._estimator(dt)

    def _estimator(self, dt):
        k = dt / max(self.est_tau, 1e-6)
        k = min(1.0, k)
        meas = self.p + self.rng.normal(0, self.noise, 3)
        v_new = self.p_est.copy()
        self.p_est = self.p_est + k * (meas - self.p_est)
        self.v_est = self.v_est + k * ((self.p_est - v_new) / max(dt, 1e-6) - self.v_est)


class Clock:
    """Advances the plant in real-ish time, with the measured loop jitter."""

    def __init__(self, cfs, jitter=0.0011, rate_scale=1.111, substep=0.001, rng=None):
        self.crazyflies = cfs
        self.t = 0.0
        self.jitter = float(jitter)
        self.rate_scale = float(rate_scale)   # hardware ran 89.9 Hz vs 100 Hz asked
        self.substep = float(substep)
        self.rng = rng if rng is not None else np.random.default_rng(1)

    def time(self):
        return self.t

    def sleep(self, duration):
        # Real elapsed time is longer than requested and jitters (measured).
        d = max(0.0, float(duration) * self.rate_scale
                + self.rng.normal(0, self.jitter))
        n = max(1, int(round(d / self.substep)))
        h = d / n
        for _ in range(n):
            for cf in self.crazyflies:
                cf.integrate(h)
            self.t += h

    def sleepForRate(self, rate):
        self.sleep(1.0 / rate)

    def isShutdown(self):
        return False


class Swarm:
    """Stands in for pycrazyswarm.Crazyswarm."""

    def __init__(self, initial_position, seed=0):
        rng = np.random.default_rng(seed)
        cf = Quadrotor(initial_position, rng=rng)
        self.allcfs = type("AllCFs", (), {"crazyflies": [cf]})()
        self.timeHelper = Clock([cf], rng=rng)


# Backwards-compatible aliases (earlier local scripts used these names).
HwCrazyflie = Quadrotor
HwTimeHelper = Clock
HwSwarm = Swarm
