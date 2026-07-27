"""Offline stand-in for the vehicle: it follows the streamed setpoint.

This replaced ``deploy_sim/plant.py`` (the "calibrated plant") on 2026-07-26.

WHY THE PLANT WAS REMOVED
-------------------------
Its three fitted parameters -- ``est_tau=0.08``, ``noise=0.0017``,
``rate_scale=1.111`` -- were identified from a single flight recorded BEFORE a
control-loop timing bug was fixed. In that loop each substep advanced the
reference by a nominal ``dt_sub`` while ~1.25x that elapsed in wall time, so the
streamed ``p_ref`` and ``v_ref`` contradicted each other and the onboard
controller chased a velocity the reference never had. ``est_tau`` absorbed that
as if it were vehicle lag. The plant was, in large part, a model of the bug.

Once the loop was fixed and measured against hardware (Crazyflie 2.1 + Vicon,
four live-SafeMPPI flights, gamma 0.1/0.3/0.5/1.0, 343 control cycles):

    measured deviation from the streamed reference : 27-31 mm RMSE, 52 mm max
    the same quantity predicted by the old plant   : 176-242 mm

5.7-8.2x pessimistic. It wrongly predicted that gamma 0.5 and gamma 1.0 would
collide with the obstacle; both flew with clearance to spare. A model that wrong
is worse than no model, because it silently rejects good controllers.

WHAT THIS IS INSTEAD
--------------------
The vehicle follows the setpoint. There are **no fitted parameters**.

That is justified by measurement, not by modelling: at the speeds this harness
is used for, "the drone is where the setpoint says" is accurate to ~3 cm, an
order of magnitude below the tolerances that decide success (arrival radius,
obstacle margin, geofence margin).

The residual is deliberately NOT simulated. A simulated residual becomes a
number you start trusting, which is exactly how the old plant caused harm. Use
:data:`RESIDUAL_MAX_M` as a margin budget instead: if a trajectory leaves less
room than that against the floor, the fence or an obstacle, it is too tight to
call safe from simulation alone.

VALIDITY
--------
The measurements behind these constants were taken at <=0.91 m/s and <=0.50
m/s^2 commanded -- about 9% of the vehicle's lateral acceleration authority.
Near the thrust limit, or for direction changes faster than the lateral position
loop (omega_n ~ 0.63 rad/s), the vehicle can no longer match the setpoint and
this stand-in becomes optimistic. Re-measure before trusting it there.
"""
from __future__ import annotations

import numpy as np

# Measured over four post-fix hardware flights, 2026-07-26.
RESIDUAL_RMSE_M = 0.030      # |measured - streamed reference|, RMSE
RESIDUAL_MAX_M = 0.052       # worst single sample
RESIDUAL_BIAS_Z_M = 0.021    # the vehicle sits consistently ABOVE the reference
VALID_ENVELOPE = "<=0.91 m/s, <=0.50 m/s^2 commanded (~9% of lateral authority)"


class Vehicle:
    """Same call surface as ``pycrazyswarm.Crazyflie``; follows cmdFullState."""

    def __init__(self, pos, rng=None):
        self.p = np.asarray(pos, float).copy()
        self.p_ref = self.p.copy()
        self.mode = "idle"
        self._goto = None
        self.id = 1

    # -- what a controller reads -------------------------------------------
    def position(self):
        return self.p.copy()

    # -- what a controller commands ----------------------------------------
    def cmdFullState(self, pos, vel, acc, yaw, omega):
        self.mode = "fullstate"
        self.p_ref = np.asarray(pos, float)

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
        pass

    # -- integration --------------------------------------------------------
    def _start_goto(self, target, duration):
        self.mode = "goto"
        self._goto = (self.p.copy(), target, max(1e-3, float(duration)), 0.0)

    def integrate(self, dt):
        if self.mode == "goto" and self._goto is not None:
            p0, p1, dur, elapsed = self._goto
            elapsed = min(dur, elapsed + dt)
            s = elapsed / dur
            self.p = p0 + (p1 - p0) * (3 * s ** 2 - 2 * s ** 3)   # smoothstep
            self._goto = (p0, p1, dur, elapsed)
        elif self.mode == "fullstate":
            self.p = self.p_ref.copy()        # the vehicle IS the setpoint


class Clock:
    """Simulated clock. Time advances only inside ``sleep``.

    A caller that spends real wall time computing (a planner solve, a network
    forward pass) should charge it explicitly with ``sleep(measured_seconds)``;
    otherwise the simulation models thinking as free and can never reproduce a
    missed control deadline.
    """

    def __init__(self, cfs, substep=0.002, rng=None):
        self.crazyflies = cfs
        self.t = 0.0
        self.substep = float(substep)

    def time(self):
        return self.t

    def sleep(self, duration):
        d = max(0.0, float(duration))
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
    """Stands in for ``pycrazyswarm.Crazyswarm``."""

    def __init__(self, initial_position, seed=0):
        cf = Vehicle(initial_position)
        self.allcfs = type("AllCFs", (), {"crazyflies": [cf]})()
        self.timeHelper = Clock([cf])


def margin_verdict(name, margin_m):
    """Does a simulated margin survive the measured tracking residual?"""
    if margin_m is None or not np.isfinite(margin_m):
        return f"{name}: n/a"
    if margin_m <= 0:
        return f"{name}: {margin_m:+.3f} m  VIOLATED in the plan itself"
    if margin_m < RESIDUAL_MAX_M:
        return (f"{name}: {margin_m:+.3f} m  thinner than the measured "
                f"{RESIDUAL_MAX_M * 1e3:.0f} mm worst-case tracking error")
    if margin_m < 3 * RESIDUAL_MAX_M:
        return f"{name}: {margin_m:+.3f} m  tight (<3x the residual)"
    return f"{name}: {margin_m:+.3f} m  ok"
