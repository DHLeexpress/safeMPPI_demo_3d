"""Deployment harness: run a controller the way it is actually flown.

The point of this module is that the loop below is *the same loop used on real
hardware*. Only the plant differs. So testing a controller here exercises the
things that actually break flights — replan rate, setpoint streaming, tracking
lag, geofences, speed limits, state-estimate dropouts — rather than just the
planning geometry.

CONTROLLER PROTOCOL — anything with these two methods works:

    controller.reset()
    action, info = controller.plan(state, goal, gamma, seed=...)

      state  : np.ndarray (6,)  [x, y, z, vx, vy, vz]
      goal   : np.ndarray (3,)
      action : np.ndarray (3,)  commanded ACCELERATION [m/s^2]
      info   : dict (unused here; may be empty)

Acceleration is a *feedforward*, not a motor command: each 10 Hz plan is turned
into a kinematically consistent full-state setpoint that is streamed at ~100 Hz

    v_ref <- v_ref + u dt        p_ref <- p_ref + v_ref dt        a_ref = u

and handed to `cmdFullState(p_ref, v_ref, u, ...)`, which the vehicle's onboard
position controller tracks.

SAFETY LAYERS (all also active on hardware):
  * SOFT geofence   -> stop and land in place
  * HARD geofence   -> emergency stop
  * over-speed      -> land;  runaway speed -> emergency stop
  * state-estimate frozen/invalid -> land (a stale pose looks perfectly healthy:
    finite, inside the fence, zero finite-difference velocity, while the vehicle
    keeps flying — this is a real failure mode, not a hypothetical one)
  * every commanded position clamped inside the fence, with a margin
"""
from __future__ import annotations

import numpy as np


class SafetyAbort(Exception):
    """Raised when a safety layer fires. `hard` means cut motors, else land."""

    def __init__(self, msg, hard=False):
        super().__init__(msg)
        self.hard = hard


def geofence(cfg, env, start, goal, *, soft=None, hard=None, xy_margin=0.15,
             z_min=0.4, z_max=1.7):
    """(soft, hard) boxes. Explicit `soft` wins; else the config's optional
    top-level "safety" block; else the taskspace shrunk by a margin and widened
    just enough to contain start and goal."""
    b = np.asarray(env.bounds, float)
    spec = cfg.raw.get("safety", {}) if isinstance(getattr(cfg, "raw", None), dict) else {}
    lo = soft[0] if soft is not None else spec.get("safe_min")
    hi = soft[1] if soft is not None else spec.get("safe_max")
    if lo is not None and hi is not None:
        S = np.array([[float(lo[i]), float(hi[i])] for i in range(3)])
        H = np.stack([S[:, 0] - 0.15, S[:, 1] + 0.15], axis=1)
    else:
        S = np.array([[b[0, 0] + xy_margin, b[0, 1] - xy_margin],
                      [b[1, 0] + xy_margin, b[1, 1] - xy_margin],
                      [z_min, z_max]])
        buf = float(cfg.taskspace.reach_radius) + 0.2
        for pt in (np.asarray(start, float)[:3], np.asarray(goal, float)):
            for i in range(3):
                S[i, 0] = min(S[i, 0], max(pt[i] - buf, b[i, 0]))
                S[i, 1] = max(S[i, 1], min(pt[i] + buf, b[i, 1]))
        H = np.stack([np.maximum(S[:, 0] - 0.1, b[:, 0] - 0.05),
                      np.minimum(S[:, 1] + 0.1, b[:, 1] + 0.05)], axis=1)
    return S, (np.asarray(hard, float) if hard is not None else H)


def _inside(p, box):
    return bool(np.all(p >= box[:, 0]) and np.all(p <= box[:, 1]))


def _clamp(p, box):
    return np.minimum(np.maximum(p, box[:, 0]), box[:, 1])


def run(controller, env, cfg, swarm, *, goal=None, start=None, gamma=0.3, seed=0,
        soft=None, hard=None, stream_rate=100.0, max_speed=0.7, max_vz=0.3,
        vel_filter=0.4, accel_smooth=0.4, cmd_margin=0.25, max_lead=0.3,
        state_timeout=0.3, fault_freeze_at=None, takeoff_duration=4.0,
        settle=2.0, verbose=True):
    """Fly `controller` from start to goal on `swarm`. Returns a result dict."""
    dt = cfg.safemppi.dt
    goal = np.asarray(env.goal if goal is None else goal, float)
    start = np.asarray(env.start[:3] if start is None else start, float)
    flight_z = float(start[2])
    S, H = geofence(cfg, env, start, goal, soft=soft, hard=hard)
    cmd_box = np.stack([S[:, 0] + cmd_margin, S[:, 1] - cmd_margin], axis=1)
    for pt in (start, goal):                       # never clamp the goal away
        cmd_box[:, 0] = np.minimum(cmd_box[:, 0], pt)
        cmd_box[:, 1] = np.maximum(cmd_box[:, 1], pt)
    substeps = max(1, int(round(stream_rate * dt)))
    dt_sub = dt / substeps

    clock, cf = swarm.timeHelper, swarm.allcfs.crazyflies[0]
    log, t0, reached, outcome = [], clock.time(), False, "step limit"

    def say(msg):
        if verbose:
            print(f"[deploy] {msg}", flush=True)

    say(f"start {np.round(start,2)} -> goal {np.round(goal,2)}  gamma={gamma}")
    say(f"soft fence x[{S[0,0]:+.2f},{S[0,1]:+.2f}] y[{S[1,0]:+.2f},{S[1,1]:+.2f}] "
        f"z[{S[2,0]:.2f},{S[2,1]:.2f}]   cap {max_speed} m/s (vz {max_vz})")

    try:
        cf.takeoff(targetHeight=flight_z, duration=takeoff_duration)
        clock.sleep(takeoff_duration + 1.0)
        cf.goTo(np.array([start[0], start[1], flight_z]), yaw=0.0, duration=4.0)
        clock.sleep(4.0 + settle)

        if fault_freeze_at is not None:            # inject a state-estimate dropout
            real_pos, t_ref, frz = cf.position, [clock.time()], {}
            def frozen(_rp=real_pos, _t=t_ref, _f=frz):
                p = np.asarray(_rp(), float)
                if clock.time() - _t[0] >= fault_freeze_at:
                    _f.setdefault("p", p.copy())
                    return _f["p"]
                return p
            cf.position = frozen
            say(f"FAULT INJECTION: state estimate freezes after {fault_freeze_at}s")

        controller.reset()
        p_meas = np.asarray(cf.position(), float)
        p_ref, p_prev = p_meas.copy(), p_meas.copy()
        v_ref = np.zeros(3)
        v = np.zeros(3)
        u_prev = np.zeros(3)
        frozen_cycles = 0
        max_frozen = max(1, int(round(state_timeout / dt)))
        say("running")

        for step in range(cfg.taskspace.max_steps):
            p_meas = np.asarray(cf.position(), float)
            if not np.all(np.isfinite(p_meas)) or np.linalg.norm(p_meas) < 1e-6:
                raise SafetyAbort("invalid state estimate", hard=True)
            v = vel_filter * ((p_meas - p_prev) / dt) + (1.0 - vel_filter) * v
            # A frozen pose is the dangerous case: it looks entirely healthy.
            frozen_cycles = frozen_cycles + 1 if np.array_equal(p_meas, p_prev) else 0
            if frozen_cycles >= max_frozen:
                raise SafetyAbort(
                    f"state estimate frozen {frozen_cycles * dt:.2f}s at "
                    f"{np.round(p_meas,2)} -> flying blind", hard=False)
            p_prev = p_meas
            speed = float(np.linalg.norm(v))
            if not _inside(p_meas, H):
                raise SafetyAbort(f"left HARD fence at {np.round(p_meas,2)}", hard=True)
            if not _inside(p_meas, S):
                raise SafetyAbort(f"left SOFT fence at {np.round(p_meas,2)}", hard=False)
            if speed > 2.5 * max_speed:
                raise SafetyAbort(f"runaway speed {speed:.2f} m/s", hard=True)
            if speed > 1.8 * max_speed:
                raise SafetyAbort(f"over-speed {speed:.2f} m/s", hard=False)

            # Reference governor: keep the reference near the vehicle so it can
            # never run away, and bleed the measurement into v_ref.
            p_ref = p_meas + np.clip(p_ref - p_meas, -max_lead, max_lead)
            v_ref = 0.85 * v_ref + 0.15 * v

            action, _ = controller.plan(np.concatenate([p_meas, v_ref]), goal, gamma,
                                        seed=seed * 100_000 + step)
            u = accel_smooth * np.asarray(action, float) + (1.0 - accel_smooth) * u_prev
            u_prev = u

            ob = (np.asarray(env.spheres, float)[0] if len(env.spheres)
                  else np.full(4, np.nan))
            for _ in range(substeps):
                v_ref = v_ref + u * dt_sub
                sv = float(np.linalg.norm(v_ref))
                if sv > max_speed:
                    v_ref = v_ref * (max_speed / sv)
                v_ref[2] = float(np.clip(v_ref[2], -max_vz, max_vz))
                p_ref = _clamp(p_ref + v_ref * dt_sub, cmd_box)
                cf.cmdFullState(p_ref.tolist(), v_ref.tolist(), u.tolist(), 0.0, [0.0, 0.0, 0.0])
                clock.sleep(dt_sub)
                cur = np.asarray(cf.position(), float)
                log.append({"t": clock.time() - t0, "phase": "run",
                            "x": cur[0], "y": cur[1], "z": cur[2],
                            "tx": goal[0], "ty": goal[1], "tz": goal[2],
                            "cx": p_ref[0], "cy": p_ref[1], "cz": p_ref[2],
                            "ux": u[0], "uy": u[1], "uz": u[2],
                            "ox": ob[0], "oy": ob[1], "oz": ob[2], "orad": ob[3]})

            if float(np.linalg.norm(p_meas - goal)) < cfg.taskspace.reach_radius:
                reached, outcome = True, "reached goal"
                say(f"reached goal at step {step + 1} (t={(step + 1) * dt:.2f}s)")
                break
        _land(cf, clock, say)

    except SafetyAbort as e:
        outcome = f"{'HARD' if e.hard else 'SOFT'} ABORT: {e}"
        say(f"!! {outcome}")
        if e.hard:
            getattr(cf, "emergency", lambda: None)()
        else:
            _land(cf, clock, say)
    except KeyboardInterrupt:
        outcome = "interrupted"
        _land(cf, clock, say)

    return {"log": log, "reached": reached, "outcome": outcome,
            "soft": S, "hard": H, "goal": goal, "start": start, "gamma": gamma}


def _land(cf, clock, say):
    try:
        getattr(cf, "notifySetpointsStop", lambda: None)()
        cf.land(targetHeight=0.04, duration=4.0)
        clock.sleep(5.0)
        say("landed")
    except Exception:
        getattr(cf, "emergency", lambda: None)()


def summarize(res, env, verbose=True):
    """Margins and clearances that decide whether a run is flight-worthy."""
    L = res["log"]
    if not L:
        return {}
    P = np.array([[r["x"], r["y"], r["z"]] for r in L])
    t = np.array([r["t"] for r in L])
    S = res["soft"]
    clr = env.obstacle_clearance(P)
    # Differentiate over a ~0.1 s window, NOT sample-to-sample: at 100 Hz with
    # millimetre-level state noise a raw finite difference reports speeds ~2x
    # the true value (noise/dt), which looks alarming and means nothing.
    k = max(1, int(round(0.1 / max(np.median(np.diff(t)), 1e-6))))
    sp = (np.linalg.norm(P[k:] - P[:-k], axis=1) /
          np.maximum(t[k:] - t[:-k], 1e-9)) if len(P) > k else np.zeros(1)
    out = {
        "outcome": res["outcome"], "reached": res["reached"],
        "duration_s": float(t[-1] - t[0]),
        "closest_to_goal_m": float(np.linalg.norm(P - res["goal"], axis=1).min()),
        # clearance is measured to the CONFIGURED sphere, which normally already
        # includes vehicle radius + safety margin, so >0 means outside that shell
        "clearance_beyond_safety_sphere_m": (
            float(np.nanmin(clr)) if np.isfinite(clr).any() else None),
        "peak_speed_mps": float(sp.max()) if len(sp) else 0.0,
        "fence_margin_m": float(min((P - S[:, 0]).min(), (S[:, 1] - P).min())),
        "peak_z_m": float(P[:, 2].max()),
    }
    if verbose:
        print("\n--- deployment summary ---")
        for k, v in out.items():
            print(f"  {k:26s} {v}")
        if out["clearance_beyond_safety_sphere_m"] is not None and \
                out["clearance_beyond_safety_sphere_m"] < 0.0:
            print("  WARNING: entered the configured safety sphere")
        if out["fence_margin_m"] < 0.1:
            print("  WARNING: fence margin under 0.10 m — expect an abort on hardware")
    return out
