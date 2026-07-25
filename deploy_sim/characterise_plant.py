#!/usr/bin/env python3
"""Characterise the deployment plant: step response, state lag, overshoot law.

Regenerates docs/assets/plant_characterisation.png. Run it after changing any
plant parameter so the documented behaviour stays honest:

    python deploy_sim/characterise_plant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plant import KD, KP, Quadrotor  # noqa: E402

INK, MUTED = "#1f2430", "#6b7280"
OUT = REPO_ROOT / "docs" / "assets" / "plant_characterisation.png"


def step(axis, target, tau=0.08, noise=0.0017, T=12.0, dt=0.001):
    cf = Quadrotor([0, 0, 0.6], est_tau=tau, noise=noise)
    cf.mode = "fullstate"
    ref = np.array([0.0, 0.0, 0.6])
    ref[axis] = target
    cf.p_ref, cf.v_ref, cf.a_ff = ref, np.zeros(3), np.zeros(3)
    P, E = [], []
    for _ in range(int(T / dt)):
        cf.integrate(dt)
        P.append(cf.p.copy())
        E.append(cf.p_est.copy())
    return np.arange(len(P)) * dt, np.asarray(P), np.asarray(E)


def ramp_overshoot(v, stop_z=1.6, tau=0.08, dt=0.001, T=12.0):
    """Climb at `v` until stop_z, then hold: how far past stop_z does it go?"""
    cf = Quadrotor([0, 0, 0.6], est_tau=tau, noise=0.0)
    cf.mode = "fullstate"
    z, peak = 0.6, 0.6
    for _ in range(int(T / dt)):
        z = min(stop_z, z + v * dt)
        cf.p_ref = np.array([0.0, 0.0, z])
        cf.v_ref = np.array([0.0, 0.0, v if z < stop_z else 0.0])
        cf.a_ff = np.zeros(3)
        cf.integrate(dt)
        peak = max(peak, cf.p[2])
    return peak - stop_z


def _tidy(ax):
    ax.grid(alpha=0.16)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#d7dbe0")


def main():
    wn_z, z_z = float(np.sqrt(KP[2])), float(KD[2] / (2 * np.sqrt(KP[2])))
    wn_x, z_x = float(np.sqrt(KP[0])), float(KD[0] / (2 * np.sqrt(KP[0])))
    print(f"z : wn={wn_z:.3f} rad/s  zeta={z_z:.3f}")
    print(f"xy: wn={wn_x:.3f} rad/s  zeta={z_x:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")

    # 1) step response — overshoot/time-to-peak go in the legend, not as
    #    annotations, so the two curves' labels cannot collide.
    for axis, tgt, name, kp, kd, c in [
        (2, 1.4, "z", KP[2], KD[2], "#d1495b"),
        (0, 0.8, "xy", KP[0], KD[0], "#3f7ad0"),
    ]:
        t, P, _ = step(axis, tgt)
        y = (P[:, axis] - P[0, axis]) / (tgt - P[0, axis])
        i = int(y.argmax())
        ax[0].plot(t, y, color=c, lw=2,
                   label=f"{name}  kp={kp:g} kd={kd:g}\n"
                         f"   {100 * (y[i] - 1):.0f}% overshoot at {t[i]:.2f}s")
        ax[0].plot(t[i], y[i], "o", color=c, ms=5)
    ax[0].axhline(1.0, color=MUTED, ls="--", lw=1)
    ax[0].set_xlabel("t [s]", fontsize=9, color=MUTED)
    ax[0].set_ylabel("normalised response", fontsize=9, color=MUTED)
    ax[0].set_title("step response: both position loops are underdamped",
                    fontsize=10, weight="bold", color=INK)
    ax[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    _tidy(ax[0])

    # 2) what the controller actually sees
    t, P, E = step(2, 1.4)
    m = (t > 1.0) & (t < 3.0)
    ax[1].plot(t[m], P[m, 2], color=INK, lw=2, label="true")
    ax[1].plot(t[m], E[m, 2], color="#f59f00", lw=1.5, label="estimate (lag + noise)")
    ax[1].set_xlabel("t [s]", fontsize=9, color=MUTED)
    ax[1].set_ylabel("z [m]", fontsize=9, color=MUTED)
    ax[1].set_title("the controller sees a lagged, noisy state",
                    fontsize=10, weight="bold", color=INK)
    ax[1].legend(fontsize=8, loc="lower right")
    _tidy(ax[1])

    # 3) the overshoot law that motivates a separate vertical-speed cap
    vs = np.linspace(0.1, 0.8, 8)
    ov = [ramp_overshoot(v) for v in vs]
    ax[2].plot(vs, ov, "o-", color="#d1495b", lw=2, label="plant")
    ax[2].plot(vs, vs / wn_z, "--", color=MUTED, lw=1.6,
               label=r"rule  $v_{climb}/\omega_n$")
    ax[2].set_xlabel("commanded climb rate [m/s]", fontsize=9, color=MUTED)
    ax[2].set_ylabel("altitude overshoot [m]", fontsize=9, color=MUTED)
    ax[2].set_title("why vertical speed gets its own cap",
                    fontsize=10, weight="bold", color=INK)
    ax[2].legend(fontsize=8, loc="upper left")
    _tidy(ax[2])

    fig.suptitle(f"deploy_sim plant — altitude loop $\\omega_n$={wn_z:.2f} rad/s, "
                 f"$\\zeta$={z_z:.2f}", fontsize=12, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=145)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
