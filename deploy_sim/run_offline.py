#!/usr/bin/env python3
"""Offline deployment test: fly a controller on a realistic quadrotor model.

No simulator install, no ROS, no motion capture, no vehicle — only the packages
already in requirements.txt.

    python deploy_sim/run_offline.py --config configs/crazyflie_mppi_corner.json

Test your own controller instead of the bundled one:

    python deploy_sim/run_offline.py --controller mypkg.ctrl:MyController

`--controller` takes `module:attribute`, where the attribute is either a class
constructed as `Cls(cfg.safemppi, env, device=...)` or a factory called with the
same arguments. It needs `.reset()` and
`.plan(state, goal, gamma, seed=...) -> (accel, info)`; see deploy_sim/harness.py.

Check robustness against a state-estimate dropout (this is what a real mocap
outage looks like to the controller):

    python deploy_sim/run_offline.py --fault-freeze-at 3.0
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from safe_mppi.config import load_config          # noqa: E402
from safe_mppi.environment import TaskEnvironment  # noqa: E402

import harness                                     # noqa: E402
from vehicle import Swarm                          # noqa: E402


def build_controller(spec, cfg, env, device):
    if not spec:
        from safe_mppi.controller import Mode1SafeMPPI
        return Mode1SafeMPPI(cfg.safemppi, env, device=device)
    mod, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--controller must look like 'module:Attribute'")
    obj = getattr(importlib.import_module(mod), attr)
    return obj(cfg.safemppi, env, device=device)


def save(res, env, cfg, out_dir, gamma, want_gif, want_polytope):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"offline_{stamp}"
    log = res["log"]
    if not log:
        return None
    cols = list(log[0])
    with open(f"{base}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(log)
    arr = {k: np.asarray([r[k] for r in log]) for k in cols if k != "phase"}
    np.savez_compressed(f"{base}.npz", **arr)
    print(f"[deploy] wrote {base}.csv / .npz")
    if want_polytope:
        try:
            import viz
            P = np.array([[r["x"], r["y"], r["z"]] for r in log])
            t = np.array([r["t"] for r in log])
            OB = np.array([[r["ox"], r["oy"], r["oz"], r["orad"]] for r in log])
            OB = OB.reshape(len(P), 1, 4)
            if not np.isfinite(OB).any():
                OB = None
            png = viz.render_static(P, env, cfg.safemppi, gamma,
                                    f"{base}_polytope.png", title=base.name, t=t,
                                    ceiling=float(res["soft"][2, 1]), obstacles=OB)
            print(f"[deploy] wrote {png}")
            if want_gif:
                gif = viz.render_gif(P, env, cfg.safemppi, gamma, f"{base}_polytope.gif",
                                     t=t, title=base.name,
                                     ceiling=float(res["soft"][2, 1]), obstacles=OB)
                print(f"[deploy] wrote {gif}")
        except Exception as exc:
            print(f"[deploy] (figures skipped: {exc!r})")
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/crazyflie_mppi_corner.json"))
    ap.add_argument("--controller", default=None, help="module:Attribute of your controller")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=str(REPO_ROOT / "outputs" / "deploy_sim"))
    ap.add_argument("--gif", action="store_true", help="also render the orbiting GIF")
    ap.add_argument("--no-figures", action="store_true")
    # deployment tuning (same defaults as the flight configuration)
    ap.add_argument("--max-speed", type=float, default=0.7)
    ap.add_argument("--max-vz", type=float, default=0.3)
    ap.add_argument("--accel-smooth", type=float, default=0.4)
    ap.add_argument("--stream-rate", type=float, default=100.0)
    ap.add_argument("--fault-freeze-at", type=float, default=None,
                    help="freeze the state estimate after N s (mocap-dropout test)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = TaskEnvironment(cfg)
    gamma = args.gamma if args.gamma is not None else float(cfg.data.gammas[0])
    seed = args.seed if args.seed is not None else int(cfg.data.seed_start)
    controller = build_controller(args.controller, cfg, env, args.device)

    swarm = Swarm(np.array([env.start[0], env.start[1], 0.0]), seed=seed)
    res = harness.run(controller, env, cfg, swarm, gamma=gamma, seed=seed,
                      max_speed=args.max_speed, max_vz=args.max_vz,
                      accel_smooth=args.accel_smooth, stream_rate=args.stream_rate,
                      fault_freeze_at=args.fault_freeze_at, verbose=not args.quiet)
    harness.summarize(res, env, verbose=not args.quiet)
    save(res, env, cfg, Path(args.output), gamma, args.gif, not args.no_figures)
    return 0 if res["reached"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
