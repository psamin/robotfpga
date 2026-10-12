"""GPU simulator benchmark for the SGS teacher: MJX's JAX and Warp backends on the block-sort scene.

Batches reset-buffer states, holds each state's ctrl for --ticks 30 Hz ticks (10 physics steps
each), and reports env ticks per second for each batch size. It also checks agreement with CPU
MuJoCo: the median and max cube displacement difference after the rollout, and whether held
cubes stay held. Failures are recorded per backend and batch size, not raised.

    sbatch --export=ALL,CMD="scripts/bench_sim.py --out runs/bench_sim.json" cluster/script.sbatch
"""

import argparse
import os
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from armlab.rl.buffer import build
from armlab.sim.env import SCENE, SUBSTEPS
from armlab.util.runmeta import write_runmeta

CUBES = ("cube_red", "cube_blue")
# Warp preallocates contact buffers (naconmax: total over the batch) and constraint rows (njmax:
# per world). Expert rollouts from 300 buffer states peaked at 32 contacts and 190 rows per world;
# overflow drops contacts and is only printed as a warning, so check the job log for "overflow".
CONTACTS_PER_WORLD, NJMAX = 48, 256


def cpu_rollout(m, buf, i, ticks):
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = buf["qpos"][i], buf["qvel"][i], buf["ctrl"][i]
    mujoco.mj_forward(m, d)
    for _ in range(ticks * SUBSTEPS):
        mujoco.mj_step(m, d)
    return np.stack([d.body(c).xpos for c in CUBES])


def make_rollout(m, impl, ticks, batch):
    mx = mjx.put_model(m, impl=impl)
    sizes = {"naconmax": CONTACTS_PER_WORLD * batch, "njmax": NJMAX} if impl == "warp" else {}
    d0 = mjx.make_data(m, impl=impl, **sizes)
    ids = jp.array([m.body(c).id for c in CUBES])

    def rollout(qpos, qvel, ctrl):
        dx = mjx.forward(mx, d0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        dx, _ = jax.lax.scan(lambda d, _: (mjx.step(mx, d), None), dx, None, length=ticks * SUBSTEPS)
        return dx.xpos[ids]

    return jax.jit(jax.vmap(rollout))


def bench(m, buf, impl, batches, ticks, parity):
    out, n = {}, len(buf["qpos"])
    for b in batches:
        idx = np.arange(b) % n
        args = [jp.asarray(buf[k][idx]) for k in ("qpos", "qvel", "ctrl")]
        try:  # report any backend failure instead of stopping the benchmark
            rollout = make_rollout(m, impl, ticks, b)
            t0 = time.perf_counter()
            xpos = jax.block_until_ready(rollout(*args))
            compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            xpos = np.asarray(jax.block_until_ready(rollout(*args)))
            run_s = time.perf_counter() - t0
        except Exception as e:
            out[b] = {"error": f"{type(e).__name__}: {e}"[:500]}
            continue
        out[b] = {"compile_s": round(compile_s, 1), "ticks_per_s": round(b * ticks / run_s)}
        print(impl, b, out[b], flush=True)
    k = min(parity, n)
    cpu = np.stack([cpu_rollout(m, buf, i, ticks) for i in range(k)])
    rollout = make_rollout(m, impl, ticks, k)
    xpos = np.asarray(rollout(*[jp.asarray(buf[f][:k]) for f in ("qpos", "qvel", "ctrl")]))
    err = np.linalg.norm(xpos - cpu, axis=-1)
    lifted = cpu[..., 2] > 0.03  # cubes the CPU run still holds in the air
    out["parity"] = {
        "states": k,
        "cube_err_median_mm": round(float(np.median(err)) * 1e3, 2),
        "cube_err_max_mm": round(float(err.max()) * 1e3, 2),
        "held_kept": f"{int((xpos[..., 2][lifted] > 0.03).sum())}/{int(lifted.sum())}",
    }
    print(impl, out["parity"], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impls", nargs="+", default=["jax", "warp"])
    ap.add_argument("--batches", nargs="+", type=int, default=[256, 1024, 4096, 16384])
    ap.add_argument("--ticks", type=int, default=30, help="30 Hz control ticks per rollout")
    ap.add_argument("--states", type=int, default=512, help="reset-buffer states to tile over")
    ap.add_argument("--parity", type=int, default=64, help="states compared against CPU MuJoCo")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("devices:", jax.devices(), flush=True)
    buf, _ = build(args.states, seed=0, workers=min(os.cpu_count() or 1, 16))
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    results = {i: bench(m, buf, i, args.batches, args.ticks, args.parity) for i in args.impls}
    results["devices"] = [str(d) for d in jax.devices()]
    write_runmeta(args.out, vars(args), seed=0, results=results)


if __name__ == "__main__":
    main()
