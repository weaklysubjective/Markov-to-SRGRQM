#!/usr/bin/env python3
"""
STRICT PP 512x512 PF010 minimal orchestrator (PPV1).

This is a packaging/runner only.
It introduces NO new physics, NO PDE, NO Poisson/Laplacian, NO GR ansatz, NO regression.

Pipeline:
  A) (Guard) Do NOT run dense tau-geometry at 512.
  B) Build edges/masks at scale with a fixed, sane default.
  C) Auto-pick Shapiro IDs and run Shapiro.

Assumptions:
  - You already have:
      ca_trace_to_poset_v3_trace_gpu_v1.py
      src/gr_strict_pp_scalar/PP_build_edges_from_trace_v1.py
      src/gr_strict_pp_scalar/make_pfv1_masks.py
      src/gr_strict_pp_vector/PP_vector_scale_admissibility_probe_v1.py
      src/gr_strict_pp_vector/PP_vector_feeder_shear_markov_v1.py
      src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_v1.py
  - Working directory is repo root.
"""

import argparse
import os
import subprocess
import sys
import numpy as np

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def exists(path):
    return path is not None and os.path.exists(path)

def auto_ids_from_mass(mass_mask_path, H, W):
    m = np.load(mass_mask_path).astype(bool).reshape(H, W)
    rows, cols = np.where(m)
    if len(rows) == 0:
        raise ValueError("Mass mask is empty; cannot auto-pick Shapiro IDs.")
    r0 = int(np.round(rows.mean()))
    c0 = int(np.round(cols.mean()))

    def idx(r, c): return r * W + c

    # Fixed, simple, reproducible picks:
    # - through path: horizontal chord across mass center
    # - around path: far-corner diagonal
    d = max(40, W // 16)

    src_through = idx(r0, max(0, c0 - d))
    dst_through = idx(r0, min(W - 1, c0 + d))

    src_around  = idx(0, 0)
    dst_around  = idx(H - 1, W - 1)

    return (r0, c0, src_through, dst_through, src_around, dst_around)

def main():
    ap = argparse.ArgumentParser(description="STRICT PP PF010 512x512 PPV1 minimal runner")
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--case", type=str, default="strong_pf010")

    # Experience/trace generation controls
    ap.add_argument("--X", type=int, default=262144)  # 512*512
    ap.add_argument("--T", type=int, default=2000000)
    ap.add_argument("--eps_trace", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")

    # Fixed edge build settings (NO sweep)
    ap.add_argument("--neighbors", type=int, choices=[4,8], default=4)
    ap.add_argument("--rule", type=str, default="uphill_topk",
                    choices=["gradient_topk", "uphill_topk", "threshold", "sym_topk"])
    ap.add_argument("--k_out", type=int, default=2)  # fixed default for scale

    # Mass selection
    ap.add_argument("--mass_topk", type=int, default=80)
    ap.add_argument("--min_ring_nodes", type=int, default=12)

    # Vector thresholds
    ap.add_argument("--min_edges_used_vec", type=int, default=200)

    # Paths (default naming convention)
    ap.add_argument("--trace_out", type=str, default=None)
    ap.add_argument("--edges_curved_out", type=str, default=None)
    ap.add_argument("--edges_flat_out", type=str, default=None)
    ap.add_argument("--mass_out", type=str, default=None)
    ap.add_argument("--orbit_out", type=str, default=None)

    args = ap.parse_args()

    H, W = args.H, args.W
    case = args.case

    if args.X != H * W:
        print(f"[WARN] X ({args.X}) != H*W ({H*W}). Proceeding anyway.")

    # --- Default filenames ---
    trace_out = args.trace_out or f"trace_weights_ca_v3_{case}_{H}x{W}.txt"

    edges_curved_out = args.edges_curved_out or f"edges_ca_v3_{case}_{H}x{W}_PPV1.txt"
    edges_flat_out   = args.edges_flat_out   or f"edges_ca_v3_flat_{H}x{W}_PPV1.txt"

    mass_out  = args.mass_out  or f"PP_mass_mask_{H}x{W}_{case}_PPV1.npy"
    orbit_out = args.orbit_out or f"PP_orbit_ring_mask_{H}x{W}_{case}_PPV1.npy"

    # Reports
    trace_report = f"CA_POSV3_TRACE_{case}_{H}x{W}_eps{int(args.eps_trace*100):03d}_GPU.json"
    curved_report = f"src/gr_strict_pp_scalar/edges_{case}_{H}x{W}_PPV1.json"
    flat_report   = f"src/gr_strict_pp_scalar/edges_flat_{H}x{W}_PPV1.json"
    masks_report  = f"src/gr_strict_pp_scalar/masks_{case}_{H}x{W}_PPV1.json"

    vec_adm_report = f"src/gr_strict_pp_vector/PP_vector_admissibility_{H}_{case}_PPV1.json"
    feeder_report  = f"src/gr_strict_pp_vector/PP_vector_feeder_shear_{H}_{case}_v1_PPV1.json"
    shapiro_report = f"src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_{H}_{case}_PPV1.json"

    # --- Step 1: Trace weights (GPU-capable) ---
    if not exists(trace_out):
        run([
            sys.executable, "./ca_trace_to_poset_v3_trace_gpu_v1.py",
            "--X", str(args.X),
            "--T", str(args.T),
            "--eps-trace", str(args.eps_trace),
            "--seed", str(args.seed),
            "--device", str(args.device),
            "--out-trace", trace_out,
            "--report", trace_report,
        ])
    else:
        print(f"[SKIP] trace exists: {trace_out}")

    # --- Step 2: Curved edges from trace ---
    if not exists(edges_curved_out):
        run([
            sys.executable, "src/gr_strict_pp_scalar/PP_build_edges_from_trace_v1.py",
            "--trace_weights", trace_out,
            "--case", case,
            "--H", str(H), "--W", str(W),
            "--neighbors", str(args.neighbors),
            "--rule", args.rule,
            "--k_out", str(args.k_out),
            "--edges_out", edges_curved_out,
            "--report_out", curved_report,
        ])
    else:
        print(f"[SKIP] curved edges exist: {edges_curved_out}")

    # --- Step 3: Flat edges from trace (same trace is OK for PP bookkeeping) ---
    # If you have a dedicated flat trace in your tree, swap trace_out here.
    if not exists(edges_flat_out):
        run([
            sys.executable, "src/gr_strict_pp_scalar/PP_build_edges_from_trace_v1.py",
            "--trace_weights", trace_out,
            "--case", "flat",
            "--H", str(H), "--W", str(W),
            "--neighbors", str(args.neighbors),
            "--rule", args.rule,
            "--k_out", str(args.k_out),
            "--edges_out", edges_flat_out,
            "--report_out", flat_report,
        ])
    else:
        print(f"[SKIP] flat edges exist: {edges_flat_out}")

    # --- Step 4: Masks (topk + auto-band) ---
    if not exists(mass_out) or not exists(orbit_out):
        run([
            sys.executable, "src/gr_strict_pp_scalar/make_pfv1_masks.py",
            "--edges", edges_curved_out,
            "--trace_weights", trace_out,
            "--H", str(H), "--W", str(W),
            "--mass_topk", str(args.mass_topk),
            "--auto_band",
            "--min_ring_nodes", str(args.min_ring_nodes),
            "--mass_out", mass_out,
            "--orbit_out", orbit_out,
            "--report_out", masks_report,
        ])
    else:
        print(f"[SKIP] masks exist: {mass_out} / {orbit_out}")

    # --- Step 5: Vector admissibility ---
    run([
        sys.executable, "src/gr_strict_pp_vector/PP_vector_scale_admissibility_probe_v1.py",
        "--edges_curved", edges_curved_out,
        "--mass_mask", mass_out,
        "--H", str(H), "--W", str(W),
        "--output", vec_adm_report,
    ])

    # --- Step 6: Feeder shear ---
    run([
        sys.executable, "src/gr_strict_pp_vector/PP_vector_feeder_shear_markov_v1.py",
        "--edges_curved", edges_curved_out,
        "--mass_mask", mass_out,
        "--H", str(H), "--W", str(W),
        "--output", feeder_report,
    ])

    # --- Step 7: Shapiro IDs auto-pick + run ---
    r0, c0, src_t, dst_t, src_a, dst_a = auto_ids_from_mass(mass_out, H, W)
    print("\n[INFO] Shapiro auto-IDs")
    print("mass_center_row_col", r0, c0)
    print("src_through", src_t)
    print("dst_through", dst_t)
    print("src_around ", src_a)
    print("dst_around ", dst_a)

    run([
        sys.executable, "src/gr_strict_pp_scalar/PP_Shapiro_markov_tau_v1.py",
        "--edges_flat", edges_flat_out,
        "--edges_curved", edges_curved_out,
        "--H", str(H), "--W", str(W),
        "--src_through", str(src_t),
        "--dst_through", str(dst_t),
        "--src_around", str(src_a),
        "--dst_around", str(dst_a),
        "--output", shapiro_report,
    ])

    # --- Step 8: Print the safe next τ-geometry command ---
    print("\n[BLOCKED BY DESIGN] τ-geometry dense build at 512 is not allowed.")
    print("Next safe step is sparse τ-geometry (CPU).")
    print("When your sparse v4 file exists, run:")
    print(
        f"\npython src/gr_strict_pp_scalar/PP_markov_tau_geometry_v4_sparse_cpu.py "
        f"--edges_flat {edges_flat_out} "
        f"--edges_curved {edges_curved_out} "
        f"--trace_weights {trace_out} "
        f"--H {H} --W {W} "
        f"--mass_mask {mass_out} "
        f"--report src/gr_strict_pp_scalar/PP_markov_tau_geometry_{H}_{case}_PPV1.json "
        f"--npz_out src/gr_strict_pp_scalar/PP_markov_tau_geometry_{H}_{case}_PPV1.npz"
    )

    print("\nDONE.")

if __name__ == "__main__":
    main()

