#!/usr/bin/env python3
import argparse, json

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="STRICT PP vector evidence pack v1 with admissibility gates.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--azimuthal_json", required=False, default=None)
    ap.add_argument("--frame_dragging_json", required=False, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    P = {
        "case": args.case,
        "notes": (
            "STRICT PP vector pack v1. Aggregates vector observables with admissibility gates. "
            "No PDE, no Laplacian/Poisson, no GR ansatz, no regression."
        )
    }

    # --- Azimuthal flux ---
    if args.azimuthal_json:
        az = load(args.azimuthal_json)
        P["azimuthal_flux"] = az
        applicable = bool(az.get("VECTOR_AZIMUTHAL_FLUX_APPLICABLE"))
        PAS = az.get("PASS_vector_azimuthal_flux_PP")
        P["PASS_azimuthal_or_NA"] = True if not applicable else bool(PAS)
    else:
        P["azimuthal_flux"] = None
        P["PASS_azimuthal_or_NA"] = True

    # --- Frame-dragging proxy ---
    if args.frame_dragging_json:
        fd = load(args.frame_dragging_json)
        P["frame_dragging"] = fd
        applicable = bool(fd.get("FRAME_DRAGGING_APPLICABLE"))
        PAS = fd.get("PASS_frame_dragging_PP")
        P["PASS_frame_dragging_or_NA"] = True if not applicable else bool(PAS)
    else:
        P["frame_dragging"] = None
        P["PASS_frame_dragging_or_NA"] = True

    # Overall vector PASS (strict PP logic)
    P["overall_PASS_vector_strict_PP_v1"] = bool(
        P["PASS_azimuthal_or_NA"] and P["PASS_frame_dragging_or_NA"]
    )

    with open(args.output, "w") as f:
        json.dump(P, f, indent=2, sort_keys=True)

    print("WROTE", args.output)
    print("overall_PASS_vector_strict_PP_v1 =", P["overall_PASS_vector_strict_PP_v1"])

if __name__ == "__main__":
    main()
