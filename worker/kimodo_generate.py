from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beyond Motion Kimodo worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser.parse_args()


def configure_imports() -> None:
    extension_root = Path(__file__).resolve().parents[1]
    vendor_root = extension_root / "_vendor"
    kimodo_root = extension_root / "vendor" / "kimodo"
    motion_correction_root = kimodo_root / "MotionCorrection" / "python"
    if vendor_root.is_dir():
        site.addsitedir(str(vendor_root))
    for path in (motion_correction_root, kimodo_root):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> int:
    args = parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if request.get("enable_mps_fallback", True) else "0"

    configure_imports()

    import torch
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.device_utils import resolve_torch_device
    from kimodo.tools import seed_everything

    requested_device = request.get("device", "auto")
    device = resolve_torch_device(requested_device, torch_mod=torch)
    model = load_model(request["model_name"], device=device, default_family="Kimodo")
    constraint_lst = load_constraints_lst(request["constraints"], model.skeleton, device=device)
    if request.get("seed") is not None:
        seed_everything(int(request["seed"]))

    cfg_type = request.get("cfg_type")
    cfg_kwargs = {}
    if cfg_type == "nocfg":
        cfg_kwargs["cfg_type"] = "nocfg"
    elif cfg_type == "regular":
        cfg_kwargs["cfg_type"] = "regular"
        cfg_kwargs["cfg_weight"] = float(request.get("cfg_text_weight", 2.0))
    elif cfg_type == "separated":
        cfg_kwargs["cfg_type"] = "separated"
        cfg_kwargs["cfg_weight"] = [
            float(request.get("cfg_text_weight", 2.0)),
            float(request.get("cfg_constraint_weight", 2.0)),
        ]

    output = model(
        request["prompt"],
        int(request["num_frames"]),
        num_denoising_steps=int(request["diffusion_steps"]),
        constraint_lst=constraint_lst,
        post_processing=bool(request.get("post_processing", False)),
        **cfg_kwargs,
    )

    def as_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    response_path = Path(args.response)
    output_npz = response_path.with_name("output_motion.npz")
    np.savez_compressed(
        output_npz,
        local_rot_mats=as_numpy(output["local_rot_mats"]),
        root_positions=as_numpy(output["root_positions"]),
        global_rot_mats=as_numpy(output["global_rot_mats"]),
        posed_joints=as_numpy(output["posed_joints"]),
        foot_contacts=as_numpy(output["foot_contacts"]),
        smooth_root_pos=as_numpy(output["smooth_root_pos"]),
        global_root_heading=as_numpy(output["global_root_heading"]),
    )
    response_path.write_text(
        json.dumps(
            {
                "output_npz": str(output_npz),
                "device": device,
                "requested_device": requested_device,
                "fps": float(model.fps),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
