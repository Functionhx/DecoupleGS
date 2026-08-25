#!/usr/bin/env python3
"""Compare VAD's native dataloader with the closed-loop FIFO adapter.

The native test configuration loads point clouds and map annotations for
evaluation, although VAD's released camera-only ``forward_test`` does not
consume them.  This audit retains the official image transforms, metadata,
ego history, and command while dropping evaluation-only file loaders so it can
run on the camera subset used by the DecoupleGS reproduction.

Run from the VAD_SIM repository in the ``decouplegs-ad`` environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--adapter-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def protocol_records(path: Path) -> tuple[dict[str, dict], list[str]]:
    payload = json.loads(path.read_text())
    frames = payload["frames"]
    return (
        {str(frame["sample_token"]): frame for frame in frames},
        [str(frame["sample_token"]) for frame in frames],
    )


def adapter_records(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    return {
        str(record["sample_token"]): record
        for record in payload["records"]
        if record.get("sample_token") is not None
    }


def inference_pipeline(cfg) -> list[dict]:
    """Keep official VAD image transforms and ego input formatting only."""

    original = cfg.data.test.pipeline
    load_images = copy.deepcopy(
        next(step for step in original if step["type"] == "LoadMultiViewImageFromFiles")
    )
    normalize = copy.deepcopy(
        next(step for step in original if step["type"] == "NormalizeMultiviewImage")
    )
    augmentation = copy.deepcopy(
        next(step for step in original if step["type"] == "MultiScaleFlipAug3D")
    )
    augmentation["transforms"] = [
        step
        for step in augmentation["transforms"]
        if step["type"]
        in {
            "RandomScaleImageMultiViewImage",
            "PadMultiViewImage",
            "CustomDefaultFormatBundle3D",
            "CustomCollect3D",
        }
    ]
    collect = next(
        step
        for step in augmentation["transforms"]
        if step["type"] == "CustomCollect3D"
    )
    collect["keys"] = ["img", "ego_his_trajs", "ego_fut_cmd"]
    return [load_images, normalize, augmentation]


def unwrap_single(value, name: str):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} has {len(value)} test augmentations; expected one")
        return value[0]
    return value


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet3d.datasets import build_dataset

    protocol, protocol_order = protocol_records(args.protocol)
    adapter = adapter_records(args.adapter_result)

    cfg = Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = str(args.ann_file.resolve())
    cfg.data.test.data_root = str(args.data_root.resolve())
    cfg.data.test.pipeline = inference_pipeline(cfg)
    samples_per_gpu = int(cfg.data.test.pop("samples_per_gpu", 1))

    dataset = build_dataset(cfg.data.test)
    # Vector-map labels are evaluation-only and require no model input.
    dataset.is_vis_on_test = False
    dataset_tokens = [str(info["token"]) for info in dataset.data_infos]
    if dataset_tokens != protocol_order:
        raise ValueError("filtered annotation order does not match the protocol")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    records = []
    for index, data in enumerate(data_loader):
        token = dataset_tokens[index]
        # VAD's fork expects ``img_metas`` to retain the outer test-time
        # augmentation list, while its image and ego tensors are already
        # unwrapped.  This asymmetry matches the fork's forward_test contract.
        model_inputs = {
            "img": unwrap_single(data["img"], "img"),
            "img_metas": data["img_metas"],
            "ego_his_trajs": unwrap_single(data["ego_his_trajs"], "ego_his_trajs"),
            "ego_fut_cmd": unwrap_single(data["ego_fut_cmd"], "ego_fut_cmd"),
        }
        with torch.inference_mode():
            result = model(return_loss=False, rescale=True, **model_inputs)
        bbox = result[0]["pts_bbox"]
        command = int(torch.argmax(bbox["ego_fut_cmd"]).item())
        increments = bbox["ego_fut_preds"][command].detach().cpu().numpy()
        plan = np.cumsum(increments, axis=0).astype(np.float64)

        frame = protocol[token]
        gt = np.asarray(frame["gt_traj"], dtype=np.float64)[: len(plan)]
        mask = np.asarray(frame["gt_mask"], dtype=bool)[: len(plan)]
        ade = float(np.linalg.norm(plan[mask] - gt[mask], axis=-1).mean())

        adapter_record = adapter.get(token)
        if adapter_record is None:
            adapter_ade = None
            adapter_delta_mean = None
            adapter_delta_max = None
            adapter_command = None
        else:
            adapter_plan = np.asarray(adapter_record["plan_traj"], dtype=np.float64)[
                : len(plan), :2
            ]
            delta = np.linalg.norm(plan - adapter_plan, axis=-1)
            adapter_ade = float(adapter_record["ade_to_logged_gt_m"])
            adapter_delta_mean = float(delta.mean())
            adapter_delta_max = float(delta.max())
            adapter_command = int(np.argmax(np.asarray(frame["planner_info"]["planner_state"]["vad_ego_fut_cmd"])))

        records.append(
            {
                "sequence_index": index,
                "sample_token": token,
                "command_index": command,
                "adapter_command_index": adapter_command,
                "official_plan_traj": plan.tolist(),
                "gt_traj": gt.tolist(),
                "gt_mask": mask.tolist(),
                "official_ade_to_logged_gt_m": ade,
                "adapter_ade_to_logged_gt_m": adapter_ade,
                "official_to_adapter_plan_mean_m": adapter_delta_mean,
                "official_to_adapter_plan_max_m": adapter_delta_max,
            }
        )
        print(
            f"[{index + 1:02d}/{len(dataset_tokens):02d}] {token} "
            f"official ADE={ade:.3f}m adapter delta="
            f"{'n/a' if adapter_delta_mean is None else f'{adapter_delta_mean:.3f}m'}",
            flush=True,
        )
        del result
        torch.cuda.empty_cache()

    official_ades = [record["official_ade_to_logged_gt_m"] for record in records]
    paired = [
        record["official_to_adapter_plan_mean_m"]
        for record in records
        if record["official_to_adapter_plan_mean_m"] is not None
    ]
    command_matches = [
        record["command_index"] == record["adapter_command_index"]
        for record in records
        if record["adapter_command_index"] is not None
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "vad_official_dataloader_vs_fifo_adapter",
        "samples": len(records),
        "official_mADE_to_logged_gt_m": float(np.mean(official_ades)),
        "official_to_adapter_plan_mean_m": None if not paired else float(np.mean(paired)),
        "command_match_rate": None if not command_matches else float(np.mean(command_matches)),
        "records": records,
        "inputs": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "ann_file": str(args.ann_file.resolve()),
            "data_root": str(args.data_root.resolve()),
            "protocol": str(args.protocol.resolve()),
            "adapter_result": (
                None if args.adapter_result is None else str(args.adapter_result.resolve())
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
