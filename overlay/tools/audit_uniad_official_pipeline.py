#!/usr/bin/env python3
"""Compare UniAD's native dataloader with the closed-loop FIFO adapter.

Run this script from the UniAD_SIM repository in the ``decouplegs-ad``
environment.  It intentionally feeds only the arguments accepted by
``UniAD.forward_test``; the released UniAD_SIM test pipeline collects several
ground-truth tensors that this fork's inference signature does not accept.
"""

from __future__ import annotations

import argparse
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


def _protocol_records(path: Path) -> tuple[dict[str, dict], list[str]]:
    payload = json.loads(path.read_text())
    records = {str(frame["sample_token"]): frame for frame in payload["frames"]}
    return records, [str(frame["sample_token"]) for frame in payload["frames"]]


def _adapter_records(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    return {
        str(record["sample_token"]): record
        for record in payload["records"]
        if record.get("sample_token") is not None
    }


def main() -> None:
    args = parse_args()

    # The script is normally invoked by absolute path while cwd is UniAD_SIM.
    # Keep that repository importable for its ``projects`` package.
    sys.path.insert(0, str(Path.cwd()))
    from mmdet.datasets import replace_ImageToTensor
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet3d.datasets import build_dataset

    protocol, protocol_order = _protocol_records(args.protocol)
    adapter = _adapter_records(args.adapter_result)

    cfg = Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = str(args.ann_file.resolve())
    cfg.data.test.data_root = str(args.data_root.resolve())
    cfg.data.test.pipeline[0].img_root = str(args.data_root.resolve())
    samples_per_gpu = int(cfg.data.test.pop("samples_per_gpu", 1))
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
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
    accepted = ("img", "img_metas", "l2g_t", "l2g_r_mat", "timestamp", "command")
    for index, data in enumerate(data_loader):
        token = dataset_tokens[index]
        model_inputs = {key: data[key] for key in accepted}
        # MultiScaleFlipAug3D wraps the image tensor in a one-element test-time
        # augmentation list.  UniAD_SIM's forward_test implementation in this
        # fork expects the tensor itself (the FIFO adapter already supplies it
        # in that form).
        if isinstance(model_inputs["img"], (list, tuple)):
            if len(model_inputs["img"]) != 1:
                raise ValueError("the audit supports exactly one test-time augmentation")
            model_inputs["img"] = model_inputs["img"][0]
        if isinstance(model_inputs["timestamp"], (list, tuple)):
            if len(model_inputs["timestamp"]) != 1:
                raise ValueError("the audit supports a batch size of one")
            model_inputs["timestamp"] = model_inputs["timestamp"][0]
        for key in ("l2g_t", "l2g_r_mat"):
            if isinstance(model_inputs[key], (list, tuple)):
                if len(model_inputs[key]) != 1:
                    raise ValueError("the audit supports a batch size of one")
                model_inputs[key] = model_inputs[key][0]
            # Match the FIFO adapter and UniAD's float32 reference points.
            model_inputs[key] = model_inputs[key].float()
        with torch.inference_mode():
            result = model(
                return_loss=False,
                rescale=True,
                **model_inputs,
            )
        plan = (
            result[0]["planning"]["result_planning"]["sdc_traj"][0, :, :2]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        frame = protocol[token]
        gt = np.asarray(frame["gt_traj"], dtype=np.float64)[: len(plan)]
        mask = np.asarray(frame["gt_mask"], dtype=bool)[: len(plan)]
        ade = float(np.linalg.norm(plan[mask] - gt[mask], axis=-1).mean())

        adapter_record = adapter.get(token)
        if adapter_record is None:
            adapter_ade = None
            adapter_delta_mean = None
            adapter_delta_max = None
        else:
            adapter_plan = np.asarray(adapter_record["plan_traj"], dtype=np.float64)[
                : len(plan), :2
            ]
            adapter_error = np.linalg.norm(plan - adapter_plan, axis=-1)
            adapter_ade = float(adapter_record["ade_to_logged_gt_m"])
            adapter_delta_mean = float(adapter_error.mean())
            adapter_delta_max = float(adapter_error.max())

        records.append(
            {
                "sequence_index": index,
                "sample_token": token,
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
    payload = {
        "schema_version": 1,
        "benchmark": "uniad_official_dataloader_vs_fifo_adapter",
        "samples": len(records),
        "official_mADE_to_logged_gt_m": float(np.mean(official_ades)),
        "official_to_adapter_plan_mean_m": None if not paired else float(np.mean(paired)),
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
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
