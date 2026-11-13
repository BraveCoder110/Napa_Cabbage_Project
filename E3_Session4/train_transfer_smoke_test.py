"""Verify that pretrained ResNet18 U-Net can train on real paired images.

This is a short engineering test, not a segmentation performance experiment.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "E3_Session3"))
from train_smoke_test import PairedZipDataset, combined_loss  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from resnet18_unet import ResNet18UNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--random-init-diagnostic", action="store_true", help="Diagnostic only: disable pretrained weights")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.size < 32 or args.size % 32 != 0 or args.lr <= 0:
        parser.error("Invalid parameters: size >=32 divisible by 32; positive steps, batch-size and lr")
    for path in (args.images, args.masks):
        if not Path(path).is_file():
            parser.error(f"Missing data ZIP: {path}")
    out = Path(args.output)
    if out.exists():
        parser.error(f"Output path exists; choose a new directory to prevent overwriting: {out}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    dataset = PairedZipDataset(args.images, args.masks, "train", args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    try:
        model = ResNet18UNet(pretrained=not args.random_init_diagnostic).cpu()
    except Exception as exc:
        raise RuntimeError("Failed to load pretrained weights. Run Session 4.1 cache check before retrying.") from exc
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    print(f"Device: CPU | Training pairs: {len(dataset)} | Pretrained encoder loaded: {model.pretrained_encoder}", flush=True)

    history = []
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            images, targets, names = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets, names = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != targets.shape:
            raise RuntimeError(f"Output/target mismatch: {logits.shape} / {targets.shape}")
        loss = combined_loss(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not torch.isfinite(grad):
            raise RuntimeError("Non-finite gradient detected")
        before = model.final_up[-1].weight.detach().clone()
        optimizer.step()
        delta = (model.final_up[-1].weight.detach() - before).abs().max().item()
        if delta <= 0:
            raise RuntimeError("Optimizer failed to update output head")
        history.append({"step": step, "loss": loss.item(), "max_head_update": delta, "image_names": ";".join(names)})
        print(f"Step {step}/{args.steps} | loss={loss.item():.4f} | max head update={delta:.7f} | {list(names)}", flush=True)

    out.mkdir(parents=True, exist_ok=False)
    # Store CPU tensors for portable, safer weights_only loading.
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args), "steps_completed": args.steps, "purpose": "smoke_test_only",
                "pretrained_encoder": model.pretrained_encoder}, out / "resnet18_unet_smoke_test.pth")
    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "max_head_update", "image_names"])
        writer.writeheader()
        writer.writerows(history)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {out / 'resnet18_unet_smoke_test.pth'}")
    print(f"Saved log: {out / 'training_log.csv'}")
    print("PASS: Real-data ResNet18 U-Net training smoke test (NOT a performance evaluation).")


if __name__ == "__main__":
    main()
