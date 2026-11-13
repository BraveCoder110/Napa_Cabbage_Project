"""CPU smoke test for the lightweight binary segmentation U-Net."""

import argparse
import math
import torch
from models.unet import LightweightUNet


def main():
    parser = argparse.ArgumentParser(description="Validate U-Net shape and gradients on CPU")
    parser.add_argument("--size", type=int, default=128, help="Input height and width")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()
    if args.size < 8 or args.batch_size < 1:
        parser.error("--size must be >= 8 and --batch-size must be >= 1")

    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = LightweightUNet(base_channels=args.base_channels).cpu()
    model.eval()
    images = torch.rand(args.batch_size, 3, args.size, args.size)

    # A smoke test checks tensor behavior, not segmentation accuracy.
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
    expected_shape = (args.batch_size, 1, args.size, args.size)
    assert tuple(logits.shape) == expected_shape, (logits.shape, expected_shape)
    assert torch.isfinite(logits).all().item(), "Non-finite output detected"
    assert 0 <= probabilities.min().item() <= probabilities.max().item() <= 1

    # Check whether gradients can flow through the model.
    model.train()
    output = model(images)
    target = torch.randint(0, 2, output.shape).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
    assert math.isfinite(loss.item())
    loss.backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all().item()

    count = sum(parameter.numel() for parameter in model.parameters())
    print("Device: CPU")
    print(f"Input shape:  {tuple(images.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Model parameters: {count:,}")
    print(f"Output logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"Probability range: [{probabilities.min().item():.4f}, {probabilities.max().item():.4f}]")
    print("Forward pass: PASS")
    print("Backward pass: PASS")
    print("NOTE: Random inputs/targets do not measure model accuracy.")


if __name__ == "__main__":
    main()
