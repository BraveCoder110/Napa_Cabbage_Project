"""CPU forward/backward test for the transfer-learning segmentation model."""
import argparse
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.resnet18_unet import ResNet18UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-encoder", action="store_true", help="Only check architecture without official weights")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = ResNet18UNet(pretrained=not args.random_encoder).cpu()
    model.eval()  # BatchNorm uses stored running statistics for this small test batch.
    x = torch.rand(2, 3, 128, 128)
    labels = torch.randint(0, 2, (2, 1, 128, 128)).float()
    logits = model(x)
    assert logits.shape == labels.shape, (logits.shape, labels.shape)
    assert torch.isfinite(logits).all(), "Non-finite logits"
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    assert torch.isfinite(loss), "Non-finite loss"
    assert model.enc4[0].conv1.weight.grad is not None, "No encoder gradient"
    assert model.final_up[-1].weight.grad is not None, "No decoder gradient"
    params = sum(p.numel() for p in model.parameters())
    print(f"Device: CPU | torchvision encoder: ResNet18")
    print(f"Pretrained encoder loaded: {model.pretrained_encoder}")
    print(f"Input: {tuple(x.shape)} | Logits: {tuple(logits.shape)}")
    print(f"Parameters: {params:,} | BCE loss: {loss.item():.4f}")
    print("PASS: Forward shape, finite loss, encoder and decoder backward gradients.")
    if args.random_encoder:
        print("NOTE: Random-encoder mode validates architecture only; NOT transfer learning.")


if __name__ == "__main__":
    main()
