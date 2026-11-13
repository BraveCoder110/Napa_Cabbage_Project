"""Session 4.1: inspect torchvision and check a ResNet18 encoder on CPU.

Does NOT train on any dataset, alter prior experiments, or download weights
unless --download-weights is explicitly supplied.
"""
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights


def check_weight_cache(weights):
    """Verify whether the expected official pretrained weight file is cached."""
    cache = Path(torch.hub.get_dir()) / 'checkpoints'
    candidate = cache / Path(urlparse(weights.url).path).name
    if not candidate.is_file():
        return candidate, False, 'not cached'
    # Verify the SHA256 prefix embedded in the official torchvision filename.
    suffix = candidate.stem.rsplit('-', 1)[-1]
    if len(suffix) == 8 and all(ch in '0123456789abcdef' for ch in suffix.lower()):
        digest = hashlib.file_digest(candidate.open('rb'), 'sha256').hexdigest()
        if not digest.startswith(suffix):
            return candidate, False, 'cached file failed hash verification'
    return candidate, True, 'cached and hash-checked'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-weights', action='store_true',
                        help='Explicitly download official ImageNet ResNet18 pretrained weights if needed.')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(42)
    print(f'PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}')
    print(f'Device: CPU | CUDA available: {torch.cuda.is_available()}')
    weights = ResNet18_Weights.DEFAULT
    path, cached, status = check_weight_cache(weights)
    print(f'Official pretrained encoder: ResNet18_Weights.DEFAULT')
    print(f'Weight cache: {path}')
    print(f'Weight status: {status}')
    if args.download_weights and not cached:
        print('Downloading official pretrained weights (explicit opt-in)...', flush=True)
        try:
            resnet18(weights=weights)
        except Exception as exc:
            raise RuntimeError('Pretrained-weight download failed; verify internet access and retry.') from exc
        path, cached, status = check_weight_cache(weights)
        print(f'Weight status after download: {status}')
    # Architecture-only check: does not claim to use transfer learning.
    model = resnet18(weights=None)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    features = []
    z = model.conv1(x)
    z = model.bn1(z)
    z = model.relu(z)
    features.append(tuple(z.shape))
    z = model.maxpool(z)
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        z = layer(z)
        features.append(tuple(z.shape))
    print(f'Input shape: {tuple(x.shape)}')
    print(f'Encoder feature shapes: {features}')
    print('PASS: ResNet18 CPU encoder forward pass.')
    if cached:
        print('PASS: Pretrained weights are ready for Session 4.2.')
    else:
        print('NEXT: For real transfer learning, obtain pretrained weights using --download-weights.')
        print('NOTE: Running weights=None is NOT transfer learning.')


if __name__ == '__main__':
    main()
