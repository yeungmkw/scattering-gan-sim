"""Small dependency and device sanity check."""

from __future__ import annotations

import importlib


REQUIRED_MODULES = [
    "matplotlib",
    "numpy",
    "cv2",
    "PIL",
    "scipy",
    "torch",
    "torchvision",
]


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    for name in REQUIRED_MODULES:
        print(f"{name}: {module_version(name)}")

    import torch

    print(f"torch device cuda: {torch.cuda.is_available()}")
    print(f"torch device mps: {torch.backends.mps.is_available()}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.rand(2, 1, 16, 16, device=device)
    print(f"sanity tensor: shape={tuple(x.shape)} device={x.device} mean={x.mean().item():.4f}")


if __name__ == "__main__":
    main()
