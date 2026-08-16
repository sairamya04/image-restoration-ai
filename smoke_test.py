"""Small architecture check that does not require the training dataset or weights."""

import torch

from models.integrated_model import IntegratedRestorationModel


model = IntegratedRestorationModel().eval()
input_tensor = torch.randn(1, 1, 128, 128)

with torch.inference_mode():
    output = model(input_tensor)

expected = (1, 1, 256, 256)
assert tuple(output.shape) == expected, (output.shape, expected)

parameters = sum(p.numel() for p in model.parameters())
print(f"Output shape : {tuple(output.shape)}")
print(f"Parameters   : {parameters:,} ({parameters / 1e6:.6f} M)")
print("Smoke test passed.")
