# ============================================================
# FINAL INTEGRATED MODEL
# NAFNet + DEGRADATION CONDITIONING + DETAIL/SR
# Exact architecture used for the final trained checkpoint.
# ============================================================

import torch
import torch.nn as nn

from models.nafnet import NAFNet


class DetailSRHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.detail_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        main = self.main_branch(x)
        detail = self.detail_branch(x)
        combined = torch.cat([main, detail], dim=1)
        return self.fusion(combined)


class DegradationAnalyzer(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.embedding = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.embedding(x)


class IntegratedRestorationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nafnet = NAFNet(
            img_channel=1,
            width=32,
            middle_blk_num=2,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1]
        )
        self.analyzer = DegradationAnalyzer(embedding_dim=16)
        self.gamma = nn.Sequential(
            nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 1)
        )
        self.beta = nn.Sequential(
            nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 1)
        )
        self.sr_head = DetailSRHead()

        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.zeros_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)

    def forward(self, x, return_details=False):
        embedding = self.analyzer(x)
        features = self.nafnet(x)
        gamma = self.gamma(embedding)[:, :, None, None]
        beta = self.beta(embedding)[:, :, None, None]
        features = features * (1.0 + gamma) + beta
        output = self.sr_head(features)
        if return_details:
            return output, embedding, gamma, beta
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegratedRestorationModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Integrated model created")
    print(f"Total parameters: {total_params / 1e6:.6f} M")
    x = torch.randn(1, 1, 128, 128, device=device)
    with torch.no_grad():
        y, embedding, gamma, beta = model(x, return_details=True)
    print("Input:", tuple(x.shape))
    print("NAFNet output:", tuple(model.nafnet(x).shape))
    print("Embedding:", tuple(embedding.shape))
    print("Gamma:", tuple(gamma.shape))
    print("Beta:", tuple(beta.shape))
    print("Final output:", tuple(y.shape))
    assert y.shape[-2] == x.shape[-2] * 2
    assert y.shape[-1] == x.shape[-1] * 2
    print("✓ DIMENSIONS CORRECT")
