# ============================================================
# FINAL INTEGRATED MODEL
# NAFNet + DEGRADATION CONDITIONING + DETAIL/SR
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DETAIL / SR HEAD
# ============================================================

class DetailSRHead(nn.Module):

    def __init__(self):

        super().__init__()

        # Main reconstruction
        self.main_branch = nn.Sequential(

            nn.Conv2d(
                1, 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2)
        )

        # Dedicated detail reconstruction
        self.detail_branch = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                16, 16,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                16, 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2)
        )

        # Main + detail fusion
        self.fusion = nn.Sequential(

            nn.Conv2d(
                2, 16,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                16, 1,
                kernel_size=3,
                padding=1
            )
        )


    def forward(self, x):

        main = self.main_branch(x)

        detail = self.detail_branch(x)

        combined = torch.cat(
            [main, detail],
            dim=1
        )

        return self.fusion(
            combined
        )


# ============================================================
# DEGRADATION ANALYZER
#
# Learns a compact representation of the degradation.
# ============================================================

class DegradationAnalyzer(nn.Module):

    def __init__(
        self,
        embedding_dim=16
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                16, 32,
                3,
                stride=2,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                32, 64,
                3,
                stride=2,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                64, 64,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.embedding = nn.Sequential(

            nn.Linear(
                64,
                32
            ),

            nn.GELU(),

            nn.Linear(
                32,
                embedding_dim
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = x.flatten(1)

        return self.embedding(x)


# ============================================================
# INTEGRATED RESTORATION MODEL
# ============================================================

class IntegratedRestorationModel(
    nn.Module
):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------------
        # NAFNet backbone
        # --------------------------------------------------------

        self.nafnet = NAFNet(
            img_channel=1,
            width=32,
            middle_blk_num=2,
            enc_blk_nums=[
                1, 1, 1, 1
            ],
            dec_blk_nums=[
                1, 1, 1, 1
            ]
        )

        # --------------------------------------------------------
        # Degradation analyzer
        # --------------------------------------------------------

        self.analyzer = (
            DegradationAnalyzer(
                embedding_dim=16
            )
        )

        # --------------------------------------------------------
        # Dynamic gamma
        # --------------------------------------------------------

        self.gamma = nn.Sequential(

            nn.Linear(
                16,
                16
            ),

            nn.GELU(),

            nn.Linear(
                16,
                1
            )
        )

        # --------------------------------------------------------
        # Dynamic beta
        # --------------------------------------------------------

        self.beta = nn.Sequential(

            nn.Linear(
                16,
                16
            ),

            nn.GELU(),

            nn.Linear(
                16,
                1
            )
        )

        # --------------------------------------------------------
        # Detail / SR reconstruction
        # --------------------------------------------------------

        self.sr_head = DetailSRHead()

        # --------------------------------------------------------
        # Start with neutral modulation
        #
        # gamma = 0
        # beta  = 0
        #
        # Therefore initially:
        #
        # F' = F
        # --------------------------------------------------------

        nn.init.zeros_(
            self.gamma[-1].weight
        )

        nn.init.zeros_(
            self.gamma[-1].bias
        )

        nn.init.zeros_(
            self.beta[-1].weight
        )

        nn.init.zeros_(
            self.beta[-1].bias
        )


    def forward(
        self,
        x,
        return_details=False
    ):

        # --------------------------------------------------------
        # Analyze degradation
        # --------------------------------------------------------

        embedding = self.analyzer(x)

        # --------------------------------------------------------
        # NAFNet
        # --------------------------------------------------------

        features = self.nafnet(x)

        # --------------------------------------------------------
        # Dynamic modulation
        # --------------------------------------------------------

        gamma = self.gamma(
            embedding
        )

        beta = self.beta(
            embedding
        )

        gamma = gamma[
            :, :, None, None
        ]

        beta = beta[
            :, :, None, None
        ]

        # --------------------------------------------------------
        # F' = F(1 + gamma) + beta
        # --------------------------------------------------------

        features = (
            features *
            (1.0 + gamma)
            +
            beta
        )

        # --------------------------------------------------------
        # 2× detail/SR reconstruction
        # --------------------------------------------------------

        output = self.sr_head(
            features
        )

        if return_details:

            return (
                output,
                embedding,
                gamma,
                beta
            )

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = IntegratedRestorationModel().to(
    device
)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("=" * 70)
print("INTEGRATED MODEL CREATED")
print("=" * 70)

print(
    f"Total parameters: "
    f"{total_params / 1e6:.6f} M"
)


# ============================================================
# FORWARD-PASS SANITY TEST
# ============================================================

lr_test, gt_test = next(
    iter(val_loader)
)

lr_test = lr_test.to(
    device
)

gt_test = gt_test.to(
    device
)

with torch.no_grad():

    pred_test, embedding, gamma, beta = (
        model(
            lr_test,
            return_details=True
        )
    )


print()
print("FORWARD TEST")
print("-" * 70)

print(
    "Input:",
    tuple(lr_test.shape)
)

print(
    "NAFNet output:",
    tuple(
        model.nafnet(
            lr_test
        ).shape
    )
)

print(
    "Embedding:",
    tuple(
        embedding.shape
    )
)

print(
    "Gamma:",
    tuple(
        gamma.shape
    )
)

print(
    "Beta:",
    tuple(
        beta.shape
    )
)

print(
    "Final output:",
    tuple(
        pred_test.shape
    )
)

print(
    "Ground truth:",
    tuple(
        gt_test.shape
    )
)

print("=" * 70)

# ------------------------------------------------------------
# HARD CHECK
# ------------------------------------------------------------

assert pred_test.shape == gt_test.shape, (
    f"OUTPUT/GROUND-TRUTH MISMATCH: "
    f"{pred_test.shape} vs {gt_test.shape}"
)

print("✓ DIMENSIONS CORRECT")
print("✓ READY FOR TRAINING")