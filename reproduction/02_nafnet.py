# ============================================================
# RECREATE THE EXACT NAFNET USED IN OUR NOTEBOOK
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# LayerNorm2d
# ============================================================

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):

        ctx.eps = eps

        N, C, H, W = x.size()

        mu = x.mean(
            1,
            keepdim=True
        )

        var = (
            (x - mu)
            .pow(2)
            .mean(
                1,
                keepdim=True
            )
        )

        y = (
            x - mu
        ) / torch.sqrt(
            var + eps
        )

        ctx.save_for_backward(
            y,
            var,
            weight
        )

        y = (
            weight.view(
                1, C, 1, 1
            ) * y
            +
            bias.view(
                1, C, 1, 1
            )
        )

        return y


    @staticmethod
    def backward(
        ctx,
        grad_output
    ):

        eps = ctx.eps

        y, var, weight = (
            ctx.saved_tensors
        )

        N, C, H, W = (
            grad_output.size()
        )

        g = (
            grad_output *
            weight.view(
                1, C, 1, 1
            )
        )

        mean_g = g.mean(
            1,
            keepdim=True
        )

        mean_gy = (
            g * y
        ).mean(
            1,
            keepdim=True
        )

        grad_input = (
            1.0 /
            torch.sqrt(
                var + eps
            )
        ) * (
            g
            -
            y * mean_gy
            -
            mean_g
        )

        grad_weight = (
            grad_output * y
        ).sum(
            dim=(0, 2, 3)
        )

        grad_bias = (
            grad_output
            .sum(
                dim=(0, 2, 3)
            )
        )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None
        )


class LayerNorm2d(nn.Module):

    def __init__(
        self,
        channels,
        eps=1e-6
    ):

        super().__init__()

        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.ones(
                    channels
                )
            )
        )

        self.register_parameter(
            "bias",
            nn.Parameter(
                torch.zeros(
                    channels
                )
            )
        )

        self.eps = eps


    def forward(self, x):

        return LayerNormFunction.apply(
            x,
            self.weight,
            self.bias,
            self.eps
        )


# ============================================================
# SimpleGate
# ============================================================

class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(
            2,
            dim=1
        )

        return x1 * x2


# ============================================================
# NAFBlock
# Exact structure from notebook
# ============================================================

class NAFBlock(nn.Module):

    def __init__(
        self,
        c,
        DW_Expand=2,
        FFN_Expand=2,
        drop_out_rate=0.
    ):

        super().__init__()

        dw_channel = (
            c * DW_Expand
        )

        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True
        )

        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        # --------------------------------------------------------
        # Simplified Channel Attention
        # --------------------------------------------------------

        self.sca = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True
            )
        )

        self.sg = SimpleGate()

        # --------------------------------------------------------
        # Feed Forward Network
        # --------------------------------------------------------

        ffn_channel = (
            FFN_Expand * c
        )

        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        self.norm1 = LayerNorm2d(
            c
        )

        self.norm2 = LayerNorm2d(
            c
        )

        self.dropout1 = (
            nn.Dropout(
                drop_out_rate
            )
            if drop_out_rate > 0
            else nn.Identity()
        )

        self.dropout2 = (
            nn.Dropout(
                drop_out_rate
            )
            if drop_out_rate > 0
            else nn.Identity()
        )

        self.beta = nn.Parameter(
            torch.zeros(
                (1, c, 1, 1)
            ),
            requires_grad=True
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                (1, c, 1, 1)
            ),
            requires_grad=True
        )


    def forward(self, inp):

        x = inp

        x = self.norm1(
            x
        )

        x = self.conv1(
            x
        )

        x = self.conv2(
            x
        )

        x = self.sg(
            x
        )

        x = (
            x *
            self.sca(x)
        )

        x = self.conv3(
            x
        )

        x = self.dropout1(
            x
        )

        y = (
            inp
            +
            x * self.beta
        )

        x = self.conv4(
            self.norm2(y)
        )

        x = self.sg(
            x
        )

        x = self.conv5(
            x
        )

        x = self.dropout2(
            x
        )

        return (
            y
            +
            x * self.gamma
        )


# ============================================================
# NAFNet
# Exact configuration used in our project
# ============================================================

class NAFNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[]
    ):

        super().__init__()

        self.intro = nn.Conv2d(
            img_channel,
            width,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.ending = nn.Conv2d(
            width,
            img_channel,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.encoders = nn.ModuleList()

        self.decoders = nn.ModuleList()

        self.middle_blks = nn.ModuleList()

        self.ups = nn.ModuleList()

        self.downs = nn.ModuleList()

        # --------------------------------------------------------
        # Encoder
        # --------------------------------------------------------

        chan = width

        for num in enc_blk_nums:

            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan
                        )
                        for _ in range(num)
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    chan,
                    chan * 2,
                    kernel_size=2,
                    stride=2
                )
            )

            chan = chan * 2

        # --------------------------------------------------------
        # Middle blocks
        # --------------------------------------------------------

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(
                    chan
                )
                for _ in range(
                    middle_blk_num
                )
            ]
        )

        # --------------------------------------------------------
        # Decoder
        # --------------------------------------------------------

        for num in dec_blk_nums:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        chan,
                        chan * 2,
                        kernel_size=1,
                        bias=False
                    ),
                    nn.PixelShuffle(2)
                )
            )

            chan = chan // 2

            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan
                        )
                        for _ in range(num)
                    ]
                )
            )

        self.padder_size = 2 ** len(
            self.encoders
        )


    def forward(self, inp):

        B, C, H, W = inp.shape

        x = self.intro(
            inp
        )

        encs = []

        # --------------------------------------------------------
        # Encoder
        # --------------------------------------------------------

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):

            x = encoder(
                x
            )

            encs.append(
                x
            )

            x = down(
                x
            )

        # --------------------------------------------------------
        # Middle
        # --------------------------------------------------------

        x = self.middle_blks(
            x
        )

        # --------------------------------------------------------
        # Decoder
        # --------------------------------------------------------

        for decoder, up, enc_skip in zip(
            self.decoders,
            self.ups,
            encs[::-1]
        ):

            x = up(
                x
            )

            x = x + enc_skip

            x = decoder(
                x
            )

        x = self.ending(
            x
        )

        x = x + inp

        return x


# ============================================================
# VERIFY EXACT PROJECT CONFIGURATION
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

test_nafnet = NAFNet(
    img_channel=1,
    width=32,
    middle_blk_num=2,
    enc_blk_nums=[
        1, 1, 1, 1
    ],
    dec_blk_nums=[
        1, 1, 1, 1
    ]
).to(device)

print("=" * 70)
print("NAFNET RECONSTRUCTED")
print("=" * 70)

print(
    "NAFNet:",
    "NAFNet" in globals()
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in test_nafnet.parameters()
    ) / 1e6,
    "M"
)

# Test forward
dummy = torch.randn(
    1,
    1,
    128,
    128,
    device=device
)

with torch.no_grad():

    dummy_out = test_nafnet(
        dummy
    )

print(
    "Input shape :",
    tuple(dummy.shape)
)

print(
    "Output shape:",
    tuple(dummy_out.shape)
)

print("=" * 70)