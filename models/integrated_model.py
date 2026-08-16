import torch
import torch.nn as nn
from .nafnet import NAFNet


class DetailSRHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_branch = nn.Sequential(nn.Conv2d(1,4,3,padding=1), nn.PixelShuffle(2))
        self.detail_branch = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.GELU(),
            nn.Conv2d(16,16,3,padding=1), nn.GELU(),
            nn.Conv2d(16,4,3,padding=1), nn.PixelShuffle(2)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.GELU(), nn.Conv2d(16,1,3,padding=1)
        )
    def forward(self,x):
        main=self.main_branch(x); detail=self.detail_branch(x)
        return self.fusion(torch.cat([main,detail],dim=1))


class DegradationAnalyzer(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.GELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(64,64,3,padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.embedding=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,embedding_dim))
    def forward(self,x):
        x=self.features(x).flatten(1)
        return self.embedding(x)


class IntegratedRestorationModel(nn.Module):
    """Validated final model: NAFNet + degradation-conditioned modulation + detail/SR head."""
    def __init__(self):
        super().__init__()
        self.nafnet=NAFNet(img_channel=1,width=32,middle_blk_num=2,
                           enc_blk_nums=[1,1,1,1],dec_blk_nums=[1,1,1,1])
        self.analyzer=DegradationAnalyzer(embedding_dim=16)
        self.gamma=nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,1))
        self.beta=nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,1))
        self.sr_head=DetailSRHead()
        nn.init.zeros_(self.gamma[-1].weight); nn.init.zeros_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight); nn.init.zeros_(self.beta[-1].bias)

    def forward(self,x,return_details=False):
        embedding=self.analyzer(x)
        features=self.nafnet(x)
        gamma=self.gamma(embedding)[:,:,None,None]
        beta=self.beta(embedding)[:,:,None,None]
        features=features*(1.0+gamma)+beta
        output=self.sr_head(features)
        if return_details:
            return output,embedding,gamma,beta
        return output
