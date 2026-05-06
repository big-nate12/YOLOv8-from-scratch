import torch
import torch.nn as nn
from camadas import Conv_BN, C2f

class PANet(nn.Module):
    """
    Constrói o neck da YOLOv8 (PANet)
    """
    def __init__(self):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Caminho Top-Down
        # up(p5)[512] + p4[512] = 1024 -> sai 512
        self.c2f_n4 = C2f(c_in=1024, c_out=512, n=3, shortcut=False)
        
        # up(n4)[512] + p3[256] = 768 -> sai 256 (Saída Small)
        self.c2f_small = C2f(c_in=768, c_out=256, n=3, shortcut=False)
        
        # Caminho Bottom-Up
        self.down_n3 = Conv_BN(c_in=256, c_out=256, kernel_size=3, stride=2)
        
        # down(n3)[256] + n4[512] = 768 -> sai 512 (Saída Medium)
        self.c2f_medium = C2f(c_in=768, c_out=512, n=3, shortcut=False)
        
        self.down_n4 = Conv_BN(c_in=512, c_out=512, kernel_size=3, stride=2)
        
        # down(n4)[512] + p5[512] = 1024 -> sai 512 (Saída Large)
        self.c2f_large = C2f(c_in=1024, c_out=512, n=3, shortcut=False)

    def forward(self, p3, p4, p5):
        # 1. TOP-DOWN
        up_p5 = self.upsample(p5)
        concat_1 = torch.cat([up_p5, p4], dim=1)
        n4 = self.c2f_n4(concat_1)
        
        up_n4 = self.upsample(n4)
        concat_2 = torch.cat([up_n4, p3], dim=1)
        out_small = self.c2f_small(concat_2)
        
        # 2. BOTTOM-UP
        down_n3 = self.down_n3(out_small)
        concat_3 = torch.cat([down_n3, n4], dim=1)
        out_medium = self.c2f_medium(concat_3)
        
        down_n4 = self.down_n4(out_medium)
        concat_4 = torch.cat([down_n4, p5], dim=1)
        out_large = self.c2f_large(concat_4)
        
        return out_small, out_medium, out_large