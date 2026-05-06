import torch
import torch.nn as nn
from camadas import Conv_BN, C2f, SPPF

class CSPDarknet(nn.Module):
    """
    Retorna os feature maps [P3, P4, P5]
    Assumindo entrada de 3 canais (RGB).
    """
    def __init__(self):
        super().__init__()
        
        # --- P1 (Stem) ---
        self.p1 = Conv_BN(3, 64, kernel_size=3, stride=2)
        
        # --- P2 ---
        self.p2_conv = Conv_BN(64, 128, kernel_size=3, stride=2)
        self.p2_c2f = C2f(128, 128, n=3)
        
        # --- P3 (Objetos Pequenos) ---
        self.p3_conv = Conv_BN(128, 256, kernel_size=3, stride=2)
        self.p3_c2f = C2f(256, 256, n=6)
        
        # --- P4 (Objetos Médios) ---
        self.p4_conv = Conv_BN(256, 512, kernel_size=3, stride=2)
        self.p4_c2f = C2f(512, 512, n=6)
        
        # --- P5 (Objetos Grandes) ---
        self.p5_conv = Conv_BN(512, 512, kernel_size=3, stride=2)
        self.p5_c2f = C2f(512, 512, n=3)
        self.sppf = SPPF(512, 512)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2_c2f(self.p2_conv(x))
        
        p3 = self.p3_c2f(self.p3_conv(x))
        p4 = self.p4_c2f(self.p4_conv(p3))
        
        x = self.p5_c2f(self.p5_conv(p4))
        p5 = self.sppf(x)
        
        return p3, p4, p5
