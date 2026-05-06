# %%

import torch
import torch.nn as nn
from camadas import Conv_BN

class YOLOHead(nn.Module):
    """Head desacoplado para uma escala da YOLOv8."""

    def __init__(self, c_in, c_cls, num_classes, dfl_module = None):  
        super().__init__()
        c_box = 64
        self.box_conv1 = Conv_BN(c_in, c_box, kernel_size=3)
        self.box_conv2 = Conv_BN(c_box, c_box, kernel_size=3)
        self.box_out = nn.Conv2d(c_box, 64, kernel_size=1)
        
        self.cls_conv1 = Conv_BN(c_in, c_cls, kernel_size=3)
        self.cls_conv2 = Conv_BN(c_cls, c_cls, kernel_size=3)
        self.cls_out = nn.Conv2d(c_cls, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.dfl = dfl_module

    def forward(self, x):
        # 1. Box Path (Extração)
        x_box = self.box_conv1(x)
        x_box = self.box_conv2(x_box)
        x_box = self.box_out(x_box) # Saída: 64 canais

        # 2. DFL (Converte 64 canais em 4 coordenadas reais)
        if self.dfl is not None:
            B, _, H, W = x_box.shape
            # [B, 4, 16, H, W] -> Softmax nos 16 bins
            x_box = x_box.view(B, 4, 16, H, W).softmax(2)
            # Colapsa os bins: Saída [B, 4, H, W][cite: 5]
            x_box = self.dfl(x_box.view(-1, 16, H, W)).view(B, 4, H, W)

        # 3. Class Path (Gera os 80 canais das classes)
        x_cls = self.cls_conv1(x)
        x_cls = self.cls_conv2(x_cls)
        # self.cls_out transforma 256 -> 80 canais
        cls_out = self.sigmoid(self.cls_out(x_cls)) 
        
        # RETORNO: x_box (4 canais) e cls_out (80 canais)
        return x_box, cls_out
# %%
