import torch
import torch.nn as nn

class Conv_BN(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU (Swish)"""
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        # O padding matemático da YOLOv8 (k // 2)
        padding = kernel_size // 2 
        
        self.conv = nn.Conv2d(
            in_channels=c_in, 
            out_channels=c_out, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Bottleneck utilizado nos blocos C2f"""
    def __init__(self, c_in, c_out, shortcut=True):
        super().__init__()
        self.cv1 = Conv_BN(c_in, c_out, kernel_size=3)
        self.cv2 = Conv_BN(c_out, c_out, kernel_size=3)
        # O shortcut só ocorre se a entrada e a saída tiverem o mesmo tamanho
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Arquitetura do bloco C2f"""
    def __init__(self, c_in, c_out, n=1, shortcut=True):
        super().__init__()
        self.c = c_out // 2  # c_hidden
        self.cv1 = Conv_BN(c_in, c_out, kernel_size=1)
        # A entrada final tem (2 + n) tensores de tamanho c_hidden
        self.cv2 = Conv_BN((2 + n) * self.c, c_out, kernel_size=1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        # Chunk divide no meio ao longo da dimensão 1 (canais)
        y = list(self.cv1(x).chunk(2, 1))
        for module in self.m:
            y.append(module(y[-1]))
        
        # Concatena na dimensão 1
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Bloco SPPF para extração de features"""
    def __init__(self, c_in, c_out):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = Conv_BN(c_in, c_hidden, kernel_size=1)
        self.cv2 = Conv_BN(c_hidden * 4, c_out, kernel_size=1)
        # Padding 2 em kernel 5 equivale a 'same'
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))