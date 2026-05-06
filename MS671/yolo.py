# %%
import torch
import torch.nn as nn
from backbone import CSPDarknet
from neck import PANet
from head import YOLOHead

class YOLOv8(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Instancia o Backbone
        self.backbone = CSPDarknet()
        
        # 2. Instancia o Neck
        self.neck = PANet()
        
        # Extra: Criação do DFL
        self.dfl = nn.Conv2d(16, 1, kernel_size=1, bias=False) 

        # 3. Instancia os Heads (c_in mapeado para as saídas do neck)
        self.head_small = YOLOHead(c_in=256, c_cls = 256, num_classes=num_classes, dfl_module = self.dfl)
        self.head_medium = YOLOHead(c_in=512, c_cls = 256, num_classes=num_classes, dfl_module = self.dfl)
        self.head_large = YOLOHead(c_in=512, c_cls = 256, num_classes=num_classes, dfl_module = self.dfl)

          


    def forward(self, x):
        # 1. Fluxo do Backbone e Neck
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        
        # 2. Obtém as predições de cada escala
        # Cada head deve retornar uma tupla (box, cls)
        # box shape: [B, 64, H, W] | cls shape: [B, 80, H, W]
        head_p3 = self.head_small(n3)
        head_p4 = self.head_medium(n4)
        head_p5 = self.head_large(n5)
        
        results = []
        for box, cls in [head_p3, head_p4, head_p5]:
            # Junta os 4 canais de box com os 80 de classe
            combined = torch.cat([box, cls], dim=1) # Formato: [B, 84, H, W]
            
            # Achata H e W para o formato final [B, 84, 8400]
            combined = combined.view(combined.shape[0], combined.shape[1], -1)
            results.append(combined)
            
        return torch.cat(results, dim=2)

# %%
# --- Instanciando o modelo ---
from load_weights import extract_weights

# --- Carregando os pesos ---

model = YOLOv8(num_classes = 80)
extract_weights(model, "yolov8l.pt")
# %%
# Teste final de saida
model.eval()

entrada_teste = torch.rand(1,3,640,640)

with torch.no_grad():
    saida = model(entrada_teste)

classes = saida[:, 4:, :]
print(f"Novo maior valor de classe: {classes.max().item():.4f}")

# %%

