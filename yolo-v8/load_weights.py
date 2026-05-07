import torch
from ultralytics import YOLO

# ------------------------------------------------------
# MAPEAMENTO DE ÍNDICES DO MODELO OFICIAL PARA O CUSTOM
# ------------------------------------------------------

_indices = {
    0:  "backbone.p1",
    1:  "backbone.p2_conv",
    2:  "backbone.p2_c2f",
    3:  "backbone.p3_conv",
    4:  "backbone.p3_c2f",
    5:  "backbone.p4_conv",
    6:  "backbone.p4_c2f",
    7:  "backbone.p5_conv",
    8:  "backbone.p5_c2f",
    9:  "backbone.sppf",
    12: "neck.c2f_n4",
    15: "neck.c2f_small",
    16: "neck.down_n3",
    18: "neck.c2f_medium",
    19: "neck.down_n4",
    21: "neck.c2f_large",
}

# Mapeamento dos indices referentes a saída da rede, o índice 22 no modelo original 
_head_scale_map = {
    "cv2.0": "head_small.box",
    "cv2.1": "head_medium.box",
    "cv2.2": "head_large.box",
    "cv3.0": "head_small.cls",
    "cv3.1": "head_medium.cls",
    "cv3.2": "head_large.cls",
}

def _convert_key(oficial_key):
    """Converte uma chave do state_dict oficial (sem o prefixo 'model.') para o nome customizado"""
    parts = oficial_key.split(".")
    try:
        idx = int(parts[0])
    except ValueError:
        return None

    if idx == 22:  # Head
        if len(parts) >= 4 and parts[1] in ["cv2", "cv3"]:
            branch = parts[1]
            scale = parts[2]
            rest = ".".join(parts[3:])
            prefix = _head_scale_map.get(f"{branch}.{scale}")
            if not prefix:
                return None
            sub_parts = rest.split(".")
            idx_intern = sub_parts[0]
            intern_map = {
                "0": "conv1",
                "1": "conv2",
                "2": "out"
            }
            name = intern_map.get(idx_intern)
            if not name:
                return None
            if name == "out":
                return f"{prefix}_{name}.{sub_parts[1]}"
            else:
                if sub_parts[1].isdigit():
                    return f"{prefix}_{name}.{'.'.join(sub_parts[2:])}"
            return f"{prefix}_{name}.{'.'.join(sub_parts[1:])}"
        elif parts[1] == "dfl":
            if parts[-1] == "weight":
                return "dfl.weight"
            
        return None

    # Backbone e Neck
    prefix = _indices.get(idx)
    if not prefix:
        return None
    sub_path = ".".join(parts[1:])
    return f"{prefix}.{sub_path}"

def create_map(state_dict_official, state_dict_custom):
    """Retorna dicionário {chaves_customs: tensor_oficial} com todos os shapes compatíveis"""

    map = {}

    for official_key, official_tensor in state_dict_official.items():
        key_without_model = official_key.replace("model.","",1)
        custom_key = _convert_key(key_without_model)

        if custom_key and custom_key in state_dict_custom:
            if official_tensor.shape == state_dict_custom[custom_key].shape:
                map[custom_key] = official_tensor
            else: 
                print(f"Shape mismatch: {custom_key}")

        else:
            if custom_key:
                print(f"Chave não encontrada: {custom_key}")

    return map

def extract_weights (model, path = "yolov8l.pt"):
    """
    Carrega os pesos de um arquivo .pt do Ultralytics para a rede customizada
    """   

    print(f"Carregando os pesos de {path}...\n")
    official_model = YOLO(path)
    official_state = official_model.model.model.state_dict()
    custom_state = model.state_dict()

    weights = create_map(official_state, custom_state)
    model.load_state_dict(weights, strict = False)

    total_params = sum(tensor.numel() for tensor in weights.values())

    print(f"Sucesso!\n") 
    print(f"Camadas pareadas: {len(weights)}\n")
    print(f"Total de parâmetros carregados: {total_params}") 

