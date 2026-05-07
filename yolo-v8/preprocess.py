from PIL import Image
import numpy as np
import torch

# Adaptado do código do professor João Florindo:


def letterbox_image(image, size=(640, 640)):

    iw, ih = image.size
    w, h = size
    # Encontra a escala ideal sem distorcer
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # Redimensiona mantendo a proporção
    image = image.resize((nw, nh), Image.BICUBIC)

    # Cria um fundo cinza neutro e cola a imagem no centro
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def reverter_escala_caixas(boxes, img_size, original_shape):
    """Remove o efeito do letterbox e reajusta as caixas para o tamanho original da imagem."""
    iw, ih = original_shape
    w, h = img_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    # Calcula o tamanho das barras cinzas no formato normalizado (0 a 1)
    dx = (w - nw) / 2.0 / w
    dy = (h - nh) / 2.0 / h
    scale_w = nw / w
    scale_h = nh / h

    # Remove as barras (Lembrando que o formato é [y_min, x_min, y_max, x_max])
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dy) / scale_h
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dx) / scale_w

    # Multiplica pelas dimensões exatas da imagem original
    boxes[:, [0, 2]] *= ih # Multiplica o eixo Y pela Altura
    boxes[:, [1, 3]] *= iw # Multiplica o eixo X pela Largura

    return boxes


def scale_boxes(boxes, image_shape):
    height, width = image_shape
    image_dims = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    return boxes * image_dims


def preprocess_image(img_path, model_image_size=(640, 640)):
    # 1. Carrega a imagem original
    image = Image.open(img_path).convert('RGB')

    # 2. Aplica o Letterbox (mantém proporção) em vez de distorcer
    boxed_image = letterbox_image(image, model_image_size)

    # 3. Converte para Tensor
    image_data = np.array(boxed_image, dtype='float32') / 255.0
    image_data = image_data[:, :, ::-1].copy() # RGB para BGR

    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = torch.from_numpy(image_data).unsqueeze(0)

    # Retorna a imagem ORIGINAL (para desenho) e os dados quadrados (para a rede)
    return image, image_data

