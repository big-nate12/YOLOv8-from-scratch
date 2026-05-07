import random
from bisect import bisect
import colorsys
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from constants import CLASS_NAMES

# Adaptado do código do professor João Florindo:


def generate_colors():
    hsv_tuples = [(x / len(CLASS_NAMES), 1., 1.) for x in range(len(CLASS_NAMES))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    return colors


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


def draw_box(image, box, label, color, font):
    thickness = (image.size[0] + image.size[1]) // 300
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), label, font=font)
    label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    
    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    
    if left >= right or top >= bottom:
        return
    
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])
    
    for j in range(thickness):
        if left+j >= right-j or top+j >= bottom-j:
            break
        draw.rectangle([left+j, top+j, right-j, bottom-j], outline=color)
       
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
    draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)


def draw_boxes(image, boxes, scores, classes):
    colors = generate_colors()
    font = ImageFont.load_default((image.size[0] + image.size[1]) // 100)
    for i, c in reversed(list(enumerate(classes))):
        predicted_class = CLASS_NAMES[c]
        score = scores[i].cpu().item()
        label = f'{predicted_class} {score:.2f}'
        draw_box(image, boxes[i].cpu().numpy(), label, colors[c], font)


# Original:


def find_absolute_boxes(rel_boxes, strides=(8, 16, 32), image_size=(640, 640)):
    # rel_boxes tem shape (N, 4)
    all_boxes = []
    width, height = image_size
    i = 0
    for stride in strides:
        centers = torch.cartesian_prod(torch.arange(stride//2, height, stride),
                                       torch.arange(stride//2, width, stride))
        offsets = stride * rel_boxes[i : i + centers.shape[0], :]
        boxes = torch.stack([(centers[:,0] - offsets[:,1])/height,   # top
                             (centers[:,1] - offsets[:,0])/width,    # left
                             (centers[:,0] + offsets[:,3])/height,   # bottom
                             (centers[:,1] + offsets[:,2])/width]).T # right
        all_boxes.append(boxes)
        i += centers.shape[0]
    if i != rel_boxes.shape[0]:
        raise ValueError('rel_boxes tem o comprimento errado')
    return torch.cat(all_boxes)


def decode_output(output):
    output = output.T
    boxes = find_absolute_boxes(output[:,:4])
    scores, classes = torch.max(output[:,4:], dim=-1)
    return boxes, scores, classes


def iou(boxa, boxb):
    xa1, ya1, xa2, ya2 = boxa
    areaa = (xa2 - xa1) * (ya2 - ya1)
    xb1, yb1, xb2, yb2 = boxb
    areab = (xb2 - xb1) * (yb2 - yb1)
    left = max(xa1, xb1)
    right = min(xa2, xb2)
    width = max(0, right - left)
    top = max(ya1, yb1)
    bottom = min(ya2, yb2)
    height = max(0, bottom - top)
    intersection = width * height
    union = areaa + areab - intersection
    return intersection / union


def filter_iou(boxes, classes, iou_threshold):
    # Faz non-max suppression por IOU
    # Presume que boxes já estão ordenadas por score decrescente
    # Faz o IOU em cada classe separadamente
    # retorna uma mask True nos que são mantidos
    kept = [True] * len(boxes)
    for i in range(len(boxes)):
        for j in range(i):
            if (kept[j]
                and classes[j] == classes[i]
                and iou(boxes[i], boxes[j]) > iou_threshold):
                kept[i] = False
                break
    return kept


def nms(boxes, scores, classes, score_threshold, iou_threshold):
    # Presume que boxes já estão ordenadas por score decrescente

    # Limpa tensores
    valid_mask = torch.isfinite(boxes).all(dim=-1) & torch.isfinite(scores).all(dim=-1)
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    classes = classes[valid_mask]



    # Filtragem por Confiança
    i = bisect(list(-scores), -score_threshold)
    boxes = boxes[:i]
    scores = scores[:i]
    classes = classes[:i]

    # Filtragem por IOU
    iou_mask = filter_iou(boxes, classes, iou_threshold)
    boxes = boxes[iou_mask]
    scores = scores[iou_mask]
    classes = classes[iou_mask]

    return boxes, scores, classes


def postprocess_output(output, img_size, original_shape,
                       score_threshold, iou_threshold):
    boxes, scores, classes = decode_output(output)
    inds = torch.tensor(sorted(range(len(scores)),
                               key=(lambda i: scores[i]),
                               reverse=True))
    boxes, scores, classes = boxes[inds], scores[inds], classes[inds]
    boxes, scores, classes = nms(boxes, scores, classes,
                                 score_threshold, iou_threshold)
    boxes = reverter_escala_caixas(boxes, img_size, original_shape)
    return boxes, scores, classes
