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

def manual_nms(boxes, scores, iou_threshold):
    indices = sorted(range(len(boxes)), key=(lambda i: scores[i]), reverse=True)
    suppressed = [False] * len(indices)
    for ki in range(len(indices)):
        i = indices[ki]
        for kj in range(ki):
            if suppressed[kj]:
                continue
            j = indices[kj]
            if iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed[ki] = True
                break
    return [indices[j] for j in range(len(indices)) if not suppressed[j]]
