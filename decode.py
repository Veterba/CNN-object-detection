import torch
from torchvision.ops import nms

from loss import ANCHORS


def decode(predictions, num_classes=8, anchors=ANCHORS,
           conf_threshold=0.5, iou_threshold=0.45):
    """
    Turn one image's raw head output into a list of detections.

    predictions: [1, num_anchors*(5+num_classes), grid, grid]  raw logits (single image)
    returns: list of {"box": [left, top, right, bottom] in [0,1],
                      "class_id": int, "score": float}
    Boxes are normalized to [0,1]; the caller scales them to frame pixels.
    This is the exact inverse of how loss.py encodes targets.
    """
    batch_size, _, grid_size, _ = predictions.shape
    assert batch_size == 1, "decode expects a single image (batch size 1)"
    num_anchors = anchors.shape[0]
    anchors = anchors.to(predictions.device)

    # [anchor, row, col, 5+num_classes] — same channel layout loss.py reads
    cells = (predictions
             .view(num_anchors, 5 + num_classes, grid_size, grid_size)
             .permute(0, 2, 3, 1)
             .contiguous())

    raw_center = cells[..., 0:2]   # cell-relative center, pre-sigmoid
    raw_size = cells[..., 2:4]     # log-space size relative to anchor
    raw_objectness = cells[..., 4]
    raw_class = cells[..., 5:]

    # grid coordinate of each cell, broadcast across anchors
    col_index = torch.arange(grid_size, device=predictions.device).view(1, 1, grid_size)
    row_index = torch.arange(grid_size, device=predictions.device).view(1, grid_size, 1)

    # un-transform back to normalized image coordinates
    center_x = (torch.sigmoid(raw_center[..., 0]) + col_index) / grid_size
    center_y = (torch.sigmoid(raw_center[..., 1]) + row_index) / grid_size
    box_width = torch.exp(raw_size[..., 0]) * anchors[:, 0].view(num_anchors, 1, 1)
    box_height = torch.exp(raw_size[..., 1]) * anchors[:, 1].view(num_anchors, 1, 1)

    objectness = torch.sigmoid(raw_objectness)
    class_probs = torch.softmax(raw_class, dim=-1)
    best_class_prob, best_class_id = class_probs.max(dim=-1)
    confidence = objectness * best_class_prob

    keep = confidence > conf_threshold
    if keep.sum() == 0:
        return []

    center_x = center_x[keep]
    center_y = center_y[keep]
    box_width = box_width[keep]
    box_height = box_height[keep]
    confidence = confidence[keep]
    best_class_id = best_class_id[keep]

    # center+size -> corner coordinates
    left = center_x - box_width / 2
    top = center_y - box_height / 2
    right = center_x + box_width / 2
    bottom = center_y + box_height / 2
    boxes = torch.stack([left, top, right, bottom], dim=1)

    # drop overlapping duplicates of the same object
    survivors = nms(boxes, confidence, iou_threshold)

    detections = []
    for index in survivors:
        detections.append({
            "box": boxes[index].tolist(),
            "class_id": int(best_class_id[index]),
            "score": float(confidence[index]),
        })
    return detections
