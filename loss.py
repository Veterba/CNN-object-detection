import torch
import torch.nn.functional as F

# normalized (w, h) anchor priors — rough sizes for the 3 anchor slots in the head
ANCHORS = torch.tensor([[0.20, 0.25], [0.45, 0.50], [0.80, 0.75]])


def yolo_loss(preds, targets, num_classes=8, anchors=ANCHORS,
              lambda_coord=5.0, lambda_noobj=0.02):
    """
    preds:   [B, A*(5+num_classes), S, S]  raw head output
    targets: tuple of dicts, each {"boxes": [n,4] xyxy in [0,1], "labels": [n]}
    returns: scalar loss
    """
    B, _, S, _ = preds.shape
    A = anchors.shape[0]
    device = preds.device
    anchors = anchors.to(device)

    # [B, A, S, S, 5+num_classes] ; channels laid out anchor-major
    p = preds.view(B, A, 5 + num_classes, S, S).permute(0, 1, 3, 4, 2).contiguous()
    pred_xy = torch.sigmoid(p[..., 0:2])   # offset within cell, [0,1]
    pred_wh = p[..., 2:4]                   # log-space vs anchor
    pred_obj = p[..., 4]                    # objectness logit
    pred_cls = p[..., 5:]                   # class logits

    tgt_xy = torch.zeros(B, A, S, S, 2, device=device)
    tgt_wh = torch.zeros(B, A, S, S, 2, device=device)
    tgt_cls = torch.zeros(B, A, S, S, dtype=torch.long, device=device)
    obj_mask = torch.zeros(B, A, S, S, dtype=torch.bool, device=device)

    for b, t in enumerate(targets):
        boxes, labels = t["boxes"], t["labels"]
        if boxes.numel() == 0:
            continue
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        gx = (cx * S).long().clamp(0, S - 1)
        gy = (cy * S).long().clamp(0, S - 1)

        for i in range(boxes.shape[0]):
            wh = torch.stack([bw[i], bh[i]])
            inter = torch.min(wh, anchors).prod(1)
            union = wh.prod() + anchors.prod(1) - inter
            a = (inter / union).argmax()          # responsible anchor = best IoU
            yy, xx = gy[i], gx[i]
            obj_mask[b, a, yy, xx] = True
            tgt_xy[b, a, yy, xx, 0] = cx[i] * S - xx
            tgt_xy[b, a, yy, xx, 1] = cy[i] * S - yy
            tgt_wh[b, a, yy, xx, 0] = torch.log(bw[i] / anchors[a, 0] + 1e-6)
            tgt_wh[b, a, yy, xx, 1] = torch.log(bh[i] / anchors[a, 1] + 1e-6)
            tgt_cls[b, a, yy, xx] = labels[i]

    box_loss = (F.mse_loss(pred_xy[obj_mask], tgt_xy[obj_mask], reduction="sum")
                + F.mse_loss(pred_wh[obj_mask], tgt_wh[obj_mask], reduction="sum"))

    obj_t = obj_mask.float()
    obj_loss = F.binary_cross_entropy_with_logits(
        pred_obj[obj_mask], obj_t[obj_mask], reduction="sum")
    noobj_loss = F.binary_cross_entropy_with_logits(
        pred_obj[~obj_mask], obj_t[~obj_mask], reduction="sum")

    cls_loss = F.cross_entropy(pred_cls[obj_mask], tgt_cls[obj_mask], reduction="sum")

    return (lambda_coord * box_loss + obj_loss + lambda_noobj * noobj_loss + cls_loss) / B
