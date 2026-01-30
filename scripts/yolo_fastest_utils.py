"""
yolo_fastest_utils.py

Utility functions for YOLO-Fastest V1 training and inference.

https://github.com/dog-qiuqiu/Yolo-Fastest

Full disclosure: this was written almost entirely by Claude AI. If you find any
issues with the script, please let me know.

License: Apache-2.0
"""

import torch
import numpy as np


def get_anchors_from_config(model):
    """
    Extract anchors and masks from parsed YOLO config.
    
    Args:
        model: YoloFastest model with parsed .blocks attribute
        
    Returns:
        anchors: List of (width, height) tuples for all anchors
        masks: List of lists, anchor indices for each scale
    """
    anchors = None
    masks = []
    
    for block in model.blocks:
        if block['type'] == 'yolo':
            if anchors is None:
                anchor_vals = [int(x.strip()) for x in block['anchors'].split(',')]
                anchors = [(anchor_vals[i], anchor_vals[i+1]) for i in range(0, len(anchor_vals), 2)]
            mask = [int(x.strip()) for x in block['mask'].split(',')]
            masks.append(mask)
    
    return anchors, masks


def box_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: List or tensor [x1, y1, x2, y2]
        box2: List or tensor [x1, y1, x2, y2]
        
    Returns:
        IoU value (float)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def bbox_iou_tensor(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    """
    Compute IoU (and variants) between two sets of boxes.
    
    Args:
        box1: Tensor of shape [N, 4]
        box2: Tensor of shape [N, 4]
        x1y1x2y2: If True, boxes are [x1, y1, x2, y2]. If False, [cx, cy, w, h]
        GIoU, DIoU, CIoU: Which IoU variant to compute
        eps: Small value to avoid division by zero
        
    Returns:
        IoU tensor of shape [N]
    """
    if not x1y1x2y2:
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        # Convex hull (smallest enclosing box)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        
        if CIoU or DIoU:
            # Diagonal distance squared of convex hull
            c2 = cw ** 2 + ch ** 2 + eps
            # Center distance squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + eps)
                return iou - (rho2 / c2 + v * alpha)
        else:
            # GIoU
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area

    return iou


def box_iou_wh(wh1, wh2):
    """
    Compute IoU based on width/height only (assuming same center).
    Used for anchor-target matching.
    
    Args:
        wh1: Tensor [N, 2] - anchor widths/heights
        wh2: Tensor [M, 2] - target widths/heights
        
    Returns:
        Tensor [N, M] - IoU matrix
    """
    # Intersection (min of each dimension, assuming centered)
    inter_w = torch.min(wh1[:, 0:1], wh2[:, 0:1].T)
    inter_h = torch.min(wh1[:, 1:2], wh2[:, 1:2].T)
    inter = inter_w * inter_h

    # Union
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-16)


def build_targets(targets, anchors, anchor_masks, grid_sizes, img_size, device, batch_size):
    """
    Build target tensors for YOLO loss computation.
    
    Args:
        targets: Tensor [N, 6] with (batch_idx, class, cx, cy, w, h) normalized 0-1
        anchors: List of (w, h) tuples for all anchors (in pixels for original input size)
        anchor_masks: List of lists, anchor indices for each scale
        grid_sizes: List of grid sizes for each scale (e.g., [6, 12])
        img_size: Input image size (assumes square)
        device: Torch device
        batch_size: Actual batch size from predictions
        
    Returns:
        List of target dicts, one per scale
    """
    print(f"DEBUG build_targets: batch_size = {batch_size}")
    if len(targets) == 0:
        return None
    
    num_scales = len(grid_sizes)
    
    # Convert anchors to tensor
    anchors_tensor = torch.tensor(anchors, dtype=torch.float32, device=device)
    
    # Scale anchors to current image size (anchors are defined for 320x320)
    scale_factor = img_size / 320
    anchors_scaled = anchors_tensor * scale_factor
    
    all_targets = []
    
    for scale_idx in range(num_scales):
        grid_size = grid_sizes[scale_idx]
        mask = anchor_masks[scale_idx]
        num_anchors = len(mask)
        scale_anchors = anchors_scaled[mask]
        
        # Initialize tensors
        obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, dtype=torch.bool, device=device)
        tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
        ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
        tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
        th = torch.zeros(batch_size, num_anchors, grid_size, grid_size, device=device)
        tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, dtype=torch.long, device=device)
        
        # For CIoU loss
        tbox_list = []
        indices_list = []
        
        # Process each target
        batch_idx = targets[:, 0].long()
        cls_idx = targets[:, 1].long()
        
        # Scale to grid coordinates
        gx = targets[:, 2] * grid_size
        gy = targets[:, 3] * grid_size
        gw = targets[:, 4] * img_size
        gh = targets[:, 5] * img_size
        
        # Grid cell indices
        gi = gx.long().clamp(0, grid_size - 1)
        gj = gy.long().clamp(0, grid_size - 1)
        
        # Find best anchor for each target (based on shape only)
        target_wh = torch.stack([gw, gh], dim=1)
        ious = box_iou_wh(scale_anchors, target_wh)
        best_anchor_idx = ious.argmax(dim=0)
        
        # Assign targets
        for t_idx in range(len(targets)):
            b = batch_idx[t_idx]
            a = best_anchor_idx[t_idx]
            gj_idx = gj[t_idx]
            gi_idx = gi[t_idx]
            
            # Set masks
            obj_mask[b, a, gj_idx, gi_idx] = True
            noobj_mask[b, a, gj_idx, gi_idx] = False
            
            # Target offsets
            tx[b, a, gj_idx, gi_idx] = gx[t_idx] - gi_idx.float()
            ty[b, a, gj_idx, gi_idx] = gy[t_idx] - gj_idx.float()
            
            # Target size (log scale relative to anchor)
            tw[b, a, gj_idx, gi_idx] = torch.log(gw[t_idx] / scale_anchors[a, 0] + 1e-16)
            th[b, a, gj_idx, gi_idx] = torch.log(gh[t_idx] / scale_anchors[a, 1] + 1e-16)
            
            # Target class
            tcls[b, a, gj_idx, gi_idx] = cls_idx[t_idx]
            
            # For CIoU: store target box in grid-relative coords
            tbox_list.append(torch.tensor([
                gx[t_idx] - gi_idx.float(),
                gy[t_idx] - gj_idx.float(),
                gw[t_idx],
                gh[t_idx]
            ], device=device))
            
            indices_list.append((b, a, gj_idx, gi_idx))
        
        # Stack target boxes
        if tbox_list:
            tbox = torch.stack(tbox_list, dim=0)
            indices = (
                torch.tensor([x[0] for x in indices_list], device=device),
                torch.tensor([x[1] for x in indices_list], device=device),
                torch.tensor([x[2] for x in indices_list], device=device),
                torch.tensor([x[3] for x in indices_list], device=device)
            )
        else:
            tbox = torch.zeros(0, 4, device=device)
            indices = (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device)
            )
        
        all_targets.append({
            'obj_mask': obj_mask,
            'noobj_mask': noobj_mask,
            'tx': tx,
            'ty': ty,
            'tw': tw,
            'th': th,
            'tcls': tcls,
            'tbox': tbox,
            'indices': indices,
            'anchors': scale_anchors
        })
    
    return all_targets


def decode_predictions(outputs, anchors, anchor_masks, img_size, conf_thresh=0.5, num_channels=3):
    """
    Decode raw model outputs into bounding boxes.
    
    Args:
        outputs: List of tensors [batch, 18, H, W] for each scale
        anchors: List of (w, h) tuples for all anchors
        anchor_masks: List of lists, anchor indices for each scale
        img_size: Input image size (assumes square)
        conf_thresh: Confidence threshold for filtering
        num_channels: Number of input channels (3 for RGB, 1 for grayscale)
        
    Returns:
        List of detections per image, each detection is [x1, y1, x2, y2, conf, class]
    """
    batch_size = outputs[0].shape[0]
    device = outputs[0].device
    
    # Scale anchors
    scale_factor = img_size / 320
    anchors_scaled = [(w * scale_factor, h * scale_factor) for w, h in anchors]
    
    all_detections = [[] for _ in range(batch_size)]
    
    for scale_idx, output in enumerate(outputs):
        _, channels, h, w = output.shape
        num_anchors = len(anchor_masks[scale_idx])
        mask = anchor_masks[scale_idx]
        
        # Reshape: [batch, 18, H, W] -> [batch, 3, 6, H, W] -> [batch, 3, H, W, 6]
        pred = output.view(batch_size, num_anchors, -1, h, w)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract components
        tx = torch.sigmoid(pred[..., 0])
        ty = torch.sigmoid(pred[..., 1])
        tw = pred[..., 2]
        th = pred[..., 3]
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid_x = grid_x.to(device).float()
        grid_y = grid_y.to(device).float()
        
        # Decode box coordinates
        for b in range(batch_size):
            for a_idx, a in enumerate(mask):
                anchor_w, anchor_h = anchors_scaled[a]
                
                for j in range(h):
                    for i in range(w):
                        obj = obj_conf[b, a_idx, j, i].item()
                        
                        if obj < conf_thresh:
                            continue
                        
                        # Decode position
                        cx = (tx[b, a_idx, j, i].item() + i) / w * img_size
                        cy = (ty[b, a_idx, j, i].item() + j) / h * img_size
                        bw = torch.exp(tw[b, a_idx, j, i]).item() * anchor_w
                        bh = torch.exp(th[b, a_idx, j, i]).item() * anchor_h
                        
                        # Convert to x1, y1, x2, y2
                        x1 = cx - bw / 2
                        y1 = cy - bh / 2
                        x2 = cx + bw / 2
                        y2 = cy + bh / 2
                        
                        # Class confidence
                        if cls_conf.shape[-1] > 0:
                            cls_scores = cls_conf[b, a_idx, j, i]
                            cls_score, cls_id = cls_scores.max(0)
                            conf = obj * cls_score.item()
                            cls_id = cls_id.item()
                        else:
                            conf = obj
                            cls_id = 0
                        
                        all_detections[b].append([x1, y1, x2, y2, conf, cls_id])
    
    return all_detections


def nms(detections, iou_thresh=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        detections: List of [x1, y1, x2, y2, conf, class]
        iou_thresh: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        detections = [
            det for det in detections
            if box_iou(best[:4], det[:4]) < iou_thresh
        ]

    return keep


def compute_map(model, data_loader, anchors, anchor_masks, img_size, device, 
                conf_thresh=0.25, iou_thresh=0.5, num_channels=3):
    """
    Compute mAP@0.5 on a dataset.
    
    Args:
        model: YOLO model
        data_loader: DataLoader with (images, labels) batches
        anchors: List of anchor (w, h) tuples
        anchor_masks: List of anchor masks per scale
        img_size: Input image size
        device: Torch device
        conf_thresh: Confidence threshold for predictions
        iou_thresh: IoU threshold for mAP
        num_channels: Number of input channels (3 for RGB, 1 for grayscale)
        
    Returns:
        mAP value, total predictions, total ground truth
    """
    model.eval()

    all_pred_boxes = []
    all_true_boxes = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.float().to(device)
            batch_size = imgs.shape[0]

            # Get predictions
            outputs = model(imgs)
            predictions = decode_predictions(
                outputs, anchors, anchor_masks, img_size, conf_thresh, num_channels
            )

            # Apply NMS to each image
            for b in range(batch_size):
                pred_boxes = nms(predictions[b], iou_thresh=0.5)
                all_pred_boxes.append(pred_boxes)

                # Get ground truth boxes for this image
                img_labels = labels[labels[:, 0] == b]
                true_boxes = []
                for lbl in img_labels:
                    cx = lbl[2].item() * img_size
                    cy = lbl[3].item() * img_size
                    w = lbl[4].item() * img_size
                    h = lbl[5].item() * img_size
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    class_id = int(lbl[1].item())
                    true_boxes.append([x1, y1, x2, y2, class_id])
                all_true_boxes.append(true_boxes)

    # Calculate AP
    tp_list = []
    conf_list = []
    total_gt = sum(len(gt) for gt in all_true_boxes)

    for pred_boxes, true_boxes in zip(all_pred_boxes, all_true_boxes):
        matched = [False] * len(true_boxes)

        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(true_boxes):
                if matched[gt_idx]:
                    continue
                iou = box_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp_list.append(1)
                matched[best_gt_idx] = True
            else:
                tp_list.append(0)
            conf_list.append(pred[4])

    # Sort by confidence
    if len(conf_list) == 0:
        return 0.0, 0, total_gt

    sorted_indices = np.argsort(-np.array(conf_list))
    tp_list = np.array(tp_list)[sorted_indices]

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(1 - tp_list)

    recalls = tp_cumsum / total_gt if total_gt > 0 else tp_cumsum
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall) / 11

    total_predictions = len(conf_list)

    return ap, total_predictions, total_gt