"""
yolo_fastest_loss.py

Loss function for YOLO-Fastest V1 training.

https://github.com/dog-qiuqiu/Yolo-Fastest

Full disclosure: this was written almost entirely by Claude AI. If you find any
issues with the script, please let me know.

License: Apache-2.0
"""

import torch
import torch.nn as nn

from yolo_fastest_utils import build_targets, bbox_iou_tensor


class YoloLoss(nn.Module):
    """
    YOLO-Fastest V1 loss function.
    
    Combines:
    - CIoU loss for bounding box regression
    - BCE loss for objectness
    - BCE loss for classification
    """
    
    def __init__(self, anchors, anchor_masks, num_classes=1, img_size=192):
        super().__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Grid sizes for each scale
        self.grid_sizes = [img_size // 32, img_size // 16]  # [6, 12]
        
        # Loss weights
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 0.5
        
        # Scale balance weights
        self.balance = [1.0, 0.4]
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Compute YOLO loss.
        
        Args:
            predictions: List of [batch, 18, H, W] tensors (one per scale)
            targets: [N, 6] tensor with (batch_idx, class, cx, cy, w, h) normalized
            
        Returns:
            total_loss, box_loss, obj_loss, cls_loss
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        # %%%TEST
        print(f"DEBUG: batch_size from predictions = {batch_size}")
        print(f"DEBUG: targets shape = {targets.shape}")
        if len(targets) > 0:
            print(f"DEBUG: max batch_idx in targets = {targets[:, 0].max().item()}")
        
        # Build targets for all scales
        if len(targets) > 0:
            all_targets = build_targets(
                targets, 
                self.anchors,
                self.anchor_masks,
                self.grid_sizes, 
                self.img_size,
                device,
                batch_size
            )
        else:
            all_targets = None
        
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        
        # Process each scale
        for scale_idx, pred in enumerate(predictions):
            grid_size = self.grid_sizes[scale_idx]
            num_anchors = len(self.anchor_masks[scale_idx])
            
            # Reshape: [batch, 18, H, W] -> [batch, 3, 6, H, W] -> [batch, 3, H, W, 6]
            pred = pred.view(batch_size, num_anchors, -1, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Extract predictions
            tx = pred[..., 0]
            ty = pred[..., 1]
            tw = pred[..., 2]
            th = pred[..., 3]
            obj_pred = pred[..., 4]
            cls_pred = pred[..., 5:]
            
            if all_targets is not None:
                scale_targets = all_targets[scale_idx]
                obj_mask = scale_targets['obj_mask']
                print(f"DEBUG: obj_mask shape = {obj_mask.shape}")
                print(f"DEBUG: obj_pred shape = {obj_pred.shape}")
                noobj_mask = scale_targets['noobj_mask']
                tbox = scale_targets['tbox']
                indices = scale_targets['indices']
                tcls = scale_targets['tcls']
                scale_anchors = scale_targets['anchors']
                
                num_obj = obj_mask.sum().item()
                
                # Box loss (CIoU)
                if num_obj > 0:
                    b, a, gj, gi = indices
                    
                    # Get predictions for positive samples
                    pxy = torch.stack([
                        torch.sigmoid(tx[b, a, gj, gi]),
                        torch.sigmoid(ty[b, a, gj, gi])
                    ], dim=1)
                    
                    pwh = torch.stack([
                        torch.exp(tw[b, a, gj, gi]) * scale_anchors[a, 0],
                        torch.exp(th[b, a, gj, gi]) * scale_anchors[a, 1]
                    ], dim=1)
                    
                    pbox = torch.cat([pxy, pwh], dim=1)  # [num_obj, 4]
                    
                    # CIoU loss
                    ciou = bbox_iou_tensor(pbox, tbox, x1y1x2y2=False, CIoU=True)
                    box_loss = (1.0 - ciou).mean()
                else:
                    box_loss = torch.tensor(0.0, device=device)
                
                # Objectness loss
                obj_target = obj_mask.float()
                obj_loss_all = self.bce(obj_pred, obj_target)
                
                # Weighted sum: positive + lambda_noobj * negative
                obj_loss_pos = obj_loss_all[obj_mask].sum()
                obj_loss_neg = obj_loss_all[noobj_mask].sum()
                
                if num_obj > 0:
                    obj_loss = (obj_loss_pos + self.lambda_noobj * obj_loss_neg) / num_obj
                else:
                    obj_loss = self.lambda_noobj * obj_loss_neg / (noobj_mask.sum() + 1e-16)
                
                # Classification loss
                if num_obj > 0 and self.num_classes > 0:
                    b, a, gj, gi = indices
                    cls_target = torch.zeros_like(cls_pred[b, a, gj, gi])
                    cls_target[range(len(tcls[b, a, gj, gi])), tcls[b, a, gj, gi]] = 1.0
                    cls_loss = self.bce(cls_pred[b, a, gj, gi], cls_target).mean()
                else:
                    cls_loss = torch.tensor(0.0, device=device)
            else:
                # No targets - only compute objectness loss for background
                box_loss = torch.tensor(0.0, device=device)
                obj_target = torch.zeros_like(obj_pred)
                obj_loss = self.bce(obj_pred, obj_target).mean()
                cls_loss = torch.tensor(0.0, device=device)
            
            # Accumulate with scale balance
            total_box_loss += box_loss * self.balance[scale_idx]
            total_obj_loss += obj_loss * self.balance[scale_idx]
            total_cls_loss += cls_loss * self.balance[scale_idx]
        
        # Apply loss weights
        total_box_loss *= self.lambda_box
        total_obj_loss *= self.lambda_obj
        total_cls_loss *= self.lambda_cls
        
        total_loss = total_box_loss + total_obj_loss + total_cls_loss
        
        return total_loss, total_box_loss, total_obj_loss, total_cls_loss