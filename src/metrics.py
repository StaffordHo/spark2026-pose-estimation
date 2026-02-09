"""
Evaluation metrics for spacecraft pose estimation.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def position_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between predicted and ground-truth positions.
    
    Args:
        pred: Predicted translation (B, 3)
        gt: Ground-truth translation (B, 3)
        
    Returns:
        Position error per sample (B,)
    """
    return torch.norm(pred - gt, dim=1)


def rotation_error_deg(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation error in degrees between predicted and ground-truth quaternions.
    
    Args:
        q_pred: Predicted quaternion (B, 4)
        q_gt: Ground-truth quaternion (B, 4)
        
    Returns:
        Rotation error in degrees per sample (B,)
    """
    # Normalize quaternions
    q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + 1e-8)
    q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute angular distance
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    
    angle_rad = 2.0 * torch.acos(dot)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg


def compute_metrics(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute all pose estimation metrics.
    
    Args:
        pred: Dictionary with "translation" and "quaternion"
        target: Dictionary with "translation" and "quaternion"
        
    Returns:
        Dictionary with metric values
    """
    pos_err = position_error(pred["translation"], target["translation"])
    rot_err = rotation_error_deg(pred["quaternion"], target["quaternion"])
    
    metrics = {
        "pos_error_mean": pos_err.mean().item(),
        "pos_error_median": pos_err.median().item(),
        "pos_error_std": pos_err.std().item(),
        "rot_error_mean": rot_err.mean().item(),
        "rot_error_median": rot_err.median().item(),
        "rot_error_std": rot_err.std().item(),
    }
    
    # Add percentile metrics
    pos_np = pos_err.cpu().numpy()
    rot_np = rot_err.cpu().numpy()
    
    for p in [50, 90, 95]:
        metrics[f"pos_error_p{p}"] = float(np.percentile(pos_np, p))
        metrics[f"rot_error_p{p}"] = float(np.percentile(rot_np, p))
    
    return metrics


class MetricTracker:
    """
    Tracks and aggregates metrics over batches.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.pos_errors = []
        self.rot_errors = []
        self.count = 0
    
    def update(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        """Add batch of predictions to tracker."""
        pos_err = position_error(pred["translation"], target["translation"])
        rot_err = rotation_error_deg(pred["quaternion"], target["quaternion"])
        
        self.pos_errors.append(pos_err.detach().cpu())
        self.rot_errors.append(rot_err.detach().cpu())
        self.count += pos_err.size(0)
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics."""
        if self.count == 0:
            return {}
        
        pos_all = torch.cat(self.pos_errors)
        rot_all = torch.cat(self.rot_errors)
        
        return {
            "pos_error_mean": pos_all.mean().item(),
            "pos_error_median": pos_all.median().item(),
            "pos_error_std": pos_all.std().item(),
            "rot_error_mean": rot_all.mean().item(),
            "rot_error_median": rot_all.median().item(),
            "rot_error_std": rot_all.std().item(),
        }
