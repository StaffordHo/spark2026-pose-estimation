"""
Loss functions for spacecraft pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def quaternion_angular_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute angular distance between two quaternions.
    
    Args:
        q1, q2: Quaternions of shape (B, 4) in format (qx, qy, qz, qw)
        
    Returns:
        Angular distance in radians, shape (B,)
    """
    # Compute dot product (absolute value to handle double cover)
    dot = torch.abs(torch.sum(q1 * q2, dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Angular distance = 2 * arccos(|q1 · q2|)
    angle = 2.0 * torch.acos(dot)
    
    return angle


def geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic loss for quaternion rotation.
    
    This loss measures the angular distance on SO(3) between predicted
    and ground-truth rotations.
    
    Args:
        q_pred: Predicted quaternion (B, 4), normalized
        q_gt: Ground-truth quaternion (B, 4), normalized
        
    Returns:
        Mean geodesic loss (scalar)
    """
    angle = quaternion_angular_distance(q_pred, q_gt)
    return angle.mean()


def quaternion_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Simple quaternion loss handling double-cover ambiguity.
    
    Uses min(||q_pred - q_gt||, ||q_pred + q_gt||) to handle the fact
    that q and -q represent the same rotation.
    """
    # Normalize predictions
    q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute both distances
    loss_pos = F.mse_loss(q_pred, q_gt, reduction='none').sum(dim=1)
    loss_neg = F.mse_loss(q_pred, -q_gt, reduction='none').sum(dim=1)
    
    # Take minimum
    loss = torch.min(loss_pos, loss_neg)
    
    return loss.mean()


class PoseLoss(nn.Module):
    """
    Combined loss for pose estimation.
    
    L_total = w_trans * L_translation + w_rot * L_rotation
    """
    
    def __init__(
        self,
        trans_weight: float = 1.0,
        rot_weight: float = 1.0,
        trans_loss_type: str = "smooth_l1",  # "mse" or "smooth_l1"
        rot_loss_type: str = "geodesic"  # "geodesic" or "mse"
    ):
        super().__init__()
        
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.trans_loss_type = trans_loss_type
        self.rot_loss_type = rot_loss_type
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined pose loss.
        
        Args:
            pred: Dictionary with "translation" (B, 3) and "quaternion" (B, 4)
            target: Dictionary with "translation" (B, 3) and "quaternion" (B, 4)
            
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual losses for logging
        """
        # Translation loss
        if self.trans_loss_type == "smooth_l1":
            trans_loss = F.smooth_l1_loss(pred["translation"], target["translation"])
        else:
            trans_loss = F.mse_loss(pred["translation"], target["translation"])
        
        # Rotation loss
        if self.rot_loss_type == "geodesic":
            rot_loss = geodesic_loss(pred["quaternion"], target["quaternion"])
        else:
            rot_loss = quaternion_loss(pred["quaternion"], target["quaternion"])
        
        # Combined loss
        total_loss = self.trans_weight * trans_loss + self.rot_weight * rot_loss
        
        loss_dict = {
            "loss_total": total_loss,
            "loss_translation": trans_loss,
            "loss_rotation": rot_loss
        }
        
        return total_loss, loss_dict


class PoseLossUncertainty(nn.Module):
    """
    Pose loss with learned uncertainty weighting.
    
    Uses aleatoric uncertainty to weight the losses.
    Reference: Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning?"
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute uncertainty-weighted pose loss.
        """
        trans_pred = pred["translation"]
        trans_gt = target["translation"]
        trans_log_var = pred["trans_log_var"]
        
        quat_pred = pred["quaternion"]
        quat_gt = target["quaternion"]
        quat_log_var = pred["quat_log_var"]
        
        # Translation loss with uncertainty
        trans_precision = torch.exp(-trans_log_var)
        trans_loss = (trans_precision * (trans_pred - trans_gt) ** 2 + trans_log_var).mean()
        
        # Compute quaternion error (handling double cover)
        quat_error_pos = (quat_pred - quat_gt) ** 2
        quat_error_neg = (quat_pred + quat_gt) ** 2
        quat_error = torch.min(quat_error_pos.sum(dim=1), quat_error_neg.sum(dim=1))
        
        # Rotation loss with uncertainty
        quat_precision = torch.exp(-quat_log_var.mean(dim=1))
        rot_loss = (quat_precision * quat_error + quat_log_var.mean(dim=1)).mean()
        
        total_loss = trans_loss + rot_loss
        
        loss_dict = {
            "loss_total": total_loss,
            "loss_translation": trans_loss,
            "loss_rotation": rot_loss,
            "trans_uncertainty": trans_log_var.exp().mean(),
            "rot_uncertainty": quat_log_var.exp().mean()
        }
        
        return total_loss, loss_dict
