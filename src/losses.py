"""
Loss functions for spacecraft pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# =============================================================================
# 6D Rotation Representation Utilities (Zhou et al., CVPR 2019)
# =============================================================================

def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix via Gram-Schmidt.
    
    Args:
        rot_6d: (B, 6) - first two columns of rotation matrix, flattened
        
    Returns:
        Rotation matrix (B, 3, 3)
    """
    a1 = rot_6d[:, 0:3]  # First column
    a2 = rot_6d[:, 3:6]  # Second column
    
    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation (first two columns).
    
    Args:
        matrix: (B, 3, 3) rotation matrix
        
    Returns:
        (B, 6) - first two columns flattened
    """
    return matrix[:, :, :2].reshape(-1, 6)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (xyzw format).
    
    Args:
        matrix: (B, 3, 3) rotation matrix
        
    Returns:
        (B, 4) quaternion in (qx, qy, qz, qw) format
    """
    batch_size = matrix.shape[0]
    m = matrix
    
    # Shepperd's method for numerical stability
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    
    quat = torch.zeros(batch_size, 4, device=matrix.device, dtype=matrix.dtype)
    
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4*qw
    quat[:, 3] = 0.25 * s  # qw
    quat[:, 0] = (m[:, 2, 1] - m[:, 1, 2]) / (s + 1e-10)  # qx
    quat[:, 1] = (m[:, 0, 2] - m[:, 2, 0]) / (s + 1e-10)  # qy
    quat[:, 2] = (m[:, 1, 0] - m[:, 0, 1]) / (s + 1e-10)  # qz
    
    # Handle edge cases where trace is small
    cond1 = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]) & (trace <= 0)
    cond2 = (m[:, 1, 1] > m[:, 2, 2]) & ~cond1 & (trace <= 0)
    cond3 = (trace <= 0) & ~cond1 & ~cond2
    
    if cond1.any():
        s1 = torch.sqrt(torch.clamp(1.0 + m[cond1, 0, 0] - m[cond1, 1, 1] - m[cond1, 2, 2], min=1e-10)) * 2
        quat[cond1, 3] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / (s1 + 1e-10)
        quat[cond1, 0] = 0.25 * s1
        quat[cond1, 1] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / (s1 + 1e-10)
        quat[cond1, 2] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / (s1 + 1e-10)
    
    if cond2.any():
        s2 = torch.sqrt(torch.clamp(1.0 + m[cond2, 1, 1] - m[cond2, 0, 0] - m[cond2, 2, 2], min=1e-10)) * 2
        quat[cond2, 3] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / (s2 + 1e-10)
        quat[cond2, 0] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / (s2 + 1e-10)
        quat[cond2, 1] = 0.25 * s2
        quat[cond2, 2] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / (s2 + 1e-10)
    
    if cond3.any():
        s3 = torch.sqrt(torch.clamp(1.0 + m[cond3, 2, 2] - m[cond3, 0, 0] - m[cond3, 1, 1], min=1e-10)) * 2
        quat[cond3, 3] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / (s3 + 1e-10)
        quat[cond3, 0] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / (s3 + 1e-10)
        quat[cond3, 1] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / (s3 + 1e-10)
        quat[cond3, 2] = 0.25 * s3
    
    # Normalize
    quat = quat / (quat.norm(dim=1, keepdim=True) + 1e-10)
    
    return quat


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (xyzw format) to rotation matrix.
    
    Args:
        q: (B, 4) quaternion in (qx, qy, qz, qw) format
        
    Returns:
        (B, 3, 3) rotation matrix
    """
    q = q / (q.norm(dim=1, keepdim=True) + 1e-10)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    matrix = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)
    ], dim=1).reshape(-1, 3, 3)
    
    return matrix


def geodesic_loss_matrix(rot_6d_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic rotation loss using rotation matrices.
    Computes the angle of R_pred^T @ R_gt (should be identity for perfect prediction).
    
    Args:
        rot_6d_pred: (B, 6) predicted 6D rotation
        q_gt: (B, 4) ground-truth quaternion (xyzw)
        
    Returns:
        Mean geodesic loss (scalar)
    """
    R_pred = rotation_6d_to_matrix(rot_6d_pred)
    R_gt = quaternion_to_matrix(q_gt)
    
    # R_diff = R_pred^T @ R_gt
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    
    # Angle from trace: angle = arccos((trace(R_diff) - 1) / 2)
    # IMPORTANT: cast to float32 for numerical stability under AMP float16
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    trace = trace.float()  # ensure float32 for acos precision
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    
    return angle.mean()


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


class PoseLoss6D(nn.Module):
    """
    Pose loss using 6D continuous rotation representation.
    
    Uses geodesic loss on rotation matrices instead of quaternion loss.
    Reference: Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
    """
    
    def __init__(
        self,
        trans_weight: float = 1.0,
        rot_weight: float = 3.0,
        trans_loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.trans_loss_type = trans_loss_type
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred: Dict with "translation" (B, 3), "rotation_6d" (B, 6), "quaternion" (B, 4)
            target: Dict with "translation" (B, 3), "quaternion" (B, 4)
        """
        # Translation loss
        if self.trans_loss_type == "smooth_l1":
            trans_loss = F.smooth_l1_loss(pred["translation"], target["translation"])
        else:
            trans_loss = F.mse_loss(pred["translation"], target["translation"])
        
        # Rotation loss using 6D representation
        rot_loss = geodesic_loss_matrix(pred["rotation_6d"], target["quaternion"])
        
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
