"""
Proba-2 satellite 3D model and keypoint projection utilities.

Defines 8 bounding-box keypoints on the Proba-2 satellite (~0.6x0.6x0.8m)
and provides functions to project them to 2D image coordinates.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ── Proba-2 Satellite Bounding Box Keypoints (body frame) ──────────────
# Approximate dimensions: 0.6m × 0.6m × 0.8m
# Origin at center of satellite body
PROBA2_KEYPOINTS_3D = np.array([
    [-0.3, -0.3, -0.4],  # 0: front-bottom-left
    [ 0.3, -0.3, -0.4],  # 1: front-bottom-right
    [ 0.3,  0.3, -0.4],  # 2: front-top-right
    [-0.3,  0.3, -0.4],  # 3: front-top-left
    [-0.3, -0.3,  0.4],  # 4: back-bottom-left
    [ 0.3, -0.3,  0.4],  # 5: back-bottom-right
    [ 0.3,  0.3,  0.4],  # 6: back-top-right
    [-0.3,  0.3,  0.4],  # 7: back-top-left
], dtype=np.float64)

N_KEYPOINTS = 8

# ── Camera Intrinsics (from camera.json) ───────────────────────────────
CAMERA_MATRIX = np.array([
    [1258.6057531097028, 0.0, 640.0],
    [0.0, 1258.6057531097028, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def project_keypoints(keypoints_3d, quaternion, translation, K=None):
    """
    Project 3D keypoints to 2D image coordinates using pose and camera intrinsics.
    
    Args:
        keypoints_3d: (N, 3) 3D keypoint coordinates in body frame
        quaternion: (4,) quaternion [Qx, Qy, Qz, Qw] (scipy convention)
        translation: (3,) translation [Tx, Ty, Tz]
        K: (3, 3) camera intrinsic matrix (default: CAMERA_MATRIX)
    
    Returns:
        keypoints_2d: (N, 2) projected pixel coordinates (x, y)
        visibility: (N,) boolean mask — True if keypoint is in front of camera
    """
    if K is None:
        K = CAMERA_MATRIX
    
    # Build rotation matrix from quaternion
    R = Rotation.from_quat(quaternion).as_matrix()  # (3, 3)
    t = np.asarray(translation).reshape(3, 1)       # (3, 1)
    
    # Transform keypoints from body frame to camera frame
    # P_cam = R @ P_body + t
    kp = np.asarray(keypoints_3d)  # (N, 3)
    P_cam = (R @ kp.T + t).T      # (N, 3)
    
    # Check visibility (z > 0 means in front of camera)
    visibility = P_cam[:, 2] > 0
    
    # Project to image plane: p = K @ P_cam / P_cam_z
    P_cam_safe = P_cam.copy()
    P_cam_safe[~visibility, 2] = 1.0  # Avoid division by zero
    
    p_homog = (K @ P_cam_safe.T).T  # (N, 3)
    keypoints_2d = p_homog[:, :2] / p_homog[:, 2:3]  # (N, 2)
    
    return keypoints_2d, visibility


def generate_heatmaps(keypoints_2d, visibility, heatmap_size, image_size, sigma=2.0):
    """
    Generate Gaussian heatmaps for each keypoint.
    
    Args:
        keypoints_2d: (N, 2) pixel coordinates in original image space
        visibility: (N,) boolean visibility mask
        heatmap_size: int, output heatmap resolution (e.g. 64)
        image_size: (H, W) original image dimensions
        sigma: float, Gaussian standard deviation in heatmap pixels
    
    Returns:
        heatmaps: (N, heatmap_size, heatmap_size) float32 array
    """
    N = len(keypoints_2d)
    H_img, W_img = image_size
    heatmaps = np.zeros((N, heatmap_size, heatmap_size), dtype=np.float32)
    
    # Scale factor from image space to heatmap space
    sx = heatmap_size / W_img
    sy = heatmap_size / H_img
    
    for i in range(N):
        if not visibility[i]:
            continue
        
        # Scale keypoint to heatmap coordinates
        kx = keypoints_2d[i, 0] * sx
        ky = keypoints_2d[i, 1] * sy
        
        # Check if keypoint is within heatmap bounds (with some margin)
        if kx < -3 * sigma or kx >= heatmap_size + 3 * sigma:
            continue
        if ky < -3 * sigma or ky >= heatmap_size + 3 * sigma:
            continue
        
        # Generate 2D Gaussian
        x = np.arange(heatmap_size, dtype=np.float32)
        y = np.arange(heatmap_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        
        gaussian = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = gaussian
    
    return heatmaps
