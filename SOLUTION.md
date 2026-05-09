# SPARK 2026: Spacecraft Pose Estimation (9th Place Global Solution)

This repository contains the source code and configuration for our 9th place solution in the **SPARK 2026 Spacecraft Pose Estimation Challenge**. Our approach focuses on leveraging event-based vision for high-precision pose estimation in space environments.

## The Challenge
The goal of the SPARK 2026 challenge is to estimate the 6-DoF pose (3D translation and 4D quaternion rotation) of a spacecraft from a stream of event data. Event cameras provide high-temporal resolution and high-dynamic range, which are critical for capturing motion in the harsh lighting conditions of space.

## Technical Journey & Evolution

### 1. Baseline & Dataset Representation
We started by representing the event streams as **event frames** (accumulated polarity surfaces).
- **Backbone**: Initial experiments used ResNet-50 and EfficientNet-B0.
- **Representation**: 2-channel event frames (ON/OFF accumulations).

### 2. Iteration & Architecture Refinement
As the competition progressed, we realized that translation and rotation required different optimization strategies.
- **Loss Functions**: We transitioned from simple L1 loss to a combined loss using **SmoothL1** for translation and a specialized **quaternion distance loss** for rotation.
- **Backbone Upgrade**: Moved to more powerful backbones like **EfficientNet-B3** and **B4** to capture finer features.
- **FPN Integration**: Added Feature Pyramid Networks (FPN) to better handle the multi-scale nature of the spacecraft in the frame.

### 3. Keypoint-Based Experiments
We explored a keypoint-based approach (`src/models/keypoint_posenet.py`) where the model predicts the 2D projections of known 3D points on the spacecraft. While this provided strong geometric constraints, direct regression ultimately proved more robust for the final ensemble.

### 4. The "Hybrid" Breakthrough
Our most significant performance jump came from identifying that different training runs specialized in different components of the pose:
- **Version 7 (v7)**: Consistently achieved the lowest Translation Error.
- **Version 15 (v15)**: Achieved the most stable Rotation Error.

We implemented a **Hybrid Strategy** (`create_final_submissions.py`) that combined the translation vectors from `v7` with the quaternions from `v15`.

### 5. Temporal Smoothing (SLERP)
Event data is sequential. We observed high-frequency jitter in the raw rotation predictions. To fix this, we applied **Spherical Linear Interpolation (SLERP)** (`smooth_predictions.py`) across the predicted quaternions. This smoothed out the trajectory and significantly reduced the Orientation Error.

### 6. Final Submission Strategy
The winning submission (`hybrid_v7T_smoothedV15R.zip`) was a combination of:
1.  **Translation**: Best performing model on translation (`rtx4090_v7.yaml`).
2.  **Rotation**: Best performing model on rotation (`rtx4090_v15.yaml`).
3.  **Post-Processing**: SLERP smoothing on the rotation sequence.
4.  **TTA (Test Time Augmentation)**: Multi-scale and flip augmentation during inference.

## Final Results
Our hybrid approach achieved:
- **Rank**: 9th Place Globally
- **Competition**: [SPARK 2026 - Codabench](https://www.codabench.org/competitions/12706/#/results-tab)

## Repository Structure
- `config/`: Configuration files for all model versions (`v1` to `v18`).
- `src/models/`: Implementation of `PoseNet`, `FPN`, and `KeypointHead`.
- `smooth_predictions.py`: Post-processing script for SLERP.
- `create_final_submissions.py`: Logic for ensembling and hybrid pose construction.
- `predict_ensemble.py`: Multi-model inference pipeline.

## Reproducing the Results
1.  **Training**: Run `python train.py --config config/rtx4090_v7.yaml` and `v15.yaml`.
2.  **Prediction**: Generate predictions for both models using `predict.py`.
3.  **Hybridization**: Use `create_final_submissions.py` to merge the results.
4.  **Smoothing**: Run `smooth_predictions.py` on the resulting CSV.
