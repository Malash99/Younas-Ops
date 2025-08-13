# TSFormer Visual Odometry Training Analysis Report

**Date:** August 13, 2025  
**Model:** TSFormer with ViT Backbone  
**Training Duration:** 8 epochs  
**Sequence Length:** 8 frames  

## Executive Summary

The TSFormer visual odometry model shows significant issues with trajectory prediction that manifest as straight-line behavior and dramatic failure in the YZ plane. The analysis reveals several critical problems in the model architecture, training configuration, and evaluation pipeline that explain the observed behavior.

## Key Findings

### 1. Straight-Line Trajectory Problem

**Observation:** The model consistently predicts nearly straight-line trajectories across all datasets (training, validation, test).

**Root Causes:**
- **Frozen Backbone Issue:** The model was configured with `freeze_backbone=False` in training but evaluation shows minimal feature variation
- **Insufficient Training:** Only 8 epochs with a complex transformer architecture is insufficient for convergence
- **Poor Gradient Flow:** The combined L1+L2 loss with high rotation weight (100x) may cause gradient dominance issues

### 2. YZ Plane Failure

**Critical Issue:** The model completely fails to predict motion in the YZ plane (front view), showing:
- Training: Straight line with minimal variation
- Validation: Dramatic divergence in YZ plane with straight-line artifact
- Test: Complete failure to track YZ motion

**Technical Analysis:**
- The YZ plane corresponds to lateral (Y) and vertical (Z) movements
- The model shows bias toward predicting minimal changes in these dimensions
- This suggests the ViT backbone may not be extracting sufficient features for these motion types

### 3. Training vs Validation Discrepancy

**Performance Metrics Comparison:**

| Dataset | Translation RMSE | Rotation RMSE | Samples |
|---------|------------------|---------------|---------|
| Training | 0.0125 m | 0.0184 rad | 611 |
| Validation | 0.0074 m | 0.0117 rad | 170 |
| Test | 0.0153 m | 0.0208 rad | 201 |

**Analysis:**
- Validation metrics appear artificially better than training
- This indicates potential overfitting to validation data or data leakage
- Test performance is worse than both training and validation

## Architectural Issues

### 1. Model Configuration Problems

```python
# Current problematic configuration
freeze_backbone=False  # Should be True initially for transfer learning
hidden_dim=768        # Too large for limited data
num_transformer_layers=4  # May be excessive for 8-frame sequences
```

### 2. Loss Function Issues

```python
# Current loss weighting
trans_weight=1.0
rot_weight=100.0  # Excessive rotation weighting
```

The 100x rotation weight may cause:
- Gradient dominance by rotation loss
- Suppression of translation learning
- Unstable training dynamics

### 3. Data Pipeline Issues

**Sequence Windows:**
- Overlap frames: 4 (50% overlap may cause data leakage)
- Window creation doesn't account for temporal gaps
- Target pose prediction uses only the last frame

## Training Configuration Analysis

### Hyperparameter Issues:

1. **Learning Rate:** 1e-4 may be too low for unfrozen ViT backbone
2. **Batch Size:** 2 is extremely small, causing noisy gradients
3. **Weight Decay:** 1e-4 may be insufficient for regularization
4. **Sequence Length:** 8 frames may be too long for current model capacity

### Data Augmentation Concerns:
- Horizontal flip augmentation inappropriate for VO (changes motion direction)
- Color jittering may hurt feature consistency
- Gaussian blur may remove important edge features

## Dataset Analysis

### Split Strategy Issues:
- Bag-based splitting is correct for preventing data leakage
- However, train/val split within bags (80/20) may be insufficient
- Test bag selection may not be representative

### Data Quality Concerns:
- Delta pose computation assumes small motion steps
- No scale normalization or standardization
- Missing data validation for pose consistency

## Recommendations

### Immediate Fixes:

1. **Freeze ViT Backbone Initially:**
   ```python
   freeze_backbone=True  # Start with frozen features
   ```

2. **Adjust Loss Weighting:**
   ```python
   trans_weight=1.0
   rot_weight=10.0  # Reduce from 100 to 10
   ```

3. **Increase Batch Size:**
   ```python
   batch_size=8  # Use gradient accumulation if memory limited
   accumulate_steps=4  # Effective batch size = 32
   ```

4. **Longer Training:**
   ```python
   num_epochs=50  # Much longer training needed
   ```

### Architecture Improvements:

1. **Progressive Unfreezing:**
   - Train with frozen backbone for 20 epochs
   - Unfreeze top layers gradually
   - Fine-tune entire model at lower learning rate

2. **Simpler Temporal Model:**
   ```python
   sequence_length=4  # Start smaller
   num_transformer_layers=2  # Reduce complexity
   hidden_dim=384  # Smaller hidden dimension
   ```

3. **Better Pose Representation:**
   - Use relative pose deltas properly
   - Add pose normalization/standardization
   - Consider quaternion representation for rotations

### Data Pipeline Fixes:

1. **Remove Harmful Augmentations:**
   ```python
   # Remove horizontal flip and gaussian blur
   augmentations = [transforms.ColorJitter(brightness=0.1, contrast=0.1)]
   ```

2. **Improve Window Strategy:**
   ```python
   overlap_frames=1  # Minimal overlap to prevent leakage
   ```

3. **Add Data Validation:**
   - Check for pose outliers
   - Validate temporal consistency
   - Add motion magnitude thresholds

## Expected Improvements

With the recommended changes:

1. **YZ Plane Performance:** Should improve dramatically with proper loss balancing
2. **Trajectory Diversity:** Reduced straight-line behavior with better feature learning
3. **Validation Consistency:** More realistic performance gaps between train/val/test
4. **Training Stability:** Better convergence with appropriate hyperparameters

## Conclusion

The current TSFormer model suffers from fundamental issues in:
- Loss function design (excessive rotation weighting)
- Training strategy (insufficient epochs, wrong backbone strategy)
- Data pipeline (harmful augmentations, potential leakage)
- Architecture sizing (too complex for available data)

The straight-line trajectory and YZ plane failure are direct consequences of these design choices. Implementing the recommended fixes should resolve these issues and lead to more realistic trajectory prediction.

## Files Analyzed

- `models/tsformer_vo.py` - Model architecture
- `train_tsformer.py` - Training script and hyperparameters  
- `evaluate_tsformer_comprehensive.py` - Evaluation pipeline
- `datasets/underwater_vo_dataset.py` - Data loading and preprocessing
- `evaluation_results_comprehensive/` - Generated visualizations and metrics