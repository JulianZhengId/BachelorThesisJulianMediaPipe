# Accessible Hand Gesture Instruction Pipeline Using MediaPipe and Unreal Engine
## Bachelor Thesis Expose: Games Programming

---

## Executive Summary

This thesis addresses a critical accessibility gap in gesture-based instruction authoring for serious applications (medical rehabilitation, physiotherapy, educational training). Current workflows require expertise in specialized animation software (Unreal Engine AnimBlueprints, motion capture systems), creating barriers for domain experts without technical backgrounds. The proposed pipeline enables non-technical users to author hand gesture instructions through simple video recording, leveraging **MediaPipe hand landmark detection** for data extraction, **temporal filtering algorithms** for noise reduction, and **Inverse Kinematics (IK) systems** in Unreal Engine for real-time skeletal animation synthesis.

The system is decomposed into three interconnected technical domains: (1) **input acquisition and landmark extraction** via MediaPipe, (2) **kinematic data preprocessing and filtering**, and (3) **skeletal retargeting and IK-driven animation synthesis** in Unreal Engine 5.

---

## 1. Problem Statement and Motivation

### 1.1 Domain Context: Gesture-Based Instruction Authoring

Hand gesture instruction pipelines are critical in domains including:
- **Medical/Physiotherapy**: Patient rehabilitation protocols requiring precise hand movement specifications
- **Occupational Therapy**: Fine motor skill training with visual reference standards
- **Sign Language Education**: Standardized gesture capture and reproduction
- **VR/AR Interfaces**: Gesture-based training and simulation

Current state-of-the-art approaches require **motion capture hardware** (optical, magnetic) or **manual keyframe animation** in professional software, both economically and technically prohibitive for practitioners without animation expertise.

### 1.2 The Accessibility Problem

**Barrier 1 – Technical Expertise**: Creating animations in Unreal Engine requires:
- Knowledge of skeletal mesh architecture and bone hierarchies
- Competency in AnimGraph editing and state machines
- Understanding of IK constraints, blending, and procedural animation
- Experience with retargeting and skeletal remapping

**Barrier 2 – Acquisition Cost**: Professional mocap systems ($50K–$500K+) are inaccessible to small clinical practices, educational institutions, or independent developers.

**Barrier 3 – Authoring Friction**: Iterating on gesture specifications requires re-entering Unreal Engine, modifying keyframes, and re-exporting—a slow feedback loop.

### 1.3 Proposed Solution Architecture

The pipeline abstracts gestural authoring into a **3-stage workflow**:

1. **Capture**: Non-technical user records gesture video (smartphone camera)
2. **Extract & Smooth**: Python pipeline processes video, extracts hand landmarks, applies noise filtering
3. **Synthesize**: Filtered joint data feeds Unreal Engine IK system, generating smooth skeletal animation

This decoupling allows gesture domain experts to work entirely outside Unreal Engine, while animation synthesis becomes **automated and deterministic**.

---

## 2. Technical Component 1: MediaPipe Hand Landmark Extraction

### 2.1 MediaPipe Hands Architecture Overview

**MediaPipe Hands** is Google's real-time, lightweight hand pose estimation solution designed for mobile and edge devices. The system operates in two stages:

#### Stage 1: Palm Detection
- **Detector Model**: Region-based CNN (conceptually similar to Faster R-CNN, optimized for real-time inference)
- **Input**: Full video frame in RGB format (arbitrary resolution, typically 1280×720 or 1920×1080 for instructional video)
- **Output**: Palm bounding box with rotation normalization
- **Design Rationale**: Palm-centric detection is **rotation-invariant** compared to full-hand detection, providing stable initialization across diverse hand articulations

**Architectural advantages**:
- Achieves **95.7% average precision** (AP) on palm detection benchmarks through:
  - Square-anchor palm modeling (3–5× fewer anchors than aspect-ratio-specific approaches)
  - Encoder-decoder feature extractor for multi-scale context
  - Focal loss optimization to handle high anchor density variance

#### Stage 2: Hand Landmark Regression
- **Regressor Model**: Dense neural network operating on high-resolution ROI crop
- **Input**: Cropped palm region (typical input size 192×192 or 224×224 pixels)
- **Output**: **21 3D hand keypoints** (21 = 4 fingers × 4 joints + 1 thumb with special indexing + 1 wrist)
- **Coordinate System**: (x, y, z) where x,y are normalized to [0,1] relative to bounding box, z is **normalized depth** derived from training data

**Landmark Topology (MediaPipe Hand Joint Indexing)**:
```
0:  Wrist
1:  Thumb CMC (Carpometacarpal)
2:  Thumb MCP (Metacarpophalangeal)
3:  Thumb IP (Interphalangeal)
4:  Thumb Tip

5:  Index Finger MCP
6:  Index Finger PIP (Proximal Interphalangeal)
7:  Index Finger DIP (Distal Interphalangeal)
8:  Index Finger Tip

[Similar indexing for Middle, Ring, Pinky fingers]

17: Pinky MCP
18: Pinky PIP
19: Pinky DIP
20: Pinky Tip
```

### 2.2 Training Data and Accuracy Characteristics

**Training Dataset Composition**:
- ~30K real-world annotated images (manual 3D coordinate annotation)
- Synthetic hand models rendered over diverse backgrounds
- Ground truth 3D coordinates from:
  - Manual annotation (2D image evidence)
  - Depth maps (where available)
  - GHUM hand model fitting (morphable hand model optimization)

**Inference Accuracy**:
- Mean 3D landmark localization error: ~8–15 mm for typical working distances (30–100 cm)
- Robustness to occlusions: Maintains tracking with up to 40% hand self-occlusion
- Temporal prediction**: Uses previous-frame landmarks to predict next palm ROI, achieving 60% compute reduction on static hands

### 2.3 Primary Limitation: Jitter and Temporal Noise

**Noise Characteristics**:
The key limitation for instruction authoring is **high-frequency jitter** in extracted landmarks, particularly:

1. **Detection Jitter**: Palm detection bounding box oscillates frame-to-frame by 2–5 pixels
2. **Regression Jitter**: Individual landmark coordinates fluctuate by 0.5–2 cm even during static hand posture
3. **Frequency Composition**: Jitter predominantly concentrated in **10–30 Hz band**, well above natural hand gesture bandwidth (< 5 Hz for controlled movements)

**Physical Source**:
- CNN regression inherent quantization and interpolation artifacts
- Camera sensor noise propagating through detection pipeline
- Shallow hand pose ambiguities (self-similar configurations at different depths/rotations)

**Impact on Downstream Retargeting**:
Raw landmarks directly fed to IK solver cause:
- High-frequency skeletal tremor in Unreal Engine animation
- IK constraint oscillation (joint chains "jittering" to satisfy contradictory target positions across frames)
- Unsuitability for clinical/educational applications requiring smooth reference gestures

**Solution Strategy**: Temporal filtering (covered in Section 3)

---

## 3. Technical Component 2: Temporal Filtering and Kinematic Data Preprocessing

### 3.1 Filter Selection Rationale: Frequency Domain Analysis

Hand gesture motion during controlled instruction demonstrations exhibits **characteristic frequency content**:
- **Genuine gesture motion**: 0.5–5 Hz (typical voluntary hand movement bandwidth)
- **Respiration-induced micro-motion**: 0.2–0.5 Hz
- **Sensor/detection jitter**: 10–50 Hz (high-frequency noise)

**Filter Design Objective**: Remove 10–50 Hz jitter while **preserving sub-5 Hz gesture dynamics**.

### 3.2 Filter Candidates and Comparative Analysis

#### Option 1: Exponential Moving Average (EMA)
**Mathematical Formulation**:
\[
\hat{x}_t = \alpha x_t + (1 - \alpha) \hat{x}_{t-1}
\]

Where:
- \(\hat{x}_t\) = filtered position at frame t
- \(x_t\) = raw measurement
- \(\alpha\) = smoothing coefficient ∈ (0,1), typically 0.1–0.3 for mocap

**Characteristics**:
- **Computational Cost**: O(1) per frame, minimal memory (single state vector per landmark)
- **Latency**: \(\tau_{\text{group}} = \frac{1-\alpha}{\alpha f_s}\) where \(f_s\) is sampling frequency
- **Typical Performance**: Reduces jitter amplitude by 30–50%, introduces 100–300 ms output lag
- **Limitation**: Simple low-pass with roll-off of −6 dB/octave; insufficient for aggressive noise suppression

**Use Case**: Suitable only for non-critical applications; clinical/educational use is questionable due to motion distortion.

---

#### Option 2: One Euro Filter (Recommended for This Application)
**Mathematical Formulation** (Infinite Impulse Response with velocity-adaptive smoothing):

The One Euro Filter operates on principles of **dynamic parameter tuning**:

1. **Low-Pass Filter on Position**:
\[
\text{filter}_{\text{pos}} = \frac{\beta}{1 + \beta} x_t + \frac{1}{1 + \beta} \hat{x}_{t-1}
\]

2. **Low-Pass Filter on Velocity** (derivative smoothing):
\[
v_t = \frac{x_t - \hat{x}_{t-1}}{\Delta t}
\]
\[
\text{filter}_{\text{vel}} = \frac{\beta_v}{1 + \beta_v} v_t + \frac{1}{1 + \beta_v} \hat{v}_{t-1}
\]

3. **Adaptive Cutoff Frequency**:
\[
\omega_c = 2 \pi f_c^{\min} + \alpha |\text{filter}_{\text{vel}}|
\]

Where:
- \(f_c^{\min}\) = minimum cutoff frequency (e.g., 1 Hz)
- \(\alpha\) = velocity scaling factor (0.001–0.01)
- \(\beta = \frac{2 \pi \omega_c \Delta t}{2 \pi \omega_c \Delta t + 1}\)

**Key Insight**: When velocity is low (static hand), cutoff frequency decreases, maximizing smoothing. When velocity is high (fast gesture), cutoff increases, preserving motion fidelity.

**Characteristics**:
- **Noise Suppression**: Achieves 70–90% jitter amplitude reduction
- **Latency**: Typically 50–100 ms at \(f_c^{\min} = 1\) Hz, significantly lower than EMA
- **Latency-Smoothness Trade-off**: Tunable via \(f_c^{\min}\) and \(\alpha\) parameters
- **Gesture Preservation**: Maintains sharp transitions (e.g., finger extension) better than simple low-pass

**Implementation Considerations**:
- Per-landmark filtering: Apply independently to all 21 × 3 coordinate dimensions
- Temporal state accumulation: Requires buffering previous position/velocity estimates
- Reference Implementation**: GitHub libraries available (jaantollander/OneEuroFilter); well-suited to Python preprocessing pipeline

**Recommended Parameters for Gesture Instruction**:
- \(f_c^{\min} = 1.0\) Hz (minimum cutoff for slow hand stabilization)
- \(\alpha = 0.008\) (gentle velocity coupling for smooth gesture onset/offset)
- \(\Delta t = \frac{1}{f_s}\) where \(f_s \in [30, 120]\) fps

---

#### Option 3: Kalman Filter
**Mathematical Formulation** (discrete constant-velocity model):

**Prediction**:
\[
\mathbf{x}_t^- = \mathbf{F} \mathbf{x}_{t-1}^+ + \mathbf{w}_{t-1}
\]
\[
\mathbf{P}_t^- = \mathbf{F} \mathbf{P}_{t-1}^+ \mathbf{F}^T + \mathbf{Q}
\]

**Update**:
\[
\mathbf{K}_t = \mathbf{P}_t^- \mathbf{H}^T (\mathbf{H} \mathbf{P}_t^- \mathbf{H}^T + \mathbf{R})^{-1}
\]
\[
\mathbf{x}_t^+ = \mathbf{x}_t^- + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H} \mathbf{x}_t^-)
\]

Where:
- \(\mathbf{F}\) = state transition matrix (encodes constant-velocity assumption)
- \(\mathbf{Q}\) = process noise covariance (hand acceleration unpredictability)
- \(\mathbf{R}\) = measurement noise covariance (MediaPipe jitter)
- \(\mathbf{H}\) = measurement matrix (identity for direct landmark observation)

**Characteristics**:
- **Theoretical Optimality**: Minimum mean-square error under linear Gaussian assumptions
- **Computational Cost**: O(n²) per frame for n-dimensional state; for 21 × 3 landmarks, **computationally expensive** (O(27² ≈ 729) operations per hand per frame)
- **Tuning Complexity**: Requires careful Q/R covariance calibration; poor tuning leads to lag or overshooting
- **Velocity Estimation**: Directly estimates hand velocity, useful for gesture classification
- **Limitation for Gesture**: Assumes constant velocity—abrupt gesture transitions (e.g., snap closing fingers) are over-smoothed if Q is underestimated

**Use Case**: Justified when velocity is critical downstream; otherwise, One Euro Filter offers better latency/smoothing trade-off.

---

#### Option 4: Butterworth Low-Pass Filter
**Transfer Function** (frequency domain):
\[
H(f) = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}}
\]

Where:
- \(f_c\) = cutoff frequency (e.g., 5 Hz)
- \(n\) = filter order (typically 2–4)

**Implementation** (time domain via IIR recursion):
\[
y_t = b_0 x_t + b_1 x_{t-1} + b_2 x_{t-2} - a_1 y_{t-1} - a_2 y_{t-2}
\]

Coefficients \(b_0, b_1, b_2, a_1, a_2\) derived from standard IIR design (bilinear transformation).

**Characteristics**:
- **Frequency Response**: Maximally flat passband (no ripple), smooth roll-off (−20n dB/decade)
- **Computational Cost**: O(1) per frame, minimal memory (maintains 2 previous samples)
- **Latency**: \(\tau \approx \frac{1}{2 \pi f_c}\) per filter stage; stacking higher orders increases lag
- **Gesture Preservation**: Risk of ringing artifacts (overshooting) on sharp transitions with higher orders
- **Advantage**: Well-understood, widely supported in signal processing libraries (scipy.signal)

**Recommended Parameters**:
- Order n = 2 or 3 (higher order risks ringing; lower order insufficient attenuation)
- Cutoff \(f_c = 5–8\) Hz (preserves gesture dynamics, attenuates jitter)
- **Total Group Delay**: ~125–200 ms at 5 Hz cutoff

---

### 3.3 Comparative Summary and Selection

| Characteristic | EMA | One Euro | Kalman | Butterworth |
|---|---|---|---|---|
| **Jitter Reduction** | 30–50% | 70–90% | 80–95% | 65–85% |
| **Output Latency** | High (100–300 ms) | **Low (50–100 ms)** | Medium (50–150 ms) | Medium (125–200 ms) |
| **Computational Cost** | Very Low | Very Low | **High** | Low |
| **Tuning Complexity** | Very Low | **Low** | High | Medium |
| **Gesture Fidelity** | Poor (over-smoothed) | **Excellent** | Good | Good |
| **Real-time Capability** | ✓ | **✓** | ✓ | ✓ |

**Selection Rationale for This Pipeline**: **One Euro Filter** is optimal because:
1. **Latency-critical application**: 50–100 ms output lag acceptable for offline instruction authoring
2. **Gesture fidelity**: Velocity-adaptive smoothing preserves sharp finger transitions
3. **Implementation accessibility**: Simple Python implementation, no library dependencies
4. **Parametric control**: Two intuitive hyperparameters (\(f_c^{\min}, \alpha\)) with clear physical interpretation

---

### 3.4 Preprocessing Pipeline Architecture

**Sequential Processing Stages**:

```
Video Input (MP4/MOV)
        ↓
Frame Extraction (OpenCV)
        ↓
Per-Frame MediaPipe Detection
  ├─ Palm Detection
  └─ 21 Landmark Regression
        ↓
Raw Landmark Sequence [T, 21, 3]
        ↓
One Euro Filter (per-landmark, per-dimension)
        ↓
Smoothed Landmark Sequence [T, 21, 3]
        ↓
JSON/CSV Export (Frame-indexed keypoint data)
        ↓
Unreal Engine Import
```

**Data Format Specification** (JSON output):
```json
{
  "metadata": {
    "source_video": "gesture_demo.mp4",
    "fps": 30,
    "total_frames": 150,
    "filter_config": {
      "filter_type": "one_euro",
      "fc_min": 1.0,
      "alpha": 0.008
    }
  },
  "frames": [
    {
      "frame_id": 0,
      "timestamp_ms": 0,
      "landmarks": [
        {"joint_id": 0, "x": 0.45, "y": 0.52, "z": 0.31, "confidence": 0.998},
        {"joint_id": 1, "x": 0.48, "y": 0.61, "z": 0.28, "confidence": 0.995},
        ...
      ]
    },
    ...
  ]
}
```

**Confidence Scoring**: MediaPipe provides per-landmark confidence (0–1), indicating detection reliability. Optionally, filter confidence as secondary pre-filtering step.

---

## 4. Technical Component 3: Inverse Kinematics and Skeletal Retargeting in Unreal Engine

### 4.1 Hand Skeletal Architecture

**Skeletal Hierarchy Structure** (standard for humanoid rigs in Unreal Engine 5):

```
Hand_Root (Wrist)
├── Thumb_00 (CMC joint)
│   └── Thumb_01 (MCP joint)
│       └── Thumb_02 (IP joint)
│           └── Thumb_Tip
├── Index_00 (MCP joint)
│   └── Index_01 (PIP joint)
│       └── Index_02 (DIP joint)
│           └── Index_Tip
├── Middle_00 (MCP joint)
│   └── Middle_01 (PIP joint)
│       └── Middle_02 (DIP joint)
│           └── Middle_Tip
├── Ring_00 (MCP joint)
│   └── Ring_01 (PIP joint)
│       └── Ring_02 (DIP joint)
│           └── Ring_Tip
└── Pinky_00 (MCP joint)
    └── Pinky_01 (PIP joint)
        └── Pinky_02 (DIP joint)
            └── Pinky_Tip
```

**Architectural Assumption**: Unreal Engine mannequin includes detailed finger bone chains (typical configuration: 5 bones per finger = 25 bones total for hand).

**Mapping Strategy - MediaPipe ↔ Unreal Skeleton**:

| MediaPipe Landmark | Semantic Joint | Unreal Bone | Purpose |
|---|---|---|---|
| 0 | Wrist | Hand_Root | Root position anchor |
| 1–4 | Thumb CMC→IP | Thumb_00→Thumb_02 | Thumb articulation |
| 5–8 | Index MCP→DIP | Index_00→Index_02 | Index finger articulation |
| 9–12 | Middle MCP→DIP | Middle_00→Middle_02 | Middle finger articulation |
| 13–16 | Ring MCP→DIP | Ring_00→Ring_02 | Ring finger articulation |
| 17–20 | Pinky MCP→DIP | Pinky_00→Pinky_02 | Pinky finger articulation |

**Coordinate Frame Conversion**:
- **MediaPipe**: Camera-relative coordinates; x (left-right), y (up-down), z (depth toward camera)
- **Unreal Engine**: World-space coordinates; forward (X), right (Y), up (Z)
- **Transformation Required**: Rigid transformation (rotation + translation) from camera frame to UE world frame

---

### 4.2 IK Rig Design: Multi-Bone Finger Chains

**Problem Statement**: MediaPipe provides **21 3D landmark positions** (wrist + 20 finger/palm joints). Direct kinematic retargeting would require **joint angle estimation** from positions, which is **ill-posed** (multiple rotation angles can yield same endpoint position due to joint DOF redundancy).

**Solution: Inverse Kinematics (IK)**
Instead of computing joint rotations directly, use **constrained optimization** to find bone rotations that satisfy:
- **End-effector (finger tip) reaches MediaPipe landmark position**
- **Intermediate joint (knuckle) follows intermediate MediaPipe position**
- **Biomechanical joint limits** (e.g., fingers don't bend backward)

### 4.3 IK Constraint Architecture in Unreal Engine 5

#### Skeletal Control Node: Two-Bone IK

**Configuration** (per-finger IK chain):

```
Bone Chain Structure:
  Proximal Bone (MCP joint) ← solves for rotation
      ↓
  Middle Bone (PIP joint) ← automatic result
      ↓
  Distal Bone (DIP joint) ← **Effector target** (finger tip)
```

**Two-Bone IK Node Parameters**:

| Parameter | Definition | Typical Value |
|---|---|---|
| **IK Bone** | Distal bone (finger tip bone, e.g., Index_02) | Index_02 |
| **Effector Location** | 3D world position (from MediaPipe tip landmark) | Landmark[8].xyz |
| **Joint Target Location** | PIP joint position (intermediate knuckle, from landmark[7]) | Landmark[7].xyz |
| **Allow Stretching** | Permit finger elongation if effector unreachable | False (preserve anatomy) |
| **Allow Twist** | Enable rotational freedom around bone axis | True (natural hand twist) |
| **Twist Axis** | Axis of twist in local bone space | Bone-dependent (typically X) |
| **Maintain Effector Rel Rot** | Preserve finger tip orientation relative to effector | True |
| **Start Stretch Ratio** | Threshold before stretching activates | 0.9 (90% of limb length) |
| **Max Stretch Scale** | Maximum elongation multiplier | 1.15 (15% max stretch) |

**Mathematical Formulation** (simplified):

Given:
- Proximal bone position \(\mathbf{P}\)
- Distal bone position \(\mathbf{D}\)
- Target effector position \(\mathbf{T}\) (finger tip from MediaPipe)
- Bone segment lengths \(L_1, L_2\)

Solve for rotation angles \(\theta_1, \theta_2\) such that:
\[
\mathbf{D} + L_2 \mathbf{R}(\theta_1, \theta_2) \hat{\mathbf{z}} = \mathbf{T}
\]

(Where \(\mathbf{R}\) is rotation matrix parameterized by joint angles)

Unreal's Two-Bone IK uses **analytic closed-form solution** (FABRIK variant):
1. Compute distance \(d = |\mathbf{T} - \mathbf{P}|\)
2. If \(d > L_1 + L_2\): stretching case (or limit) → solve constrained problem
3. Otherwise: apply law of cosines to find joint angles \(\theta_1, \theta_2\)

**Computational Complexity**: O(1) per finger per frame (analytic, non-iterative)

---

### 4.4 IK Rig Setup Workflow

**Step 1: Create Base IK Rig Asset**

In Unreal Editor:
```
Skeletal Mesh → IK Rig → Create IK Rig
```

**Step 2: Define Joint Chains**

For each finger, create a **Goal Chain**:
```
Right-Click on Root Bone → Set as Retarget Root

For each finger:
  Right-Click on proximal bone (e.g., Index_00)
    → New Goal Chain "RightIndex"
    → Drag to distal bone (Index_02)
    → Confirm
```

**Step 3: Configure Effector Goals**

```
In IK Rig Details:
  Effector Settings
    ├─ [For each finger]
    │   ├─ Effector Name: "RightIndex_Tip"
    │   ├─ Target Bone: Index_02 (distal)
    │   └─ Enable Debug Visualization
```

**Step 4: Bone Settings and Constraints**

Per finger, configure biomechanical limits:
```
Right-Click on MCP bone → Add IK Settings:
  ├─ Rotation Limit Type: "Free" or "Locked Axis"
  │   (Typically "Free" for wrist/base; "Locked" for twist-only joints)
  ├─ Preferred Angle Offset: [set to match T-pose]
  └─ Effector Weight: 1.0
```

**Step 5: Create Animation Blueprint**

In Animation Graph:
```
[Input] Pose → [Skeletal Control] IK Retargeter → [Output] Final Pose

IK Retargeter Settings:
  ├─ IK Rig Asset: <created in Step 1>
  ├─ Use Foot Lock: False (no ground contact)
  ├─ Enable Foot Lock Curves: False
  └─ Alpha: 1.0 (full IK application)
```

---

### 4.5 Data Flow from Python to IK Animation

**Complete Integration Pipeline**:

```
Python Preprocessing
  ↓
Smoothed Landmarks (JSON) [T frames × 21 joints × 3 coords]
  ↓
Unreal Python API / File Import
  ├─ Read JSON frame-by-frame
  ├─ Convert MediaPipe coords → World coords
  └─ Write to UE Animation Sequence
  ↓
Animation Sequence (Native UE Format)
  ├─ 150 frames (example)
  ├─ Bone curve data for each skeleton joint
  └─ [Wrist position, Finger IK targets × 5]
  ↓
AnimGraph with IK Retargeter
  ├─ Loads Animation Sequence
  ├─ Evaluates IK Retargeter Skeletal Control
  └─ Outputs final skeletal poses
  ↓
Mannequin Visual (Real-time or Preview)
  ├─ Smooth hand animation
  ├─ Artifact-free finger articulation
  └─ Publication-ready instruction video
```

---

### 4.6 Practical Implementation Considerations

#### Coordinate Space Alignment

**MediaPipe Output Coordinate System**:
- Origin: Camera principal point (center of image)
- X: Rightward (pixel columns)
- Y: Downward (pixel rows)
- Z: Forward (toward camera, normalized 0–1 within hand region)
- Units: Image-normalized (0–1) or pixel coordinates (0–image width/height)

**Unreal Engine World Space**:
- Origin: Level spawn point
- X: Forward
- Y: Right
- Z: Up
- Units: Unreal Units (1 UU ≈ 1 cm)

**Transformation Pipeline**:
```python
# Pseudocode
def transform_mediapipe_to_ue(mp_landmark, camera_matrix, extrinsic_matrix):
    # mp_landmark: [x, y, z] in image coordinates (0–1 normalized)
    
    # 1. Denormalize to pixel coordinates
    pixel_x = mp_landmark.x * image_width
    pixel_y = mp_landmark.y * image_height
    pixel_z = mp_landmark.z * 100  # depth scaling (arbitrary units)
    
    # 2. Camera intrinsic inversion (backproject to 3D)
    camera_ray = camera_matrix_inv @ [pixel_x, pixel_y, 1.0].T
    depth_scale = pixel_z / ||camera_ray||  # normalized depth → actual metric depth
    world_point_camera_frame = depth_scale * camera_ray
    
    # 3. Camera extrinsic transformation (camera frame → world frame)
    world_point = extrinsic_matrix @ world_point_camera_frame
    
    # 4. Scale to Unreal Units and apply offset
    ue_position = (world_point + position_offset) * scale_factor
    
    return ue_position
```

**Calibration Requirement**: Single-time setup to compute camera intrinsics (focal length, principal point) and extrinsic pose (position/orientation in Unreal level). OpenCV calibration tools or direct measurement.

---

#### IK Solver Stability and Convergence

**Potential Issues**:

1. **Unreachable Targets**: If MediaPipe landmark falls outside the geometric reach of finger chain
   - **Solution**: Clamp effector to closest reachable position
   - **Parameter**: `Allow Stretching = True` with `Max Stretch Scale = 1.15` permits controlled elongation

2. **Singular Configurations**: When effector lies on the proximal bone axis, IK solution is ambiguous
   - **Solution**: Use `Joint Target Location` (intermediate knuckle position) to bias solution
   - **Frequency**: Rare with filtered, continuous motion

3. **Flickering Between Solutions**: Multiple IK solutions exist; discrete frame-to-frame switching causes visual pop
   - **Solution**: Blend-space smoothing in Animation Blueprint; continuity via filtering (addressed in Section 3)

---

## 5. System Integration and Data Flow

### 5.1 End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  Non-Technical User Records Gesture (Smartphone/Webcam Video)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  EXTRACTION LAYER (Python)                       │
│                                                                  │
│  1. Video Frame Decode (OpenCV)                                 │
│  2. Per-Frame MediaPipe Hand Detection & Landmark Regression     │
│  3. Temporal Filtering (One Euro Filter)                        │
│  4. Confidence Validation & Outlier Removal                     │
│  5. Coordinate System Transformation                            │
│  6. JSON/FBX Export                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────────────────┐  ┌────────────▼─────────────────┐
│   VALIDATION LAYER         │  │  OPTIONALLY: Manual Refinement│
│  Jitter Visualization      │  │  (Expert Fine-tune Poses)     │
│  Frame Confidence Analysis │  │                              │
└───────┬────────────────────┘  └────────────┬─────────────────┘
        │                                     │
        └─────────────────┬────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────────┐
│              SYNTHESIS LAYER (Unreal Engine)                    │
│                                                                │
│  1. Import Landmark Sequence                                   │
│  2. Create Animation Sequence (Native UE Format)                │
│  3. Apply IK Retargeter Skeletal Control                       │
│  4. Real-time IK Solving for All Fingers                       │
│  5. Generate Final Skeletal Mesh Animation                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│  Publication-Ready Hand Gesture Instruction Animation           │
│  (Video Export, Loop-Ready Asset, Documentation)                │
└──────────────────────────────────────────────────────────────────┘
```

---

### 5.2 Python Implementation Skeleton

**Module Structure**:

```python
# gesture_pipeline.py

import cv2
import mediapipe as mp
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class OneEuroFilter:
    """One Euro Filter implementation for landmark smoothing"""
    
    def __init__(self, fc_min: float = 1.0, alpha: float = 0.008, fps: int = 30):
        self.fc_min = fc_min
        self.alpha = alpha
        self.fps = fps
        self.dt = 1.0 / fps
        
        self.prev_position = None
        self.prev_velocity = np.zeros(3)
        
    def filter(self, position: np.ndarray) -> np.ndarray:
        """Apply One Euro Filter to 3D position"""
        
        if self.prev_position is None:
            self.prev_position = position.copy()
            return position
        
        # Velocity computation
        velocity = (position - self.prev_position) / self.dt
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Adaptive cutoff frequency
        omega_c = 2 * np.pi * (self.fc_min + self.alpha * velocity_magnitude)
        
        # Smoothing coefficient
        beta = (2 * np.pi * omega_c * self.dt) / (2 * np.pi * omega_c * self.dt + 1)
        
        # Filter position
        filtered_position = beta * position + (1 - beta) * self.prev_position
        
        # Filter velocity for next iteration
        filtered_velocity = beta * velocity + (1 - beta) * self.prev_velocity
        
        # Update state
        self.prev_position = filtered_position.copy()
        self.prev_velocity = filtered_velocity.copy()
        
        return filtered_position

class GesturePipeline:
    """Complete gesture extraction and preprocessing pipeline"""
    
    def __init__(self, video_path: str, output_json: str, fps: int = 30):
        self.video_path = video_path
        self.output_json = output_json
        self.fps = fps
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize filters (per landmark)
        self.filters = [OneEuroFilter(fc_min=1.0, alpha=0.008, fps=fps) 
                        for _ in range(21)]
    
    def extract_landmarks(self) -> List[dict]:
        """Extract and filter hand landmarks from video"""
        
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0
        frames_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            results = self.mp_hands.process(frame_rgb)
            
            frame_data = {
                "frame_id": frame_id,
                "timestamp_ms": (frame_id / self.fps) * 1000,
                "landmarks": []
            }
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for joint_id, landmark in enumerate(hand_landmarks.landmark):
                        # Raw coordinates
                        raw_pos = np.array([landmark.x, landmark.y, landmark.z])
                        
                        # Apply One Euro Filter
                        filtered_pos = self.filters[joint_id].filter(raw_pos)
                        
                        frame_data["landmarks"].append({
                            "joint_id": joint_id,
                            "x": float(filtered_pos[0]),
                            "y": float(filtered_pos[1]),
                            "z": float(filtered_pos[2]),
                            "confidence": float(landmark.z)  # z as confidence proxy
                        })
            
            frames_data.append(frame_data)
            frame_id += 1
        
        cap.release()
        return frames_data
    
    def export_json(self, frames_data: List[dict]):
        """Export landmark data to JSON"""
        
        metadata = {
            "source_video": self.video_path,
            "fps": self.fps,
            "total_frames": len(frames_data),
            "filter_config": {
                "filter_type": "one_euro",
                "fc_min": 1.0,
                "alpha": 0.008
            }
        }
        
        output = {"metadata": metadata, "frames": frames_data}
        
        with open(self.output_json, 'w') as f:
            json.dump(output, f, indent=2)
    
    def run(self):
        """Execute complete pipeline"""
        print("Extracting landmarks...")
        frames_data = self.extract_landmarks()
        
        print(f"Exporting {len(frames_data)} frames to {self.output_json}...")
        self.export_json(frames_data)
        
        print("Pipeline complete.")

# Usage
if __name__ == "__main__":
    pipeline = GesturePipeline(
        video_path="gesture_demo.mp4",
        output_json="landmarks_filtered.json",
        fps=30
    )
    pipeline.run()
```

---

## 6. Validation and Quality Assurance

### 6.1 Quantitative Metrics

**Filtering Effectiveness**:
- **Signal-to-Noise Ratio (SNR)** improvement post-filtering
- **Root Mean Square Error (RMSE)** of filtered vs. reference (ground truth from manual annotation)
- **Frequency spectrum analysis**: Verify attenuation in 10–50 Hz band

**IK Accuracy**:
- **End-effector error**: Distance between MediaPipe tip landmark and actual rendered finger tip
- **Joint angle consistency**: Verify intermediate knuckle positions match expectation

### 6.2 Qualitative Validation

- **Visual inspection**: Render gesture animation in Unreal and compare with source video
- **Jitter elimination**: Subjective assessment that tremor is absent
- **Gesture fidelity**: Confirm sharp transitions (e.g., finger snap) are preserved
- **Domain expert review**: Physiotherapist/instructor validates instruction suitability

---

## 7. Conclusions and Future Work

### 7.1 Key Contributions

1. **Accessibility Pipeline**: Non-technical users can author hand gesture instructions via smartphone video
2. **Noise Filtering Architecture**: One Euro Filter balances jitter reduction and latency
3. **IK-Driven Synthesis**: Fully automated skeletal animation from 3D landmarks
4. **Reproducible Framework**: Documented pipeline with clear technical rationale

### 7.2 Limitations and Future Directions

**Current Limitations**:
- Single-hand tracking only; multi-hand extension requires separate MediaPipe instances
- Camera calibration required for accurate 3D reconstruction
- IK solver assumes fixed finger topology (extensible to different skeletal structures)

**Future Enhancements**:
- Real-time preview in Unreal Engine during capture
- Multi-hand simultaneous tracking and cross-hand interaction synthesis
- Deep learning-based end-to-end gesture-to-animation network
- Integration with gesture classification systems for automated instruction repository

---

## References and Technical Standards

**Core Technologies**:
- MediaPipe Hands: Google AI Research (https://google.github.io/mediapipe/solutions/hands.html)
- Unreal Engine 5 Documentation: IK Retargeting (https://dev.epicgames.com/)
- One Euro Filter: Casiez et al., "1€ Filter" (https://github.com/jaantollander/OneEuroFilter)

**Key Research**:
- Lee & Shin (1999): "Real-time continuous motion blending based on motion similarity" — quaternion-based motion smoothing
- Perona & Malik (1990): Anisotropic diffusion filtering (theoretical foundation for adaptive smoothing)
- Skogstad et al. (2013): "Filtering Motion Capture Data for Real-Time Applications"

**Standards**:
- OpenGL/DirectX coordinate conventions (Unreal Engine usage)
- FBX skeletal animation format (interchange standard)

---

**Document Version**: 1.0  
**Date**: December 2025  
**Status**: Bachelor Thesis Expose (Ready for Academic Review)
