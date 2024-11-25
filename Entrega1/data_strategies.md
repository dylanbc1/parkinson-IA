# Data Collection and Processing Strategies

## Data Processing Pipeline Overview

This document details our comprehensive approach to data collection, processing, and feature engineering, with particular emphasis on the extraction and calculation of biomechanical features from video data.

## Raw Data Collection

### Video Capture System
Our data collection system utilizes a webcam-based approach implemented in the `VideoDataCollector` class with the following specifications:
- Frame rate: 30 FPS
- Resolution: Native webcam resolution
- Format: MP4 with H.264 encoding
- File naming convention: `subject_[ID]_[action]_[timestamp].mp4`

### Action Categories
Five distinct actions are captured:
1. Walking towards camera
2. Walking away from camera
3. Turning
4. Sitting down
5. Standing up

## Feature Engineering Pipeline

### 1. Pose Estimation and Landmark Extraction

#### MediaPipe Integration
The system employs MediaPipe's pose estimation model to extract 33 key landmarks. Critical landmarks include:
- Head: nose (0)
- Upper body: shoulders (11, 12), elbows (13, 14), wrists (15, 16)
- Lower body: hips (23, 24), knees (25, 26), ankles (27, 28)

#### Landmark Normalization
```python
def _normalize_landmarks(self, pose_landmarks, frame_shape):
    h, w = frame_shape[:2]
    # Normalize to [-1, 1] range for scale invariance
    x = (x - w/2) / (w/2)
    y = (y - h/2) / (h/2)
    z = z / (w/2)
```

### 2. Feature Extraction

#### Joint Angle Calculations
Joint angles are calculated using vector operations between relevant landmarks:

```python
def _calculate_angle(self, p1, p2, p3):
    """
    Calculates the angle between three points using the dot product formula:
    cos(θ) = (v1 · v2) / (|v1| |v2|)
    """
    v1 = p1 - p2  # Vector from point 2 to point 1
    v2 = p3 - p2  # Vector from point 2 to point 3
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = np.degrees(np.arccos(cos_angle))
```

Key angles calculated:
1. **Knee Angles**
   - Right knee: hip(23) → knee(25) → ankle(27)
   - Left knee: hip(24) → knee(26) → ankle(28)

2. **Hip Angles**
   - Right hip: shoulder(11) → hip(23) → knee(25)
   - Left hip: shoulder(12) → hip(24) → knee(26)

3. **Elbow Angles**
   - Right elbow: shoulder(11) → elbow(13) → wrist(15)
   - Left elbow: shoulder(12) → elbow(14) → wrist(16)

### 3. Postural Analysis Features

#### Trunk Inclination
```python
def calculate_trunk_angle(self, landmarks):
    """Calculate forward/backward lean of the trunk"""
    spine_vector = landmarks[23] - landmarks[11]  # Right hip to right shoulder
    vertical = np.array([0, 1, 0])
    return self._calculate_angle_with_vertical(spine_vector, vertical)
```

#### Body Symmetry
```python
def calculate_leg_symmetry(self, landmarks):
    """Calculate symmetry ratio between left and right legs"""
    left_leg_length = np.linalg.norm(landmarks[24] - landmarks[26])
    right_leg_length = np.linalg.norm(landmarks[23] - landmarks[25])
    return left_leg_length / right_leg_length
```

### 4. Motion Features

#### Velocity Calculations
The system calculates velocities using a sliding window approach:
```python
def _calculate_motion_features(self, landmark_history):
    velocities = []
    key_points = [23, 24, 25, 26]  # Hips and knees
    
    for i in range(len(landmark_history)-1):
        curr = landmark_history[i]
        next_frame = landmark_history[i+1]
        
        for kp in key_points:
            vel = np.linalg.norm(next_frame[kp] - curr[kp])
            velocities.append(vel)
```

### 5. Temporal Processing

#### Sliding Window Implementation
- Window size: 15 frames (0.5 seconds at 30 FPS)
- Features calculated per window:
  ```python
  window_features = [
      np.mean(window),
      np.std(window),
      np.max(window),
      np.min(window),
      np.median(window)
  ]
  ```

#### Signal Smoothing
```python
def _smooth_signal(self, signal_data, window_length=5):
    """Apply rolling average smoothing to reduce noise"""
    return pd.Series(signal_data).rolling(
        window=window_length, 
        center=True, 
        min_periods=1
    ).mean().values
```

## Feature Normalization and Scaling

### Standard Scaling
```python
def normalize_features(self, features):
    """Z-score normalization for numerical features"""
    scaler = StandardScaler()
    return scaler.fit_transform(features)
```

### Robust Scaling for Angles
- Angle features are normalized to [0, 180] range
- Special handling for angle wrapping at 360 degrees

## Quality Control Measures

### Data Validation
1. **Landmark Confidence**
   - Minimum detection confidence: 0.5
   - Minimum tracking confidence: 0.5

2. **Signal Quality**
   - Outlier detection for joint angles
   - Velocity thresholds for movement validation
   - Frame dropout detection

### Error Handling
```python
def validate_features(self, features):
    """Validate extracted features for quality assurance"""
    # Check for NaN values
    if np.isnan(features).any():
        return False
        
    # Validate angle ranges
    for angle in self.angle_features:
        if features[angle] < 0 or features[angle] > 180:
            return False
            
    return True
```

## Optimization Considerations

### Performance Optimizations
1. **Vectorized Operations**
   - Use of NumPy for efficient array operations
   - Optimized angle calculations using vector operations

2. **Memory Management**
   - Efficient use of deque for sliding windows
   - Proper cleanup of video resources
   - Streaming processing to handle long videos

### Real-time Processing
1. **Feature Selection**
   - Focus on most discriminative features
   - Efficient feature calculation ordering
   - Early stopping for invalid frames

2. **Buffer Management**
   - Circular buffer for landmark history
   - Efficient memory allocation for feature arrays
   - Optimized sliding window implementation