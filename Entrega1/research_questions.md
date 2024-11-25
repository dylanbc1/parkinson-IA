# Research Questions and Objectives

## Main Research Questions

### 1. Human Action Recognition
- How can we develop a robust system that accurately identifies and classifies five basic human actions (walking towards the camera, walking away, turning, sitting down, standing up) in real-time?
- Which human movement characteristics are most relevant for distinguishing between these specific actions?
- How can we ensure the system maintains its accuracy under different capture conditions?

### 2. Postural and Biomechanical Analysis
- Which joint angles are most informative for characterizing each of the studied actions?
- How can we effectively quantify movement quality through metrics such as body symmetry and postural stability?
- What temporal patterns in joint movements are characteristic of each action?

### 3. Real-Time Processing
- How can we optimize the balance between model accuracy and processing speed to maintain a fluid real-time experience?
- Which preprocessing strategies are most effective for handling variability in capture conditions?

## Specific Objectives

### 1. Technical Development
- Implement a robust video capture system that allows consistent movement data collection
- Develop a processing pipeline that extracts relevant features from body landmarks
- Create a classification model capable of distinguishing between the five target actions with high accuracy
- Optimize the system for real-time operation while maintaining acceptable accuracy

### 2. Data Analysis
- Identify and validate the most discriminative features for each type of action
- Evaluate the relative importance of different feature types (joint angles, velocities, postural features)
- Analyze inter-subject variability in action execution

### 3. Practical Implementation
- Develop a user interface that allows clear visualization of predictions and postural metrics
- Establish a capture protocol that considers visual field limitations and required space
- Implement quality control mechanisms for data capture

## Working Hypotheses

1. Temporal movement characteristics (velocities and accelerations) are as important as static postural features for accurate action classification.

2. The combination of multiple feature types (joint angles, velocities, postural measurements) will significantly improve classification accuracy compared to using a single feature type.

3. Using an appropriate temporal window in feature processing will capture the temporal dynamics necessary to distinguish between similar actions.

## Scope and Limitations

### Project Scope
- Recognition of five specific actions
- Real-time analysis of joint movements
- Evaluation of basic postural features
- Real-time visualization of results

### Identified Limitations
- Specific space and visual field requirements for capture
- Variability in lighting and background conditions
- Real-time processing constraints
- Need for calibration across different subjects

## Relevance and Impact

This project has both academic and practical relevance:

1. **Academic**
   - Contributes to understanding human movement patterns
   - Explores real-time video processing techniques
   - Develops methodologies for biomechanical feature extraction

2. **Practical**
   - Lays groundwork for movement analysis applications
   - Demonstrates viability of real-time postural monitoring systems
   - Provides insights into technical requirements for similar systems

## Success Metrics

To evaluate project success, the following will be considered:

1. **System Accuracy**
   - Overall accuracy > 90% on test set
   - F1-Score > 0.85 for each class
   - Response time < 100ms per frame

2. **Robustness**
   - Consistency across different lighting conditions
   - Ability to handle variations in movement execution
   - Stability in landmark tracking

3. **Usability**
   - Clear and responsive interface
   - Real-time feedback
   - Ease of use for non-technical users