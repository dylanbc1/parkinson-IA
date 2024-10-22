# Parkinson project using AI 

## Abstract
This project aims to develop a software tool capable of analyzing specific activities such as walking, turning, sitting, and standing, and tracking joint movements and postural inclinations. The system processes video data, extracts key joint positions using MediaPipe, and calculates joint angles to detect and classify activities. The project uses Python, OpenCV, and MediaPipe for real-time analysis, with a focus on improving diagnostic tools for Parkinson's disease through activity monitoring.

## 1. Introduction
- **Context**: Parkinson's disease is a neurodegenerative disorder affecting movement. Monitoring joint movements and postural changes can help in early diagnosis. This project aims to develop an AI-based solution to monitor and analyze such activities.
- **Problem Description**: The project focuses on detecting five key activities (walking forward, walking back, turning, sitting, standing) using joint movement analysis. The system uses real-time video data captured via a webcam, tracks key joint landmarks, and classifies activities.
- **Objective**: To build an AI tool that classifies activities and tracks joint movements, specifically targeting key joints like hips, knees, and wrists for Parkinson’s analysis.
- **Interesting Aspects**: The challenge lies in processing real-time video, calculating precise joint angles, and achieving robust classification under varying conditions (e.g., different perspectives, lighting).

## 2. Theory
- **Key Concepts**: 
  - **MediaPipe** is used to detect joint landmarks in video frames.
  - **Joint Angles**: Using vectors between key points (e.g., hip, knee, and ankle), angles such as knee flexion are computed.
  - **Trunk Tilt**: The angle between the shoulders and the horizontal axis helps analyze postural inclinations.
  - **CRISP-DM Methodology**: The project follows this model, focusing on data collection, processing, and classification.

## 3. Methodology
- **Data Collection**:
   - Videos are collected using the `VideoDataCollector` class, which captures videos of subjects performing predefined activities.
   - Key activities include walking, turning, sitting, and standing. Each video is saved in the format: `sujeto_<id>_<activity>_<timestamp>.mp4`.
  
- **Data Processing**:
   - The **SmartVideoProcessor** class processes raw video data and extracts joint positions using **MediaPipe**.
   - **Preprocessing**: Frames from the videos are converted to RGB and passed to MediaPipe for pose detection.
   - **Landmark Extraction**: Key joints like hips, knees, and wrists are tracked using MediaPipe’s pose estimation.
   - **Angle Calculation**: Joint angles (e.g., knee angle, trunk tilt) are calculated using vector mathematics between joints.
  
- **Joint Tracking**:
   - For each frame in the video, landmarks are extracted, and angles are computed.
   - Example:
     - **Knee Angle**: Using the positions of the hip, knee, and ankle.
     - **Trunk Tilt**: Calculated by measuring the angle between the shoulders.

- **Classification Model**:
   - The extracted features (joint angles and postures) will be used for activity classification.
   - **Model Choice**: Various classifiers like SVM, Random Forest, and XGBoost are considered for classifying activities based on joint movement.

## 4. Results
- **Model Performance**:
   - _Results will be added after training the classification models._
   - Key metrics: Accuracy, Precision, Recall, F1-Score.

## 5. Results Analysis
- **Model Behavior**:
   - _To be added: Analysis of model performance and generalization capabilities._

## 6. Validation & Evaluation
- **Real-Time Processing**:
   - The system uses a webcam for real-time video processing and displays classified activities.
   - Joint angles are computed and displayed in real time.

- **Performance Testing**:
   - Tests will be conducted on various subjects performing the activities. The system's predictions will be compared against manually labeled ground truth data.

## 7. Conclusions and Future Work
- **Summary**:
   - We have built a system capable of extracting joint angles and classifying physical activities in real time.
   - MediaPipe is effectively used for pose estimation, and the system shows promise for tracking joint movements relevant to Parkinson's disease.
  
- **Future Directions**:
   - Improve the robustness of the system for different environments.
   - Add more activities for comprehensive movement analysis.
   - Explore more sophisticated models for higher accuracy and real-time performance optimization.

## 8. Ethical Considerations
- The system deals with potentially sensitive data (video of individuals), so privacy and consent must be ensured.
- Ethical considerations include ensuring no bias in activity detection and addressing privacy concerns during data collection.

## 9. References
1. MediaPipe Documentation: https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419
2. LabelStudio: https://labelstud.io/
3. CVAT vs. LabelStudio comparison: https://medium.com/cvat-ai/cvat-vs-labelstudio-which-one-is-better-b1a0d333842e
