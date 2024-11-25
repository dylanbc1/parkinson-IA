import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time
import logging
from pathlib import Path
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeActionRecognizer:
    def __init__(self, model_path, preprocessor_path, window_size=15):
        self.window_size = window_size
        self.setup_model(model_path)
        self.setup_mediapipe()
        self.setup_preprocessor(preprocessor_path)
        self.frame_buffer = deque(maxlen=window_size)
        self.prediction_buffer = deque(maxlen=5)
        self.last_prediction = None
        self.confidence_threshold = 0.6
        self.landmark_buffer = deque(maxlen=window_size)


    def setup_model(self, model_path):
        logger.info("Cargando modelo...")
        model_info = joblib.load(model_path)
        self.model = model_info['model']
        self.classes = model_info.get('classes', None)
        logger.info(f"Modelo cargado exitosamente. Clases: {self.classes}")
        
    def setup_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
    def setup_preprocessor(self, preprocessor_path):
        logger.info("Cargando preprocesador...")
        config = joblib.load(preprocessor_path)
        self.scaler = config['scaler']
        self.selected_features = config['selected_features']
        logger.info("Preprocesador cargado exitosamente")

    def _extract_normalized_landmarks(self, pose_landmarks, frame_shape):
        """Extrae y normaliza landmarks."""
        landmarks = {}
        h, w = frame_shape[:2]
        
        # Mapear índices a nombres significativos
        landmark_mapping = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
        }
        
        for name, idx in landmark_mapping.items():
            landmark = pose_landmarks.landmark[idx]
            # Convertir a coordenadas de píxeles y normalizar
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z * w
            
            # Normalizar a [-1, 1]
            x = (x - w/2) / (w/2)
            y = (y - h/2) / (h/2)
            z = z / (w/2)
            
            landmarks[name] = np.array([x, y, z])
        
        return landmarks


    def _calculate_features(self, landmarks):
        """Calcula todas las características necesarias."""
        features = {}
        
        # 1. Ángulos de articulaciones
        joint_angles = self._calculate_joint_angles(landmarks)
        features.update(joint_angles)
        
        # 2. Características posturales
        posture_features = self._calculate_posture_features(landmarks)
        features.update(posture_features)
        
        return features

    def _calculate_joint_angles(self, landmarks):
        """Calcula ángulos entre articulaciones."""
        angles = {}
        
        # Definir tripletes de puntos para calcular ángulos
        angle_configs = {
            'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
            'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
            'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
            'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
            'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist')
        }
        
        for angle_name, (p1_name, p2_name, p3_name) in angle_configs.items():
            try:
                p1 = landmarks[p1_name]
                p2 = landmarks[p2_name]
                p3 = landmarks[p3_name]
                
                angles[angle_name] = self._calculate_angle(p1, p2, p3)
            except KeyError as e:
                logger.warning(f"No se pudo calcular {angle_name}: {str(e)}")
                angles[angle_name] = 0
        
        return angles


    def _calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

    def _calculate_posture_features(self, landmarks):
        """Calcula características posturales."""
        features = {}
        
        try:
            # Inclinación del tronco
            spine_vector = landmarks['right_hip'] - landmarks['right_shoulder']
            vertical = np.array([0, 1, 0])
            features['trunk_angle'] = self._calculate_angle_with_vertical(spine_vector, vertical)
            
            # Simetría de las piernas
            left_leg_length = np.linalg.norm(landmarks['left_hip'] - landmarks['left_knee'])
            right_leg_length = np.linalg.norm(landmarks['right_hip'] - landmarks['right_knee'])
            features['leg_symmetry'] = left_leg_length / (right_leg_length + 1e-6)  # Evitar división por cero
            
            # Altura relativa
            head_height = landmarks['nose'][1]
            ankle_height = (landmarks['left_ankle'][1] + landmarks['right_ankle'][1]) / 2
            features['relative_height'] = head_height - ankle_height
            
            # Ancho de la postura
            stance_width = np.linalg.norm(landmarks['left_ankle'] - landmarks['right_ankle'])
            features['stance_width'] = stance_width
            
        except KeyError as e:
            logger.warning(f"Error calculando características posturales: {str(e)}")
            features.update({
                'trunk_angle': 0,
                'leg_symmetry': 1,
                'relative_height': 0,
                'stance_width': 0
            })
        
        return features

    def _calculate_velocity_features(self, current_landmarks):
        """Calcula características de velocidad."""
        features = {}
        
        if len(self.frame_buffer) > 0:
            last_features = self.frame_buffer[-1]
            
            # Calcular velocidades para características existentes
            for feature_name in last_features:
                if isinstance(last_features[feature_name], (int, float)):
                    try:
                        velocity = float(last_features[feature_name]) - float(current_landmarks[feature_name])
                        features[f'{feature_name}_velocity'] = velocity
                    except (KeyError, TypeError):
                        continue
        
        return features

    def _calculate_angle_with_vertical(self, vector, vertical):
        """Calcula el ángulo entre un vector y la vertical."""
        cos_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle
    
    def predict(self, features):
        """Realiza predicción usando el modelo entrenado."""
        self.frame_buffer.append(features)
        
        if len(self.frame_buffer) < self.window_size:
            return "Esperando...", 0.0
        
        try:
            X = self._prepare_features_for_prediction()
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = np.max(probabilities)
            
            if confidence >= self.confidence_threshold:
                self.prediction_buffer.append(prediction)
                if self.prediction_buffer:
                    prediction = max(set(self.prediction_buffer), 
                                  key=list(self.prediction_buffer).count)
                    self.last_prediction = prediction
            else:
                prediction = self.last_prediction if self.last_prediction else "Baja confianza"
            
            if self.classes is not None and isinstance(prediction, (int, np.integer)):
                prediction = self.classes[prediction]
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return "Error", 0.0
    
    def process_frame(self, frame):
        """Procesa un frame y extrae características."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = self._extract_normalized_landmarks(results.pose_landmarks, frame.shape)
            features = self._calculate_features(landmarks)
            return features, results.pose_landmarks
        
        return None, None
    
    def _prepare_features_for_prediction(self):
        """Prepara características para predicción."""
        try:
            # Convertir buffer a DataFrame
            df = pd.DataFrame(list(self.frame_buffer))
            
            # Verificar que tenemos todas las características necesarias
            missing_features = set(self.selected_features) - set(df.columns)
            if missing_features:
                logger.warning(f"Características faltantes: {missing_features}")
                # Añadir características faltantes con valores 0
                for feature in missing_features:
                    df[feature] = 0
            
            # Seleccionar y ordenar características
            features = []
            for feature in self.selected_features:
                window = df[feature].values
                
                # Características estadísticas
                features.extend([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window),
                    np.median(window)
                ])
                
                # Características de cambio
                features.extend([
                    np.mean(np.diff(window)) if len(window) > 1 else 0,
                    np.mean(np.diff(np.diff(window))) if len(window) > 2 else 0
                ])
            
            X = np.array(features).reshape(1, -1)
            return self.scaler.transform(X)
            
        except Exception as e:
            logger.error(f"Error preparando características: {str(e)}")
            raise
    
    def run(self):
        """Ejecuta el reconocimiento en tiempo real."""
        cap = cv2.VideoCapture(0)
        logger.info(f"FPS de la cámara: {cap.get(cv2.CAP_PROP_FPS)}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                start_time = time.time()
                features, landmarks = self.process_frame(frame)
                
                if features is not None:
                    # Realizar predicción
                    prediction, confidence = self.predict(features)
                    
                    # Calcular FPS
                    process_time = time.time() - start_time
                    current_fps = 1.0 / process_time if process_time > 0 else 0
                    
                    # Dibujar resultados
                    self._draw_results(frame, landmarks, prediction, confidence, current_fps)
                else:
                    # Si no se detectaron landmarks
                    cv2.putText(frame, "No se detecta persona", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Mostrar frame
                cv2.imshow('Action Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            logger.error(f"Error en ejecución: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_results(self, frame, landmarks, prediction, confidence, fps):
        """Dibuja resultados en el frame."""
        # Dibujar pose
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Dibujar predicción y FPS
        cv2.putText(frame, f'Accion: {prediction}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confianza: {confidence:.2f}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    model_path = "modelos_entrenados/best_model.joblib"
    preprocessor_path = "datos_procesados/preprocessor.joblib"
    
    try:
        recognizer = RealTimeActionRecognizer(model_path, preprocessor_path)
        recognizer.run()
    except Exception as e:
        logger.error(f"Error en el programa principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()