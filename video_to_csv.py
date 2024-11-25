import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import math
import re

class ActionVideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Aumentamos la complejidad para mejor precisión
        )
        
    def process_video_file(self, video_path):
        """
        Procesa un archivo de video y extrae características.
        
        Args:
            video_path (Path): Ruta al archivo de video
        Returns:
            list: Lista de diccionarios con características por frame
        """
        # Extraer información del nombre del archivo
        filename = video_path.stem
        parts = filename.split('_')
        subject_id = parts[1]
        action = parts[2]  # caminar_hacia, caminar_regreso, girar, etc.
        
        print(f"Procesando {filename}...")
        frames_data = []
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        prev_landmarks = None
        window_size = 5  # Ventana temporal para características
        landmark_history = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = self._normalize_landmarks(results.pose_landmarks, frame.shape)
                landmark_history.append(landmarks)
                
                # Mantener solo los últimos N frames para la ventana temporal
                if len(landmark_history) > window_size:
                    landmark_history.pop(0)
                
                if len(landmark_history) == window_size:
                    features = self._extract_features(
                        landmarks, 
                        landmark_history,
                        frame_count,
                        subject_id,
                        action
                    )
                    frames_data.append(features)
                
                prev_landmarks = landmarks
                frame_count += 1
            
        cap.release()
        return frames_data
    
    def _normalize_landmarks(self, pose_landmarks, frame_shape):
        """Normaliza las coordenadas de los landmarks."""
        landmarks = []
        h, w = frame_shape[:2]
        
        for landmark in pose_landmarks.landmark:
            # Convertir a coordenadas de píxeles y normalizar
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z * w  # Usar ancho para la profundidad también
            
            # Normalizar a un rango [-1, 1]
            x = (x - w/2) / (w/2)
            y = (y - h/2) / (h/2)
            z = z / (w/2)
            
            landmarks.append([x, y, z])
            
        return np.array(landmarks)
    
    def _extract_features(self, current_landmarks, landmark_history, frame_count, subject_id, action):
        """
        Extrae características completas incluyendo información temporal.
        """
        # Características base
        features = {
            'frame': frame_count,
            'subject_id': subject_id,
            'action': action,
        }
        
        # 1. Ángulos de articulaciones clave
        joint_angles = self._calculate_joint_angles(current_landmarks)
        features.update(joint_angles)
        
        # 2. Velocidades y aceleraciones usando la ventana temporal
        motion_features = self._calculate_motion_features(landmark_history)
        features.update(motion_features)
        
        # 3. Características posturales
        posture_features = self._calculate_posture_features(current_landmarks)
        features.update(posture_features)
        
        # 4. Características de forma global
        shape_features = self._calculate_shape_features(current_landmarks)
        features.update(shape_features)
        
        return features
    
    def _calculate_joint_angles(self, landmarks):
        """Calcula ángulos entre articulaciones clave."""
        angles = {}
        
        # Definir articulaciones para calcular ángulos
        joint_sets = {
            'right_knee': [23, 25, 27],  # cadera-rodilla-tobillo derecho
            'left_knee': [24, 26, 28],   # cadera-rodilla-tobillo izquierdo
            'right_hip': [11, 23, 25],   # hombro-cadera-rodilla derecho
            'left_hip': [12, 24, 26],    # hombro-cadera-rodilla izquierdo
            'right_elbow': [11, 13, 15], # hombro-codo-muñeca derecho
            'left_elbow': [12, 14, 16],  # hombro-codo-muñeca izquierdo
        }
        
        for name, (p1, p2, p3) in joint_sets.items():
            angle = self._calculate_angle(
                landmarks[p1],
                landmarks[p2],
                landmarks[p3]
            )
            angles[f'{name}_angle'] = angle
            
        return angles
    
    def _calculate_motion_features(self, landmark_history):
        """Calcula características de movimiento usando la ventana temporal."""
        features = {}
        
        if len(landmark_history) < 2:
            return {'mean_velocity': 0, 'max_velocity': 0}
            
        # Calcular velocidades entre frames consecutivos
        velocities = []
        for i in range(len(landmark_history)-1):
            curr = landmark_history[i]
            next_frame = landmark_history[i+1]
            
            # Usar puntos clave específicos para velocidad
            key_points = [23, 24, 25, 26]  # caderas y rodillas
            for kp in key_points:
                vel = np.linalg.norm(next_frame[kp] - curr[kp])
                velocities.append(vel)
        
        features['mean_velocity'] = np.mean(velocities)
        features['max_velocity'] = np.max(velocities)
        
        return features
    
    def _calculate_posture_features(self, landmarks):
        """Calcula características posturales."""
        features = {}
        
        # Inclinación del tronco
        spine_vector = landmarks[23] - landmarks[11]  # cadera derecha a hombro derecho
        vertical = np.array([0, 1, 0])
        
        trunk_angle = self._calculate_angle_with_vertical(spine_vector, vertical)
        features['trunk_angle'] = trunk_angle
        
        # Simetría del cuerpo
        left_leg_length = np.linalg.norm(landmarks[24] - landmarks[26])  # cadera-rodilla izq
        right_leg_length = np.linalg.norm(landmarks[23] - landmarks[25]) # cadera-rodilla der
        features['leg_symmetry'] = left_leg_length / right_leg_length
        
        return features
    
    def _calculate_shape_features(self, landmarks):
        """Calcula características de la forma global del cuerpo."""
        features = {}
        
        # Altura total (distancia vertical entre tobillos y cabeza)
        height = landmarks[0][1] - np.mean([landmarks[27][1], landmarks[28][1]])
        features['body_height'] = height
        
        # Ancho de la postura (distancia entre tobillos)
        width = np.linalg.norm(landmarks[27] - landmarks[28])
        features['stance_width'] = width
        
        return features
    
    def _calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_angle_with_vertical(self, vector, vertical):
        """Calcula el ángulo entre un vector y la vertical."""
        cos_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle

def process_videos_directory(directory_path, output_file):
    """
    Procesa todos los videos en el directorio y genera un dataset.
    
    Args:
        directory_path (str): Ruta al directorio con los videos
        output_file (str): Ruta donde guardar el dataset CSV
    """
    processor = ActionVideoProcessor()
    all_data = []
    directory = Path(directory_path)
    
    # Procesar cada video en el directorio
    for video_file in directory.glob('*.mp4'):
        try:
            frame_data = processor.process_video_file(video_file)
            all_data.extend(frame_data)
        except Exception as e:
            print(f"Error procesando {video_file}: {str(e)}")
            continue
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(all_data)
    
    # Agregar timestamp de procesamiento
    from datetime import datetime
    df['processing_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar dataset
    df.to_csv(output_file, index=False)
    print(f"\nDataset guardado en {output_file}")
    print(f"Total de frames procesados: {len(df)}")
    print("\nResumen de acciones:")
    print(df['action'].value_counts())
    
    # Mostrar algunas estadísticas básicas
    print("\nEstadísticas de características principales:")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_columns].describe())

# Ejemplo de uso
if __name__ == "__main__":
    # Ajusta estas rutas según tu estructura de archivos
    process_videos_directory(
        "dataset_raw/",
        "dataset_acciones_detallado.csv"
    )