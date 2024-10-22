import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path

class VideoProcessor:
    def __init__(self, input_dir="dataset_raw", output_dir="dataset_processed"):
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Articulaciones clave mencionadas en el proyecto
        self.key_joints = [
            'LEFT_HIP', 'RIGHT_HIP',           # Caderas
            'LEFT_KNEE', 'RIGHT_KNEE',         # Rodillas
            'LEFT_ANKLE', 'RIGHT_ANKLE',       # Tobillos
            'LEFT_WRIST', 'RIGHT_WRIST',       # Muñecas
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', # Hombros
            'NOSE'                             # Cabeza
        ]
        
    def extract_landmarks(self, frame):
        """Extrae landmarks de un frame usando MediaPipe."""
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Crear diccionario de landmarks
            landmarks_dict = {}
            for joint in self.key_joints:
                idx = getattr(self.mp_pose.PoseLandmark, joint)
                landmark = results.pose_landmarks.landmark[idx]
                landmarks_dict[joint] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            return landmarks_dict
        return None

    def calculate_angles(self, landmarks_dict):
        """Calcula ángulos relevantes entre articulaciones."""
        angles = {}
        
        # Ángulo de rodillas
        for side in ['LEFT', 'RIGHT']:
            hip = landmarks_dict[f'{side}_HIP']
            knee = landmarks_dict[f'{side}_KNEE']
            ankle = landmarks_dict[f'{side}_ANKLE']
            
            # Calcular vectores
            v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
            v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
            
            # Calcular ángulo
            angle = np.degrees(np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 
                -1.0, 1.0
            )))
            angles[f'{side}_KNEE_ANGLE'] = angle
        
        # Calcular inclinación lateral del tronco
        left_shoulder = np.array([landmarks_dict['LEFT_SHOULDER']['x'],
                                landmarks_dict['LEFT_SHOULDER']['y']])
        right_shoulder = np.array([landmarks_dict['RIGHT_SHOULDER']['x'],
                                 landmarks_dict['RIGHT_SHOULDER']['y']])
        
        shoulder_vector = right_shoulder - left_shoulder
        horizontal = np.array([1, 0])
        
        trunk_angle = np.degrees(np.arccos(np.clip(
            np.dot(shoulder_vector, horizontal) / np.linalg.norm(shoulder_vector),
            -1.0, 1.0
        )))
        angles['TRUNK_TILT'] = trunk_angle
        
        return angles

    def process_video(self, video_path):
        """Procesa un video y extrae landmarks y ángulos."""
        # Extraer información del nombre del archivo
        filename = video_path.stem
        subject_id, activity, timestamp = filename.split('_', 2)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Preparar datos para este video
        video_data = {
            'metadata': {
                'subject_id': subject_id,
                'activity': activity,
                'timestamp': timestamp,
                'total_frames': frame_count
            },
            'frames': []
        }
        
        # Procesar cada frame
        with tqdm(total=frame_count, desc=f"Procesando {filename}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extraer landmarks
                landmarks = self.extract_landmarks(frame)
                if landmarks:
                    # Calcular ángulos
                    angles = self.calculate_angles(landmarks)
                    
                    # Guardar datos del frame
                    frame_data = {
                        'landmarks': landmarks,
                        'angles': angles
                    }
                    video_data['frames'].append(frame_data)
                
                pbar.update(1)
        
        cap.release()
        return video_data

    def process_all_videos(self):
        """Procesa todos los videos en el directorio de entrada."""
        video_files = list(self.input_dir.glob('*.mp4'))
        print(f"Encontrados {len(video_files)} videos para procesar")
        
        for video_path in video_files:
            print(f"\nProcesando {video_path.name}")
            
            # Procesar video
            video_data = self.process_video(video_path)
            
            # Guardar resultados
            output_path = self.output_dir / f"{video_path.stem}_processed.json"
            with open(output_path, 'w') as f:
                json.dump(video_data, f, indent=2)
            
            print(f"Datos guardados en {output_path}")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_all_videos()