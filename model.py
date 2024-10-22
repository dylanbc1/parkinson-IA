import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

class SmartVideoProcessor:
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
        
        # Crear archivo de registro si no existe
        self.log_file = self.output_dir / "processing_log.json"
        self.processed_videos = self.load_processing_log()
        
        # Definir mapeo de actividades
        self.activity_mapping = {
            'caminar_hacia': 0,
            'caminar_regreso': 1,
            'girar': 2,
            'sentarse': 3,
            'levantarse': 4
        }
        
        # Guardar el mapeo de actividades
        with open(self.output_dir / 'activity_mapping.json', 'w') as f:
            json.dump(self.activity_mapping, f, indent=2)
        
        # Articulaciones clave
        self.key_joints = [
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'NOSE'
        ]
        
        # Imprimir información de configuración
        print("Actividades configuradas:")
        for activity, idx in self.activity_mapping.items():
            print(f"{idx}: {activity}")

    def load_processing_log(self):
        """Carga el registro de videos procesados."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}

    def update_processing_log(self, video_path, status="processed"):
        """Actualiza el registro de procesamiento."""
        self.processed_videos[video_path.name] = {
            "processed_date": datetime.now().isoformat(),
            "status": status
        }
        with open(self.log_file, 'w') as f:
            json.dump(self.processed_videos, f, indent=2)

    def needs_processing(self, video_path):
        """Verifica si un video necesita ser procesado."""
        if video_path.name in self.processed_videos:
            output_path = self.output_dir / f"{video_path.stem}_processed.json"
            if output_path.exists():
                return False
        return True

    def extract_landmarks(self, frame):
        """Extrae landmarks de un frame usando MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks_dict = {}
            for joint in self.key_joints:
                idx = getattr(self.mp_pose.PoseLandmark, joint)
                landmark = results.pose_landmarks.landmark[idx]
                # Guardar coordenadas en formato para el modelo
                landmarks_dict[f"{joint}_x"] = landmark.x
                landmarks_dict[f"{joint}_y"] = landmark.y
                landmarks_dict[f"{joint}_z"] = landmark.z
                landmarks_dict[f"{joint}_visibility"] = landmark.visibility
            return landmarks_dict
        return None

    def calculate_angles(self, landmarks_dict):
        """Calcula ángulos relevantes entre articulaciones."""
        angles = {}
        
        # Ángulo de rodillas
        for side in ['LEFT', 'RIGHT']:
            # Extraer coordenadas
            hip_x = landmarks_dict[f'{side}_HIP_x']
            hip_y = landmarks_dict[f'{side}_HIP_y']
            knee_x = landmarks_dict[f'{side}_KNEE_x']
            knee_y = landmarks_dict[f'{side}_KNEE_y']
            ankle_x = landmarks_dict[f'{side}_ANKLE_x']
            ankle_y = landmarks_dict[f'{side}_ANKLE_y']
            
            # Calcular vectores
            v1 = np.array([hip_x - knee_x, hip_y - knee_y])
            v2 = np.array([ankle_x - knee_x, ankle_y - knee_y])
            
            # Calcular ángulo
            angle = np.degrees(np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 
                -1.0, 1.0
            )))
            angles[f'{side}_KNEE_ANGLE'] = angle
        
        # Inclinación lateral del tronco
        left_shoulder_x = landmarks_dict['LEFT_SHOULDER_x']
        left_shoulder_y = landmarks_dict['LEFT_SHOULDER_y']
        right_shoulder_x = landmarks_dict['RIGHT_SHOULDER_x']
        right_shoulder_y = landmarks_dict['RIGHT_SHOULDER_y']
        
        shoulder_vector = np.array([right_shoulder_x - left_shoulder_x,
                                  right_shoulder_y - left_shoulder_y])
        horizontal = np.array([1, 0])
        
        trunk_angle = np.degrees(np.arccos(np.clip(
            np.dot(shoulder_vector, horizontal) / np.linalg.norm(shoulder_vector),
            -1.0, 1.0
        )))
        angles['TRUNK_TILT'] = trunk_angle
        
        return angles

    def identify_activity(self, filename):
        """Identifica la actividad desde el nombre del archivo."""
        parts = filename.split('_')
        if len(parts) < 2:
            return None
            
        # Buscar la actividad en el nombre del archivo
        for activity in self.activity_mapping.keys():
            if activity in filename:
                return activity
                
        return None

    def process_video(self, video_path):
        """Procesa un video y extrae landmarks y ángulos."""
        filename = video_path.stem
        
        # Identificar actividad
        activity = self.identify_activity(filename)
        if not activity:
            raise ValueError(f"No se pudo identificar la actividad en {filename}")
        
        activity_id = self.activity_mapping[activity]
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_data = {
            'metadata': {
                'filename': filename,
                'activity': activity,
                'activity_id': activity_id,
                'total_frames': frame_count
            },
            'frames': []
        }
        
        with tqdm(total=frame_count, desc=f"Procesando {filename}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.extract_landmarks(frame)
                if landmarks:
                    angles = self.calculate_angles(landmarks)
                    frame_data = {
                        'landmarks': landmarks,
                        'angles': angles
                    }
                    video_data['frames'].append(frame_data)
                
                pbar.update(1)
        
        cap.release()
        return video_data

    def process_all_videos(self):
        """Procesa solo los videos nuevos en el directorio de entrada."""
        video_files = list(self.input_dir.glob('*.mp4'))
        new_videos = [v for v in video_files if self.needs_processing(v)]
        
        print(f"Encontrados {len(video_files)} videos totales")
        print(f"Videos nuevos para procesar: {len(new_videos)}")
        
        if not new_videos:
            print("No hay nuevos videos para procesar")
            return
        
        for video_path in new_videos:
            print(f"\nProcesando {video_path.name}")
            try:
                # Procesar video
                video_data = self.process_video(video_path)
                
                # Guardar resultados
                output_path = self.output_dir / f"{video_path.stem}_processed.json"
                with open(output_path, 'w') as f:
                    json.dump(video_data, f, indent=2)
                
                # Actualizar registro
                self.update_processing_log(video_path)
                print(f"Datos guardados en {output_path}")
                
            except Exception as e:
                print(f"Error procesando {video_path.name}: {str(e)}")
                self.update_processing_log(video_path, status="error")

        print("\nProcesamiento completado.")
        print("Por favor, verifique que las actividades se hayan identificado correctamente en los archivos JSON generados.")

if __name__ == "__main__":
    processor = SmartVideoProcessor()
    processor.process_all_videos()