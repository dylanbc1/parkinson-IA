import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import mediapipe as mp
from datetime import datetime

class ModelValidator:
    def __init__(self, models_dir="models", data_dir="dataset_final"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Cargar modelo y metadata
        self.load_model_and_metadata()
        
        # Cargar datos de prueba
        self.load_test_data()
        
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Definir los landmarks clave
        self.key_joints = [
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'NOSE'
        ]
        
    def load_model_and_metadata(self):
        """Carga el modelo entrenado y su información asociada."""
        self.model = joblib.load(self.models_dir / 'best_model.pkl')
        
        with open(self.models_dir / 'model_info.json', 'r') as f:
            self.model_info = json.load(f)
            
        self.activity_mapping_inv = {int(v): k for k, v in self.model_info['activity_mapping'].items()}
        
        print("Actividades que el modelo puede clasificar:")
        for idx, activity in self.activity_mapping_inv.items():
            print(f"{idx}: {activity}")
            
    def load_test_data(self):
        """Carga los datos de prueba."""
        self.test_data = pd.read_csv(self.data_dir / 'test_data.csv')
        self.X_test = self.test_data.drop('activity_encoded', axis=1)
        self.y_test = self.test_data['activity_encoded']
        
        # Guardar los nombres de las características en el orden correcto
        self.feature_columns = self.X_test.columns
        
    def extract_features(self, results):
        """Extrae las características en el mismo formato que los datos de entrenamiento."""
        if not results.pose_landmarks:
            return None
            
        landmarks_dict = {}
        
        # Extraer solo los landmarks clave
        for joint in self.key_joints:
            idx = getattr(self.mp_pose.PoseLandmark, joint)
            landmark = results.pose_landmarks.landmark[idx]
            landmarks_dict[f"{joint}_x"] = landmark.x
            landmarks_dict[f"{joint}_y"] = landmark.y
            landmarks_dict[f"{joint}_z"] = landmark.z
            landmarks_dict[f"{joint}_visibility"] = landmark.visibility
            
        # Calcular ángulos
        # (Aquí puedes agregar el cálculo de ángulos si los usaste en el entrenamiento)
        
        # Crear DataFrame con las características en el orden correcto
        features = pd.DataFrame([landmarks_dict])
        
        # Asegurar que tenemos todas las columnas necesarias en el orden correcto
        features = features.reindex(columns=self.feature_columns, fill_value=0)
        
        return features
        
    def analyze_predictions(self):
        """Analiza las predicciones del modelo en el conjunto de prueba."""
        predictions = self.model.predict(self.X_test)
        
        results_df = pd.DataFrame({
            'Real': [self.activity_mapping_inv[y] for y in self.y_test],
            'Predicción': [self.activity_mapping_inv[y] for y in predictions]
        })
        
        print("\nDistribución de predicciones:")
        pred_distribution = results_df['Predicción'].value_counts()
        for activity, count in pred_distribution.items():
            print(f"{activity}: {count} predicciones ({count/len(predictions)*100:.1f}%)")
        
        results_df['Correcto'] = results_df['Real'] == results_df['Predicción']
        
        print("\nAnálisis de predicciones incorrectas:")
        incorrect_predictions = results_df[~results_df['Correcto']]
        if len(incorrect_predictions) > 0:
            print("\nPredicciones incorrectas:")
            print(incorrect_predictions)
        else:
            print("No se encontraron predicciones incorrectas en el conjunto de prueba")
        
        plt.figure(figsize=(10, 8))
        conf_matrix = confusion_matrix(results_df['Real'], results_df['Predicción'])
        sns.heatmap(conf_matrix, 
                    annot=True, 
                    fmt='d',
                    xticklabels=sorted(self.activity_mapping_inv.values()),
                    yticklabels=sorted(self.activity_mapping_inv.values()))
        plt.title('Matriz de Confusión Detallada')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(self.models_dir / 'detailed_confusion_matrix.png')
        
        return results_df
    
    def test_real_time(self, duration=30):
        """Prueba el modelo en tiempo real usando la cámara."""
        print("\nIniciando prueba en tiempo real...")
        print("Presiona 'q' para salir")
        
        cap = cv2.VideoCapture(0)
        start_time = datetime.now()
        predictions_buffer = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar frame con MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extraer características usando la misma estructura que en entrenamiento
                features = self.extract_features(results)
                
                if features is not None:
                    # Hacer predicción
                    prediction = self.model.predict(features)[0]
                    activity = self.activity_mapping_inv[prediction]
                    
                    # Agregar a buffer de predicciones
                    predictions_buffer.append(activity)
                    if len(predictions_buffer) > 10:
                        predictions_buffer.pop(0)
                    
                    # Mostrar predicción más común en el buffer
                    if predictions_buffer:
                        current_prediction = max(set(predictions_buffer), key=predictions_buffer.count)
                        cv2.putText(frame, f"Actividad: {current_prediction}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Dibujar landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('Prueba en Tiempo Real', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if (datetime.now() - start_time).seconds >= duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    validator = ModelValidator()
    
    print("\n=== Análisis del Modelo ===")
    results_df = validator.analyze_predictions()
    
    print("\n=== Prueba en Tiempo Real ===")
    response = input("¿Desea realizar una prueba en tiempo real? (s/n): ")
    if response.lower() == 's':
        validator.test_real_time()

if __name__ == "__main__":
    main()