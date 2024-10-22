import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PostureDataPreprocessor:
    def __init__(self, input_dir="dataset_processed", output_dir="dataset_final"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scaler = None
    
    def load_json_files(self):
        """Carga todos los archivos JSON y los convierte en un DataFrame."""
        print("Cargando archivos JSON...")
        
        all_data = []
        activities = set()
        
        for json_file in self.input_dir.glob("*_processed.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extraer metadata
                metadata = data['metadata']
                activity = metadata['activity']
                activities.add(activity)
                
                # Procesar cada frame
                for frame_idx, frame_data in enumerate(data['frames']):
                    row_data = {
                        'filename': metadata['filename'],
                        'activity': activity,
                        'frame_idx': frame_idx,
                    }
                    
                    # Agregar landmarks
                    for joint, coords in frame_data['landmarks'].items():
                        row_data[f"{joint}"] = coords
                    
                    # Agregar ángulos
                    for angle_name, value in frame_data['angles'].items():
                        row_data[angle_name] = value
                    
                    all_data.append(row_data)
        
        df = pd.DataFrame(all_data)
        print(f"\nActividades encontradas: {sorted(activities)}")
        print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} características")
        return df
    
    def extract_features(self, df):
        """Extrae características adicionales de los datos."""
        print("\nExtrayendo características adicionales...")
        
        # Velocidades relativas entre articulaciones clave
        joint_pairs = [
            ('LEFT_KNEE', 'LEFT_HIP'),
            ('RIGHT_KNEE', 'RIGHT_HIP'),
            ('LEFT_ANKLE', 'LEFT_KNEE'),
            ('RIGHT_ANKLE', 'RIGHT_KNEE')
        ]
        
        for joint1, joint2 in joint_pairs:
            df[f'velocity_{joint1}_{joint2}'] = df.groupby('filename').apply(
                lambda x: np.sqrt(
                    (x[f'{joint1}_x'].diff()**2) + 
                    (x[f'{joint1}_y'].diff()**2)
                )
            ).reset_index(level=0, drop=True)
        
        # Distancias entre articulaciones
        for joint1, joint2 in joint_pairs:
            df[f'distance_{joint1}_{joint2}'] = np.sqrt(
                (df[f'{joint1}_x'] - df[f'{joint2}_x'])**2 +
                (df[f'{joint1}_y'] - df[f'{joint2}_y'])**2
            )
        
        return df
    
    def clean_data(self, df):
        """Limpia los datos eliminando valores atípicos y NaN."""
        print("\nLimpiando datos...")
        
        # Eliminar filas con valores NaN
        initial_rows = df.shape[0]
        df = df.dropna()
        print(f"Filas eliminadas por NaN: {initial_rows - df.shape[0]}")
        
        # Eliminar valores atípicos usando IQR para ángulos
        angle_cols = ['LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE', 'TRUNK_TILT']
        for col in angle_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        
        print(f"Dataset limpio: {df.shape[0]} muestras")
        return df
    
    def normalize_data(self, df):
        """Normaliza las características numéricas."""
        print("\nNormalizando datos...")
        
        # Identificar columnas numéricas, excluyendo identificadores y etiquetas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols 
                       if col not in ['frame_idx'] 
                       and not col.endswith('_visibility')]
        
        # Normalizar usando StandardScaler
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def prepare_for_training(self, df):
        """Prepara los datos para entrenamiento."""
        print("\nPreparando datos para entrenamiento...")
        
        # Codificar la variable objetivo (actividad)
        le = LabelEncoder()
        df['activity_encoded'] = le.fit_transform(df['activity'])
        
        # Convertir el mapeo de actividades a tipos nativos de Python
        activity_mapping = {
            str(activity): int(code) 
            for activity, code in zip(le.classes_, le.transform(le.classes_))
        }
        
        # Guardar el mapeo
        with open(self.output_dir / 'activity_mapping.json', 'w') as f:
            json.dump(activity_mapping, f, indent=2)
        
        print("\nMapeo de actividades:")
        for activity, code in activity_mapping.items():
            print(f"{code}: {activity}")
        
        # Seleccionar características para entrenamiento
        feature_cols = [col for col in df.columns 
                       if col not in ['activity', 'activity_encoded', 'filename', 
                                    'frame_idx']]
        
        X = df[feature_cols]
        y = df['activity_encoded']
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convertir a tipos nativos de Python antes de guardar
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(self.output_dir / 'train_data.csv', index=False)
        test_data.to_csv(self.output_dir / 'test_data.csv', index=False)
        
        # Guardar el scaler
        import joblib
        joblib.dump(self.scaler, self.output_dir / 'scaler.pkl')
        
        print(f"\nDatos guardados en {self.output_dir}")
        print(f"Conjunto de entrenamiento: {X_train.shape}")
        print(f"Conjunto de prueba: {X_test.shape}")
        
        # Guardar lista de características
        feature_list = list(X_train.columns)
        with open(self.output_dir / 'feature_list.json', 'w') as f:
            json.dump(feature_list, f, indent=2)
        
        return X_train, X_test, y_train, y_test
    
    def process(self):
        """Ejecuta todo el pipeline de preprocesamiento."""
        # Cargar datos
        df = self.load_json_files()
        
        # Extraer características
        df = self.extract_features(df)
        
        # Limpiar datos
        df = self.clean_data(df)
        
        # Normalizar datos
        df = self.normalize_data(df)
        
        # Preparar para entrenamiento
        return self.prepare_for_training(df)

if __name__ == "__main__":
    preprocessor = PostureDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.process()