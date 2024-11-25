import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionPreprocessor:
    def __init__(self, window_size=15):
        """
        Inicializa el preprocesador.
        
        Args:
            window_size (int): Tamaño de la ventana para features secuenciales
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        
    def preprocess_dataset(self, input_path, output_path):
        """
        Procesa el dataset completo y guarda los resultados.
        
        Args:
            input_path (str): Ruta al CSV con los datos crudos
            output_path (str): Ruta donde guardar los datos procesados
        """
        logger.info(f"Iniciando preprocesamiento del dataset: {input_path}")
        
        # Cargar datos
        df = pd.read_csv(input_path)
        logger.info(f"Dataset cargado: {df.shape[0]} muestras")
        
        # Seleccionar y preparar features
        X_processed, y_encoded = self._prepare_features(df)
        
        # Guardar datos procesados
        self._save_processed_data(X_processed, y_encoded, output_path)
        
        logger.info("Preprocesamiento completado")
        return X_processed, y_encoded
    
    def _prepare_features(self, df):
        """
        Prepara las características para el entrenamiento.
        """
        # 1. Seleccionar features relevantes
        angle_features = [col for col in df.columns if 'angle' in col]
        velocity_features = [col for col in df.columns if 'velocity' in col]
        posture_features = ['trunk_angle', 'leg_symmetry', 'body_height', 'stance_width']
        
        self.selected_features = angle_features + velocity_features + posture_features
        
        # 2. Aplicar suavizado y calcular características temporales
        X = self._process_temporal_features(df)
        
        # 3. Codificar etiquetas
        y = self.label_encoder.fit_transform(df['action'])
        
        logger.info(f"Features seleccionados: {len(self.selected_features)}")
        logger.info(f"Clases encontradas: {self.label_encoder.classes_}")
        
        return X, y
    
    def _process_temporal_features(self, df):
        """
        Procesa características temporales usando ventanas deslizantes.
        """
        # 1. Suavizar señales
        smoothed_data = {}
        for feature in self.selected_features:
            smoothed_data[feature] = self._smooth_signal(df[feature].values)
        
        # 2. Crear ventanas temporales
        windows = []
        for i in range(len(df) - self.window_size + 1):
            window_features = []
            
            for feature in self.selected_features:
                # Ventana de valores
                window = smoothed_data[feature][i:i + self.window_size]
                
                # Características estadísticas de la ventana
                window_features.extend([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window),
                    np.median(window)
                ])
                
                # Características de cambio
                if i > 0:
                    delta = np.mean(np.diff(window))
                    delta2 = np.mean(np.diff(np.diff(window)))
                else:
                    delta = delta2 = 0
                    
                window_features.extend([delta, delta2])
            
            windows.append(window_features)
        
        # 3. Convertir a array y normalizar
        X = np.array(windows)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def _smooth_signal(self, signal_data, window_length=5):
        """
        Aplica suavizado a una señal para reducir ruido.
        """
        return pd.Series(signal_data).rolling(
            window=window_length, 
            center=True, 
            min_periods=1
        ).mean().values
    
    def _save_processed_data(self, X, y, output_path):
        """
        Guarda los datos procesados y los parámetros del preprocesador.
        """
        # Crear directorio si no existe
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar datos procesados
        np.savez(output_path,
                 X=X,
                 y=y,
                 feature_names=self.selected_features,
                 classes=self.label_encoder.classes_)
        
        # Guardar parámetros del preprocesador
        preprocessor_path = output_dir / 'preprocessor.joblib'
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'selected_features': self.selected_features,
            'window_size': self.window_size
        }, preprocessor_path)
        
        logger.info(f"Datos procesados guardados en: {output_path}")
        logger.info(f"Preprocesador guardado en: {preprocessor_path}")
        
    def load_preprocessor(self, path):
        """
        Carga un preprocesador guardado previamente.
        """
        config = joblib.load(path)
        self.scaler = config['scaler']
        self.label_encoder = config['label_encoder']
        self.selected_features = config['selected_features']
        self.window_size = config['window_size']
        
        return self

def analyze_dataset(processed_data_path):
    """
    Analiza el dataset procesado y muestra estadísticas útiles.
    """
    data = np.load(processed_data_path, allow_pickle=True)  # Habilitar pickle
    X, y = data['X'], data['y']
    feature_names = data['feature_names']
    classes = data['classes']
    
    logger.info("\nEstadísticas del Dataset:")
    logger.info(f"Dimensiones de X: {X.shape}")
    logger.info(f"Número de clases: {len(classes)}")
    logger.info("\nDistribución de clases:")
    for i, class_name in enumerate(classes):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100
        logger.info(f"{class_name}: {count} muestras ({percentage:.2f}%)")
    
    logger.info("\nEstadísticas de features:")
    for i, feature in enumerate(feature_names):
        logger.info(f"\n{feature}:")
        logger.info(f"Media: {X[:, i].mean():.3f}")
        logger.info(f"Std: {X[:, i].std():.3f}")
        logger.info(f"Min: {X[:, i].min():.3f}")
        logger.info(f"Max: {X[:, i].max():.3f}")


if __name__ == "__main__":
    # Rutas de archivos
    input_path = "dataset_acciones_detallado.csv"  # Tu archivo CSV original
    output_path = "datos_procesados/data_processed.npz"
    
    # Crear y ejecutar preprocesador
    preprocessor = ActionPreprocessor(window_size=15)
    X, y = preprocessor.preprocess_dataset(input_path, output_path)
    
    # Analizar resultados
    analyze_dataset(output_path)