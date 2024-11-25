import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import time
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionClassifierTrainer:
    def __init__(self, data_path, models_dir="modelos_entrenados"):
        """
        Inicializa el entrenador de clasificadores.
        
        Args:
            data_path: Ruta a los datos procesados
            models_dir: Directorio donde guardar los modelos
        """
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def load_data(self):
        """Carga los datos procesados."""
        logger.info("Cargando datos procesados...")
        data = np.load(self.data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        # Asegurarse de que X e y tengan el mismo número de muestras
        min_samples = min(len(X), len(y))
        self.X = X[:min_samples]
        self.y = y[:min_samples]
        
        self.classes = data['classes']
        
        logger.info(f"Dimensiones de X: {self.X.shape}")
        logger.info(f"Dimensiones de y: {self.y.shape}")
        logger.info(f"Clases únicas: {np.unique(self.y)}")
        
        # Validar que tenemos el mismo número de muestras
        assert len(self.X) == len(self.y), "Las dimensiones de X e y no coinciden"
        
        # Split de datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        logger.info(f"Datos de entrenamiento - X: {self.X_train.shape}, y: {self.y_train.shape}")
        logger.info(f"Datos de prueba - X: {self.X_test.shape}, y: {self.y_test.shape}")
        
    def train_models(self):
        """Entrena y evalúa múltiples modelos."""
        models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
            }
        }
        
        for name, config in models.items():
            logger.info(f"\nEntrenando {name}...")
            try:
                self._train_and_evaluate_model(name, config['model'], config['params'])
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
        
        if self.best_model is not None:
            logger.info(f"\nMejor modelo: {self.best_model_name} con score: {self.best_score:.4f}")
        else:
            logger.warning("No se pudo entrenar ningún modelo correctamente")
        
    def _train_and_evaluate_model(self, name, model, param_grid):
        """Entrena y evalúa un modelo específico."""
        # Grid Search con validación cruzada
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        # Evaluar modelo
        y_pred = grid_search.predict(self.X_test)
        
        # Medir tiempo de inferencia
        start_time = time.time()
        for _ in range(100):  # Promedio de 100 predicciones
            grid_search.predict(self.X_test[:1])
        inference_time = (time.time() - start_time) / 100
        
        # Calcular y mostrar métricas detalladas
        self._print_detailed_metrics(name, grid_search, y_pred, train_time, inference_time)
        
        # Guardar modelo
        self._save_model(name, grid_search, train_time, inference_time)
        
        # Actualizar mejor modelo si corresponde
        if grid_search.best_score_ > self.best_score:
            self.best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            self.best_model_name = name
    
    def _print_detailed_metrics(self, name, grid_search, y_pred, train_time, inference_time):
        """Imprime métricas detalladas del modelo."""
        logger.info(f"\nResultados detallados para {name}:")
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        logger.info(f"F1-Score: {grid_search.best_score_:.4f}")
        logger.info(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
        logger.info(f"Tiempo promedio de inferencia: {inference_time*1000:.2f} ms")
        
        logger.info("\nMatriz de confusión:")
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        logger.info(f"\n{conf_matrix}")
        
        logger.info("\nReporte de clasificación:")
        logger.info(classification_report(self.y_test, y_pred, target_names=self.classes))
    
    def _save_model(self, name, grid_search, train_time, inference_time):
        """Guarda el modelo y sus métricas."""
        model_info = {
            'model': grid_search.best_estimator_,
            'params': grid_search.best_params_,
            'score': grid_search.best_score_,
            'train_time': train_time,
            'inference_time': inference_time,
            'classes': self.classes
        }
        
        model_path = self.models_dir / f"{name}_model.joblib"
        joblib.dump(model_info, model_path)
        logger.info(f"Modelo guardado en: {model_path}")
    
    def save_best_model(self):
        """Guarda el mejor modelo y metadatos importantes."""
        if self.best_model is not None:
            model_info = {
                'model': self.best_model,
                'classes': self.classes,
                'performance': self.best_score
            }
            
            best_model_path = self.models_dir / "best_model.joblib"
            joblib.dump(model_info, best_model_path)
            logger.info(f"\nMejor modelo guardado en: {best_model_path}")
        else:
            logger.warning("No hay mejor modelo para guardar")

def train_and_evaluate():
    """Función principal para entrenar y evaluar modelos."""
    data_path = "datos_procesados/data_processed.npz"
    models_dir = "modelos_entrenados"
    
    trainer = ActionClassifierTrainer(data_path, models_dir)
    
    try:
        trainer.load_data()
        trainer.train_models()
        trainer.save_best_model()
    except Exception as e:
        logger.error(f"Error en el proceso de entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_evaluate()