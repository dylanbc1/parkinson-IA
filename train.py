import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from pathlib import Path
import json
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PostureModelTrainer:
    def __init__(self, data_dir="dataset_final", output_dir="models"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cargar datos
        self.load_data()
        
        # Definir modelos y sus hiperparámetros
        self.models = {
            'svm': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            }
        }
        
        self.best_models = {}
    
    def load_data(self):
        """Carga los datos de entrenamiento y prueba."""
        print("Cargando datos...")
        
        self.train_data = pd.read_csv(self.data_dir / 'train_data.csv')
        self.test_data = pd.read_csv(self.data_dir / 'test_data.csv')
        
        # Separar características y etiquetas
        self.X_train = self.train_data.drop('activity_encoded', axis=1)
        self.y_train = self.train_data['activity_encoded']
        self.X_test = self.test_data.drop('activity_encoded', axis=1)
        self.y_test = self.test_data['activity_encoded']
        
        # Cargar mapeo de actividades
        with open(self.data_dir / 'activity_mapping.json', 'r') as f:
            self.activity_mapping = json.load(f)
        
        print(f"Datos cargados - Características de entrenamiento: {self.X_train.shape}")
    
    def train_and_evaluate(self):
        """Entrena y evalúa todos los modelos."""
        results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\nEntrenando {model_name}...")
            
            # Búsqueda de hiperparámetros con validación cruzada
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Guardar el mejor modelo
            self.best_models[model_name] = grid_search.best_estimator_
            
            # Evaluar en conjunto de prueba
            y_pred = grid_search.predict(self.X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"Mejores parámetros para {model_name}: {grid_search.best_params_}")
            print(f"F1-Score: {f1:.4f}")
        
        return results
    
    def plot_results(self, results):
        """Genera visualizaciones de los resultados."""
        # Comparación de métricas
        metrics_df = pd.DataFrame({
            model: {
                'Accuracy': results[model]['accuracy'],
                'Precision': results[model]['precision'],
                'Recall': results[model]['recall'],
                'F1-Score': results[model]['f1']
            }
            for model in results.keys()
        }).T
        
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar', ylim=(0, 1))
        plt.title('Comparación de Métricas por Modelo')
        plt.xlabel('Modelo')
        plt.ylabel('Puntuación')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png')
        
        # Matrices de confusión
        for model_name, model_results in results.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                model_results['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[v for k, v in self.activity_mapping.items()],
                yticklabels=[v for k, v in self.activity_mapping.items()]
            )
            plt.title(f'Matriz de Confusión - {model_name}')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'confusion_matrix_{model_name}.png')
    
    def save_best_model(self, results):
        """Guarda el mejor modelo basado en F1-Score."""
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        best_model = self.best_models[best_model_name]
        
        # Guardar modelo
        joblib.dump(best_model, self.output_dir / f'best_model.pkl')
        
        # Guardar información del modelo
        model_info = {
            'model_type': best_model_name,
            'parameters': results[best_model_name]['best_params'],
            'metrics': {
                'accuracy': results[best_model_name]['accuracy'],
                'precision': results[best_model_name]['precision'],
                'recall': results[best_model_name]['recall'],
                'f1': results[best_model_name]['f1']
            },
            'feature_names': list(self.X_train.columns),
            'activity_mapping': self.activity_mapping,
            'training_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\nMejor modelo ({best_model_name}) guardado en {self.output_dir}")
        return best_model_name, best_model
    
    def train(self):
        """Ejecuta el pipeline completo de entrenamiento."""
        # Entrenar y evaluar modelos
        results = self.train_and_evaluate()
        
        # Generar visualizaciones
        self.plot_results(results)
        
        # Guardar mejor modelo
        best_model_name, best_model = self.save_best_model(results)
        
        return best_model_name, best_model, results

if __name__ == "__main__":
    trainer = PostureModelTrainer()
    best_model_name, best_model, results = trainer.train()
    
    # Mostrar reporte final
    print("\nReporte Final:")
    print(f"Mejor modelo: {best_model_name}")
    print(f"Métricas del mejor modelo:")
    for metric, value in results[best_model_name].items():
        if metric not in ['confusion_matrix', 'best_params']:
            print(f"{metric}: {value:.4f}")