## **Entrenamiento de Modelos**
El siguiente paso consistió en entrenar modelos de clasificación supervisada para identificar movimientos con alta precisión.

### **División del Conjunto de Datos**
- **80%** de los datos se usaron para entrenamiento.
- **20%** se reservaron para pruebas.
- Se utilizó **estratificación** para garantizar una representación equilibrada de todas las clases.

### **Modelos Entrenados**
Se seleccionaron dos modelos principales debido a su desempeño comprobado en tareas similares:
1. **Random Forest:** 
   - Un enfoque basado en múltiples árboles de decisión, conocido por su robustez en datos tabulares y su resistencia al sobreajuste.
2. **XGBoost:** 
   - Un modelo avanzado que implementa gradiente boosting, ideal para capturar relaciones complejas en los datos.

### **Optimización de Hiperparámetros**
- Se utilizó **GridSearchCV** con validación cruzada (5 folds) para ajustar hiperparámetros clave:
  - **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`.
  - **XGBoost:** `n_estimators`, `max_depth`, `learning_rate`.
- La métrica objetivo fue el **F1-score ponderado**, dado que equilibra precisión y recall, especialmente importante para conjuntos de datos con clases desbalanceadas.

### **Evaluación de Modelos**
- Durante el entrenamiento, se midieron las siguientes métricas:
  - **F1-score**: Indicador principal del desempeño.
  - **Reporte de clasificación:** Incluye precisión, recall y F1 por clase.
  - **Matriz de confusión:** Analiza los errores específicos de cada clase.
- Se evaluó también la eficiencia:
  - **Tiempo de entrenamiento**: Duración necesaria para ajustar el modelo.
  - **Tiempo promedio de inferencia**: Crucial para asegurar que el sistema funcione en tiempo real.

### **Resultados del Entrenamiento**
El mejor modelo identificado se seleccionó en función de su F1-score y balance entre precisión y recall. Este modelo, junto con sus parámetros, fue guardado en formato `.joblib` para su posterior uso en la etapa de inferencia.