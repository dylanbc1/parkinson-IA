## **Documentación Técnica**

### **Arquitectura del Proyecto**
1. **Estructura del código:**
   - `train.py`: Script principal para el entrenamiento y evaluación de modelos.
   - `preprocess.py`: Módulo de procesamiento y transformación de datos crudos.
   - `real_time_inference.py`: Script para realizar inferencia en tiempo real.
2. **Almacenamiento de modelos:**
   - Los modelos entrenados y sus metadatos se guardan en formato `.joblib`, permitiendo su reutilización eficiente.
3. **Formato de datos:**
   - Los datos procesados se almacenan en formato `.npz`, con tres componentes principales: `X` (características), `y` (etiquetas), y `classes` (nombres de las clases objetivo).

### **Procesos Automatizados**
1. **Entrenamiento:**
   - Se automatizó el ajuste de hiperparámetros utilizando `GridSearchCV`.
   - Los mejores modelos se seleccionan y guardan automáticamente.
2. **Inferencia:**
   - El script de inferencia convierte los frames de video en características procesables y realiza predicciones utilizando los modelos entrenados.
3. **Evaluación:**
   - Métricas como F1-score, matriz de confusión y tiempos de inferencia se registran para análisis y comparación.

### **Próximos Pasos Técnicos**
1. Implementar técnicas avanzadas de selección de características para mejorar la precisión y reducir tiempos de inferencia.
2. Expandir el conjunto de datos para incluir movimientos adicionales y mejorar el balance de clases.
3. Desarrollar una interfaz de usuario amigable para facilitar el uso del sistema por personas no técnicas.