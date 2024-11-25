## **Reporte Final**

### **Reducción de Características**
La selección y reducción de características desempeñaron un papel fundamental en el rendimiento y la eficiencia de los modelos entrenados:
1. **Técnicas utilizadas:**
   - Métodos de selección manual basados en la relevancia de características, priorizando aquellas que mejor representaban la dinámica de los movimientos.
   - Métodos algorítmicos como evaluación de importancia de características mediante modelos Random Forest.
2. **Resultados:**
   - Eliminación de características redundantes o no informativas, lo que resultó en una representación más compacta del problema, reduciendo la dimensionalidad de los datos sin sacrificar la precisión.

### **Evaluación de Resultados**
1. **Modelos entrenados:**
   - Los modelos Random Forest y XGBoost mostraron el mejor desempeño durante las fases de prueba.
2. **Métricas y resultados:**
   - Las métricas específicas y resultados detallados se encuentran en los archivos `.joblib` almacenados en el directorio de modelos entrenados (`modelos_entrenados`). Cada archivo incluye el modelo, sus parámetros óptimos y su desempeño medido por F1-score ponderado.
3. **Observaciones clave:**
   - Las clases con menor representación en el conjunto de datos requieren recolección adicional para mejorar el desempeño.
   - El tiempo de inferencia promedio satisface los requerimientos para aplicaciones en tiempo real.