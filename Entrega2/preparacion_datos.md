## **Preparación de Datos**
El primer paso crítico para garantizar la calidad de la solución fue la recolección y preparación de los datos. Este proceso incluyó la extracción de características, limpieza, y almacenamiento eficiente.

### **Obtención de Datos**
1. **Fuente de datos:**
   - Videos que registran movimientos específicos. Estos fueron procesados con **MediaPipe**, una herramienta que permite extraer posiciones clave de articulaciones del cuerpo y calcular ángulos entre ellas.
2. **Transformación a formato tabular:**
   - Los datos procesados se convirtieron en un conjunto estructurado de características para representar cada movimiento en términos matemáticos y geométricos.
   - Cada muestra incluye variables como ángulos, velocidades y dinámicas temporales de las posturas.

### **Procesamiento de Datos**
1. **Limpieza:**
   - Eliminación de valores atípicos que podrían distorsionar el aprendizaje del modelo.
   - Validación para asegurar la consistencia entre las características y las etiquetas.
2. **Generación de características dinámicas:**
   - Uso de ventanas deslizantes para calcular métricas como la media, desviación estándar, y diferencias temporales, que capturan la evolución de los movimientos.
3. **Estandarización y normalización:**
   - Transformación de las características a una escala común usando técnicas como `StandardScaler`, asegurando un impacto uniforme de cada variable en el modelo.
4. **Etiquetado de datos:**
   - Las clases se mapearon a valores numéricos utilizando `LabelEncoder` para facilitar el procesamiento.
5. **Almacenamiento final:**
   - Los datos procesados se guardaron en un archivo comprimido `.npz`, que incluye:
     - `X`: Características normalizadas.
     - `y`: Etiquetas de las clases.
     - `classes`: Nombres de las clases objetivo.