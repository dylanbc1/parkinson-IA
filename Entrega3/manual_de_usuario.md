## **Manual de Usuario**

### **Requisitos del Sistema**
1. **Hardware:**
   - CPU de al menos 4 núcleos.
   - Opcional: GPU para acelerar el procesamiento en aplicaciones con alta demanda de tiempo real.
2. **Software:**
   - Python 3.9 o superior.
   - Dependencias listadas en `requirements.txt`.

### **Instrucciones de Uso**
1. **Entrenamiento de modelos:**
   - Proporcione los datos procesados en formato `.npz`.
   - Ejecute el script `train.py` para entrenar y evaluar los modelos.
   - Los modelos entrenados se guardarán automáticamente en el directorio `modelos_entrenados`.
2. **Inferencia en tiempo real:**
   - Ejecute el script `real_time_inference.py`, que utiliza una cámara conectada al sistema.
   - Asegúrese de que los modelos entrenados estén disponibles en el directorio configurado.
   - Los movimientos detectados se mostrarán en la salida estándar o se integrarán a la interfaz deseada.
3. **Modificaciones adicionales:**
   - Los parámetros de entrenamiento y los modelos pueden ajustarse modificando los archivos de configuración y los scripts correspondientes.