## **Plan de Despliegue**
El objetivo es implementar el sistema en un entorno de producción que permita realizar inferencias en tiempo real.

### **Infraestructura Propuesta**
1. **Script de Inferencia (`RealTimeInference.py`):**
   - Procesa frames de video capturados en tiempo real.
   - Utiliza MediaPipe para extraer características relevantes y el modelo entrenado para clasificar movimientos.
2. **Hardware recomendado:**
   - GPU para acelerar el procesamiento en aplicaciones con altas demandas de tiempo real.
3. **Formato del modelo:**
   - El modelo entrenado se almacena en formato `.joblib`, que permite carga eficiente y compatibilidad con el entorno de inferencia.
4. **Pruebas finales:**
   - Verificar la precisión y tiempos de respuesta en condiciones reales de operación.