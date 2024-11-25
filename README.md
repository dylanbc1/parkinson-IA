# **Proyecto Parkinson utilizando IA - Reporte Final**

## **Resumen**
Este proyecto tiene como objetivo desarrollar una herramienta de software capaz de analizar actividades específicas como caminar, girar, sentarse y levantarse, rastrear los movimientos articulares y medir inclinaciones posturales. El sistema procesa datos de video, extrae las posiciones clave de las articulaciones mediante MediaPipe y calcula los ángulos articulares para detectar y clasificar actividades. Utilizando Python, OpenCV y MediaPipe, el proyecto se centra en mejorar las herramientas de diagnóstico de la enfermedad de Parkinson a través del monitoreo de actividades en tiempo real.

---

## **1. Introducción**

### **Contexto**
La enfermedad de Parkinson es un trastorno neurodegenerativo que afecta el movimiento. El monitoreo de los cambios posturales y movimientos articulares puede facilitar el diagnóstico temprano. Este proyecto busca desarrollar una solución basada en inteligencia artificial para monitorear y analizar estas actividades.

### **Descripción del Problema**
El sistema se centra en detectar cinco actividades clave (caminar hacia adelante, caminar hacia atrás, girar, sentarse y levantarse) mediante el análisis de movimientos articulares en videos en tiempo real capturados con una cámara. El sistema rastrea puntos clave en las articulaciones, analiza los ángulos entre estas, y clasifica las actividades detectadas.

### **Objetivo**
Construir una herramienta de IA que clasifique actividades físicas específicas y rastree los movimientos articulares relevantes, priorizando las articulaciones clave como caderas, rodillas y muñecas para el análisis relacionado con la enfermedad de Parkinson.

### **Aspectos Interesantes**
El desafío radica en procesar videos en tiempo real, calcular con precisión los ángulos articulares y garantizar una clasificación robusta bajo diferentes condiciones, como variaciones en las perspectivas y la iluminación.

---

## **2. Teoría**

### **Conceptos Clave**
- **MediaPipe**: Tecnología utilizada para detectar puntos clave (landmarks) de las articulaciones en cada fotograma del video.
- **Ángulos articulares**: Cálculo de ángulos entre vectores definidos por puntos clave, como la flexión de la rodilla.
- **Inclinación del tronco**: Ángulo entre los hombros y el eje horizontal, útil para analizar cambios posturales.
- **Metodología CRISP-DM**: El proyecto sigue esta metodología para estructurar las fases de recolección, procesamiento y clasificación de datos.

---

## **3. Metodología**

### **Recolección de Datos**
- **Captura de videos**: 
  - Se utiliza la clase `VideoDataCollector` para capturar videos de sujetos realizando actividades predefinidas.
  - Las actividades incluyen caminar, girar, sentarse y levantarse. Los videos se guardan con el formato: `sujeto_<id>_<actividad>_<timestamp>.mp4`.

### **Procesamiento de Datos**
- **Extracción de Landmarks**:
  - Los videos se procesan con la clase `SmartVideoProcessor`, que utiliza **MediaPipe** para detectar puntos clave de las articulaciones.
  - Cada fotograma se convierte a formato RGB antes de pasar por el modelo de detección de poses de MediaPipe.

- **Cálculo de Ángulos**:
  - Los ángulos articulares, como la flexión de la rodilla o la inclinación del tronco, se calculan mediante vectores definidos entre los puntos clave.
  - Ejemplos:
    - **Ángulo de rodilla**: Determinado usando las posiciones de la cadera, la rodilla y el tobillo.
    - **Inclinación del tronco**: Calculado midiendo el ángulo entre los hombros y el eje horizontal.

### **Modelos de Clasificación**
- **Extracción de Características**:
  - Los ángulos articulares y posturas detectadas se utilizan como entrada para los modelos de clasificación.
- **Modelos Utilizados**:
  - Se evaluaron varios clasificadores, incluidos SVM, Random Forest y XGBoost, para identificar la actividad basada en movimientos articulares.

---

## **4. Resultados**

### **Desempeño del Modelo**
- Los modelos entrenados, junto con sus métricas y secuencias, están almacenados en archivos `.joblib` en el directorio del proyecto. Estos archivos incluyen las configuraciones de los modelos y sus métricas principales (precisión, F1-score, entre otras).

### **Análisis Preliminar**
- Los resultados muestran que Random Forest tuvo un desempeño consistente, aunque se requieren datos adicionales para mejorar la precisión en clases menos representadas.

---

## **5. Análisis de Resultados**

### **Comportamiento del Modelo**
- Los modelos muestran un desempeño sólido en condiciones de prueba. Sin embargo, la capacidad de generalización puede mejorarse mediante un conjunto de datos más diverso y representativo.

### **Evaluación en Tiempo Real**
- El sistema clasifica actividades en tiempo real utilizando datos capturados con una cámara conectada al sistema.
- Las predicciones son consistentes para actividades simples, pero los movimientos complejos pueden requerir refinamiento adicional.

---

## **6. Validación y Evaluación**

### **Pruebas de Campo**
- El sistema se probó con múltiples sujetos realizando actividades predefinidas. Las predicciones se compararon manualmente con datos etiquetados como referencia.

### **Desafíos Técnicos**
- Las variaciones en iluminación y perspectiva del video representan un desafío para mantener la precisión del modelo.
- Los tiempos de inferencia actuales cumplen con los requerimientos de aplicaciones en tiempo real.

---

## **7. Conclusiones y Trabajo Futuro**

### **Resumen**
- Este proyecto desarrolló un sistema que extrae ángulos articulares y clasifica actividades físicas en tiempo real.
- MediaPipe demostró ser una herramienta efectiva para la estimación de poses, y el sistema tiene potencial para monitorear movimientos relevantes para el diagnóstico de Parkinson.

### **Futuro**
1. Incorporar más actividades para un análisis integral de movimientos.
2. Explorar modelos más avanzados para mejorar la precisión y robustez del sistema.
3. Optimizar el sistema para entornos variados, incluyendo condiciones de baja iluminación.

---

## **8. Consideraciones Éticas**

- **Privacidad y Consentimiento**:
  - Los datos de video son sensibles, por lo que se debe garantizar la privacidad de los sujetos.
  - Todo uso del sistema debe realizarse con el consentimiento informado de los participantes.
  
- **Imparcialidad**:
  - Es importante asegurar que el sistema no muestre sesgos en la detección de actividades debido a diferencias demográficas o físicas.

---

## **9. Referencias**
1. MediaPipe Documentation: https://ai.google.dev/edge/mediapipe/solutions/guide?hl=es-419  
2. LabelStudio: https://labelstud.io/  
3. Comparación de CVAT vs. LabelStudio: https://medium.com/cvat-ai/cvat-vs-labelstudio-which-one-is-better-b1a0d333842e  
