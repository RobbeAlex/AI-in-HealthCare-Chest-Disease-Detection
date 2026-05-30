# Documentación del Proyecto de Detección de Enfermedades Torácicas

## 1. Dataset Elegido

**Fuentes de Datos:** Se realizó una fusión de dos conjuntos de datos de Kaggle para robustecer el modelo:
1. *COVID-19 Radiography Database* (Tawsifur Rahman)
2. *Chest X-Ray Images - Pneumonia* (Paul Mooney)

**Descripción del Conjunto:** Se utilizó esta combinación masiva enfocada en la clasificación de patologías torácicas a través de radiografías de tórax (X-Rays), garantizando mayor variabilidad clínica. 

**Clases Extraídas y Mapeo:**
A partir de la combinación de ambos datasets, se categorizaron las imágenes en cuatro clases finales (según consta en `mapeo_clases.json`):
1. `Bacterial Pneumonia` (Neumonía Bacteriana)
2. `Healthy` (Sanos/Normal)
3. `Viral Pneumonia` (Neumonía Viral)
4. `covid-19` (COVID-19)

**Preprocesamiento y Auditoría de Imágenes (`procesar_dataset.py`, `limpiar_dataset.py`):**
- Las imágenes fueron auditadas automáticamente para detectar archivos corruptos, y se eliminaron archivos de caché del sistema operativo (como `thumbs.db`).
- Se forzó una conversión al espacio de color `RGB` para garantizar la compatibilidad (ya que algunas radiografías podían estar en escala de grises original).
- Se redimensionaron al tamaño objetivo de `224x224` píxeles empleando interpolación LANCZOS (preparándolas para la arquitectura de la red).
- Se guardaron en formato `JPEG` con una compresión de alta calidad al 95%.

---

## 2. Documentación del Proceso de Optimización del Modelo

### 2.1 Ajustes de Arquitectura (Transfer Learning)
- **Modelo Base:** Se implementó una arquitectura **MobileNetV2** pre-entrenada con ImageNet. Se eligió debido a su bajo costo computacional y alta eficiencia, descartando su cabeza clasificadora original (`include_top=False`).
- **Arquitectura de Salida Personalizada:**
  - Capa `GlobalAveragePooling2D` para aplanar los mapas de características sin perder información espacial de forma abrupta.
  - Capa densa oculta de `128` neuronas con función de activación `ReLU`.
  - Capa de regularización `Dropout (0.5)` (50%) implementada de forma agresiva para forzar a la red a no depender de patrones singulares y prevenir el sobreajuste (overfitting).
  - Capa de salida con `4` neuronas y función de activación probabilística `softmax`.

### 2.2 Balanceo y Aumento de Datos (Data Augmentation)
- **Corrección de Desbalanceo de Clases:** Para corregir el sesgo provocado por tener cantidades desiguales de radiografías entre las distintas patologías, se aplicó un cálculo automático de pesos de clase (`compute_class_weight='balanced'`). Esto asignó multiplicadores en la función de costo para darle mayor importancia a los errores en las clases menos representadas.
- **Data Augmentation:** Durante el entrenamiento, las imágenes de origen se transformaron dinámicamente (`ImageDataGenerator`) para robustecer el modelo, incluyendo:
  - Rotaciones aleatorias de hasta 10 grados (`rotation_range=10`).
  - Acercamientos aleatorios (`zoom_range=0.1`).
  - Desplazamientos horizontales y verticales (`width_shift_range=0.05`, `height_shift_range=0.05`).
  - Alteración lumínica entre un 90% y 110% (`brightness_range=[0.9, 1.1]`).

### 2.3 Estrategia de Entrenamiento (Two-Phase Fine-Tuning)
El proceso de entrenamiento se ejecutó utilizando una técnica de ajuste por etapas para evitar que los pesos preentrenados convergieran erróneamente:

1. **Fase 1: Entrenamiento del Clasificador Superior (Feature Extraction)**
   - La base extractora de MobileNetV2 fue **congelada** (`trainable = False`).
   - **Optimizador:** Adam con una tasa de aprendizaje estándar de `0.001`.
   - **Épocas:** 5.
   - **Mejoras / Callbacks:** Se configuró un `EarlyStopping` con paciencia de 2 épocas monitorizando la pérdida en validación (`val_loss`) para detenerse prematuramente si no había mejora.

2. **Fase 2: Ajuste Fino Avanzado (Fine-Tuning)**
   - Se **descongelaron** específicamente las últimas 20 capas del modelo base.
   - **Optimizador:** Adam con una tasa de aprendizaje microscópica de `1e-5`. Esto permitió realizar ajustes sutiles y orientados al dominio de radiografías sin causar "olvido catastrófico" de los bordes y patrones base.
   - **Épocas:** 15.
   - **Mejoras / Callbacks:** Se empleó `EarlyStopping` (paciencia de 4) combinado con un `ModelCheckpoint` configurado con `save_best_only=True` enfocado en `val_accuracy`, logrando capturar y guardar únicamente la versión matemáticamente superior del modelo (`modelo_chest_disease.keras`).

### 2.4 Evaluación y Resultados Obtenidos
- **Generador Estricto de Prueba:** El modelo fue auditado bajo un conjunto de datos `Testing Data` con el factor de barajado desactivado (`shuffle=False`), garantizando una asignación correcta de etiquetas. Para evitar sesgos de "Out-Of-Distribution", se realizó un barajado aleatorio cruzado entre entrenamiento y prueba garantizando la misma consistencia visual en todas las imágenes.
- **Reporte de Clasificación (Resultados Finales):** Se extrajeron las métricas analíticas utilizando `classification_report`. El modelo alcanzó un **87% de precisión global**, logrando hitos clínicos críticos:
  - **100% de precisión y 87% de recall** para pulmones sanos (`Healthy`), garantizando cero falsos positivos para sanos.
  - **100% de precisión y 93% de recall** para `covid-19`.
  - **93% de recall** para `Bacterial Pneumonia`.
- **Matriz de Confusión:** Los resultados clínicos del ensayo se graficaron a través de una matriz de confusión utilizando `seaborn` con una paleta de colores térmicos azules, facilitando la visualización precisa de falsos positivos y falsos negativos por patología.
- **Integración Web:** El modelo final optimizado se empaquetó con Streamlit (`app.py`), implementando una barra de confianza porcentual y advertencias predictivas para inferencias menores al 60% de certeza.
