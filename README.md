# AI in HealthCare - Chest Disease Detection

Este proyecto implementa un modelo de inteligencia artificial basado en Deep Learning para la clasificación y detección de enfermedades torácicas (Neumonía Bacteriana, Neumonía Viral y COVID-19) o pulmones sanos a partir de radiografías de tórax (X-Rays). 

El modelo utiliza la arquitectura **MobileNetV2** pre-entrenada con ImageNet (Transfer Learning) y una aplicación web interactiva desarrollada con **Streamlit** para realizar inferencias en tiempo real.

## 🚀 Características Principales
- **Clasificación de 4 Clases:** `Bacterial Pneumonia`, `Healthy`, `Viral Pneumonia` y `COVID-19`.
- **Transfer Learning:** Uso de MobileNetV2 para eficiencia computacional y alta precisión.
- **Fine-Tuning de 2 Fases:** Descongelamiento de las capas finales para ajustarse a características de radiografías.
- **Data Augmentation:** Balanceo dinámico de datos de entrenamiento para prevenir sobreajuste.
- **Interfaz Web (Streamlit):** Una aplicación fácil de usar que permite subir radiografías y obtener una predicción con su porcentaje de confianza.

## 📋 Estructura del Proyecto

El repositorio tiene la siguiente estructura de archivos principal:

```
├── app.py                      # Aplicación web principal en Streamlit
├── train_model.py              # Script principal para entrenar el modelo
├── evaluate_model.py           # Script para evaluar el modelo y generar la matriz de confusión
├── procesar_dataset.py         # Script para limpieza, redimensionamiento y preprocesamiento
├── reorganizar_dataset.py      # Script para estructurar carpetas de entrenamiento/prueba
├── requirements.txt            # Dependencias de Python necesarias para ejecutar el proyecto
├── mapeo_clases.json           # Diccionario con el mapeo de clases e índices
├── documentacion_proyecto.md   # Documentación extensa del proceso y optimización del modelo
├── Notebook para presentacion.ipynb # Notebook interactivo (útil para la presentación o experimentación)
└── modelo_chest_disease.keras  # Modelo final pre-entrenado (¡listo para usar en app.py!)
```

*(Nota: Las carpetas de datos `Training Data/` y `Testing Data/` no se incluyen en el repositorio para no sobrecargar el almacenamiento, pero los scripts para procesarlos sí están disponibles).*

## ⚙️ Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto localmente.

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/nombre-del-repo.git
cd nombre-del-repo
```

### 2. Crear un entorno virtual (Recomendado)
Es altamente recomendado crear un entorno virtual para aislar las dependencias.
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 💻 Uso

Para arrancar la interfaz web y empezar a probar el modelo con tus propias radiografías, ejecuta:

```bash
streamlit run app.py
```
Esto abrirá automáticamente una pestaña en tu navegador web donde podrás interactuar con el sistema de predicción.

## 🧠 Entrenamiento y Evaluación (Opcional)

Si deseas volver a entrenar el modelo o evaluar su rendimiento desde cero (asegúrate de tener las imágenes en las carpetas `Training Data/` y `Testing Data/` locales):

1. **Preprocesar los datos:** `python procesar_dataset.py`
2. **Entrenar:** `python train_model.py`
3. **Evaluar:** `python evaluate_model.py` (esto actualizará el archivo de la matriz de confusión).

## 📈 Resultados

El modelo actual alcanza aproximadamente un **87% de precisión global**, con picos destacables de **100% de precisión y 87% de recall para pacientes sanos** (garantizando 0 falsos positivos para sanos) y **100% de precisión para COVID-19**. Para ver un desglose completo de métricas y la metodología de Data Augmentation, consulta [documentacion_proyecto.md](documentacion_proyecto.md).

## 📝 Licencia / Autoría
Desarrollado como proyecto para el curso de **Introducción a la IA**.
