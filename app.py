import os
import json
import streamlit as st  
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="AI Chest Disease Detection",
    page_icon="🫁",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #fafafa; }
    .stAlert { border-radius: 10px; }
    h1 { color: #1e3d59; }
    h3 { color: #17b978; }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 Diagnóstico Automatizado de Enfermedades Torácicas")
st.write("Carga una radiografía de tórax para que la red neuronal analice patrones patológicos.")

# Carga dinámica del diccionario de clases exportado en el entrenamiento
@st.cache_data
def cargar_diccionario_clases():
    try:
        with open('mapeo_clases.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

CLASES_DICT = cargar_diccionario_clases()

# Carga del modelo usando el formato nativo de Keras
@st.cache_resource
def cargar_modelo():
    try:
        return tf.keras.models.load_model('modelo_chest_disease.keras')
    except Exception as e:
        return None

model = cargar_modelo()

def preprocesar_imagen(imagen_pil):
    """Convierte la imagen a RGB, redimensiona y aplica la normalización de MobileNetV2."""
    img = imagen_pil.convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Fundamental: Rango [-1, 1] idéntico al entrenamiento
    return preprocess_input(img_array)

# Verificaciones de seguridad antes de mostrar la interfaz
if CLASES_DICT is None:
    st.error("⚠️ Falla de configuración: No se encontró 'mapeo_clases.json'. Ejecuta el script de entrenamiento primero.")
elif model is None:
    st.error("⚠️ Falla de configuración: No se encontró 'modelo_chest_disease.keras'. Ejecuta el script de entrenamiento primero.")
else:
    # Obtener el número total de clases para iterar la interfaz
    TOTAL_CLASES = len(CLASES_DICT)
    
    # Interfaz de usuario
    uploaded_file = st.file_uploader(
        "Selecciona una radiografía en formato JPG, JPEG o PNG...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Visualización de la Placa")
            st.image(image, caption="Radiografía procesada", use_container_width=True)
            
        with col2:
            st.subheader("Análisis de Red Neuronal")
            
            with st.spinner("Ejecutando inferencia..."):
                tensor_imagen = preprocesar_imagen(image)
                predicciones = model.predict(tensor_imagen)[0]
                
                # Extracción del índice y traducción a texto usando el JSON
                indice_prediccion = np.argmax(predicciones)
                clase_detectada = CLASES_DICT[str(indice_prediccion)]
                confianza_detectada = predicciones[indice_prediccion] * 100
                    
            if clase_detectada.lower() == 'healthy' or clase_detectada.lower() == 'normal':
                st.success(f"### Resultado: {clase_detectada}")
                st.write(f"Confianza del análisis: **{confianza_detectada:.2f}%**")
            else:
                st.error(f"### Hallazgo: {clase_detectada}")
                st.write(f"Confianza del análisis: **{confianza_detectada:.2f}%**")
                st.warning("⚠️ *Aviso: Este sistema de IA no sustituye un diagnóstico médico profesional.*")
                
            st.write("---")
            st.write("**Distribución de confianza por patología:**")
                
            # Generar barras de progreso dinámicamente según el JSON
            for i in range(TOTAL_CLASES):
                nombre_clase = CLASES_DICT[str(i)]
                porcentaje = float(predicciones[i])
                porcentaje_barra = max(0.0, min(1.0, porcentaje))
                    
                st.write(f"{nombre_clase} ({porcentaje*100:.1f}%)")
                st.progress(porcentaje_barra)