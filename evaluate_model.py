import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configuración para mantener la terminal limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("Iniciando evaluación del modelo...")

# 1. Cargar el diccionario de clases exportado previamente
try:
    with open('mapeo_clases.json', 'r') as f:
        diccionario_clases = json.load(f)
        # Invertir el diccionario para tener {0: 'Bacterial Pneumonia', ...}
        nombres_clases = [diccionario_clases[str(i)] for i in range(len(diccionario_clases))]
except FileNotFoundError:
    print("Error: No se encontró 'mapeo_clases.json'.")
    exit()

# 2. Cargar el modelo entrenado
try:
    model = tf.keras.models.load_model('modelo_chest_disease.keras')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# 3. Configurar el generador estricto para datos de prueba
# REGLA DE ORO: shuffle=False es obligatorio para que las predicciones 
# coincidan en orden con las etiquetas reales de las carpetas.
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    'Testing Data',
    target_size=(224, 224),
    batch_size=10,       # Lote pequeño para no saturar memoria
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False        
)

# 4. Generar predicciones matemáticas
print("\nAnalizando imágenes de prueba. Por favor espera...")
predicciones_crudas = model.predict(test_generator)

# Extraer el índice con la probabilidad más alta para cada imagen
predicciones_clases = np.argmax(predicciones_crudas, axis=1)
etiquetas_reales = test_generator.classes

# 5. Calcular las métricas detalladas
print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN CLÍNICA")
print("="*50)
if len(etiquetas_reales) < 100:
    print("ADVERTENCIA: Este reporte se generó utilizando una cantidad muy pequeña")
    print("de imágenes de prueba (Testing Data). Las métricas pueden ser inestables.")
    print("Por favor, confía principalmente en la precisión de validación (val_accuracy)")
    print("reportada durante el entrenamiento.\n")
print(classification_report(etiquetas_reales, predicciones_clases, target_names=nombres_clases))

# 6. Construir y dibujar la Matriz de Confusión
matriz = confusion_matrix(etiquetas_reales, predicciones_clases)

plt.figure(figsize=(10, 8))
# sns.heatmap dibuja la matriz con una escala de colores térmica
sns.heatmap(
    matriz, 
    annot=True,         # Muestra los números exactos en cada celda
    fmt='d',            # Formato de números enteros
    cmap='Blues',       # Paleta de colores azules médicos
    xticklabels=nombres_clases, 
    yticklabels=nombres_clases
)

plt.title('Matriz de Confusión: Diagnóstico Automatizado', pad=20, fontsize=16)
plt.ylabel('Diagnóstico Real (Lo que es)', fontsize=12, fontweight='bold')
plt.xlabel('Diagnóstico de la IA (Lo que predijo)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar la imagen de la matriz en alta calidad y mostrarla en pantalla
plt.savefig('matriz_confusion.png', dpi=300)
print("\n¡Evaluación terminada! La matriz se ha guardado como 'matriz_confusion.png'.")
plt.show()