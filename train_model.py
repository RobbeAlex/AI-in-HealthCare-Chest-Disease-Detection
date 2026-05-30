import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Configuración de variables de entorno para limpieza de terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("TensorFlow versión:", tf.__version__)

print("\nCargando y configurando los generadores de imágenes...")

# Generador de entrenamiento con Data Augmentation leve y preprocesamiento nativo
# Se añade validation_split=0.2 para reservar el 20% de los datos reales de entrenamiento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05],
    validation_split=0.2
)

# Generador de validación estricto (sin augmentation, pero con el mismo split)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Carga de datos
# Carga de datos de entrenamiento (80% del Training Data)
train_generator = train_datagen.flow_from_directory(
    'Training Data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='training',
    seed=42
)

# Carga de datos de validación (20% del Training Data)
validation_generator = validation_datagen.flow_from_directory(
    'Training Data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='validation',
    shuffle=False,
    seed=42
)

# Exportar el mapeo exacto de clases a un JSON para la aplicación web
indices_clases = {v: k for k, v in train_generator.class_indices.items()}
with open('mapeo_clases.json', 'w') as f:
    json.dump(indices_clases, f)
print(f"\nMapeo de clases exportado: {indices_clases}")

# Cálculo de pesos para evitar colapso de clase (atajos matemáticos)
print("\nCalculando pesos de clase para balancear el set de datos...")
pesos_clases_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
pesos_clases_dict = dict(enumerate(pesos_clases_array))

# Estrategia de pesos:
# Utilizaremos los pesos balanceados calculados automáticamente para evitar
# sesgos extremos hacia una sola clase.
for idx, peso in pesos_clases_dict.items():
    nombre = indices_clases[idx]
    print(f" - Clase '{nombre}': Multiplicador ajustado a {peso:.2f}x")
print("\nConstruyendo la arquitectura de red neuronal...")

# Importar MobileNetV2 sin su cabeza original
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# FASE 1: Congelar la base extractora
base_model.trainable = False

# Ensamblar la arquitectura
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.4),
    layers.Dense(4, activation='softmax')
])

print("\n=== FASE 1: Entrenando el clasificador superior (5 épocas) ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping_fase1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    class_weight=pesos_clases_dict,
    callbacks=[early_stopping_fase1]
)

print("\n=== FASE 2: Ajuste Fino / Fine-Tuning (15 épocas) ===")
# Descongelar las últimas 20 capas del modelo base
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# REGLA DE ORO: Las capas BatchNormalization DEBEN permanecer congeladas en Transfer Learning
# Si se descongelan, sus estadísticas móviles se destruyen y el modelo falla en las predicciones.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Tasa de aprendizaje microscópica para modificaciones sutiles
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping_fase2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    class_weight=pesos_clases_dict,
    callbacks=[early_stopping_fase2]
)

# Guardar en formato moderno
nombre_modelo = 'modelo_chest_disease.keras'
model.save(nombre_modelo)

print(f"\n¡Entrenamiento completado! Modelo guardado como '{nombre_modelo}'")