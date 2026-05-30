import os
import shutil
import random

# Rutas base
train_dir = r"G:\Mi unidad\Drive alumno.udg.mx\4to Semestre\Introduccion a la IA\AI in HealthCare - Chest Disease Detection With Teachable Machines\Training Data"
test_dir = r"G:\Mi unidad\Drive alumno.udg.mx\4to Semestre\Introduccion a la IA\AI in HealthCare - Chest Disease Detection With Teachable Machines\Testing Data"

clases = ["Bacterial Pneumonia", "Healthy", "Viral Pneumonia", "covid-19"]
n_test = 15

print("Iniciando reorganización del dataset...")

for clase in clases:
    train_clase = os.path.join(train_dir, clase)
    test_clase = os.path.join(test_dir, clase)
    
    # Asegurar que las carpetas existan
    if not os.path.exists(train_clase):
        os.makedirs(train_clase)
    if not os.path.exists(test_clase):
        os.makedirs(test_clase)
    
    # 1. Mover todo de Testing a Training para juntar todas las imágenes
    imagenes_movidas = 0
    for img in os.listdir(test_clase):
        ruta_origen = os.path.join(test_clase, img)
        if os.path.isfile(ruta_origen):
            # Añadimos un prefijo para evitar que se sobreescriban imágenes si casualmente tienen el mismo nombre
            nuevo_nombre = f"test_orig_{img}"
            ruta_destino = os.path.join(train_clase, nuevo_nombre)
            shutil.move(ruta_origen, ruta_destino)
            imagenes_movidas += 1
      
    # 2. Obtener la lista de todas las imágenes combinadas en Training
    archivos_train = [f for f in os.listdir(train_clase) if os.path.isfile(os.path.join(train_clase, f))]
    total_combinado = len(archivos_train)

    # Proteger contra el caso poco probable de tener menos de 15 imágenes en total
    cantidad_a_test = min(n_test, total_combinado)
    
    # 3. Seleccionar al azar las que irán a Testing
    random.seed() # Usamos una semilla completamente aleatoria
    seleccionadas = random.sample(archivos_train, cantidad_a_test)
    
    # 4. Mover la selección de vuelta a Testing
    for img in seleccionadas:
        ruta_origen = os.path.join(train_clase, img)
        ruta_destino = os.path.join(test_clase, img)
        shutil.move(ruta_origen, ruta_destino)
        
    print(f"Clase '{clase}': {total_combinado - cantidad_a_test} en Training, {cantidad_a_test} en Testing.")
    
print("\n¡Reorganización completada con éxito! Los datos ahora están bien mezclados.")
