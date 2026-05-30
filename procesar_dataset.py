import os
from PIL import Image

# 1. Rutas exactas
RUTA_ORIGEN = (
    r"G:\Mi unidad\Drive alumno.udg.mx\4to Semestre\Introduccion a la IA"
    r"\AI in HealthCare - Chest Disease Detection With Teachable Machines"
    r"\Training Data"
)
RUTA_DESTINO = "Training Data"

TAMANO_OBJETIVO = (224, 224)
 
def auditar_y_procesar_imagenes():
    """Audita la ruta de origen, clasifica, convierte y redimensiona imágenes al destino.

    Busca carpetas en RUTA_ORIGEN, determina la clase objetivo a partir
    del nombre de la carpeta y del archivo, y guarda imágenes JPEG 224x224 RGB
    en RUTA_DESTINO dentro de subcarpetas predefinidas.
    """
    print("="*50)
    print("🚀 INICIANDO MODO DE DIAGNÓSTICO Y EXTRACCIÓN")
    print("="*50)
    
    # Verificación crítica 1: ¿Existe la ruta principal?
    if not os.path.exists(RUTA_ORIGEN):
        print(
            f"❌ ERROR CRÍTICO: No se puede encontrar la ruta:\n"
            f"{RUTA_ORIGEN}"
        )
        print(
            "Revisa que el disco duro virtual de Google Drive (G:) "
            "esté conectado y sincronizado."
        )
        return

    print("✅ Ruta principal detectada exitosamente.\n")
    
    imagenes_procesadas = 0
    imagenes_ignoradas = 0
    
    # Crear carpetas destino
    carpetas_destino = ["Bacterial Pneumonia", "Healthy", "Viral Pneumonia"]
    for carpeta in carpetas_destino:
        os.makedirs(os.path.join(RUTA_DESTINO, carpeta), exist_ok=True)

    carpetas_encontradas = os.listdir(RUTA_ORIGEN)
    print(f"🔍 Carpetas o archivos encontrados en el origen: {carpetas_encontradas}\n")

    for carpeta_origen in carpetas_encontradas:
        ruta_carpeta_origen = os.path.join(RUTA_ORIGEN, carpeta_origen)

        if not os.path.isdir(ruta_carpeta_origen):
            continue

        archivos = os.listdir(ruta_carpeta_origen)
        print(f"📁 Explorando carpeta: '{carpeta_origen}' ({len(archivos)} archivos detectados)")

        # Convertimos el nombre de la carpeta a mayúsculas para evitar errores de sintaxis
        carpeta_upper = carpeta_origen.upper()

        for archivo in archivos:
            ruta_archivo_original = os.path.join(ruta_carpeta_origen, archivo)
            nombre_base = archivo.lower()
            clase_destino = None
        
            # Lógica de clasificación a prueba de mayúsculas/minúsculas
            if "NORMAL" in carpeta_upper:
                clase_destino = "Healthy"
            elif "PNEUMONIA" in carpeta_upper:
                if "virus" in nombre_base:
                    clase_destino = "Viral Pneumonia"
                elif "bacteria" in nombre_base:
                    clase_destino = "Bacterial Pneumonia"
                else:
                    imagenes_ignoradas += 1
                    continue
            else:
                continue

            # Si llegamos aquí, sabemos a dónde va la imagen
            ruta_guardado = os.path.join(RUTA_DESTINO, clase_destino)
            ruta_archivo_nuevo = os.path.join(ruta_guardado, f"kaggle_{archivo.split('.')[0]}.jpg")
  
            if os.path.exists(ruta_archivo_nuevo):
                continue

            try:
                with Image.open(ruta_archivo_original) as img:
                    img.verify() 
                
                with Image.open(ruta_archivo_original) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize(TAMANO_OBJETIVO, Image.Resampling.LANCZOS)
                    img.save(ruta_archivo_nuevo, format="JPEG", quality=95)
                    imagenes_procesadas += 1
                    
            except Exception as e:
                print(f"⚠️ Archivo dañado omitido ({archivo}): {e}")
                continue

    print("\n" + "="*50)
    print("📊 REPORTE FINAL DE EXTRACCIÓN")
    print("="*50)
    print(f"✅ Imágenes válidas integradas: {imagenes_procesadas}")
    print(f"⏭️ Archivos ignorados (sin etiqueta clara): {imagenes_ignoradas}")
    print("="*50)

if __name__ == "__main__":
    auditar_y_procesar_imagenes()