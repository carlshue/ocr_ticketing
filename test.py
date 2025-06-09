import asyncio
from pathlib import Path
import logging
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import pandas as pd
# Asegúrate de que ocr_utils.py esté en la misma carpeta o en el PYTHONPATH
from ocr_utils import (
    ensure_ocr_and_models,
    read_imagefile,
    run_ocr_with_timeout,
    process_ocr_results,
    log_system_resources,
    cleanup_memory
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ocr_main():
    logger.info("Iniciando prueba OCR con PaddleOCR...")

    # Mostrar uso inicial de recursos
    log_system_resources("Antes de OCR -> ")

    # 1. Asegurar PaddleOCR y modelos
    reader = ensure_ocr_and_models(lang='es')

    # 2. Leer la imagen
    image_path = Path("./test/wawawiwa.png")
    if not image_path.exists():
        logger.error(f"El archivo {image_path} no existe.")
        return

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    img = read_imagefile(file_bytes)
    img_copy = img.copy()  # para dibujar encima

    # 3. Ejecutar OCR con timeout
    try:
        results = await run_ocr_with_timeout(reader, img, timeout=15)
    except TimeoutError:
        logger.error("OCR se ha excedido del tiempo de espera.")
        return

    # 4. Procesar resultados
    processed = process_ocr_results(results)

    # 5. Mostrar resultados por pantalla
    logger.info("Resultados OCR:")
    for idx, (orig, clean, conf) in enumerate(
            zip(
                processed['original_texts'],
                processed['cleaned_texts'],
                processed['confidences'],
            )):
        print(f"--- Resultado {idx+1} ---")
        print(f"Texto Original: {orig}")
        print(f"Texto Limpio: {clean}")
        print(f"Confianza: {conf}\n")

    # 6. Mostrar tabla reconstruida
    logger.info("Tabla reconstruida:")
    table = processed.get("table", [])
    for r_idx, row in enumerate(table):
        logger.info(f"Fila {r_idx+1}: {row}")

    # Dibujar cajas en la imagen
    # Dibujar cajas y sus centros en la imagen
    for bbox in processed.get('bboxes', []):
        # Convertir a array numpy y asegurar enteros
        pts = np.array(bbox, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Dibujar la caja
        cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Calcular centro
        center_x = int(np.mean([point[0] for point in bbox]))
        center_y = int(np.mean([point[1] for point in bbox]))
        
        # Dibujar centro
        cv2.circle(img_copy, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=-1)
        
        
##### BEGIN CLUSTERING ########

    # Extraer los centros
    centers = []
    for bbox in processed.get('bboxes', []):
        center_x = int(np.mean([point[0] for point in bbox]))
        center_y = int(np.mean([point[1] for point in bbox]))
        centers.append([center_x, center_y])

    centers_np = np.array(centers)

    # Usar solo X para clustering por columnas (reshape necesario)
    X_coords = centers_np[:, 0].reshape(-1, 1)

    # Aplicar DBSCAN (eps: tolerancia horizontal, min_samples: mínimo puntos por cluster)
    db = DBSCAN(eps=50, min_samples=1).fit(X_coords)
    labels = db.labels_  # Cluster asignado a cada punto

    # Añadir etiquetas al centro visualmente
    for (x, y), label in zip(centers_np, labels):
        cv2.putText(
            img_copy,
            f"C{label}",
            (x + 5, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )


##### BEGIN PRINT CLUSTERING ########



    # Paso 1: Asociar cada texto con su centro y label de columna (X)
    items = []
    for idx, bbox in enumerate(processed['bboxes']):
        center_x = int(np.mean([point[0] for point in bbox]))
        center_y = int(np.mean([point[1] for point in bbox]))
        text = processed['cleaned_texts'][idx]
        col_label = labels[idx]
        items.append({'x': center_x, 'y': center_y, 'text': text, 'col': col_label})

    # Paso 2: Clusterizar las filas por coordenada Y (alineación horizontal visual)
    y_coords = np.array([[item['y']] for item in items])
    row_db = DBSCAN(eps=20, min_samples=1).fit(y_coords)
    row_labels = row_db.labels_

    # Asociar a cada item su fila detectada
    for item, row_label in zip(items, row_labels):
        item['row'] = row_label

    # Paso 3: Ordenar columnas de izquierda a derecha (por promedio X)
    column_map = {
        old: new for new, (old, _) in enumerate(sorted(
            {item['col']: [] for item in items}.items(),
            key=lambda kv: np.mean([i['x'] for i in items if i['col'] == kv[0]])
        ))
    }

    # Paso 4: Crear estructura fila/columna vacía
    max_row = max(item['row'] for item in items)
    max_col = max(column_map.values())

    # Crear matriz vacía
    table_matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # Rellenar la matriz con los textos correctos
    for item in items:
        row = item['row']
        col = column_map[item['col']]
        if table_matrix[row][col]:  # ya hay algo → concatenar
            table_matrix[row][col] += " " + item['text']
        else:
            table_matrix[row][col] = item['text']

    # Crear DataFrame y mostrar
    column_names = [f"Columna {i}" for i in range(max_col + 1)]
    df = pd.DataFrame(table_matrix, columns=column_names)

    print("\n--- Tabla resultante agrupada visualmente (clusters X, filas Y) ---")
    print(df.to_string(index=False))




##### END CLUSTERING ########


    # Mostrar imagen con resultados (sin cajas, solo texto)
    cv2.imshow("OCR Resultados", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mostrar uso de recursos tras el OCR
    log_system_resources("Después de OCR -> ")

    # 7. Liberar memoria
    cleanup_memory(img, results, processed)

if __name__ == "__main__":
    asyncio.run(test_ocr_main())