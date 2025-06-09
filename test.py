import asyncio
from pathlib import Path
import logging
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import pandas as pd
from collections import defaultdict
from math import atan2, degrees
    
    
# Asegúrate de que ocr_utils.py esté en la misma carpeta o en el PYTHONPATH
from ocr_utils import (
    ensure_ocr_and_models,
    read_imagefile,
    run_ocr_with_timeout,
    process_ocr_results,
    log_system_resources,
    cleanup_memory
)

test_path = "./test/3.png"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ocr_main():
    logger.info("Iniciando prueba OCR con PaddleOCR...")

    # Mostrar uso inicial de recursos
    log_system_resources("Antes de OCR -> ")

    # 1. Asegurar PaddleOCR y modelos
    reader = ensure_ocr_and_models(lang='es')

    # 2. Leer la imagen
    image_path = Path(test_path)
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



    # Agrupar puntos por cluster
    clusters = defaultdict(list)
    for label, pt in zip(labels, centers_np):
        clusters[label].append(pt)

    # Ordenar clusters de izquierda a derecha según la media de X
    sorted_clusters = [
        clusters[k] for k in sorted(clusters, key=lambda k: np.mean([pt[0] for pt in clusters[k]]))
    ]

    # Estimar ángulo medio horizontal
    all_angles = []
    for i in range(len(sorted_clusters)):
        for j in range(i + 1, len(sorted_clusters)):
            for pt_a in sorted_clusters[i]:
                for pt_b in sorted_clusters[j]:
                    dx = pt_b[0] - pt_a[0]
                    dy = pt_b[1] - pt_a[1]
                    if dx == 0:
                        continue
                    angle = degrees(atan2(dy, dx))
                    all_angles.append(angle)

    average_angle = np.median(all_angles)
    angle_tolerance = 6     # grados
    max_vertical_dist = 40  # píxeles

    # Conectar puntos entre columnas
    connected = set()

    for i in range(len(sorted_clusters)):
        col_a = sorted_clusters[i]
        for pt_a in col_a:
            best_match = None
            best_score = None
            best_pt = None

            for j in range(i + 1, len(sorted_clusters)):
                col_b = sorted_clusters[j]
                for pt_b in col_b:
                    dx = pt_b[0] - pt_a[0]
                    dy = pt_b[1] - pt_a[1]
                    if dx <= 0:
                        continue  # solo mirar hacia la derecha
                    angle = degrees(atan2(dy, dx))
                    angle_deviation = abs(angle - average_angle)
                    if angle_deviation > angle_tolerance:
                        continue
                    if abs(dy) > max_vertical_dist:
                        continue

                    score = abs(dy) + angle_deviation * 5  # penalizamos el ángulo
                    if best_match is None or score < best_score:
                        best_score = score
                        best_match = (tuple(pt_a), tuple(pt_b))
                        best_pt = pt_b

            if best_match and (tuple(pt_a), tuple(best_pt)) not in connected:
                cv2.line(
                    img_copy,
                    best_match[0],
                    best_match[1],
                    color=(180, 180, 180),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
                connected.add((tuple(pt_a), tuple(best_pt)))

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