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
from collections import defaultdict, deque

    
# Asegúrate de que ocr_utils.py esté en la misma carpeta o en el PYTHONPATH
from ocr_utils import *

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
    bboxes = processed.get("bboxes", [])

    # 4.1 Estimar ángulo de inclinación
    skew_angle = estimate_skew_angle_from_ocr(bboxes)
    logger.info(f"Ángulo de inclinación estimado: {skew_angle:.2f}°")

    # 4.2 Rotar imagen y cajas para corregir skew
    img, rotated_bboxes = rotate_image_and_boxes(img, bboxes, skew_angle)
    img_copy = img.copy()  # Actualizamos también img_copy
    processed["bboxes"] = rotated_bboxes


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

    # Crear grafo bidireccional a partir de las conexiones detectadas
    graph = defaultdict(list)
    for pt_a, pt_b in connected:
        graph[pt_a].append(pt_b)
        graph[pt_b].append(pt_a)

    # Buscar componentes conectados (grupos de puntos) => representan una fila del ticket
    visited = set()
    connected_rows = []  # este es el antiguo 'rows'

    for node in graph:
        if node in visited:
            continue
        queue = deque([node])
        group = []

        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            group.append(curr)
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(group) >= 2:  # filtramos grupos muy pequeños
            connected_rows.append(group)

    # Mapear coordenadas (centros) a texto y cluster asignado
    center_to_text = {}
    center_to_cluster = {}

    for (bbox, text), label in zip(zip(processed.get("bboxes", []), processed.get("cleaned_texts", [])), labels):
        center_x = int(np.mean([p[0] for p in bbox]))
        center_y = int(np.mean([p[1] for p in bbox]))
        center = (center_x, center_y)
        center_to_text[center] = text.strip()
        center_to_cluster[center] = label

    # Ordenar clusters (etiquetas de DBSCAN) de izquierda a derecha
    unique_clusters = sorted(set(labels))
    cluster_index_map = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}  # cluster -> columna

    # Construir tabla alineando textos por su cluster
    table_data = []

    for group in connected_rows:
        row_dict = {}  # columna_index -> texto
        y_vals = []

        for pt in group:
            text = center_to_text.get(pt, "")
            cluster = center_to_cluster.get(pt)
            col_idx = cluster_index_map.get(cluster, 0)
            row_dict[col_idx] = text
            y_vals.append(pt[1])

        # Construir fila con columnas fijas
        max_cols = max(cluster_index_map.values()) + 1
        row = [row_dict.get(i, "") for i in range(max_cols)]

        avg_y = np.mean(y_vals)
        table_data.append((avg_y, row))

    # Ordenar filas por eje Y (de arriba hacia abajo)
    table_data.sort(key=lambda x: x[0])

    # Crear DataFrame final
    df = pd.DataFrame([row for _, row in table_data])

    print("\n===== TABLA RECONSTRUIDA DESDE OCR (USANDO CLUSTERS COMO COLUMNAS Y ORDEN Y) =====\n")
    print(df.fillna(""))

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