import logging
import re
import numpy as np
import cv2
import asyncio
import psutil
import torch
import gc
from pathlib import Path
from math import atan2, degrees, radians, sin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_ocr_and_models(lang='es'):
    """
    Ensures that PaddleOCR is installed and that its models are downloaded.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.info("PaddleOCR not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "paddleocr"])
        from paddleocr import PaddleOCR

    # PaddleOCR manages its own model download automatically
    logger.info(f"Initializing PaddleOCR with language: {lang}")

    # The lang argument in PaddleOCR uses a different notation (e.g., 'es' or 'es+en')
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    
    # Dummy run to ensure models are downloaded
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img_path = "/tmp/dummy_image.jpg"
    cv2.imwrite(dummy_img_path, dummy_img)
    _ = ocr.ocr(dummy_img_path, cls=True)

    logger.info("PaddleOCR models downloaded and initialized.")
    return ocr


def read_imagefile(file_bytes: bytes) -> np.ndarray:
    image_data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img


async def run_ocr_with_timeout(reader, img: np.ndarray, timeout: int = 15):
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, reader.ocr, img, True, True, True),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError("OCR processing timed out.")


def desleet_text(text: str) -> tuple[str, int]:
    leet_dict = {
        '4': 'A', '@': 'A',
        '8': 'B',
        '(': 'C', '<': 'C', '{': 'C', '[': 'C',
        '3': 'E',
        '6': 'G',
        '1': 'I', '!': 'I', '|': 'I',
        '0': 'O',
        '5': 'S', '$': 'S',
        '7': 'T',
        '2': 'Z'
    }

    modificaciones_leet = 0
    tokens = re.findall(r'\S+|\s+', text)
    resultado = ""

    for token in tokens:
        if token.isspace():
            resultado += token
            continue

        if re.fullmatch(r'[\d.,]+', token):
            resultado += token
        else:
            nuevo_token = ""
            for c in token:
                c_upper = c.upper()
                if c_upper in leet_dict:
                    nuevo_token += leet_dict[c_upper]
                    modificaciones_leet += 1
                else:
                    nuevo_token += c_upper
            resultado += nuevo_token

    texto_sin_espacios = re.sub(r'\s+', '', text)
    longitud = len(texto_sin_espacios)

    if longitud > 0 and modificaciones_leet >= (longitud / 2):
        return text.upper(), 0
    else:
        return resultado, modificaciones_leet


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[^A-Za-z0-9ñÑáéíóúÁÉÍÓÚ.,€$ \-]', '', text)
    replacements = {
        'O': '0', 'l': '1', 'I': '1', 'Z': '2',
        'S': '5', 'B': '8'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


def log_system_resources(prefix=""):
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    logger.info(f"{prefix}CPU: {cpu_percent:.2f}%, RAM: {mem.percent:.2f}% "
                f"({mem.used / (1024 ** 2):.2f} MB / {mem.total / (1024 ** 2):.2f} MB)")


def process_ocr_results(results):
    original_texts, cleaned_texts, confidences, bboxes = [], [], [], []

    for line in results:
        if isinstance(line, list):
            for det in line:
                if len(det) != 2:
                    continue
                bbox, (text, conf) = det
                cleaned, _ = desleet_text(clean_text(text))
                original_texts.append(text)
                cleaned_texts.append(cleaned)
                confidences.append(round(conf, 2))
                bboxes.append(bbox)

    return {
        "original_texts": [convert_numpy_types(x) for x in original_texts],
        "cleaned_texts": [convert_numpy_types(x) for x in cleaned_texts],
        "confidences": [convert_numpy_types(x) for x in confidences],
        "bboxes": [convert_numpy_types(x) for x in bboxes],
    }


def estimate_skew_angle_from_ocr(bboxes, min_box_width=20):
    """
    Estima el ángulo de inclinación global de la imagen basándose en las cajas OCR.
    """
    angles = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        # Top-left y top-right
        (x1, y1), (x2, y2), *_ = bbox
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < min_box_width:
            continue
        angle = degrees(atan2(dy, dx))
        angles.append(angle)

    if not angles:
        return 0.0
    return np.median(angles)


def unskew_boxes(bboxes):
    """
    Aplica una rotación inversa a todas las cajas (bboxes) en función del ángulo estimado,
    rotándolas alrededor del origen (0, 0) para corregir la inclinación.
    """
    angle_deg = estimate_skew_angle_from_ocr(bboxes)
    print(f"🔁 Ángulo de inclinación detectado: {angle_deg:.2f}°")

    angle_rad = radians(-angle_deg)  # Inverso para corregir inclinación
    cos_theta = np.cos(angle_rad)
    sin_theta = sin(angle_rad)

    def rotate_point(x, y):
        new_x = x * cos_theta - y * sin_theta
        new_y = x * sin_theta + y * cos_theta
        return [new_x, new_y]

    unskewed_bboxes = []
    for bbox in bboxes:
        rotated_box = [rotate_point(x, y) for (x, y) in bbox]
        unskewed_bboxes.append(rotated_box)

    return unskewed_bboxes


def rotate_image_and_boxes(img, bboxes, angle_deg):
    """
    Rota la imagen y también ajusta las coordenadas de las cajas.
    """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    rotated_bboxes = []
    for bbox in bboxes:
        pts = np.array(bbox, dtype=np.float32)
        ones = np.ones((pts.shape[0], 1))
        pts_hom = np.hstack([pts, ones])
        rotated_pts = M.dot(pts_hom.T).T
        rotated_bboxes.append(rotated_pts.tolist())

    return rotated_img, rotated_bboxes

def cleanup_memory(*objs):
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


import asyncio
from pathlib import Path
import logging
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict, deque
import pandas as pd
from math import atan2, degrees

from ocr_utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

test_path = "./test/3.png"


def load_image(path: str):
    image_path = Path(path)
    if not image_path.exists():
        logger.error(f"El archivo {image_path} no existe.")
        return None
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    img = read_imagefile(file_bytes)
    return img


async def perform_ocr(reader, img):
    try:
        results = await run_ocr_with_timeout(reader, img, timeout=15)
        return results
    except TimeoutError:
        logger.error("OCR se ha excedido del tiempo de espera.")
        return None


def process_and_correct_skew(img, processed):
    bboxes = processed.get("bboxes", [])
    skew_angle = estimate_skew_angle_from_ocr(bboxes)
    logger.info(f"Ángulo de inclinación estimado: {skew_angle:.2f}°")
    img_rotated, rotated_bboxes = rotate_image_and_boxes(img, bboxes, skew_angle)
    processed["bboxes"] = rotated_bboxes
    return img_rotated, processed


def draw_bboxes_and_centers(img, bboxes):
    for bbox in bboxes:
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        center_x = int(np.mean([p[0] for p in bbox]))
        center_y = int(np.mean([p[1] for p in bbox]))
        cv2.circle(img, (center_x, center_y), radius=4, color=(0, 0, 255), thickness=-1)


def get_centers(bboxes):
    centers = []
    for bbox in bboxes:
        center_x = int(np.mean([p[0] for p in bbox]))
        center_y = int(np.mean([p[1] for p in bbox]))
        centers.append([center_x, center_y])
    return np.array(centers)


def cluster_centers(centers, eps=50, min_samples=1):
    X_coords = centers[:, 0].reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_coords)
    return db.labels_



#V2 
def vertical_aligned_clustering(bboxes, threshold=0):
    import numpy as np

    def bbox_edges(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), max(xs), min(ys), max(ys)

    def intersects_vertically(line_x, bbox, threshold=0):
        left, right, top, bottom = bbox_edges(bbox)
        return (left - threshold <= line_x <= right + threshold)

    def intersects_horizontally(line_y, bbox, threshold=0):
        _, _, top, bottom = bbox_edges(bbox)
        return (top - threshold <= line_y <= bottom + threshold)

    def get_vertical_lines_from_bbox(bbox):
        xs = [p[0] for p in bbox]
        return [min(xs), max(xs), int(np.mean(xs))]

    n = len(bboxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Tirar líneas verticales y unir grupos
    for i, bbox_i in enumerate(bboxes):
        vertical_lines = get_vertical_lines_from_bbox(bbox_i)

        for x_line in vertical_lines:
            intersecting = []
            for j, bbox_j in enumerate(bboxes):
                if intersects_vertically(x_line, bbox_j, threshold):
                    intersecting.append(j)

            if len(intersecting) > 1:
                for idx in intersecting[1:]:
                    union(intersecting[0], idx)

    # Generar labels
    label_map = {}
    labels = []
    current_label = 0
    for i in range(n):
        root = find(i)
        if root not in label_map:
            label_map[root] = current_label
            current_label += 1
        labels.append(label_map[root])

    return np.array(labels)


def label_clusters_on_image(img, centers, labels):
    for (x, y), label in zip(centers, labels):
        cv2.putText(
            img,
            f"C{label}",
            (x + 5, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )


def connect_clusters_lines(img, centers, labels):

    clusters = defaultdict(list)
    for label, pt in zip(labels, centers):
        clusters[label].append(pt)

    sorted_clusters = [
        clusters[k] for k in sorted(clusters, key=lambda k: np.mean([pt[0] for pt in clusters[k]]))
    ]

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
    angle_tolerance = 6
    max_vertical_dist = 40

    connected = set()
    for i, col_a in enumerate(sorted_clusters):
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
                        continue
                    angle = degrees(atan2(dy, dx))
                    if abs(angle - average_angle) > angle_tolerance:
                        continue
                    if abs(dy) > max_vertical_dist:
                        continue

                    score = abs(dy) + abs(angle - average_angle) * 5
                    if best_match is None or score < best_score:
                        best_score = score
                        best_match = (tuple(pt_a), tuple(pt_b))
                        best_pt = pt_b

            if best_match and best_match not in connected:
                cv2.line(img, best_match[0], best_match[1], color=(180, 180, 180), thickness=1, lineType=cv2.LINE_AA)
                connected.add(best_match)
    return connected


def build_connected_rows(connected):
    graph = defaultdict(list)
    for pt_a, pt_b in connected:
        graph[pt_a].append(pt_b)
        graph[pt_b].append(pt_a)

    visited = set()
    connected_rows = []
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
        if len(group) >= 2:
            connected_rows.append(group)
    return connected_rows


def build_rows_from_centers(centers, bboxes, y_threshold=0, dynamic_threshold=True):
    """
    Agrupa centros alineados horizontalmente en filas basadas en cercanía en Y.
    
    - Si dynamic_threshold=True, el threshold se calcula como la altura media de las cajas * 0.7.
    - centers: lista de [x, y]
    - bboxes: lista de listas de puntos [[(x1, y1), (x2, y2), ...]]
    """
    # Calcular threshold dinámico basado en altura media
    if dynamic_threshold and bboxes:
        heights = [max(p[1] for p in bbox) - min(p[1] for p in bbox) for bbox in bboxes]
        avg_height = np.mean(heights)
        y_threshold = avg_height * 0.7

    # Ordenar por Y
    centers = sorted(centers, key=lambda c: c[1])
    rows = []

    for pt in centers:
        added = False
        for row in rows:
            row_y = np.mean([p[1] for p in row])
            if abs(pt[1] - row_y) <= y_threshold:
                row.append(pt)
                added = True
                break
        if not added:
            rows.append([pt])

    return rows



def build_table(processed, labels, connected_rows):
    bboxes = processed["bboxes"]
    original_texts = processed["original_texts"]
    cleaned_texts = processed["cleaned_texts"]

    # Map label -> lista de índices de bboxes en esa columna
    col_to_indices = {}
    for idx, label in enumerate(labels):
        col_to_indices.setdefault(label, []).append(idx)

    # Map centro (x,y) a índice para acceder rápido
    centers = get_centers(bboxes)
    centers_tuples = [tuple(map(int, c)) for c in centers]
    center_to_idx = {c: i for i, c in enumerate(centers_tuples)}

    table_rows = []
    for row_centers in connected_rows:
        # Para cada fila, tenemos un conjunto de centros (x,y) que están conectados horizontalmente
        # Queremos ordenar por x y extraer la celda (texto) correspondiente

        # Ordenar los centros por x para fila
        row_centers_sorted = sorted(row_centers, key=lambda c: c[0])

        # Crear fila de la tabla, inicial con valores vacíos para todas las columnas
        max_col = max(labels) if len(labels) > 0 else -1
        row_data = [""] * (max_col + 1)

        for center in row_centers_sorted:
            idx = center_to_idx.get(tuple(map(int, center)))
            if idx is None:
                continue
            col = labels[idx]
            text = cleaned_texts[idx]
            # Podrías hacer merge de textos si hay más de uno en esa celda,
            # pero en esta versión simple ponemos solo uno por celda.
            if row_data[col]:
                row_data[col] += " " + text
            else:
                row_data[col] = text

        table_rows.append(row_data)

    # Construimos DataFrame
    df = pd.DataFrame(table_rows)

    return df


def build_table_from_lines(bboxes, texts, y_threshold=10, x_gap_threshold=40):
    """
    Construye la tabla a partir de las líneas agrupadas y palabras unidas.
    Aquí asumimos que cada línea es una fila y cada frase es una celda.
    """
    # Agrupa y une palabras en líneas
    lines = group_words_in_lines(bboxes, texts, y_threshold, x_gap_threshold)
    # Por simplicidad, cada línea es una fila y cada frase dentro de la línea puede dividirse por tabulaciones u otro criterio
    # Aquí retornamos un DataFrame con una columna "Item" con la línea completa
    import pandas as pd
    df = pd.DataFrame({"Item": lines})
    return df


def group_words_in_lines(bboxes, texts, y_threshold=10, x_gap_threshold=40):
    """
    Agrupa palabras en líneas, y dentro de cada línea concatena palabras que están próximas horizontalmente.
    
    Args:
        bboxes: lista de bounding boxes, cada bbox = [(x1,y1), (x2,y2), ...]
        texts: lista de textos detectados correspondiente a cada bbox
        y_threshold: máxima distancia vertical para considerar palabras en la misma línea
        x_gap_threshold: máxima distancia horizontal entre palabras para unirlas
    
    Returns:
        lines: lista de strings, cada string es una línea unida
    """
    # Calcular centro de cada bbox
    centers = [(int(np.mean([p[0] for p in bbox])), int(np.mean([p[1] for p in bbox]))) for bbox in bboxes]
    
    # Agrupar por líneas: clusterizar por Y
    lines = []
    current_line = []
    current_y = None
    
    # Ordenar por Y para ir procesando líneas de arriba hacia abajo
    sorted_items = sorted(zip(bboxes, texts, centers), key=lambda x: x[2][1])
    
    for bbox, text, (cx, cy) in sorted_items:
        if current_y is None:
            current_y = cy
            current_line = [(bbox, text, cx)]
            continue
        
        if abs(cy - current_y) <= y_threshold:
            current_line.append((bbox, text, cx))
        else:
            # Procesar línea anterior
            line_text = merge_line_words(current_line, x_gap_threshold)
            lines.append(line_text)
            # Nueva línea
            current_line = [(bbox, text, cx)]
            current_y = cy
            
    # Procesar última línea
    if current_line:
        line_text = merge_line_words(current_line, x_gap_threshold)
        lines.append(line_text)
    
    return lines


def merge_line_words(line_items, x_gap_threshold):
    """
    Une palabras en una línea, según distancia horizontal
    
    Args:
        line_items: lista de (bbox, text, cx) en la línea
        x_gap_threshold: máxima distancia horizontal entre palabras para unirlas
    
    Returns:
        string con palabras concatenadas separadas por espacio
    """
    # Ordenar por X
    line_items = sorted(line_items, key=lambda x: x[2])
    
    merged_words = []
    prev_x = None
    prev_text = ""
    
    for _, text, cx in line_items:
        if prev_x is None:
            prev_x = cx
            prev_text = text
        else:
            if cx - prev_x <= x_gap_threshold:
                # Si están cerca, unir con espacio
                prev_text += " " + text
            else:
                # Separar palabras con mucha distancia
                merged_words.append(prev_text)
                prev_text = text
            prev_x = cx
    merged_words.append(prev_text)
    
    return " ".join(merged_words)
