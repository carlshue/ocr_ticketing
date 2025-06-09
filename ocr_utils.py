import logging
import re
import numpy as np
import cv2
import asyncio
import psutil
import torch
import gc
from pathlib import Path
from math import atan2, degrees

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


def build_table(processed, labels, connected_rows):
    center_to_text = {}
    center_to_cluster = {}

    for (bbox, text), label in zip(zip(processed.get("bboxes", []), processed.get("cleaned_texts", [])), labels):
        center_x = int(np.mean([p[0] for p in bbox]))
        center_y = int(np.mean([p[1] for p in bbox]))
        center_to_text[(center_x, center_y)] = text.strip()
        center_to_cluster[(center_x, center_y)] = label

    unique_clusters = sorted(set(labels))
    cluster_index_map = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}

    table_data = []
    for group in connected_rows:
        row_dict = {}
        y_vals = []
        for pt in group:
            text = center_to_text.get(pt, "")
            cluster = center_to_cluster.get(pt)
            col_idx = cluster_index_map.get(cluster, 0)
            row_dict[col_idx] = text
            y_vals.append(pt[1])
        max_cols = max(cluster_index_map.values()) + 1
        row = [row_dict.get(i, "") for i in range(max_cols)]
        avg_y = np.mean(y_vals)
        table_data.append((avg_y, row))

    table_data.sort(key=lambda x: x[0])
    df = pd.DataFrame([row for _, row in table_data])
    return df