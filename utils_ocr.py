import os
import cv2
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher

CONFIG = {
    'max_dist_ratio': 0.05,
    'text_sim_thresh': 0.75,
    'iou_thresh': 0.3,
    'confidence_threshold': 0.4,  # EasyOCR devuelve confianza entre 0 y 1
}


def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^A-Za-z0-9ñÑáéíóúÁÉÍÓÚ.,€$ \-]', '', text)
    replacements = {
        'O': '0', 'l': '1', 'I': '1', 'Z': '2',
        'S': '5', 'B': '8'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def apply_threshold(blurred_img, thresh_type):
    if thresh_type == 'adaptive_inv':
        return cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 4)
    elif thresh_type == 'otsu_inv':
        _, thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    elif thresh_type == 'adaptive':
        return cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 4)
    else:
        raise ValueError(f"Tipo de umbral desconocido: {thresh_type}")


def generate_augmented_variants(img):
    variants = []
    inverses = []

    angles = [0, 5]
    blur_sizes = [3, 5]
    threshold_types = ['adaptive_inv', 'adaptive']
    vertical_scales = [0.9, 0.8]

    for angle in angles:
        rotated, M_inv = rotate_image_with_inverse(img, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        for blur_size in blur_sizes:
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

            for thresh_type in threshold_types:
                try:
                    thresh_img = apply_threshold(blurred, thresh_type)
                    variants.append(thresh_img)
                    inverses.append(M_inv)
                except ValueError as e:
                    print(e)

    for scale_y in vertical_scales:
        scaled_img, M_inv = scale_image_vertically_with_inverse(img, scale_y)
        gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        for thresh_type in threshold_types:
            try:
                thresh_img = apply_threshold(blurred, thresh_type)
                variants.append(thresh_img)
                inverses.append(M_inv)
            except ValueError as e:
                print(e)

    return variants, inverses


def rotate_image_with_inverse(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
    M_inv = cv2.invertAffineTransform(M)

    return rotated, M_inv


def scale_image_vertically_with_inverse(img, scale_y):
    h, w = img.shape[:2]
    scale_matrix = np.array([
        [1.0, 0, 0],
        [0, scale_y, 0]
    ], dtype=np.float32)

    new_h = int(h * scale_y)
    resized = cv2.warpAffine(img, scale_matrix, (w, new_h), borderValue=(255, 255, 255))
    inv_matrix = cv2.invertAffineTransform(scale_matrix)

    return resized, inv_matrix


def text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def iou(bbox1, bbox2):
    # EasyOCR bbox: [top-left, top-right, bottom-right, bottom-left]
    x1, y1 = bbox1[0]
    x2, y2 = bbox1[2]
    x3, y3 = bbox2[0]
    x4, y4 = bbox2[2]

    xa = max(x1, x3)
    ya = max(y1, y3)
    xb = min(x2, x4)
    yb = min(y2, y4)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def merge_ocr_results(all_ocr_results, image_shape):
    """
    Fusiona resultados OCR agrupando por texto similar.
    """
    all_results = [item for sublist in all_ocr_results for item in sublist]
    if not all_results:
        return []

    used = set()
    merged = []

    for i, box_i in enumerate(all_results):
        if i in used:
            continue
        group = [box_i]
        used.add(i)
        for j in range(i + 1, len(all_results)):
            if j in used:
                continue
            if text_similarity(box_i['text'].lower(), all_results[j]['text'].lower()) >= CONFIG['text_sim_thresh']:
                if iou(box_i['bbox'], all_results[j]['bbox']) >= CONFIG['iou_thresh']:
                    group.append(all_results[j])
                    used.add(j)
        # Fusionar grupo
        merged.append(group)

    # Construir lista final de resultados fusionados
    final_results = []
    for group in merged:
        texts = [g['text'] for g in group]
        confs = [g['conf'] for g in group]
        bboxes = [g['bbox'] for g in group]

        merged_text = max(texts, key=lambda t: len(t))
        avg_conf = np.mean(confs)

        # Combinar bbox en un rectángulo mínimo
        x_min = min([min(p[0] for p in box) for box in bboxes])
        y_min = min([min(p[1] for p in box) for box in bboxes])
        x_max = max([max(p[0] for p in box) for box in bboxes])
        y_max = max([max(p[1] for p in box) for box in bboxes])
        merged_bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

        final_results.append({
            'text': merged_text,
            'conf': avg_conf,
            'bbox': merged_bbox
        })

    return final_results


def group_lines(ocr_results):
    ocr_results.sort(key=lambda x: x['bbox'][0][1])  # y de la esquina superior izquierda
    lines = []
    current_line = []
    prev_y = -100

    for item in ocr_results:
        y = item['bbox'][0][1]
        if current_line:
            avg_height = np.mean([abs(line_item['bbox'][2][1] - line_item['bbox'][0][1]) for line_item in current_line])
            if abs(y - prev_y) > avg_height * 0.8:
                lines.append(current_line)
                current_line = []
        current_line.append(item)
        prev_y = y
    if current_line:
        lines.append(current_line)

    return [' '.join([el['text'] for el in line]) for line in lines]


def lines_to_dataframe(lines):
    df = pd.DataFrame(lines, columns=['texto'])
    return df


def detect_separator_lines(lines):
    separator_positions = []
    for idx, line in enumerate(lines):
        if re.search(r'-{2,}', line):
            separator_positions.append(idx)
    return separator_positions


def show_ocr_boxes(img, ocr_results, window_name='OCR Result'):
    img_ocr = img.copy()
    for item in ocr_results:
        bbox = item['bbox']
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_ocr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = bbox[0]
        cv2.putText(img_ocr, item['text'], (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow(window_name, img_ocr)

def resize_if_needed(img, max_height=1000):
    """
    Redimensiona la imagen si su altura es mayor a max_height,
    manteniendo la proporción de aspecto.
    
    Parámetros:
    - img: imagen numpy array
    - max_height: altura máxima permitida
    
    Retorna:
    - imagen redimensionada si fue necesario, sino la imagen original
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_img
    return img


def parse_line_to_item(text_line):
    """
    Intenta extraer cantidad, producto y precio de una línea de texto.
    Devuelve diccionario o None si no es línea válida.
    """
    # Normaliza texto para manejar X, x, espacios, comas y puntos
    line = text_line.strip()
    # Patrón para cantidad: puede estar antes o después, ej:
    # "2 x Producto 3,50", "Producto 3,50 2 X", "X5 Producto 10.00"
    # Simplificamos:
    # Buscamos cantidad: un número seguido o precedido por X o x
    qty_pattern = re.compile(r'(?i)(\d+)\s*[xX]|[xX]\s*(\d+)')
    qty_match = qty_pattern.search(line)
    quantity = None
    if qty_match:
        quantity = int(qty_match.group(1) or qty_match.group(2))
        # Eliminamos la parte cantidad del texto para extraer producto y precio
        line = qty_pattern.sub('', line).strip()

    # Buscamos precio: número con decimales o coma, al final o cerca del final
    price_pattern = re.compile(r'(\d+[.,]\d{2})$')
    price_match = price_pattern.search(line)
    price = None
    if price_match:
        price_str = price_match.group(1).replace(',', '.')
        price = float(price_str)
        # Quitamos precio del texto
        line = price_pattern.sub('', line).strip()
    
    # Producto queda lo que sobra
    product = line.strip()
    
    # Si no tenemos ni cantidad ni precio, descartamos
    if quantity is None or price is None:
        return None

    # Producto no debe ser vacio o demasiado corto
    if len(product) < 2:
        return None

    return {'producto': product, 'cantidad': quantity, 'precio': price}

def select_regions(img, window_name='Selecciona las regiones'):
    """
    Permite al usuario dibujar rectángulos sobre la imagen y devuelve las coordenadas.
    """
    selected_regions = []
    drawing = False
    ix, iy = -1, -1

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img, selected_regions

        img_copy = img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Selecciona regiones", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x_min, x_max = min(ix, x), max(ix, x)
            y_min, y_max = min(iy, y), max(iy, y)
            selected_regions.append((x_min, y_min, x_max, y_max))
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Selecciona regiones", img_copy)

    cv2.imshow("Selecciona regiones", img)
    cv2.setMouseCallback("Selecciona regiones", draw_rectangle)
    print("Dibuja con el ratón las regiones de interés y pulsa 'q' para terminar.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow("Selecciona regiones")
    return selected_regions


def process_selected_regions(image, regions, reader):
    """
    Aplica OCR sobre las regiones seleccionadas.

    Parámetros:
    - image: imagen completa (np.ndarray)
    - regions: lista de tuplas (x_min, y_min, x_max, y_max)
    - reader: instancia de easyocr.Reader

    Retorna:
    - lista de listas de resultados OCR por región
    """
    all_ocr_results = []

    print(f'[DEBUG] Procesando {len(regions)} regiones seleccionadas...')
    for idx, (x_min, y_min, x_max, y_max) in enumerate(regions):
        print(f'[DEBUG] Región {idx+1}: ({x_min}, {y_min}, {x_max}, {y_max})')
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            print(f'[WARNING] Región vacía en la región {idx+1}, se omite.')
            continue

        # Aumentos de imagen
        variants, inverses = generate_augmented_variants(roi)
        region_results = []

        for var_idx, variant in enumerate(variants):
            print(f'[DEBUG] Ejecutando OCR en la variante {var_idx+1} de la región {idx+1}...')
            ocr_output = reader.readtext(variant)
            for item in ocr_output:
                bbox, text, conf = item[0], item[1], item[2]

                # Ajustar las coordenadas a la imagen original
                adjusted_bbox = [
                    [point[0] + x_min, point[1] + y_min] for point in bbox
                ]

                region_results.append({
                    'bbox': adjusted_bbox,
                    'text': clean_text(text),
                    'conf': conf
                })

        all_ocr_results.append(region_results)
        print(f'[DEBUG] Región {idx+1}: {len(region_results)} resultados OCR encontrados.')

    print(f'[DEBUG] OCR completado para todas las regiones.')
    return all_ocr_results


def conf_to_color(conf):
    if conf < 0.5:
        return (0, 0, 255)       # Rojo BGR
    elif conf < 0.75:
        return (0, 165, 255)     # Naranja BGR
    else:
        return (0, 255, 0)       # Verde BGR

def draw_all_regions(img):
    img_copy = img.copy()
    for reg in regions_ocr:
        (x_min, y_min, x_max, y_max) = reg['region']
        conf_med = reg['conf_media']
        color = conf_to_color(conf_med)
        # Rectángulo principal
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 2)
        # Dibujar cada caja OCR dentro de la región
        for result in reg['ocr_results']:
            bbox = np.array(result['bbox']).astype(int)
            text = result['text']
            cv2.polylines(img_copy, [bbox], isClosed=True, color=color, thickness=2)
            x_text, y_text = bbox[0]
            cv2.putText(img_copy, text, (x_text, y_text - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  # Texto azul
    return img_copy

def draw_rectangle_and_ocr(event, x, y, flags, param):
    global ix, iy, drawing, use_variants, img_base, regions_ocr, img_drawn

    window_name = param['window_name']
    angulo_texto = param['angulo_texto']

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        # Limpiar regiones y redibujar desde la base
        regions_ocr.clear()  # Limpiar lista de regiones
        img_drawn = img_base.copy()  # Restaurar imagen base
        cv2.imshow(window_name, img_drawn)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Imagen temporal para mostrar rectángulo en movimiento
            img_temp = img_drawn.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 255), 2)  # Amarillo rectángulo actual
            cv2.putText(img_temp, angulo_texto, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, y_min = min(ix, x), min(iy, y)
        x_max, y_max = max(ix, x), max(iy, y)
        print(f'  -> Región seleccionada: ({x_min}, {y_min}, {x_max}, {y_max})')

        roi = img_base[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            print("  -> Región vacía, se omite.")
            return

        # OCR en la región
        if use_variants:
            print("  -> OCR con variantes activado.")
            variants, _ = generate_augmented_variants(roi)
        else:
            print("  -> OCR con variantes desactivado.")
            variants = [roi]

        ocr_results = []

        for variant in variants:
            ocr_output = reader.readtext(variant)
            for item in ocr_output:
                bbox, text, conf = item[0], item[1], item[2]
                adjusted_bbox = [[point[0] + x_min, point[1] + y_min] for point in bbox]
                # Limpiar texto y aplicar desleet
                cleaned_text = clean_text(text)
                desleeted_text, modificaciones = desleet_text(cleaned_text)

                # PRINTS
                print(f"- Modificaciones leet: {modificaciones}")
                print("===========================")
                print(desleeted_text)
                print("===========================")
                
                ocr_results.append({
                    'bbox': adjusted_bbox,
                    'text': desleeted_text,
                    'conf': conf
                })

        if ocr_results:
            conf_media = np.mean([r['conf'] for r in ocr_results])
        else:
            conf_media = 0

        regions_ocr.append({
            'region': (x_min, y_min, x_max, y_max),
            'ocr_results': ocr_results,
            'conf_media': conf_media
        })

        # Rehacer la imagen dibujada con todas las regiones y texto
        img_drawn = draw_all_regions(img_base)
        cv2.putText(img_drawn, angulo_texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, img_drawn)


def desleet_text(text):
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
    
    # Dividir texto en tokens, manteniendo separadores
    tokens = re.findall(r'\S+|\s+', text)
    resultado = ""
    
    for token in tokens:
        # Si es espacio o solo separador, copiar tal cual
        if token.isspace():
            resultado += token
            continue
        
        # Comprobar si token es número con posible coma o punto decimal
        # Regex: solo dígitos, puntos o comas (ej: 1,01 o 1000 o 3.14)
        if re.fullmatch(r'[\d.,]+', token):
            # Es número, no modificar
            resultado += token
        else:
            # Token con letras y/o números, aplicar desleet caracter a caracter
            nuevo_token = ""
            for c in token:
                c_upper = c.upper()
                if c_upper in leet_dict:
                    nuevo_token += leet_dict[c_upper]
                    modificaciones_leet += 1
                else:
                    nuevo_token += c_upper
            resultado += nuevo_token

    # Heurística: contar caracteres sin espacios para comparación
    texto_sin_espacios = re.sub(r'\s+', '', text)
    longitud = len(texto_sin_espacios)
    
    if longitud > 0 and modificaciones_leet >= (longitud / 2):
        # Si se hicieron demasiadas modificaciones, probable error
        # Devuelve el texto original en mayúsculas, modificaciones 0
        return text.upper(), 0
    else:
        return resultado, modificaciones_leet

def detectar_lineas_hough(img, min_length=50, max_gap=10):
    """
    Detecta líneas rectas en una imagen usando la Transformada de Hough.
    """
    print('[DEBUG] Ejecutando detección de líneas con Hough...')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=100,
                            minLineLength=min_length,
                            maxLineGap=max_gap)
    if lines is None:
        print('[DEBUG] No se detectaron líneas.')
        return []
    print(f'[DEBUG] {len(lines)} líneas detectadas.')
    return lines[:, 0]


def rotar_imagen(img, angulo):
    """
    Rota una imagen el ángulo especificado.
    """
    print(f'[DEBUG] Rotando imagen {angulo:.2f} grados...')
    (h, w) = img.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

