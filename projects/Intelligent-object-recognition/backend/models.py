import os
import threading
import io
import logging
import base64
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2

from .config import (
    DIGIT_MODEL_PATH,
    IMG_SIZE_GENERAL, IMG_SIZE_DIGIT,
    ANIMAL_KEYWORDS, SCENE_KEYWORDS
)

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import models as tf_models

_pytorch_model = None
_imagenet_classes = None
_digit_model = None
_yolo_model = None

_models_loaded = False
_models_loading = False
_lock = threading.Lock()

_torch_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_HARDCODED_IMAGENET_CLASSES = None


def _get_hardcoded_classes():
    global _HARDCODED_IMAGENET_CLASSES
    if _HARDCODED_IMAGENET_CLASSES is None:
        try:
            url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
            import urllib.request
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as f:
                content = f.read().decode('utf-8')
            _HARDCODED_IMAGENET_CLASSES = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception:
            _HARDCODED_IMAGENET_CLASSES = [f'class_{i}' for i in range(1000)]
    return _HARDCODED_IMAGENET_CLASSES


def _load_pytorch_model():
    global _pytorch_model, _imagenet_classes
    if _pytorch_model is None:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        _pytorch_model = models.mobilenet_v2(weights=weights)
        _pytorch_model.eval()
        _imagenet_classes = weights.meta.get('categories', None)
        if not _imagenet_classes:
            _imagenet_classes = _get_hardcoded_classes()
    return _pytorch_model


def _create_digit_model():
    model = tf_models.Sequential([
        tf_layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf_layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        tf_layers.MaxPooling2D((2, 2)),
        tf_layers.Dropout(0.25),

        tf_layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf_layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf_layers.MaxPooling2D((2, 2)),
        tf_layers.Dropout(0.25),

        tf_layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf_layers.Flatten(),
        tf_layers.Dense(256, activation='relu'),
        tf_layers.Dropout(0.5),
        tf_layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            _yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            logging.getLogger('objrec').warning('Failed to load YOLO model: %s', e)
    return _yolo_model


def _load_digit_model():
    global _digit_model
    if _digit_model is None:
        if os.path.exists(DIGIT_MODEL_PATH):
            try:
                _digit_model = keras.models.load_model(DIGIT_MODEL_PATH)
            except Exception as e:
                logging.getLogger('objrec').warning('Failed to load digit model: %s', e)
    return _digit_model


def get_model_status():
    return {
        'loaded': _models_loaded,
        'loading': _models_loading,
        'pytorch': _pytorch_model is not None,
        'digit': _digit_model is not None,
        'yolo': _yolo_model is not None,
    }


def preload_models():
    global _models_loading, _models_loaded
    with _lock:
        if _models_loading or _models_loaded:
            return
        _models_loading = True

    def _load():
        global _models_loaded, _models_loading
        try:
            _load_pytorch_model()
            _load_digit_model()
            _load_yolo_model()
            _models_loaded = True
        except Exception as e:
            logging.getLogger('objrec').warning('Model preloading partially failed: %s', e)
        finally:
            _models_loading = False

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()


def _preprocess_pytorch(image_path):
    img = Image.open(image_path).convert('RGB')
    return _torch_transform(img).unsqueeze(0)


def _preprocess_digit(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize(IMG_SIZE_DIGIT, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=(0, -1))


def _segment_digits(img_array):
    img_uint8 = (np.clip(img_array, 0, 255)).astype(np.uint8)

    border = 5
    img_uint8 = cv2.copyMakeBorder(img_uint8, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=0)

    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    binary = cv2.dilate(binary, kernel_cross, iterations=1)

    kernel_close = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    regions = []
    h, w = img_array.shape
    total_area = h * w
    min_area = max(12, total_area * 0.0002)

    for i in range(1, num_labels):
        x, y, b_w, b_h, area = stats[i]
        ratio = max(b_w, b_h) / max(min(b_w, b_h), 1)
        if area < min_area or b_w < 3 or b_h < 3 or b_w > (w + border * 2) * 0.95 or ratio > 8:
            continue

        pad = 2
        x1 = max(0, x - pad - border)
        x2 = min(w, x + b_w + pad - border)
        y1 = max(0, y - pad - border)
        y2 = min(h, y + b_h + pad - border)
        if x2 > x1 and y2 > y1:
            regions.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'area': area})

    if not regions:
        return []

    regions.sort(key=lambda r: r['x1'])

    merged = []
    for r in regions:
        if merged and r['x1'] - merged[-1]['x2'] < 2:
            merged[-1]['x2'] = max(merged[-1]['x2'], r['x2'])
            merged[-1]['y1'] = min(merged[-1]['y1'], r['y1'])
            merged[-1]['y2'] = max(merged[-1]['y2'], r['y2'])
        else:
            merged.append(r)

    return merged[:20]


def _classify_digit_region(img_array, model):
    h, w = img_array.shape
    if h < 3 or w < 3:
        return '', 0.0

    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    sq[y_off:y_off + h, x_off:x_off + w] = img_array

    pad = max(3, side // 8)
    size = side + pad * 2
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[pad:pad + side, pad:pad + side] = sq

    pil_img = Image.fromarray((np.clip(canvas * 255, 0, 255)).astype(np.uint8))
    pil_img = pil_img.resize(IMG_SIZE_DIGIT, Image.LANCZOS)
    digit_array = np.array(pil_img, dtype=np.float32) / 255.0
    digit_input = np.expand_dims(digit_array, axis=(0, -1))

    preds = model.predict(digit_input, verbose=0)

    aug_inputs = [digit_input]
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    aug_gen = ImageDataGenerator(
        rotation_range=8, width_shift_range=0.08,
        height_shift_range=0.08, zoom_range=0.08, fill_mode='nearest'
    )
    aug_iter = aug_gen.flow(np.expand_dims(digit_array, (0, -1)), np.zeros(1), batch_size=1)
    for _ in range(6):
        aug_batch, _ = next(aug_iter)
        aug_inputs.append(aug_batch)

    all_preds = []
    for inp in aug_inputs:
        all_preds.append(model.predict(inp, verbose=0)[0])

    avg_preds = np.mean(all_preds, axis=0)
    idx = int(np.argmax(avg_preds))
    return str(idx), float(avg_preds[idx])


def _filter_predictions(probs, keywords_list):
    _load_pytorch_model()
    results = []
    for idx in range(len(probs)):
        label = _imagenet_classes[idx] if idx < len(_imagenet_classes) else f'class_{idx}'
        label_lower = label.lower().replace('-', ' ').replace('_', ' ')
        for keyword in keywords_list:
            if keyword.lower() in label_lower:
                results.append({
                    'label': label.replace('_', ' ').title(),
                    'confidence': float(probs[idx])
                })
                break
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:10]


def _predict_mobilenet(image_path, top_k=5):
    model = _load_pytorch_model()
    img = _preprocess_pytorch(image_path)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    topk_prob, topk_idx = torch.topk(probs, min(top_k, len(probs)))
    results = []
    for i in range(len(topk_idx)):
        idx = topk_idx[i].item()
        label = _imagenet_classes[idx] if idx < len(_imagenet_classes) else f'class_{idx}'
        results.append({
            'label': label.replace('_', ' ').title(),
            'confidence': float(topk_prob[i].item())
        })
    return results


def _draw_boxes(image_path, detections):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    colors = ['#6366f1', '#06b6d4', '#f59e0b', '#10b981', '#ef4444',
              '#8b5cf6', '#ec4899', '#f97316', '#14b8a6', '#e11d48']
    for i, det in enumerate(detections):
        box = det['box']
        label = det['label']
        conf = det['confidence']
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f'{label} {conf:.0%}'
        try:
            font = ImageFont.truetype('arial.ttf', 16)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((x1, y1 - 20), tag, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 20), tag, fill='white', font=font)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def predict_general(image_path):
    try:
        yolo = _load_yolo_model()
        results = yolo(image_path, verbose=False)[0]
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()

            detections = []
            for i in range(len(confs)):
                if confs[i] < 0.3:
                    continue
                label = results.names.get(classes[i], 'object')
                detections.append({
                    'label': label,
                    'confidence': float(confs[i]),
                    'box': [float(v) for v in xyxy[i]]
                })

            if len(detections) == 1:
                mobilenet_results = _predict_mobilenet(image_path)
                mobile_top = mobilenet_results[0]
                img_b64 = _draw_boxes(image_path, detections)
                detections[0]['mobilenet_top'] = mobile_top
                return {
                    'type': 'single',
                    'results': mobilenet_results,
                    'top_label': mobile_top['label'],
                    'top_confidence': mobile_top['confidence'],
                    'objects': detections,
                    'annotated_image': img_b64
                }

            if len(detections) > 1:
                img_b64 = _draw_boxes(image_path, detections)
                dets_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                result_list = [{'label': d['label'], 'confidence': d['confidence'],
                               'box': d['box']} for d in dets_sorted]
                return {
                    'type': 'multi',
                    'results': result_list,
                    'top_label': ', '.join(d['label'] for d in dets_sorted[:5]),
                    'top_confidence': dets_sorted[0]['confidence'],
                    'objects': detections,
                    'annotated_image': img_b64
                }
    except Exception:
        logging.getLogger('objrec').warning('YOLO detection failed, falling back to MobileNet', exc_info=True)

    results = _predict_mobilenet(image_path)
    return {
        'type': 'single',
        'results': results,
        'top_label': results[0]['label'],
        'top_confidence': results[0]['confidence']
    }


def _group_rows(regions):
    if len(regions) <= 1:
        return [regions]

    regions = sorted(regions, key=lambda r: r['y1'])

    rows = []
    for r in regions:
        placed = False
        for row in rows:
            for existing in row:
                if r['y1'] <= existing['y2'] and existing['y1'] <= r['y2']:
                    row.append(r)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rows.append([r])

    result = []
    for row in rows:
        row.sort(key=lambda r: r['x1'])
        result.append(row)

    return result


def _enhance_digit(img_array):
    h, w = img_array.shape
    if max(h, w) > 200 or min(h, w) < 5:
        return img_array
    p_low = np.percentile(img_array, 5)
    p_high = np.percentile(img_array, 95)
    if p_high - p_low < 0.05:
        return img_array
    enhanced = np.clip((img_array - p_low) / (p_high - p_low), 0, 1)
    return enhanced


def predict_digit(image_path):
    model = _load_digit_model()
    if model is None:
        raise RuntimeError('Digit model not trained. Please run train_digits.py first.')

    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    if np.mean(img_array) > 128:
        img_array = 255.0 - img_array

    regions = _segment_digits(img_array)

    if len(regions) <= 1:
        normalized = img_array / 255.0
        if len(regions) == 1:
            r = regions[0]
            normalized = img_array[r['y1']:r['y2'], r['x1']:r['x2']] / 255.0
        pil_img = Image.fromarray((np.clip(normalized * 255, 0, 255)).astype(np.uint8))
        pil_img = pil_img.resize(IMG_SIZE_DIGIT, Image.LANCZOS)
        digit_input = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, axis=(0, -1))
        preds = model.predict(digit_input, verbose=0)
        results = []
        for i, conf in enumerate(preds[0]):
            results.append({'label': str(i), 'confidence': float(conf)})
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results, None

    rows = _group_rows(regions)
    is_multi_row = len(rows) > 1

    if is_multi_row:
        all_strings = []
        all_details = []
        for row_idx, row in enumerate(rows):
            row_str = ''
            for r in row:
                region_img = img_array[r['y1']:r['y2'], r['x1']:r['x2']] / 255.0
                region_img = _enhance_digit(region_img)
                label, conf = _classify_digit_region(region_img, model)
                row_str += label
                all_details.append({'digit': label, 'confidence': conf, 'row': row_idx + 1})
            all_strings.append(row_str)

        digit_string = '\n'.join(all_strings)
    else:
        row = rows[0]
        digit_string = ''
        all_details = []
        for r in row:
            region_img = img_array[r['y1']:r['y2'], r['x1']:r['x2']] / 255.0
            region_img = _enhance_digit(region_img)
            label, conf = _classify_digit_region(region_img, model)
            digit_string += label
            all_details.append({'digit': label, 'confidence': conf})

    if not digit_string:
        normalized = img_array / 255.0
        pil_img = Image.fromarray((normalized * 255).astype(np.uint8))
        pil_img = pil_img.resize(IMG_SIZE_DIGIT, Image.LANCZOS)
        digit_input = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, axis=(0, -1))
        preds = model.predict(digit_input, verbose=0)
        results = [{'label': str(i), 'confidence': float(conf)} for i, conf in enumerate(preds[0])]
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results, None

    results = [{'label': digit_string, 'confidence': float(np.mean([d['confidence'] for d in all_details]))}]
    for detail in all_details:
        results.append({'label': detail['digit'], 'confidence': detail['confidence']})

    extra = {'string': digit_string, 'details': all_details}
    if is_multi_row:
        extra['multi_row'] = True
    return results, extra


def predict_animal(image_path):
    model = _load_pytorch_model()
    img = _preprocess_pytorch(image_path)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return _filter_predictions(probs.cpu().numpy(), ANIMAL_KEYWORDS)


def predict_scene(image_path):
    model = _load_pytorch_model()
    img = _preprocess_pytorch(image_path)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    return _filter_predictions(probs.cpu().numpy(), SCENE_KEYWORDS)


MODES = {
    'general': {
        'name': '通用物体识别',
        'description': 'YOLO自动检测单/多物体 + MobileNetV2精细分类，支持目标框选',
        'icon': 'globe',
        'predict_fn': predict_general
    },
    'digit': {
        'name': '数字识别',
        'description': '识别手写数字串，自动分割图片中的多个数字',
        'icon': 'calculator',
        'predict_fn': predict_digit
    },
    'animal': {
        'name': '动物识别',
        'description': '识别各类动物，包括哺乳动物、鸟类、鱼类、昆虫等',
        'icon': 'paw',
        'predict_fn': predict_animal
    },
    'scene': {
        'name': '场景识别',
        'description': '识别自然与城市场景，如山水、建筑、街景等',
        'icon': 'image',
        'predict_fn': predict_scene
    }
}


def get_available_modes():
    modes_info = []
    for mode_id, mode_data in MODES.items():
        info = {
            'id': mode_id,
            'name': mode_data['name'],
            'description': mode_data['description'],
            'icon': mode_data['icon']
        }
        if mode_id == 'digit' and _load_digit_model() is None:
            info['available'] = False
            info['hint'] = '请先运行训练脚本 train_digits.py'
        else:
            info['available'] = True
        modes_info.append(info)
    return modes_info


def load_all_models():
    print('  加载通用识别模型 (PyTorch MobileNetV2)...')
    _load_pytorch_model()
    print('  加载目标检测模型 (YOLOv8)...')
    _load_yolo_model()
    print('  加载数字识别模型...')
    try:
        model = _load_digit_model()
        if model:
            print('  数字识别模型加载成功')
        else:
            print('  数字识别模型未找到，请先运行 train_digits.py')
    except Exception as e:
        print(f'  数字识别模型加载失败: {e}')
