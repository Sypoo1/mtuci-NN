#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Лабораторная работа №3: Обнаружение объектов с использованием Faster R-CNN
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.datasets import VOCDetection
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from tqdm import tqdm
import requests
import tarfile
import shutil
from pathlib import Path
import matplotlib.patches as patches

# Проверка доступности CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Загрузка предобученной модели Faster R-CNN
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model.to(device)
model.eval()

# Получение классов COCO
coco_classes = weights.meta["categories"]
print(f"Количество классов: {len(coco_classes)}")
print(f"Примеры классов: {coco_classes[:5]}")

# Трансформации для предобработки изображений
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Функция для загрузки датасета Pascal VOC 2007
def load_voc_dataset(root="./data"):
    # Создаем директорию, если она не существует
    os.makedirs(root, exist_ok=True)
    
    # Проверяем, существует ли датасет
    voc_path = os.path.join(root, "VOCdevkit", "VOC2007")
    if not os.path.exists(voc_path):
        print("Загрузка датасета Pascal VOC 2007...")
        
        # URL для загрузки тестового набора данных Pascal VOC 2007
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
        
        # Загрузка архива
        response = requests.get(url, stream=True)
        tar_path = os.path.join(root, "VOCtest_06-Nov-2007.tar")
        
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Распаковка архива
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=root)
        
        # Удаление архива
        os.remove(tar_path)
        print("Датасет успешно загружен и распакован.")
    else:
        print("Датасет Pascal VOC 2007 уже существует.")
    
    # Загрузка датасета с помощью torchvision
    voc_dataset = VOCDetection(root=root, year="2007", image_set="test", download=False, transform=None)
    print(f"Загружено {len(voc_dataset)} изображений из Pascal VOC 2007 test.")
    
    return voc_dataset

# Функция для парсинга аннотаций VOC
def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
    
    return np.array(boxes), labels

# Функция для отрисовки предсказаний на изображении
def draw_predictions(image, boxes, labels, scores=None, color="blue", width=2):
    draw = ImageDraw.Draw(image)
    
    # Попытка загрузить шрифт (если доступен)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Отрисовка прямоугольника
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)
        
        # Отрисовка метки и score (если доступен)
        label_text = labels[i]
        if scores is not None:
            label_text = f"{labels[i]} ({scores[i]:.2f})"
        
        # Отрисовка текста с фоном для лучшей видимости
        text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:]
        draw.rectangle([xmin, ymin, xmin + text_width, ymin + text_height], fill=color)
        draw.text((xmin, ymin), label_text, fill="white", font=font)
    
    return image

# Функция для расчета IoU (Intersection over Union)
def calculate_iou(pred_box, gt_box):
    x1, y1, x2, y2 = pred_box
    x1g, y1g, x2g, y2g = gt_box
    
    # Вычисление координат пересечения
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    # Вычисление площади пересечения
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Вычисление площадей боксов
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    # Вычисление площади объединения
    union_area = box1_area + box2_area - inter_area
    
    # Вычисление IoU
    return inter_area / union_area if union_area > 0 else 0

# Функция для сопоставления предсказанных и истинных боксов
def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    matches = []
    unmatched_pred = list(range(len(pred_boxes)))
    unmatched_gt = list(range(len(gt_boxes)))
    
    # Матрица IoU между всеми парами предсказанных и истинных боксов
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
    
    # Жадное сопоставление
    while len(unmatched_pred) > 0 and len(unmatched_gt) > 0:
        # Находим пару с максимальным IoU
        max_iou = 0
        max_i, max_j = -1, -1
        for i in unmatched_pred:
            for j in unmatched_gt:
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_i, max_j = i, j
        
        # Если IoU выше порога, считаем боксы совпадающими
        if max_iou >= iou_threshold:
            matches.append((max_i, max_j, max_iou))
            unmatched_pred.remove(max_i)
            unmatched_gt.remove(max_j)
        else:
            break
    
    return matches, unmatched_pred, unmatched_gt

# Функция для оценки модели на одном изображении
def evaluate_image(image_path, annotation_path, model, device, confidence_threshold=0.5, iou_threshold=0.5):
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Получение предсказаний модели
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Извлечение предсказанных боксов, меток и уверенностей
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()
    
    # Фильтрация предсказаний по порогу уверенности
    mask = pred_scores >= confidence_threshold
    filtered_boxes = pred_boxes[mask]
    filtered_labels = pred_labels[mask]
    filtered_scores = pred_scores[mask]
    
    # Преобразование индексов классов COCO в названия
    filtered_label_names = [coco_classes[label - 1] for label in filtered_labels]
    
    # Загрузка истинных боксов и меток
    gt_boxes, gt_label_names = parse_voc_annotation(annotation_path)
    
    # Сопоставление предсказанных и истинных боксов
    matches, unmatched_pred, unmatched_gt = match_boxes(filtered_boxes, gt_boxes, iou_threshold)
    
    # Подсчет TP, FP, FN
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    # Создание копий изображения для визуализации
    pred_image = image.copy()
    gt_image = image.copy()
    
    # Отрисовка предсказанных боксов
    pred_image = draw_predictions(pred_image, filtered_boxes, filtered_label_names, filtered_scores, color="red")
    
    # Отрисовка истинных боксов
    gt_image = draw_predictions(gt_image, gt_boxes, gt_label_names, color="green")
    
    # Отрисовка совпадающих, FP и FN боксов на одном изображении
    combined_image = image.copy()
    
    # Отрисовка совпадающих боксов (TP) зеленым
    for match in matches:
        i, j, _ = match
        combined_image = draw_predictions(combined_image, [filtered_boxes[i]], [filtered_label_names[i]], [filtered_scores[i]], color="green")
    
    # Отрисовка FP боксов красным
    fp_boxes = [filtered_boxes[i] for i in unmatched_pred]
    fp_labels = [filtered_label_names[i] for i in unmatched_pred]
    fp_scores = [filtered_scores[i] for i in unmatched_pred]
    if fp_boxes:
        combined_image = draw_predictions(combined_image, fp_boxes, fp_labels, fp_scores, color="red")
    
    # Отрисовка FN боксов синим
    fn_boxes = [gt_boxes[i] for i in unmatched_gt]
    fn_labels = [gt_label_names[i] for i in unmatched_gt]
    if fn_boxes:
        combined_image = draw_predictions(combined_image, fn_boxes, fn_labels, color="blue")
    
    return {
        'pred_image': pred_image,
        'gt_image': gt_image,
        'combined_image': combined_image,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'iou_scores': [iou for _, _, iou in matches] if matches else []
    }

# Функция для поиска оптимального порога уверенности
def find_optimal_threshold(dataset, model, device, num_images=5, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = {threshold: {'tp': 0, 'fp': 0, 'fn': 0} for threshold in thresholds}
    
    # Выбор случайных изображений
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for idx in tqdm(indices, desc="Оценка порогов"):
        img, annotation = dataset[idx]
        
        # Получение путей к изображению и аннотации
        img_id = annotation['annotation']['filename'].split('.')[0]
        img_path = os.path.join(dataset.root, 'VOCdevkit', 'VOC2007', 'JPEGImages', f"{img_id}.jpg")
        anno_path = os.path.join(dataset.root, 'VOCdevkit', 'VOC2007', 'Annotations', f"{img_id}.xml")
        
        # Оценка для каждого порога
        for threshold in thresholds:
            eval_result = evaluate_image(img_path, anno_path, model, device, confidence_threshold=threshold)
            results[threshold]['tp'] += eval_result['tp']
            results[threshold]['fp'] += eval_result['fp']
            results[threshold]['fn'] += eval_result['fn']
    
    # Расчет метрик для каждого порога
    metrics = {}
    for threshold in thresholds:
        tp = results[threshold]['tp']
        fp = results[threshold]['fp']
        fn = results[threshold]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return metrics

# Функция для визуализации результатов оценки порогов
def plot_threshold_metrics(metrics):
    thresholds = sorted(metrics.keys())
    precision = [metrics[t]['precision'] for t in thresholds]
    recall = [metrics[t]['recall'] for t in thresholds]
    f1 = [metrics[t]['f1'] for t in thresholds]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precision, 'b-', marker='o')
    plt.title('Precision vs Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, recall, 'r-', marker='o')
    plt.title('Recall vs Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, f1, 'g-', marker='o')
    plt.title('F1 Score vs Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(recall, precision, 'purple', marker='o')
    for i, t in enumerate(thresholds):
        plt.annotate(f"{t:.1f}", (recall[i], precision[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('threshold_metrics.png')
    plt.show()

# Функция для визуализации результатов детекции
def visualize_detection_results(results, image_names):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5*n))
    
    if n == 1:
        axes = [axes]
    
    for i, (result, name) in enumerate(zip(results, image_names)):
        axes[i][0].imshow(result['pred_image'])
        axes[i][0].set_title(f"{name} - Предсказания")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(result['gt_image'])
        axes[i][1].set_title(f"{name} - Истинные боксы")
        axes[i][1].axis('off')
        
        axes[i][2].imshow(result['combined_image'])
        axes[i][2].set_title(f"{name} - Комбинированное (TP=зеленый, FP=красный, FN=синий)")
        axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_results.png')
    plt.show()

# Основная функция для запуска оценки модели
def main():
    # Загрузка датасета
    voc_dataset = load_voc_dataset()
    
    # Выбор N изображений для оценки
    N = 5
    indices = np.random.choice(len(voc_dataset), N, replace=False)
    image_names = []
    results = []
    
    print(f"\nОценка модели на {N} случайных изображениях...")
    
    # Оценка модели на выбранных изображениях
    for idx in indices:
        img, annotation = voc_dataset[idx]
        
        # Получение путей к изображению и аннотации
        img_id = annotation['annotation']['filename'].split('.')[0]
        img_path = os.path.join(voc_dataset.root, 'VOCdevkit', 'VOC2007', 'JPEGImages', f"{img_id}.jpg")
        anno_path = os.path.join(voc_dataset.root, 'VOCdevkit', 'VOC2007', 'Annotations', f"{img_id}.xml")
        
        print(f"Обработка изображения {img_id}...")
        
        # Оценка изображения
        result = evaluate_image(img_path, anno_path, model, device)
        results.append(result)
        image_names.append(img_id)
        
        # Вывод метрик
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}")
        print(f"  Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}")
        if result['iou_scores']:
            print(f"  Средний IoU: {np.mean(result['iou_scores']):.4f}")
        print()
    
    # Визуализация результатов детекции
    visualize_detection_results(results, image_names)
    
    # Поиск оптимального порога уверенности
    print("\nПоиск оптимального порога уверенности...")
    thresholds = np.arange(0.1, 1.0, 0.1)
    metrics = find_optimal_threshold(voc_dataset, model, device, num_images=10, thresholds=thresholds)
    
    # Вывод метрик для каждого порога
    print("\nМетрики для разных порогов уверенности:")
    for threshold in sorted(metrics.keys()):
        m = metrics[threshold]
        print(f"Порог {threshold:.1f}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")
    
    # Визуализация метрик
    plot_threshold_metrics(metrics)
    
    # Определение оптимального порога по F1-мере
    best_threshold = max(metrics.keys(), key=lambda t: metrics[t]['f1'])
    print(f"\nОптимальный порог по F1-мере: {best_threshold:.1f}")
    print(f"Precision: {metrics[best_threshold]['precision']:.4f}")
    print(f"Recall: {metrics[best_threshold]['recall']:.4f}")
    print(f"F1 Score: {metrics[best_threshold]['f1']:.4f}")
    
    # Анализ баланса FP/FN
    print("\nАнализ баланса FP/FN:")
    for threshold in sorted(metrics.keys()):
        m = metrics[threshold]
        fp_fn_ratio = m['fp'] / m['fn'] if m['fn'] > 0 else float('inf')
        print(f"Порог {threshold:.1f}: FP/FN = {fp_fn_ratio:.2f} ({m['fp']}/{m['fn']})")

if __name__ == "__main__":
    main()