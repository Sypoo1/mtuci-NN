#!/usr/bin/env python
# coding: utf-8

import os
import random
import xml.etree.ElementTree as ET

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
# Раздел 2.1: Подготовка окружения
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)

# Раздел 2.2: Подготовка модели
# Определение устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используемое устройство: {device}')

# Загрузка предобученной модели Faster R-CNN
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()  # Перевод модели в режим инференса

# Получение списка классов, которые может распознавать модель
coco_classes = weights.meta["categories"]
print(f'Количество классов: {len(coco_classes)}')
print(f'Примеры классов: {coco_classes[:10]}')

# Раздел 2.3: Загрузка и предобработка изображений
# Загрузка датасета Pascal VOC 2007
voc_dataset = VOCDetection(
    root='./data',
    year='2007',
    image_set='test',
    download=True,
    transform=None
)

print(f'Размер датасета VOC: {len(voc_dataset)} изображений')

# Определение трансформаций для изображений
def get_transform():
    transforms_list = []
    # Преобразование в тензор
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)

# Раздел 2.4: Объявление методов для работы с данными
def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (int(bbox.find("xmin").text), int(bbox.find("ymin").text),
                                  int(bbox.find("xmax").text), int(bbox.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)

def draw_predictions(image, boxes, labels, scores, color="blue"):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = map(int, box)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        draw.text((xmin, ymin), f"{label} ({score:.2f})", fill=color)
    return image

# Раздел 2.5: Анализ False Positives / False Negatives
def calculate_iou(pred_box, gt_box):
    x1, y1, x2, y2 = pred_box
    x1g, y1g, x2g, y2g = gt_box

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Раздел 2.6: Оценка модели и визуализация результатов
def evaluate_model(confidence_threshold=0.5, N=5, display=True):
    # Выбор N случайных изображений из датасета
    indices = random.sample(range(len(voc_dataset)), N)

    # Статистика по FP/FN
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_gt_boxes = 0
    iou_threshold = 0.5  # Порог IoU для определения TP

    fig, axs = plt.subplots(N, 2, figsize=(15, 5*N))

    for i, idx in enumerate(indices):
        # Загрузка изображения и аннотации
        img, annotation = voc_dataset[idx]
        filename = annotation['annotation']['filename']

        # Получение путей к файлам изображения и аннотации
        img_path = os.path.join('data/VOCdevkit/VOC2007/JPEGImages', filename)
        xml_path = os.path.join('data/VOCdevkit/VOC2007/Annotations', os.path.splitext(filename)[0] + '.xml')

        # Загрузка изображения и извлечение bbox из аннотаций
        original_img = Image.open(img_path).convert("RGB")
        gt_boxes = parse_voc_annotation(xml_path)
        total_gt_boxes += len(gt_boxes)

        # Применение трансформаций и перевод на устройство
        transform = get_transform()
        img_tensor = transform(original_img).unsqueeze(0).to(device)

        # Выполнение инференса
        with torch.no_grad():
            outputs = model(img_tensor)

        # Извлечение предсказаний модели
        pred_boxes = outputs[0]['boxes'].cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].cpu().numpy()

        # Фильтрация по порогу уверенности
        mask = pred_scores >= confidence_threshold
        filtered_boxes = pred_boxes[mask]
        filtered_labels = pred_labels[mask]
        filtered_scores = pred_scores[mask]

        # Преобразование числовых меток в названия классов
        filtered_class_names = [coco_classes[label - 1] for label in filtered_labels]

        # Отрисовка предсказаний на копии изображения
        pred_img = original_img.copy()
        pred_img = draw_predictions(pred_img, filtered_boxes, filtered_class_names, filtered_scores, color="red")

        # Отрисовка ground truth на другой копии изображения
        gt_img = original_img.copy()
        gt_class_names = ["ground_truth"] * len(gt_boxes)
        gt_scores = [1.0] * len(gt_boxes)
        gt_img = draw_predictions(gt_img, gt_boxes, gt_class_names, gt_scores, color="green")

        # Подсчет метрик TP, FP, FN
        tp = 0
        assigned_gt = set()

        for pred_box in filtered_boxes:
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if j in assigned_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                tp += 1
                assigned_gt.add(best_gt_idx)
            else:
                total_false_positives += 1

        total_true_positives += tp
        total_false_negatives += (len(gt_boxes) - len(assigned_gt))

        if display:
            # Отображение изображений с предсказаниями и ground truth
            axs[i, 0].imshow(np.array(gt_img))
            axs[i, 0].set_title(f'Ground Truth - {filename}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(np.array(pred_img))
            axs[i, 1].set_title(f'Predictions (threshold={confidence_threshold})')
            axs[i, 1].axis('off')

    if display:
        plt.tight_layout()
        plt.show()

    print(f"Результаты при пороге confidence score = {confidence_threshold}:")
    print(f"True Positives: {total_true_positives}")
    print(f"False Positives: {total_false_positives}")
    print(f"False Negatives: {total_false_negatives}")
    print(f"Всего GT боксов: {total_gt_boxes}")

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return precision, recall, f1

# Запуск оценки модели
print("Оценка модели при пороге confidence score = 0.5:")
evaluate_model(confidence_threshold=0.5, N=5)

# Раздел 2.7: Поиск оптимальной конфигурации
def find_optimal_threshold(n_samples=30):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for threshold in thresholds:
        print(f"\nТестирование порога: {threshold}")
        precision, recall, f1 = evaluate_model(confidence_threshold=threshold, N=n_samples, display=False)
        results.append((threshold, precision, recall, f1))

    # Отобразить результаты
    thresholds, precisions, recalls, f1_scores = zip(*results)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'r-', label='Recall')
    plt.plot(thresholds, f1_scores, 'g-', label='F1-score')
    plt.xlabel('Порог уверенности (Confidence Threshold)')
    plt.ylabel('Значение метрики')
    plt.title('Зависимость метрик от порога уверенности')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Найти порог с лучшим F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\nОптимальный порог уверенности: {best_threshold}")
    print(f"При этом пороге F1-score: {best_f1:.4f}")
    print(f"Precision: {precisions[best_idx]:.4f}")
    print(f"Recall: {recalls[best_idx]:.4f}")

    # Отобразить баланс между FP и FN
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'bo-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall кривая')
    for i, threshold in enumerate(thresholds):
        plt.annotate(f"{threshold}", (recalls[i], precisions[i]))
    plt.grid(True)
    plt.show()

    return best_threshold

# Запуск исследования для поиска оптимального порога
print("\nПоиск оптимального порога уверенности:")
optimal_threshold = find_optimal_threshold(n_samples=30)

# Демонстрация работы модели с оптимальным порогом
print("\nДемонстрация работы модели с оптимальным порогом:")
evaluate_model(confidence_threshold=optimal_threshold, N=5)

if __name__ == "__main__":
    print("Выполнение программы завершено!")