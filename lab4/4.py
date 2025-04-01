# Лабораторная работа №4
# Вариант 2: COCO 2017
# Тема: Сегментация объектов с использованием Mask R-CNN

# Блок 2.1 - Подготовка окружения
import os
import random
import warnings

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn)
from torchvision.transforms import functional as F
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Отобразить версии используемых библиотек
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Блок 2.2 - Подготовка модели
def load_model():
    # Определение устройства для вычислений
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Загрузка предобученной модели
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights, progress=True)
    model.to(device)
    model.eval()  # перевод модели в режим инференса

    return model, device, weights

model, device, weights = load_model()

# Блок 2.3 - Загрузка и предобработка изображений
def setup_data():
    # Пути к датасету COCO 2017
    # Для запуска в Google Colab может потребоваться загрузка:
    # !wget http://images.cocodataset.org/zips/val2017.zip
    # !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    # !unzip val2017.zip
    # !unzip annotations_trainval2017.zip

    # Если датасет уже скачан, укажите путь к нему
    dataDir = 'datasets/coco'  # Можно изменить на свой путь
    dataType = 'val2017'
    annFile = f'{dataDir}/annotations/instances_{dataType}.json'

    # Инициализация COCO API для аннотаций
    coco = COCO(annFile)

    # Получение категорий объектов
    categories = coco.loadCats(coco.getCatIds())
    category_names = [cat['name'] for cat in categories]
    print(f"COCO категории: {category_names}")

    # Загрузка CocoDetection dataset
    dataset = CocoDetection(
        root=f'{dataDir}/{dataType}',
        annFile=annFile,
        transform=weights.transforms()
    )

    return coco, dataset, category_names

coco, dataset, category_names = setup_data()

# Блок 2.4 - Предсказания и извлечение масок
def get_random_images(dataset, num_images=20):
    """Выбор случайных изображений из датасета"""
    indices = random.sample(range(len(dataset)), num_images)
    return indices

def get_ground_truth_masks(coco, img_id):
    """Получение ground truth масок из COCO аннотаций"""
    # Получение аннотаций для изображения
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    masks = []
    boxes = []
    classes = []

    for ann in anns:
        # Для каждой аннотации получаем маску
        mask = coco.annToMask(ann)
        masks.append(mask)

        # Получаем bounding box
        x, y, width, height = ann['bbox']
        boxes.append([x, y, x + width, y + height])

        # Получаем класс
        class_id = ann['category_id']
        cat = coco.loadCats([class_id])[0]
        classes.append(cat['name'])

    return masks, boxes, classes

def prepare_image_for_model(image_path):
    """Подготовка изображения для модели"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = weights.transforms()(image)
    return image, image_tensor

# Блок 2.5 - Оценка модели и визуализация результатов
def run_inference(model, image_tensor, device, confidence_threshold=0.5):
    """Запуск инференса модели"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])
        predictions = predictions[0]

    # Отфильтровать предсказания по confidence score
    keep = torch.where(predictions['scores'] > confidence_threshold)[0]

    filtered_predictions = {
        'boxes': predictions['boxes'][keep].cpu(),
        'labels': predictions['labels'][keep].cpu(),
        'scores': predictions['scores'][keep].cpu(),
        'masks': predictions['masks'][keep].cpu()
    }

    return filtered_predictions

def apply_mask_threshold(mask, threshold=0.5):
    """Применение порога к маске"""
    return mask > threshold

def calculate_iou(mask1, mask2):
    """Расчет IoU между двумя масками"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def calculate_metrics(pred_mask, gt_mask):
    """Расчет метрик качества масок: IoU, Precision, Recall"""
    # Переход к бинарным маскам
    pred_mask_binary = pred_mask > 0
    gt_mask_binary = gt_mask > 0

    # Подсчет TP, FP, FN
    true_positive = np.logical_and(pred_mask_binary, gt_mask_binary).sum()
    false_positive = np.logical_and(pred_mask_binary, np.logical_not(gt_mask_binary)).sum()
    false_negative = np.logical_and(np.logical_not(pred_mask_binary), gt_mask_binary).sum()

    # Расчет IoU
    iou = calculate_iou(pred_mask_binary, gt_mask_binary)

    # Расчет Precision и Recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'fp_pixels': int(false_positive),
        'fn_pixels': int(false_negative)
    }

def visualize_results(image, predictions, gt_masks=None, gt_boxes=None, gt_classes=None,
                     mask_threshold=0.5, class_names=None):
    """Визуализация результатов предсказания и ground truth"""
    plt.figure(figsize=(16, 10))

    # Преобразуем изображение в NumPy массив для отображения
    img_np = np.array(image)

    # Отображаем исходное изображение
    plt.imshow(img_np)

    # Создаем цвета для разных классов
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Визуализация предсказаний
    for i in range(len(predictions['boxes'])):
        box = predictions['boxes'][i].numpy()
        label = predictions['labels'][i].item()
        score = predictions['scores'][i].item()
        mask = apply_mask_threshold(predictions['masks'][i, 0].numpy(), mask_threshold)

        # Получение имени класса
        class_name = class_names[label-1] if class_names else f"Class {label}"

        # Добавляем цветную маску поверх изображения
        color_mask = np.zeros_like(img_np)
        color_idx = i % len(colors)
        color = colors[color_idx]

        # Создаем RGB маску с прозрачностью
        for c in range(3):
            color_mask[:, :, c] = np.where(mask,
                                         mcolors.to_rgb(color)[c] * 255 * 0.5,
                                         img_np[:, :, c])

        # Отображаем маску с прозрачностью
        plt.imshow(np.where(mask[:, :, np.newaxis], color_mask, img_np).astype(np.uint8))

        # Рисуем bounding box
        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

        # Добавляем метку с классом и confidence score
        plt.text(box[0], box[1] - 5, f"{class_name}: {score:.2f}",
                color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.8))

    # Визуализация ground truth, если предоставлена
    if gt_masks and gt_boxes and gt_classes:
        for i, (mask, box, cls) in enumerate(zip(gt_masks, gt_boxes, gt_classes)):
            # Рисуем ground truth bounding box пунктирной линией
            color_idx = i % len(colors)
            color = colors[color_idx]

            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
            plt.gca().add_patch(rect)

            # Добавляем метку для ground truth
            plt.text(box[0], box[1] - 20, f"GT: {cls}",
                   color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.5))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def eval_image(img_idx, dataset, coco, model, device, mask_threshold=0.5, confidence_threshold=0.5):
    """Оценка модели на одном изображении и визуализация результатов"""
    # Получаем изображение и его аннотации
    img, _ = dataset[img_idx]
    img_path = dataset.root + '/' + dataset.ids[img_idx] + '.jpg'  # Корректируется в зависимости от структуры датасета
    img_id = int(dataset.ids[img_idx])

    original_image = Image.open(img_path).convert("RGB")

    # Получение ground truth масок
    gt_masks, gt_boxes, gt_classes = get_ground_truth_masks(coco, img_id)

    # Запуск инференса
    predictions = run_inference(model, img, device, confidence_threshold)

    # Визуализация результатов
    visualize_results(original_image, predictions, gt_masks, gt_boxes, gt_classes,
                     mask_threshold, class_names=category_names)

    # Расчет метрик для каждой предсказанной маски
    metrics_results = []

    for i, pred_mask in enumerate(predictions['masks']):
        pred_mask_np = apply_mask_threshold(pred_mask[0].numpy(), mask_threshold)
        pred_label = predictions['labels'][i].item()
        pred_class = category_names[pred_label-1]

        # Ищем соответствующую ground truth маску
        best_iou = 0
        best_gt_idx = -1

        for j, gt_mask in enumerate(gt_masks):
            if gt_classes[j] == pred_class:  # Сравниваем маски одного класса
                iou = calculate_iou(pred_mask_np, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

        # Расчет метрик, если нашли подходящую ground truth маску
        if best_gt_idx >= 0:
            metrics = calculate_metrics(pred_mask_np, gt_masks[best_gt_idx])
            metrics['class'] = pred_class
            metrics['score'] = predictions['scores'][i].item()
            metrics_results.append(metrics)

    return metrics_results

# Блок 2.6 - Исследование параметров маски и confidence
def compare_thresholds():
    """Исследование влияния порогов на качество сегментации"""
    # Выбираем случайные изображения
    num_images = 20
    indices = get_random_images(dataset, num_images)

    # Параметры для исследования
    mask_thresholds = [0.3, 0.5, 0.7]
    confidence_thresholds = [0.3, 0.5, 0.7, 0.9]

    # Результаты для разных порогов
    results = {}

    for mask_th in mask_thresholds:
        for conf_th in confidence_thresholds:
            key = f"mask_{mask_th}_conf_{conf_th}"
            results[key] = {
                'iou': [],
                'precision': [],
                'recall': []
            }

    # Обработка каждого изображения
    for idx in tqdm(indices, desc="Обработка изображений"):
        img, _ = dataset[idx]
        img_id = int(dataset.ids[idx])

        # Получаем ground truth
        gt_masks, _, gt_classes = get_ground_truth_masks(coco, img_id)

        for mask_th in mask_thresholds:
            for conf_th in confidence_thresholds:
                key = f"mask_{mask_th}_conf_{conf_th}"

                # Запуск инференса с текущими порогами
                predictions = run_inference(model, img, device, conf_th)

                # Расчет метрик для каждой предсказанной маски
                for i, pred_mask in enumerate(predictions['masks']):
                    pred_mask_np = apply_mask_threshold(pred_mask[0].numpy(), mask_th)
                    pred_label = predictions['labels'][i].item()
                    pred_class = category_names[pred_label-1]

                    # Поиск лучшей ground truth маски
                    best_metrics = {'iou': 0, 'precision': 0, 'recall': 0}

                    for j, gt_mask in enumerate(gt_masks):
                        if gt_classes[j] == pred_class:
                            metrics = calculate_metrics(pred_mask_np, gt_mask)
                            if metrics['iou'] > best_metrics['iou']:
                                best_metrics = metrics

                    # Сохраняем метрики
                    if best_metrics['iou'] > 0:  # Если нашли соответствие
                        results[key]['iou'].append(best_metrics['iou'])
                        results[key]['precision'].append(best_metrics['precision'])
                        results[key]['recall'].append(best_metrics['recall'])

    # Расчет средних значений
    average_results = {}
    for key, values in results.items():
        average_results[key] = {
            'avg_iou': np.mean(values['iou']) if values['iou'] else 0,
            'avg_precision': np.mean(values['precision']) if values['precision'] else 0,
            'avg_recall': np.mean(values['recall']) if values['recall'] else 0,
            'f1_score': 2 * np.mean(values['precision']) * np.mean(values['recall']) /
                      (np.mean(values['precision']) + np.mean(values['recall']))
                      if (values['precision'] and values['recall'] and
                         (np.mean(values['precision']) + np.mean(values['recall'])) > 0) else 0
        }

    # Визуализация результатов
    plt.figure(figsize=(15, 10))

    metrics = ['avg_iou', 'avg_precision', 'avg_recall', 'f1_score']
    colors = ['blue', 'green', 'red', 'purple']

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)

        x_labels = []
        y_values = []

        for key, vals in sorted(average_results.items()):
            x_labels.append(key)
            y_values.append(vals[metric])

        plt.bar(range(len(x_labels)), y_values, color=colors[i])
        plt.xticks(range(len(x_labels)), x_labels, rotation=90)
        plt.title(metric)
        plt.ylim([0, 1])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return average_results

# Демонстрационный запуск
def main():
    print("Запуск инференса на случайных изображениях из датасета COCO 2017...")

    # Выбор 20 случайных изображений
    indices = get_random_images(dataset, 20)

    # Обработка 5 изображений для быстрой демонстрации
    all_metrics = []
    for i, idx in enumerate(indices[:5]):
        print(f"\nИзображение {i+1}/5:")
        metrics = eval_image(idx, dataset, coco, model, device)
        all_metrics.extend(metrics)

    # Вывод средних метрик
    avg_iou = np.mean([m['iou'] for m in all_metrics]) if all_metrics else 0
    avg_precision = np.mean([m['precision'] for m in all_metrics]) if all_metrics else 0
    avg_recall = np.mean([m['recall'] for m in all_metrics]) if all_metrics else 0

    print("\nСредние метрики для 5 изображений:")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")

    # Исследование порогов
    print("\nИсследование влияния порогов mask_threshold и confidence_threshold...")
    average_results = compare_thresholds()

    # Вывод оптимальных параметров
    best_f1 = 0
    best_config = ""

    for key, vals in average_results.items():
        if vals['f1_score'] > best_f1:
            best_f1 = vals['f1_score']
            best_config = key

    print(f"\nОптимальные параметры (по F1-score): {best_config}")
    print(f"F1-score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
