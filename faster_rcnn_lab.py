# ----------------- 2.1 Подготовка окружения -----------------
# импорт пакетов
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

# ----------------- 2.2 Подготовка модели -----------------
# импорт модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

# Загрузка предобученной модели
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model.to(device)
model.eval()

# Получаем классы COCO для интерпретации результатов
coco_classes = weights.meta["categories"]
print(f"Количество классов: {len(coco_classes)}")


# ----------------- 2.3 Загрузка и предобработка изображений -----------------
# загрузка датасета
voc_dataset = VOCDetection(root='./data', year='2007', image_set='test', download=True)
print(f"Размер датасета: {len(voc_dataset)}")

# метод трансформации
transform = transforms.Compose([
    transforms.ToTensor(),
])


import xml.etree.ElementTree as ET

# ----------------- 2.4 Объявление методов для работы с данными -----------------
from PIL import Image, ImageDraw


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


# ----------------- 2.5 Анализ False Positives / False Negatives -----------------
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


# ----------------- 2.6 Оценка модели и визуализация результатов -----------------
# выберите N изображений 5 из датасета
N = 5
image_filenames = [voc_dataset[i][0] for i in range(N)]

# объявите цикл для проверки
results = []
iou_threshold = 0.5

for i, img_path in enumerate(image_filenames):
    # загрузите изображение и его разметку
    image, annotation = voc_dataset[i]
    xml_path = os.path.join('./data/VOCdevkit/VOC2007/Annotations', annotation['annotation']['filename'].split('.')[0] + '.xml')
    gt_boxes = parse_voc_annotation(xml_path)

    # примените трансформации
    img_tensor = transform(image).unsqueeze(0).to(device)

    # выполните прямой проход
    with torch.no_grad():
        outputs = model(img_tensor)

    # извлекаем bounding boxes, метки и confidence scores из полученных выходов модели
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()

    # установите порог confidence_threshold в эмпирическом значении
    confidence_threshold = 0.5

    # примените фильтр выходных значений на основе заданного порога
    mask = pred_scores >= confidence_threshold
    filtered_boxes = pred_boxes[mask]
    filtered_labels = pred_labels[mask]
    filtered_scores = pred_scores[mask]

    # отрисуйте bounding boxes на изображении с предсказаниями
    pred_img = image.copy()
    pred_img = draw_predictions(pred_img, filtered_boxes,
                              [coco_classes[label-1] for label in filtered_labels],
                              filtered_scores, color="blue")

    # отрисуйте bounding boxes с реальной разметкой
    gt_img = image.copy()
    gt_img = draw_predictions(gt_img, gt_boxes,
                            ["ground truth"] * len(gt_boxes),
                            [1.0] * len(gt_boxes), color="red")

    # вычислите IoU для выбранных изображений
    ious = []
    matched_gt_indices = set()

    for pred_box in filtered_boxes:
        max_iou = 0
        max_iou_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = gt_idx

        if max_iou >= iou_threshold:
            matched_gt_indices.add(max_iou_idx)

        ious.append(max_iou)

    # подсчитайте количество False Positive и False Negative для выбранных изображений
    false_positives = sum(1 for iou in ious if iou < iou_threshold)
    false_negatives = len(gt_boxes) - len(matched_gt_indices)

    # Сохранение результатов
    results.append({
        'image': annotation['annotation']['filename'],
        'pred_boxes': filtered_boxes,
        'gt_boxes': gt_boxes,
        'ious': ious,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'pred_image': pred_img,
        'gt_image': gt_img
    })

    # Вывод результатов для текущего изображения
    print(f"Изображение {i+1}: {annotation['annotation']['filename']}")
    print(f"  Количество предсказаний: {len(filtered_boxes)}")
    print(f"  Количество ground truth: {len(gt_boxes)}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Средний IoU: {np.mean(ious) if ious else 0:.4f}")
    print("-----------------------------")

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pred_img)
    plt.title(f"Предсказания (FP={false_positives})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gt_img)
    plt.title(f"Ground Truth (FN={false_negatives})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ----------------- 2.7 Поиск оптимальной конфигурации -----------------
# исследование
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fp_fn_results = []

# Выберем одно изображение для анализа
test_img_idx = 0
image, annotation = voc_dataset[test_img_idx]
xml_path = os.path.join('./data/VOCdevkit/VOC2007/Annotations',
                         annotation['annotation']['filename'].split('.')[0] + '.xml')
gt_boxes = parse_voc_annotation(xml_path)

# Трансформация изображения
img_tensor = transform(image).unsqueeze(0).to(device)

# Прямой проход через модель
with torch.no_grad():
    outputs = model(img_tensor)

# Получаем предсказания модели
pred_boxes = outputs[0]['boxes'].cpu().numpy()
pred_labels = outputs[0]['labels'].cpu().numpy()
pred_scores = outputs[0]['scores'].cpu().numpy()

# Анализ при разных порогах
for threshold in thresholds:
    # Фильтрация по порогу
    mask = pred_scores >= threshold
    filtered_boxes = pred_boxes[mask]
    filtered_labels = pred_labels[mask]
    filtered_scores = pred_scores[mask]

    # Рассчитываем IoU и считаем FP/FN
    ious = []
    matched_gt_indices = set()

    for pred_box in filtered_boxes:
        max_iou = 0
        max_iou_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = gt_idx

        if max_iou >= iou_threshold:
            matched_gt_indices.add(max_iou_idx)

        ious.append(max_iou)

    false_positives = sum(1 for iou in ious if iou < iou_threshold)
    false_negatives = len(gt_boxes) - len(matched_gt_indices)

    # Добавляем результаты
    fp_fn_results.append({
        'threshold': threshold,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_predictions': len(filtered_boxes),
        'mean_iou': np.mean(ious) if ious else 0
    })

    print(f"Порог: {threshold}")
    print(f"  Количество предсказаний: {len(filtered_boxes)}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Средний IoU: {np.mean(ious) if ious else 0:.4f}")
    print("-----------------------------")

# Визуализация результатов
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
thresholds_values = [result['threshold'] for result in fp_fn_results]
fp_values = [result['false_positives'] for result in fp_fn_results]
fn_values = [result['false_negatives'] for result in fp_fn_results]

plt.plot(thresholds_values, fp_values, 'r-', label='False Positives')
plt.plot(thresholds_values, fn_values, 'b-', label='False Negatives')
plt.xlabel('Confidence Threshold')
plt.ylabel('Количество ошибок')
plt.title('Баланс FP/FN при разных порогах')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
total_errors = [fp + fn for fp, fn in zip(fp_values, fn_values)]
plt.plot(thresholds_values, total_errors, 'g-')
plt.xlabel('Confidence Threshold')
plt.ylabel('Общее количество ошибок (FP + FN)')
plt.title('Общее количество ошибок при разных порогах')
plt.grid(True)

plt.tight_layout()
plt.show()

# Определение оптимального порога
optimal_idx = np.argmin(total_errors)
optimal_threshold = thresholds[optimal_idx]

print(f"Оптимальный порог: {optimal_threshold}")
print(f"Количество FP при оптимальном пороге: {fp_values[optimal_idx]}")
print(f"Количество FN при оптимальном пороге: {fn_values[optimal_idx]}")
print(f"Общее количество ошибок: {total_errors[optimal_idx]}")

# Анализ результатов
print("\nВыводы об оптимальном пороге:")
print(f"1. При пороге {optimal_threshold} достигается наименьшее общее число ошибок ({total_errors[optimal_idx]}).")
print("2. Слишком низкий порог приводит к большому количеству ложных срабатываний (FP).")
print("3. Слишком высокий порог приводит к большому количеству пропущенных объектов (FN).")
print("4. Выбор порога зависит от конкретной задачи - важно ли обнаружить все объекты (низкий порог) или важнее избежать ложных срабатываний (высокий порог).")