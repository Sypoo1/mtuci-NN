# импорт пакетов
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms

# импорт модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# загрузка датасета
voc_dataset = VOCDetection(root='./data', year='2007', image_set='test', download=True)

# метод трансформации
transform = transforms.Compose([
    transforms.ToTensor()
])

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

# выберите N изображений из датасета
N = 5
image_filenames = [voc_dataset[i][1]['annotation']['filename'] for i in range(N)]

# объявите цикл для проверки
for i in range(N):
    # загрузите изображение и его разметку
    image, target = voc_dataset[i]
    image_path = target['annotation']['filename']
    gt_boxes = parse_voc_annotation(target['annotation']['path'])

    # примените трансформации
    image_tensor = transform(image)

    # выполните прямой проход
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])

    # извлекаем bounding boxes, метки и confidence scores из полученных выходов модели
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()

    # установите порог confidence_threshold в эмпирическом значении
    confidence_threshold = 0.5

    # примените фильтр выходных значений на основе заданного порога
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    # отрисуйте bounding boxes на изображении с предсказаниями
    image_with_pred = draw_predictions(image.copy(), pred_boxes, pred_labels, pred_scores, color="blue")

    # отрисуйте bounding boxes с реальной разметкой
    image_with_gt = draw_predictions(image.copy(), gt_boxes, ['GT'] * len(gt_boxes), [1.0] * len(gt_boxes), color="red")

    # вычислите IoU для выбранных изображений
    ious = []
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            ious.append(iou)

    # подсчитайте количество False Positive и False Negative для выбранных изображений
    iou_threshold = 0.5
    true_positives = sum(1 for iou in ious if iou >= iou_threshold)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    print(f"Изображение {i+1}:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Средний IoU: {np.mean(ious):.3f}")
    print("-" * 50)

    # Визуализация результатов
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_with_pred)
    plt.title("Предсказания")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_gt)
    plt.title("Реальная разметка")
    plt.axis('off')
    plt.show()

# исследование
confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for threshold in confidence_thresholds:
    total_fp = 0
    total_fn = 0
    total_tp = 0

    for i in range(N):
        image, target = voc_dataset[i]
        image_tensor = transform(image)
        gt_boxes = parse_voc_annotation(target['annotation']['path'])

        with torch.no_grad():
            outputs = model([image_tensor.to(device)])

        pred_boxes = outputs[0]['boxes'].cpu().numpy()
        pred_scores = outputs[0]['scores'].cpu().numpy()

        mask = pred_scores >= threshold
        pred_boxes = pred_boxes[mask]

        ious = []
        for pred_box in pred_boxes:
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                ious.append(iou)

        iou_threshold = 0.5
        true_positives = sum(1 for iou in ious if iou >= iou_threshold)
        false_positives = len(pred_boxes) - true_positives
        false_negatives = len(gt_boxes) - true_positives

        total_fp += false_positives
        total_fn += false_negatives
        total_tp += true_positives

    results.append({
        'threshold': threshold,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'true_positives': total_tp
    })

# Визуализация результатов исследования
thresholds = [r['threshold'] for r in results]
fps = [r['false_positives'] for r in results]
fns = [r['false_negatives'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, fps, label='False Positives', marker='o')
plt.plot(thresholds, fns, label='False Negatives', marker='o')
plt.xlabel('Порог уверенности')
plt.ylabel('Количество')
plt.title('Зависимость FP/FN от порога уверенности')
plt.legend()
plt.grid(True)
plt.show()