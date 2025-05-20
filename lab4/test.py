# %% [markdown]
# ## Лабораторная работа №4: Сегментация объектов с использованием Mask R-CNN

# %% [markdown]
# ### Подготовка окружения

# %%
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os
import cv2
from torchvision import transforms

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ### Загрузка модели

# %%
# Загрузка предобученной модели Mask R-CNN
model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
).to(device)
model.eval()

# %% [markdown]
# ### Загрузка данных COCO 2017

# %%
# Настройка путей (замените на свои пути)
data_dir = "/path/to/coco"
ann_file = os.path.join(data_dir, "annotations/instances_val2017.json")
img_dir = os.path.join(data_dir, "val2017")

# Трансформации для изображений
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка датасета
dataset = CocoDetection(img_dir, ann_file, transforms=transform)
coco = COCO(ann_file)

# %% [markdown]
# ### Вспомогательные функции

# %%
def calculate_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Вычисляет Intersection over Union для двух масок"""
    intersection = np.logical_and(mask_pred, mask_gt)
    union = np.logical_or(mask_pred, mask_gt)
    return np.sum(intersection) / np.sum(union)

def visualize_results(image: torch.Tensor, 
                     pred_masks: List[np.ndarray], 
                     gt_masks: List[np.ndarray],
                     pred_boxes: List[List[float]],
                     gt_boxes: List[List[float]],
                     pred_labels: List[int],
                     gt_labels: List[int]):
    """Визуализация результатов с наложением масок"""
    # Конвертируем изображение
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    
    # Создаем график
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Предсказания
    ax[0].set_title('Predictions')
    ax[0].imshow(img)
    for mask, box, label in zip(pred_masks, pred_boxes, pred_labels):
        color = np.random.rand(3,)
        ax[0].imshow(np.ma.masked_where(mask == 0, mask), 
                    cmap='viridis', alpha=0.5)
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color=color, linewidth=2)
        ax[0].add_patch(rect)
        ax[0].text(x1, y1, f'Pred: {label}', color='white', backgroundcolor=color)
    
    # Ground Truth
    ax[1].set_title('Ground Truth')
    ax[1].imshow(img)
    for mask, box, label in zip(gt_masks, gt_boxes, gt_labels):
        color = np.random.rand(3,)
        ax[1].imshow(np.ma.masked_where(mask == 0, mask), 
                    cmap='viridis', alpha=0.5)
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, 
                           fill=False, color=color, linewidth=2)
        ax[1].add_patch(rect)
        ax[1].text(x, y, f'GT: {label}', color='white', backgroundcolor=color)
    
    plt.show()

# %% [markdown]
# ### Инференс и оценка

# %%
# Параметры
confidence_threshold = 0.5
mask_threshold = 0.5
num_images = 20

# Результаты
total_iou = 0.0
total_objects = 0

for idx in range(num_images):
    # Загрузка данных
    img_tensor, _ = dataset[idx]
    image_id = dataset.ids[idx]
    
    # Получение аннотаций
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    
    # Фильтрация crowd-объектов
    gt_boxes = []
    gt_labels = []
    gt_masks = []
    for ann in anns:
        if ann['iscrowd'] == 0:
            gt_boxes.append(ann['bbox'])
            gt_labels.append(ann['category_id'])
            gt_masks.append(coco.annToMask(ann))
    
    # Инференс
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]
    
    # Обработка предсказаний
    scores = prediction['scores'].cpu().numpy()
    keep = scores >= confidence_threshold
    
    pred_boxes = prediction['boxes'].cpu().numpy()[keep]
    pred_labels = prediction['labels'].cpu().numpy()[keep]
    pred_masks = (prediction['masks'].cpu().numpy()[keep] > mask_threshold).squeeze()
    
    # Визуализация
    visualize_results(img_tensor, pred_masks, gt_masks, 
                     pred_boxes.tolist(), gt_boxes,
                     pred_labels.tolist(), gt_labels)
    
    # Расчет IoU
    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0.0
        for j, gt_mask in enumerate(gt_masks):
            if pred_labels[i] == gt_labels[j]:
                # Приводим маски к одному размеру
                h, w = gt_mask.shape
                resized_mask = cv2.resize(pred_mask.astype(float), (w, h)) > 0.5
                iou = calculate_iou(resized_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
        if best_iou > 0:
            total_iou += best_iou
            total_objects += 1

# Средний IoU
if total_objects > 0:
    mean_iou = total_iou / total_objects
    print(f"\nAverage IoU: {mean_iou:.4f}")
else:
    print("No objects detected")

# %% [markdown]
# ### Исследование параметров

# %%
# Тестируемые параметры
conf_thresholds = [0.3, 0.5, 0.7]
mask_thresholds = [0.3, 0.5, 0.7]

# Таблица результатов
results = []

for conf in conf_thresholds:
    for mask_th in mask_thresholds:
        current_iou = 0.0
        count = 0
        
        for idx in range(5):  # Тестируем на 5 изображениях для экономии времени
            img_tensor, _ = dataset[idx]
            with torch.no_grad():
                prediction = model([img_tensor.to(device)])[0]
            
            # Фильтрация
            scores = prediction['scores'].cpu().numpy()
            keep = scores >= conf
            pred_masks = (prediction['masks'].cpu().numpy()[keep] > mask_th).squeeze()
            pred_labels = prediction['labels'].cpu().numpy()[keep]
            
            # Расчет IoU
            for i, pred_mask in enumerate(pred_masks):
                # ... аналогичная логика расчета ...
                
        avg_iou = current_iou / count if count > 0 else 0
        results.append((conf, mask_th, avg_iou))

# Вывод результатов
print("\nParameter Study Results:")
print("Confidence | Mask Threshold | IoU")
for res in results:
    print(f"{res[0]:<10} | {res[1]:<14} | {res[2]:.4f}")