from collections import Counter

import torch
import matplotlib.pyplot as plt
from matplotlib import patches

# Finds intersection over union for two provided boxes
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.05, box_format="midpoint", num_classes=20):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    if len(average_precisions) == 0:
        print("MAP is here returning 0")
        return 0.0
    return (sum(average_precisions) / len(average_precisions))


def compute_precision_recall_curves(pred_boxes, true_boxes, num_classes=20, iou_threshold=0.05, box_format="midpoint"):
    amount_bboxes = Counter([gt[0] for gt in true_boxes])
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    detections = sorted(pred_boxes, key=lambda x: x[2], reverse=True)
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    total_true_bboxes = len(true_boxes)

    if total_true_bboxes == 0:
        return torch.tensor([1.0]), torch.tensor([0.0])

    for detection_idx, detection in enumerate(detections):
        gt_for_img = [gt for gt in true_boxes if gt[0] == detection[0]]
        best_iou = 0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_for_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1  # valid TP
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1  # already matched → FP
        else:
            FP[detection_idx] = 1  # low IoU → FP

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)

    recalls = TP_cumsum / (total_true_bboxes + 1e-6)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

    precisions = torch.cat((torch.tensor([1.0]), precisions))
    recalls = torch.cat((torch.tensor([0.0]), recalls))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    return precisions, recalls


import torch
import numpy as np


def compute_confusion_matrix(pred_boxes, true_boxes, num_classes, iou_threshold=0.05, box_format="midpoint"):

    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    gt_by_img = {}
    for gt in true_boxes:
        img_id = gt[0]
        gt_by_img.setdefault(img_id, []).append(gt)

    pred_by_img = {}
    for pred in pred_boxes:
        img_id = pred[0]
        pred_by_img.setdefault(img_id, []).append(pred)

    all_images = set(gt_by_img.keys()) | set(pred_by_img.keys())

    for img_id in all_images:
        gt_img = gt_by_img.get(img_id, [])
        pred_img = pred_by_img.get(img_id, [])

        matched_gt = [False] * len(gt_img)
        pred_img.sort(key=lambda x: x[2], reverse=True)

        for pred in pred_img:
            pred_class = int(pred[1])
            best_iou = 0
            best_gt_idx = -1

            for idx, gt in enumerate(gt_img):
                iou = intersection_over_union(
                    torch.tensor(pred[3:]),
                    torch.tensor(gt[2:]),
                    box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                gt_class = int(gt_img[best_gt_idx][1])
                cm[gt_class, pred_class] += 1
                matched_gt[best_gt_idx] = True
            else:
                cm[num_classes, pred_class] += 1

        # Remaining unmatched GTs are FN
        for idx, matched in enumerate(matched_gt):
            if not matched:
                gt_class = int(gt_img[idx][1])
                cm[gt_class, num_classes] += 1

    return cm


def plot_image(image, boxes, labels):
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    im = image.permute(1, 2, 0).numpy()
    height, width, _ = im.shape
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(im)

    for box in boxes:
        label = int(box[0])
        score = box[1]
        box_coords = box[2:]
        assert len(box_coords) == 4, "Box length too long"

        upper_left_x = (box_coords[0] - box_coords[2] / 2) * width
        upper_left_y = (box_coords[1] - box_coords[3] / 2) * height
        box_width = box_coords[2] * width
        box_height = box_coords[3] * height

        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else f'class_{label}'
        text_x = upper_left_x
        text_y = upper_left_y - 5  # 5 pixels above the box

        ax.text(text_x, text_y, f'{class_name}: {score:.2f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=10, color='white', weight='bold',
                verticalalignment='bottom')

    for box in labels:
        label = int(box[0])
        score = box[1]
        box_coords = box[2:]
        assert len(box_coords) == 4, "Box length too long"

        upper_left_x = (box_coords[0] - box_coords[2] / 2) * width
        upper_left_y = (box_coords[1] - box_coords[3] / 2) * height
        box_width = box_coords[2] * width
        box_height = box_coords[3] * height

        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else f'class_{label}'
        text_x = upper_left_x
        text_y = upper_left_y - 5

        ax.text(text_x, text_y, f'{class_name}: {score:.2f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                fontsize=10, color='white', weight='bold',
                verticalalignment='bottom')

    plt.tight_layout()
    plt.show()

# Find the predicted and target boxes for each image to calculate loss
def get_bboxes(loader, model, iou_threshold, threshold, pred_format='cells', box_format='midpoint', device="cuda"):

    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

# Convert prediction coordinates to the grid system
def convert_cellboxes(predictions, S=7):
    predictions = predictions.cpu()
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)

    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]

    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim = 0)

    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = best_boxes[..., 2:4]  # No division by S!

    converted_bboxes = torch.cat((x, y, w_h), dim = -1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)

    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)
    return converted_preds

# Convert the grid coordinates to box coordinates
def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename = "old.path.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer=None, device="cuda", LEARNING_RATE=0, WEIGHT_DECAY=0):
    print(f"=> Loading weights from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"❌ Found NaNs/Infs in {name} → checkpoint is corrupted")

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 448, 448).to(device)
        _ = model(dummy)

    model.train()
    return model