from collections import Counter

import torch
import math

def intersection_over_union(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection

    return intersection / union

def boxes_to_transformed_targets(gt_boxes, anchors):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * widths
    center_y = anchors[:, 1] + 0.5 * heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_center_x - center_x) / widths
    targets_dy = (gt_center_y - center_y) / heights
    targets_dw = torch.log(gt_widths / widths)
    targets_dh = torch.log(gt_heights / heights)

    regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return regression_targets

def apply_regression_to_anchors(box_transform_pred, anchors):
    box_transform_pred = box_transform_pred.reshape(
        box_transform_pred.size(0), -1, 4)

    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * w
    center_y = anchors[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]

    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack((
        pred_box_x1,
        pred_box_y1,
        pred_box_x2,
        pred_box_y2),
        dim=2)

    return pred_boxes

def sample_positive_negative(labels, positive_count, total_count):
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels <= 0)[0]
    num_pos = positive_count
    num_pus = min(positive.numel(), num_pos)
    num_neg = total_count - num_pus
    num_neg = min(negative.numel(), num_neg)

    perm_positive_idxs = torch.randperm(positive.numel(), device = positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device = negative.device)[:num_neg]
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]

    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask

def clamp_boxes_to_img_boundary(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]

    height, width = image_shape[-2:]

    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)

    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]),
        dim=-1)
    return boxes

def transform_boxes_to_original_size(boxes, new_size, original_size):
    ratios = [torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
               / torch.tensor(s, dtype=torch.float32, device=boxes.device)
               for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    ymin = ymin * ratio_height
    xmax = xmax * ratio_width
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.05, box_format="midpoint", num_classes=21):
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
       # print("MAP is here returning 0")
        return 0.0
    return (sum(average_precisions) / len(average_precisions))

def save_checkpoint(state, filename = "faster_rcnn_voc2007.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])