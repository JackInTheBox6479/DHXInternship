from datetime import datetime

import argparse
import numpy as np
import yaml
import random

from PIL import Image
from matplotlib import pyplot as plt, patches
import xml.etree.ElementTree as ET
from torch import autocast
from model import FasterRCNN
from tqdm import tqdm
from dataset import VOCDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import *

device = torch.device('cuda') #  if torch.cuda.is_available() else 'cpu'

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train', image_dir=dataset_config['im_train_path'],
                     annotation_dir=dataset_config['ann_train_path'])
    train_dataset = DataLoader(voc, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    optimizer = torch.optim.Adam(faster_rcnn_model.parameters(), lr=train_config['lr'])
    faster_rcnn_model.to(device)
 #   load_checkpoint(torch.load("faster_rcnn_voc2007.pth.tar"), faster_rcnn_model, optimizer)

    faster_rcnn_model.train()

    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    now = datetime.now()
    current_time_str = now.strftime("%H:%M:%S")
    print("Start Time =", current_time_str)

    mean_loss = []
    mean_average_precision = []
    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()
        epoch_losses = []

        for image, target, fname in tqdm(train_dataset):
            image = image.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)

            with autocast('cuda'):
                rpn_output, frcnn_output = faster_rcnn_model(image, target)

                rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                total_loss = rpn_loss + frcnn_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("NaN/Inf detected!")
               # exit()

            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

            epoch_losses.append(total_loss.item())
            scaled_loss = total_loss / acc_steps
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(faster_rcnn_model.parameters(), max_norm=1.0)

            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1

        print('Finished epoch {}'.format(i + 1))
        optimizer.step()
        optimizer.zero_grad()

        checkpoint = {
            "state_dict": faster_rcnn_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "faster_rcnn_voc2007.pth.tar")

        loss_output = ''
        loss_output += 'RPN Classification Loss: {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))

        print(loss_output)
        print(' ')
        print(' ')
        mean_loss.append(np.mean(epoch_losses))
        scheduler.step()
    print("Done Training")
    print(f'Training Mean Average Precision: {np.mean(mean_average_precision)}')

    now2 = datetime.now()
    current_time_str2 = now2.strftime("%H:%M:%S")
    print("End Time =", current_time_str2)
    total_time = now2 - now
    if num_epochs >= 1:
        time_per_epoch = total_time / num_epochs
    else:
        time_per_epoch = 0
    print(f'Time per epoch: {time_per_epoch}, Total time: {total_time}')


    plot_loss(mean_loss)

    test_image_path = "../data/VOC2007test/JPEGImages/000018.jpg"
    annotations_path = "../data/VOC2007test/Annotations/000018.xml"
    test_and_display_image(faster_rcnn_model, test_image_path, annotations_path)

def plot_loss(mean_loss):
    plt.plot(mean_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

def test_and_display_image(model, image_path, annotations_path, confidence_threshold=0):
    VOC_CLASSES = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    image = Image.open(image_path).convert('RGB')

    tree = ET.parse(annotations_path)
    root = tree.getroot()

    gt_boxes = []
    gt_labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in VOC_CLASSES:
            label = VOC_CLASSES.index(class_name)
        else:
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        gt_boxes.append([xmin, ymin, xmax, ymax])
        gt_labels.append(label)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        class_name = VOC_CLASSES[label]
        ax.text(x1, y1 - 5, f'{class_name}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='g', alpha=0.7),
                fontsize=10, color='white', weight='bold')

    model.eval()
    image_tensor = torch.tensor(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        _, frcnn_output = model(image_tensor)

    boxes = frcnn_output['boxes'].cpu().numpy()
    labels = frcnn_output['labels'].cpu().numpy()
    scores = frcnn_output['scores'].cpu().numpy()

    mAP = mean_average_precision(np.column_stack((labels, boxes)), np.column_stack((gt_labels, gt_boxes)), iou_threshold=0)
    print(f"Test Mean Average Precision: {mAP}")

    keep = scores >= confidence_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="r", facecolor='none')
        ax.add_patch(rect)
        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else f'class_{label}'
        ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="r", alpha=0.7),
                fontsize=10, color='black')

    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for faster rcnn training")
    parser.add_argument('--config', dest='config_path', default='config.yaml', type=str)
    parser.add_argument('--test', action='store_true', help='Test a single image only')
    parser.add_argument('--image_path', type=str, help='Path to image for testing')
    args = parser.parse_args()

    if args.test:
        config = yaml.safe_load(open(args.config_path))
        model_config = config['model_params']
        dataset_config = config['dataset_params']
        model = FasterRCNN(model_config, num_classes=dataset_config['num_classes']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = torch.load("faster_rcnn_voc2007.pth.tar", map_location=device)
        load_checkpoint(checkpoint, model, optimizer)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        test_and_display_image(model, args.image_path or "../data/VOC2007test/JPEGImages/000018.jpg", "../data/VOC2007test/Annotations/000018.xml",  confidence_threshold=0)
    else:
        train(args)

""" 
cd "C:\\Users\jackc\Documents\Coding\Learning Python\DHXInternship\Week2\FasterRCNN"
python train.py --test
"""