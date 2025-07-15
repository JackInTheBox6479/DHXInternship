import torch
import argparse
import os
import numpy as np
import yaml
import random

from PIL import Image
from matplotlib import pyplot as plt, patches

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

    # print(config)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    voc = VOCDataset('train', image_dir=dataset_config['im_train_path'],
                     annotation_dir=dataset_config['ann_train_path'])
    train_dataset = DataLoader(voc, batch_size=1, shuffle=True, num_workers=4)

    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    optimizer = torch.optim.Adam(faster_rcnn_model.parameters(), lr=train_config['lr'])
    faster_rcnn_model.to(device)
    load_checkpoint(torch.load("faster_rcnn_voc2007.pth.tar"), faster_rcnn_model, optimizer)

    faster_rcnn_model.train()

    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()

        for image, target, fname in tqdm(train_dataset):
            image = image.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            rpn_output, frcnn_output = faster_rcnn_model(image, target)

            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss

            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

            loss = loss / acc_steps
            loss.backward()

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
        scheduler.step()
    print("Done Training")

    test_image_path = "../data/VOC2007test/JPEGImages/000018.jpg"
    test_and_display_image(faster_rcnn_model, test_image_path)

def test_and_display_image(model, image_path, confidence_threshold=0):
    VOC_CLASSES = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = torch.tensor(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        _, frcnn_output = model(image_tensor)

    boxes = frcnn_output['boxes'].cpu().numpy()
    labels = frcnn_output['labels'].cpu().numpy()
    scores = frcnn_output['scores'].cpu().numpy()


    keep = scores >= confidence_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.Set3(np.linspace(0, 1, 20))

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    linewidth=2, edgecolor=colors[label], facecolor='none')
        ax.add_patch(rect)
        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else f'class_{label}'
        ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[label], alpha=0.7),
                fontsize=10, color='black')

    ax.axis('off')
    ax.set_title(f'Predictions ({len(boxes)} boxes)')
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
        test_and_display_image(model, args.image_path or "../data/VOC2007test/JPEGImages/002235.jpg", confidence_threshold=0)
    else:
        train(args)

""" 
cd "C:\\Users\jackc\Documents\Coding\Learning Python\DHXInternship\Week2\FasterRCNN"
python train.py --test
"""