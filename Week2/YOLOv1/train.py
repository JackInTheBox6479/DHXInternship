from datetime import datetime
import random

import numpy as np
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import Yolov1
from dataset import VOCDataset, collate_fn
from utils import *
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-4
EPOCHS = 0
NUM_WORKERS = 8
PIN_MEMORY = False
LOAD_MODEL = True
LOAD_MODEL_FILE = "my_checkpoint.pth.tar"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    # Main training function, runs each batch
    for batch_idx, (x,y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return mean_loss

def main():
    now = datetime.now()
    current_time_str = now.strftime("%H:%M:%S")
    print("Start Time =", current_time_str)

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(LOAD_MODEL_FILE, model, optimizer=None, LEARNING_RATE=LEARNING_RATE, WEIGHT_DECAY=WEIGHT_DECAY)

    train_dataset = VOCDataset(root = "../data", image_set="trainval", transforms=transform)
    test_dataset = VOCDataset("../data", image_set="test", transforms=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    loss = []

    # Runs each epoch
    for epoch in range(EPOCHS):
        print(' ')
        print(' ')
        print(f"Epoch: {epoch + 1} out of {EPOCHS}")
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.05, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.05, box_format="midpoint")

        print(f'Train mAP: {mean_avg_prec}')
        precisions, recalls = compute_precision_recall_curves(
            pred_boxes,
            target_boxes,
            num_classes=20,
            iou_threshold=0.05
        )
        plt.figure(figsize=(6, 6))
        plt.plot(recalls, precisions, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Precision-Recall Curve")
        plt.grid()
        plt.legend()
        plt.show()

        num_classes = 20
        cm = compute_confusion_matrix(pred_boxes, target_boxes, num_classes, iou_threshold=0.05)

        VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        labels = [f"{VOC_CLASSES[i]}" for i in range(num_classes)] + ["BG"]
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Detection Confusion Matrix")
        plt.show()

        mean_loss = train_fn(train_loader, model, optimizer, loss_fn)
        loss.append(sum(mean_loss)/len(mean_loss))

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"my_checkpoint.pth.tar")

    print("Training finished!")

    now2 = datetime.now()
    current_time_str2 = now2.strftime("%H:%M:%S")
    print("End Time =", current_time_str2)
    total_time = now2 - now
    if EPOCHS >= 1:
        time_per_epoch = total_time / EPOCHS
    else:
        time_per_epoch = 0

    print(f'Time per epoch: {time_per_epoch}, Total time: {total_time}')

    plt.plot(loss)
    plt.title("Training Loss")
    draw_test_image(test_dataset, model)

# Draws an image from the test dataset
def draw_test_image(dataset, model):
    idx = random.randint(0, len(dataset) - 1)
    #idx = 11
    image, labels = dataset[idx]
    image_batch = image.unsqueeze(0).to(DEVICE)

    labels = labels.to(DEVICE)
    labels = labels.cpu().tolist()

    for box in labels:
        box.insert(1, 5)

    with torch.no_grad():
        preds = model(image_batch)

    pred_boxes = cellboxes_to_boxes(preds)
    pred_boxes = pred_boxes[0]

    #print("Pred_boxes:", pred_boxes)

    final_boxes = non_max_suppression(
        pred_boxes,
        iou_threshold=0.5,
        threshold=0.2,
        box_format="midpoint"
    )
    # Plot the image with boxes
    print(f'final_boxes: {final_boxes}')
    plot_image(image, final_boxes, labels)

if __name__ == '__main__':
    main()

