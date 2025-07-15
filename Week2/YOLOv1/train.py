from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
EPOCHS = 0
NUM_WORKERS = 8
PIN_MEMORY = True
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
        x = torch.stack([item.to(DEVICE) for item in x])
        y = torch.stack([item.to(DEVICE) for item in y])

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    # Save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=f"my_checkpoint.pth.tar")

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
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(root = "../data", image_set="trainval", transforms=transform)
    test_dataset = VOCDataset("../data", image_set="test", transforms=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True, collate_fn=collate_fn)

    loss = []

    # Runs each epoch
    for epoch in range(EPOCHS):
        print(' ')
        print(' ')
        print(f"Epoch: {epoch + 1} out of {EPOCHS}")
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        #print(f'target_boxes: {target_boxes}')
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        print(f'Train mAP: {mean_avg_prec}')

        # TODO: Figure out why this outputs zero

        loss.append(train_fn(train_loader, model, optimizer, loss_fn))

    print("Training finished!")
    if not loss: loss = [[1], [2]]
    mean_loss = [loss for sublist in loss for loss in sublist]

    x = np.arange(len(mean_loss))
    coefficients = np.polyfit(x, mean_loss, 1)
    line_of_best_fit = np.poly1d(coefficients)
    plt.plot(mean_loss)
    plt.plot(x, line_of_best_fit(x))

    now2 = datetime.now()
    current_time_str2 = now2.strftime("%H:%M:%S")
    print("End Time =", current_time_str2)
    total_time = now2 - now
    if EPOCHS >= 1:
        time_per_epoch = total_time / EPOCHS
    else:
        time_per_epoch = 0

    print(f'Time per epoch: {time_per_epoch}, Total time: {total_time}')

    draw_test_image(test_dataset, model)

# Draws an image from the test dataset
def draw_test_image(dataset, model):
    idx = random.randint(0, len(dataset) - 1)
    idx = 11
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

    final_boxes = non_max_suppression(
        pred_boxes,
        iou_threshold=0.0,
        threshold=0.0,
        box_format="midpoint"
    )
    # Plot the image with boxes
    print(f'final_boxes: {final_boxes}')
    plot_image(image, final_boxes)

if __name__ == '__main__':
    main()

