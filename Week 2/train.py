import torch
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

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

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

    for batch_idx, (x,y) in enumerate(loop):
        x = [item.to(DEVICE) for item in x]
        y = [item.to(DEVICE) for item in y]
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Finished batch {batch_idx}")

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(root = "data", image_set="trainval", transforms=transform)
    test_dataset = VOCDataset("data", image_set="test", transforms=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True, collate_fn=collate_fn)

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f'Train mAP: {mean_avg_prec}')

        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == '__main__':
    main()

