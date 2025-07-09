import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root, image_set, transforms):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms

        image_sets_file = os.path.join(
            root, f'VOC2007{image_set}', 'ImageSets', 'Main', 'trainval.txt' if image_set == "trainval" else 'test.txt'
        )
        # Read list of image ids
        with open(image_sets_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.images_dir = os.path.join(root, f'VOC2007{image_set}', 'JPEGImages')
        self.annotations_dir = os.path.join(root, f'VOC2007{image_set}', 'Annotations')

        self.class_to_idx = {
            "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
            "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10,
            "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15,
            "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Get image dimensions
        img_width, img_height = image.size

        # Load and parse annotation
        annotation_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        targets = []

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Find binding boxes and object types, add to "targets" tensor
        for obj in root.findall("object"):
            label = obj.find("name").text
            label_idx = self.class_to_idx[label]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert to YOLO format: [class, x_center, y_center, width, height]
            # All coordinates normalized to [0, 1]
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            targets.append([label_idx, x_center, y_center, width, height])

        # Convert to tensor
        if len(targets) == 0:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        else:
            targets = torch.tensor(targets, dtype=torch.float32)
            #print(len(targets))

        # Apply transforms
        image, targets = self.transforms(image, targets)
        encoded_labels = encode_labels_to_yolo_grid(targets)

        return image, encoded_labels

# Essentially normalizing them, changing coords to grid coords
def encode_labels_to_yolo_grid(label_boxes, S=7, B=2, C=20):
    yolo_tensor = torch.zeros((S, S, C + B * 5))

    for box in label_boxes:
        class_label = int(box[0].item())
        x, y, w, h = box[1:].tolist()

        i = int(x * S)
        j = int(y * S)
        i = min(i, S - 1)
        j = min(j, S - 1)

        yolo_tensor[j, i, class_label] = 1.0

        x_cell = x * S - i
        y_cell = y * S - j

        yolo_tensor[j, i, C : C + 5] = torch.tensor([x_cell, y_cell, w, h, 1.0])

    return yolo_tensor

# Custom collate function for DataLoader
def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # image tensor
        targets.append(sample[1])  # target tensor

    images = torch.stack(images, 0)
    return images, targets