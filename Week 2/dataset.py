import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(self, root, image_set, transforms):
        self.root = root
        self.image_set = image_set  # "trainval" or "test"
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
            # If no objects, return empty tensor with correct shape
            targets = torch.zeros((0, 5), dtype=torch.float32)
        else:
            targets = torch.tensor(targets, dtype=torch.float32)
            #print(len(targets))

        # Apply transforms if provided
        image, targets = self.transforms(image, targets)

        return image, targets


# Custom collate function for DataLoader
def collate_fn(batch):
    """Custom collate function for YOLO dataset"""
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # image tensor
        targets.append(sample[1])  # target tensor

    # Stack images (they should all be the same size after transforms)
    images = torch.stack(images, 0)

    # Don't stack targets - keep as list since they have different lengths
    return images, targets