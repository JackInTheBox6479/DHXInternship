import glob
import os
import random
from pathlib import Path

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import xml.etree.ElementTree as ET

def load_images_and_annotations(image_dir, annotation_dir, label2idx):
    image_infos = []

    annotation_dir = Path(annotation_dir)
    all_files = list(annotation_dir.glob("*"))
    xml_files = list(annotation_dir.glob("*.xml"))

    for annotation_file in tqdm(xml_files):
        image_info = {}
        image_info['img_id'] = os.path.basename(annotation_file).split('.xml')[0]
        image_info['filename'] = os.path.join(image_dir, '{}.jpg'.format(image_info['img_id']))
        annotation_info = ET.parse(annotation_file)
        root = annotation_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        image_info['width'] = width
        image_info['height'] = height
        detections = []

        for obj in annotation_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]

            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        image_info['detections'] = detections
        image_infos.append(image_info)
    print("Total {} images found.".format(len(image_infos)))
    return image_infos

class VOCDataset(Dataset):
    def __init__(self, split, image_dir, annotation_dir):
        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        self.images_info = load_images_and_annotations(image_dir, annotation_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        image_info = self.images_info[idx]
        image = Image.open(image_info['filename']).convert("RGB")
        to_flip = False

        # Flips random images to improve training accuracy
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_tensor = torchvision.transforms.ToTensor()(image)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in image_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in image_info['detections']])

        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2 - x1
                im_w = image_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return image_tensor, targets, image_info['filename']
