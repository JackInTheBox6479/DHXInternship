import os
import random
import cv2
import xml.etree.ElementTree as ET
import numpy as np

shuffled_images = random.sample(os.listdir('data/VOC2007trainval/JPEGImages'), len(os.listdir(
    'data/VOC2007trainval/JPEGImages')))
dog_color = (0, 0, 255)
person_color = (0, 255, 0)
car_color = (255, 0, 0)

# def manipimage(n):

def drawimage(img):
    image = cv2.imread(f'data/VOC2007trainval/JPEGImages/{img}')
    try:
        tree = ET.parse(f'data/VOC2007trainval/Annotations/{str(img)[:6]}.xml')
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != 'dog' and name != 'person' and name != 'car':
                return

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            if name == 'dog':
                box_color = dog_color
            elif name == 'person':
                box_color = person_color
            else:
                box_color = car_color

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, 2)
            cv2.putText(image, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        image = cv2.resize(image, (448, 448))
        cv2.imshow("image", image)
        cv2.waitKey(0)
    except FileNotFoundError:
        return

for img in shuffled_images:
    drawimage(img)

cv2.destroyAllWindows()
