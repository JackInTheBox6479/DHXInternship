import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import image, blur to reduce noise
img_original = cv2.imread('Test Images/test.jpg')
img_gray = cv2.imread('Test Images/test.jpg', cv2.IMREAD_GRAYSCALE)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Create black chalkboard with the same image dimension that we can draw the edges on
img_draw2 = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype='uint8')

gx = {}
gy = {}
gm = {}

# Iterate through each pixel of grayscale image, excluding edges
for x in range(img_gray.shape[1]):
    if x == 0 or x == img_gray.shape[1]-1:
        continue
    for y in range(img_gray.shape[0]):
        if y == 0 or y == img_gray.shape[0]-1:
            continue

        # Create kernel, cast to int to prevent overflow
        top_left = img_gray[y - 1, x - 1].astype(int)
        left = img_gray[y, x - 1].astype(int)
        bottom_left = img_gray[y + 1, x - 1].astype(int)

        above = img_gray[y - 1, x].astype(int)
        below = img_gray[y + 1, x].astype(int)

        top_right = img_gray[y - 1, x + 1].astype(int)
        right = img_gray[y, x + 1].astype(int)
        bottom_right = img_gray[y + 1, x + 1].astype(int)

        # Apply the kernel and solve for gradients in each direction, magnitude
        gx[x, y] = ((top_left * -1) + (left * -2) + (bottom_left * -1)) + (top_right + (right * 2) + bottom_right)
        gy[x, y] = (top_left + (above * 2) + top_right) + ((bottom_left * -1) + (below * -2) + (bottom_right * -1))
        gm[x, y] = np.sqrt(gx[x, y]**2 + gy[x, y]**2)

        # Set threshold to include only important edges
        if gm[x, y] > 100:
            img_draw2[y, x] = (255, 255, 255)

cv2.imshow('Sobel', img_draw2)

fix, axs = plt.subplots()
axs.imshow(img_draw2)

cv2.waitKey(0)
cv2.destroyAllWindows()
