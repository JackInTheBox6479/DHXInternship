import cv2
import numpy as np
import matplotlib.pyplot as plt

# Imports image of the Eiffel Tower
img_original = cv2.imread('Test Images/test.jpg')
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread('Test Images/test.jpg', cv2.IMREAD_GRAYSCALE)
# img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)


# Applies Sobel Filter
def sobel(image):
    # Calculates horizontal and vertical gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude
    gm = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize and convert to uint8
    gm = cv2.normalize(gm, None, 0, 255, cv2.NORM_MINMAX)
    gm = np.uint8(gm)

    return gm

# Applies Laplacian Filter
def laplacian(image):
    image = cv2.Laplacian(image, ksize=5, ddepth=cv2.CV_64F)
    image = np.abs(image)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(image)

    return image

# Applies Canny Filter
def canny(image, lower, upper):
    image = cv2.Canny(image, lower, upper)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(image)

    return image

# cv2.imshow('Original', img_original)
# cv2.imshow('Sobel', sobel(img_gray))
# cv2.imshow('Laplacian', laplacian(img_gray))
# cv2.imshow('Canny', canny(img_gray, 100, 300))
# cv2.imshow('Canny2', canny(img_gray, 50, 150))

# Create Plots
(fig, axs) = plt.subplots(2, 3, figsize=[40, 24])
axs[0,0].set_title('Original', fontsize=40)
axs[0,0].imshow(img_original, cmap='plasma')
axs[0, 1].set_title('Sobel', fontsize=40)
axs[0, 1].imshow(sobel(img_gray), cmap='gray')
axs[1, 0].set_title('Grayscale', fontsize=40)
axs[1, 0].imshow(img_gray, cmap='gray')
axs[1, 1].set_title('Laplacian', fontsize=40)
axs[1, 1].imshow(laplacian(img_gray), cmap='gray')
axs[0, 2].set_title('Canny', fontsize=40)
axs[0, 2].imshow(canny(img_gray, 200, 400), cmap='gray')
axs[1, 2].set_title('Canny Low Threshold', fontsize=40)
axs[1, 2].imshow(canny(img_gray, 50, 150), cmap='gray')

cv2.waitKey(0)
cv2.destroyAllWindows()


plt.subplots_adjust(wspace=.1, hspace=-.1)
plt.show()
