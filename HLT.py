import numpy as np
import cv2
from matplotlib import pyplot as plt


def hough_line(img):

    # creating theta, cos_theta, sin_theta:
    theta = np.deg2rad(np.arange(-90, 90))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # length of theta:
    num_theta = len(theta)

    # creating rhos:
    width, height = img.shape
    diagonal_len = int(round(np.sqrt(width**2 + height**2)))
    rhos = np.linspace(-diagonal_len, diagonal_len, diagonal_len * 2)

    # creating accumulator:
    accumulator = np.zeros((diagonal_len * 2, num_theta), dtype=np.uint64)
    # (row, col) indexes to edges/boundary:
    y_indexes, x_indexes = np.nonzero(img)

    # Vote in the hough accumulator for nonzero indexes in gray scale image for different (rho, theta):
    for i in range(len(x_indexes)):
        for t_index in range(num_theta):
            # Calculating rho & diagonal_len is added for a positive index value:
            rho = int(round(x_indexes[i] * cos_theta[t_index] + y_indexes[i] * sin_theta[t_index])) + diagonal_len
            accumulator[rho, t_index] += 1

    return accumulator, theta, rhos


# creating a gray scale image:
img = 255*np.ones((256, 256), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (200, 200), 1, -1)

# applying canny edge detection:
edges = cv2.Canny(img, 100, 200)

# displaying test image:
cv2.imshow('coins', img)
cv2.imshow('Canny edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# calling hough_line transform function:
accumulator, theta, rhos = hough_line(img)

# Subplotting image, Canny edge, accumulator:
plt.subplot(221)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.title('Original')
plt.subplot(222)
plt.imshow(edges, cmap='gray', interpolation='nearest')
plt.title('Canny edges')
plt.subplot(223)
plt.imshow(accumulator, cmap='jet', interpolation='nearest')
plt.title('Accumulator')
plt.show()
