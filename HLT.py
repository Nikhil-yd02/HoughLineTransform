import numpy as np
import cv2
from matplotlib import pyplot as plt


def hough_line(img):

    # creating thetas, cos_thetas, sin_thetas:
    thetas = np.deg2rad(np.arange(-90, 90))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    # length of theta:
    num_theta = len(thetas)

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
            rho = int(round(x_indexes[i] * cos_thetas[t_index] + y_indexes[i] * sin_thetas[t_index])) + diagonal_len
            accumulator[rho, t_index] += 1

    return accumulator, thetas, rhos


# creating a image:
img = np.zeros((256, 256), dtype=np.uint8)
cv2.line(img, (100, 100), (200, 200), 255)

# applying canny edge detection:
edges = cv2.Canny(img, 100, 200)

# Sub-plotting image, Canny edge:
plt.subplot(121)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.title('Original')
plt.subplot(122)
plt.imshow(edges, cmap='gray', interpolation='nearest')
plt.title('Canny edges')
plt.show()

# calling hough_line transform function:
accumulator, thetas, rhos = hough_line(img)

# peak finding based on max votes:
mav_votes = np.argmax(accumulator)
print('mav_votes =', mav_votes)
rho = rhos[int(mav_votes/accumulator.shape[1])]
theta = thetas[mav_votes % accumulator.shape[1]]
print('rho =', rho, ', theta =', np.rad2deg(theta))

# getting coordinates of the line we want to draw/project:
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*a)
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*a)

# getting the result:
cv2.line(img, (x1, y1), (x2, y2), 255)

# Sub-plotting accumulator, result:
plt.subplot(121)
plt.imshow(accumulator, origin='lower')
plt.title('Accumulator')
plt.subplot(122)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.title('result')
plt.show()
