import numpy as np
import cv2
from matplotlib import pyplot as plt

path = "E:/computer_vision/shapes.png"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris corner detection
corner_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
corner_response = cv2.dilate(corner_response, None)

# Threshold for corner detection
threshold = 0.01 * corner_response.max()
corner_image = np.copy(img)
corner_image[corner_response > threshold] = [0, 0, 255]  # Highlight corners in red

plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
plt.show()













