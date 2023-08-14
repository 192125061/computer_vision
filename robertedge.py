import cv2
import numpy as np

path = "E:/computer_vision/test3.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Define the Robert's edge detection kernels
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

# Apply Robert's edge detection
roberts_x_result = cv2.filter2D(image, cv2.CV_64F, roberts_x)
roberts_y_result = cv2.filter2D(image, cv2.CV_64F, roberts_y)

# Calculate the gradient magnitude
gradient_magnitude = np.sqrt(roberts_x_result**2 + roberts_y_result**2)

# Normalize the gradient magnitude to the range [0, 255]
normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the original and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow("Robert's Edge Detection", normalized_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
