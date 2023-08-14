import cv2
import numpy as np

# Load an image from file
path = "E:/computer_vision/test3.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Apply Prewitt edge detection
prewitt_x = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
prewitt_y = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

# Calculate the gradient magnitude
gradient_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)

# Normalize the gradient magnitude to the range [0, 255]
normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the original and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Prewitt Edge Detection', normalized_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
