import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image_path = "E:/computer_vision/test.png"
image = cv2.imread(image_path)

# Split the image into color channels
b, g, r = cv2.split(image)

# Calculate histograms for each color channel
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# Create a figure with subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(12, 10))

# Plot the input image
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Input Image')
ax1.axis('off')

# Plot the histograms
ax2.plot(hist_b, color='blue')
ax2.set_title('Blue Histogram')
ax2.set_xlabel('Bins')
ax2.set_ylabel('Frequency')

ax3.plot(hist_g, color='green')
ax3.set_title('Green Histogram')
ax3.set_xlabel('Bins')
ax3.set_ylabel('Frequency')

ax4.plot(hist_r, color='red')
ax4.set_title('Red Histogram')
ax4.set_xlabel('Bins')
ax4.set_ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
