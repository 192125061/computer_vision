import cv2
import numpy as np

image_path = "E:/computer_vision/face1.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(image)

cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

result = cv2.bitwise_and(image, mask)

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
