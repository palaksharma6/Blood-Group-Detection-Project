import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the original fingerprint image
img = cv2.imread("ann.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Light blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(blurred)

# Step 4: Adaptive Thresholding to get binary image (black ridges, white BG)
binary = cv2.adaptiveThreshold(contrast, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Step 5: Morphological closing to connect broken ridges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Step 6: Save as BMP with white background and black ridges
cv2.imwrite("ann_fingerprint.bmp", closed)

# Step 7: Display result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Processed Like Your Sample")
plt.imshow(closed, cmap='gray')
plt.axis("off")
plt.show()
