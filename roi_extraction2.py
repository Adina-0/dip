import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = './Data/4. Bougainvillea/Image-290_2024-02-05.jpg'
image = cv2.imread(image_path)

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extract the Saturation channel to emphasize color information
s_channel = hsv_image[:, :, 1]

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(s_channel, (15, 15), 0)

# Perform adaptive thresholding
threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
threshold_blurred = cv2.GaussianBlur(threshold, (111, 111), 0)
threshold2 = cv2.threshold(threshold_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Detect contours
contours, _ = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area and circularity
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    print(f"Area: {area}, Circularity: {circularity}")
    if 5000 < area < 50000 and 0 < circularity < 1:
        filtered_contours.append(contour)

# Draw filtered contours on the original image
output_image = image.copy()
cv2.drawContours(output_image, filtered_contours, -1, (0, 0, 255), 10)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Thresholded Image")
plt.imshow(threshold2, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Detected Pollen")
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
