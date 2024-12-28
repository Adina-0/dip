import cv2
import matplotlib.pyplot as plt
import preprocessing as pp

# Load image
image = cv2.imread('./Data/4. Bougainvillea/Image-296_2024-02-05.jpg')

enhanced_image = pp.apply_clahe_on_lab(image)

# Step 3: Convert to HSV and extract the saturation (S) channel
hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2HSV)
s_channel = hsv_image[:, :, 1]

# Step 4: Equalize the histogram of the saturation channel
equalized_s = cv2.equalizeHist(s_channel)

# Step 5: Binarize the saturation channel
threshold = 0.2 * 255  # Since OpenCV uses values between 0-255
_, binary_mask = cv2.threshold(equalized_s, threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)

# Step 6: Eliminate noise using morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
cleaned_mask_close = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
cleaned_mask = cv2.morphologyEx(cleaned_mask_close, cv2.MORPH_OPEN, kernel, iterations=1)

# Step 7: Apply the mask to the original image
final_result = cv2.bitwise_and(enhanced_image, enhanced_image, mask=cleaned_mask)

plt.imshow(equalized_s, cmap='gray')
plt.imshow(final_result)
plt.show()

# Find contours
contours, _ = cv2.findContours(equalized_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image, cmap='gray')
print("Number of contours found:", len(contours))

# Draw and extract ROIs
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none'))

plt.show()

