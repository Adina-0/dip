import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def pad_image(image, max_dim):
    """
    Pads the image to the specified max dimension by extending and smoothing the border pixels.
    """
    height, width = image.shape[:2]

    # Calculate padding
    top = (max_dim - height) // 2
    bottom = max_dim - height - top
    left = (max_dim - width) // 2
    right = max_dim - width - left

    # Apply the padding
    padded_image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_image


def process_image_to_black_background(image_path, max_dim=None, pad=False):
    # Load and preprocess the image
    img = cv.imread(image_path)
    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mean_values = np.mean(image, axis=(0, 1))
    height, width = image.shape[:2]
    total_pixels = height * width
    
    if max_dim is None:
        max_dim = max(height, width)
    
    # Create a blob from the image
    blob = cv.dnn.blobFromImage(image, scalefactor=1, size=(width, height), 
                                mean=mean_values, swapRB=False, crop=False)
    blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)
    
    if blob_for_plot.ndim == 2:
     gray_image = blob_for_plot
    elif blob_for_plot.ndim == 3:
    # Convert to grayscale if it has multiple channels
     gray_image = cv.cvtColor(blob_for_plot, cv.COLOR_BGR2GRAY)

    # Normalize the image to the range [0, 255] and convert to uint8
    gray_image_normalized = cv.normalize(gray_image, None, 0, 255, cv.NORM_MINMAX)
    gray_image_uint8 = gray_image_normalized.astype(np.uint8)

    # Apply Otsu's thresholding
    _, otsu_threshold = cv.threshold(gray_image_uint8, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Create a binary mask with values 0 and 255
    white_mask = (otsu_threshold == 255).astype(np.uint8) * 255
    
    if 0 <= total_pixels < 100000:
        k_size = 3
        k_size1 = 9
    elif 100000 <= total_pixels <= 200000:
        k_size = 5
        k_size1 = 10
    elif 200000 <= total_pixels <= 250000: 
        k_size = 9
        k_size1 = 18
    elif 250000 <= total_pixels <= 300000: 
        k_size = 12
        k_size1 = 25
    else: 
        k_size = 15
        k_size1 = 30
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size,k_size))

    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size1,k_size1))

    #Seperate the noise
    final_img = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel1)
    final_img = cv.erode(final_img, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv.findContours(final_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    
    # Create a binary mask
    binary_mask = np.zeros_like(white_mask, dtype=np.uint8)  # Start with all-black mask
    if largest_contour is not None:
    # Fill the largest contour with 255 (white) in the binary mask
     cv.drawContours(binary_mask, [largest_contour], -1, 255, thickness=cv.FILLED)
     binary_mask = cv.dilate(binary_mask, kernel1, iterations = 1)

    # Convert the binary mask to values of 1 and 0
    binary_image = (binary_mask // 255).astype(np.uint8)
    contour, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = contour[0]

    #Create a black mask
    mask = np.zeros_like(white_mask)  # Start with an all-black mask
    cv.drawContours(mask, contour, -1, 255, thickness=cv.FILLED)
    
    # Apply the mask to the original image
    masked_image = cv.bitwise_and(img, img, mask=mask)
    
    # Turn the background black
    background_black = masked_image.copy()
    background_black[mask == 0] = 0

    if not pad:
        return background_black, mask, contour, binary_image

    # Pad the output to the maximum dimension
    background_black = pad_image(background_black, max_dim)
    mask = pad_image(mask, max_dim)
    binary_image = pad_image(binary_image, max_dim)

    # Pad the largest contour to the maximum dimension
    top = (max_dim - height) // 2
    left = (max_dim - width) // 2
    contour += np.array([left, top])

    # plot all
    # Create a blank image with the same dimensions as the original image
    contour_image = np.zeros_like(background_black)

    # Draw the largest contour on the blank image
    if largest_contour is not None:
        cv.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), thickness=5)

    # plot all
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax1.imshow(cv.cvtColor(contour_image, cv.COLOR_BGR2RGB))  # Display the image with the contour drawn on it
    ax1.set_title('Contour')

    ax2 = fig.add_subplot(222)
    ax2.imshow(mask, cmap='gray')  # Display mask
    ax2.set_title('Mask')

    ax3 = fig.add_subplot(223)
    ax3.imshow(cv.cvtColor(background_black, cv.COLOR_BGR2RGB))  # Display background with blacked-out areas
    ax3.set_title('Background Black')

    ax4 = fig.add_subplot(224)
    ax4.imshow(binary_image, cmap='gray')  # Display binary image
    ax4.set_title('Binary Image')

    plt.show()

    return background_black, mask, contour, binary_image



def find_global_max_dimension(data_path):
    max_dim = 0
    for img_folder in os.listdir(data_path):
        if img_folder[0] == ".":
            continue
        img_folder = os.path.join(data_path, img_folder + "/")
        for img_path in os.listdir(img_folder):
            if img_path[0] == ".":
                continue
            path = os.path.join(img_folder, img_path)
            img = cv.imread(path)
            height, width = img.shape[:2]
            max_dim = max(max_dim, height, width)

    return max_dim

