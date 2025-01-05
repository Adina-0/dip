import cv2 as cv
import numpy as np

def process_image_to_black_background_old(image_path):
    # Load and preprocess the image
    img = cv.imread(image_path)
    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mean_values = np.mean(image, axis=(0, 1))
    height, width = image.shape[:2]
    
    # Create a blob from the image
    blob = cv.dnn.blobFromImage(image, scalefactor=1, size=(width, height), 
                                mean=mean_values, swapRB=False, crop=False)
    blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)
    
    threshold = 10  # Threshold for black pixel detection
    
    # Create a binary mask for black pixels
    black_pixel_mask = np.all(blob_for_plot <= threshold, axis=2)
    white_mask = (black_pixel_mask * 1).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    final_img = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)
    
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
    
    # Create a mask for the original image
    mask = np.zeros_like(white_mask)
    cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    
    # Apply the mask to the original image
    masked_image = cv.bitwise_and(img, img, mask=mask)
    
    # Turn the background black
    background_black = masked_image.copy()
    background_black[mask == 0] = 0
    # cv.imshow("Masked Result", background_black)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    return background_black, mask


def process_image_to_black_background(image_path):
    # Load and preprocess the image
    img = cv.imread(image_path)
    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_contours = img.copy()
    img_blackBackgound = img.copy()
    mean_values = np.mean(image, axis=(0, 1))
    height, width = image.shape[:2]

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

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    final_img = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)

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

    binary_image = (binary_mask // 255).astype(np.uint8)

    cv.drawContours(img_contours, [largest_contour], -1, (0, 255, 0), 5)

    # Create a black mask
    mask = np.zeros_like(white_mask)  # Start with an all-black mask
    cv.drawContours(mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    # Apply the mask to the original image
    masked_image = cv.bitwise_and(img_blackBackgound, img_blackBackgound, mask=mask)
    # Turn the background black
    background_black = masked_image.copy()
    background_black[mask == 0] = 0

    return background_black, mask, largest_contour, binary_image