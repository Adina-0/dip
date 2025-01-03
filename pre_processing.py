import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def process_image_to_black_background(image_path):
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