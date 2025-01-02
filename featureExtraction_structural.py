# extract structural features from the pollen images to be used in the classification task
# proposed features: contrast, correlation, energy, entropy, homogeneity, relative areas and objects
# following the paper: M. del Pozo-Baños, et al., Features extraction techniques for pollen grain classification, Neurocomputing (2014), http://dx.doi.org/10.1016/j.neucom.2014.05.085i

import numpy as np
import cv2
from scipy.ndimage import label


def grey_level_co_occurrence_matrix(img):
    # Define the number of grey levels
    grey_levels = 256

    # Define the co-occurrence matrix
    glcm = np.zeros((grey_levels, grey_levels))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Define the number of rows and columns in the image
    rows, cols = img_gray.shape

    # compute the co-occurrence matrix
    for i in range(rows):
        for j in range(cols):
            # don't consider the background pixels (valued 0)
            if i + 1 < rows and j + 1 < cols and img_gray[i, j] != 0 and img_gray[i + 1, j + 1] != 0:
                glcm[img_gray[i, j], img_gray[i + 1, j + 1]] += 1
                glcm[img_gray[i + 1, j + 1], img_gray[i, j]] += 1

    # Normalize the co-occurrence matrix
    glcm = glcm / np.sum(glcm)

    return glcm


def contrast(glcm):
    contrast = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            contrast += (i - j) ** 2 * glcm[i, j]

    return contrast


def correlation(glcm):
    # Calculate means (μ) for rows and columns
    i_indices = np.arange(glcm.shape[0])
    j_indices = np.arange(glcm.shape[1])

    mu_i = np.sum(i_indices * np.sum(glcm, axis=1))  # Mean for rows
    mu_j = np.sum(j_indices * np.sum(glcm, axis=0))  # Mean for columns

    # Calculate standard deviations (σ) for rows and columns
    sigma_i = np.sqrt(np.sum((i_indices - mu_i) ** 2 * np.sum(glcm, axis=1)))
    sigma_j = np.sqrt(np.sum((j_indices - mu_j) ** 2 * np.sum(glcm, axis=0)))

    # Compute the correlation
    correlation = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            correlation += ((i - mu_i) * (j - mu_j) * glcm[i, j]) / (sigma_i * sigma_j)

    return correlation



def energy(glcm):
    energy = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            energy += glcm[i, j] ** 2

    return energy


def entropy(image, channels=None):
    """
    Calculate entropy for specific channels of an image.

    Parameters:
    image (ndarray): Input image in BGR format.
    channels (list): List of channels to use ('B', 'S', 'V').

    Returns:
    float: Entropy value.
    """
    if channels is None:
        channels = ['B', 'S', 'V']
    entropy_values = []

    for channel in channels:
        # Extract the channel
        if channel == 'B':  # Blue channel from BGR
            selected_channel = image[:, :, 0]
        elif channel in ['S', 'V']:  # Convert to HSV and select channel
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if channel == 'S':
                selected_channel = hsv_image[:, :, 1]
            elif channel == 'V':
                selected_channel = hsv_image[:, :, 2]
        else:
            raise ValueError("Invalid channel. Use 'B', 'S', or 'V'.")

        # Calculate histogram counts
        hist = cv2.calcHist([selected_channel], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Normalize histogram to probabilities
        p = hist / np.sum(hist)

        # Avoid log(0) by masking zero probabilities
        p = p[p > 0]

        # Compute entropy
        entropy = -np.sum(p * np.log2(p))
        entropy_values.append(entropy)

    # Return entropy of all channels as a list
    return entropy_values


def homogeneity(glcm):
    homogeneity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneity += glcm[i, j] / (1 + np.abs(i - j))

    return homogeneity


def relative_areas_and_objects(img, thresholds=None):
    """
    Calculate relative areas and relative objects for an image at given thresholds.

    Parameters:
    image (ndarray): color image in BGR format.
    thresholds (list): List of threshold values.

    Returns:
    dict: A dictionary with thresholds as keys and (relative_area, relative_objects) as values.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    for threshold in thresholds:
        # Binarize the image
        _, img_binary = cv2.threshold(img_gray, 255 * threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Binary Image", img_binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Calculate relative area (fraction of pixels with value 1)
        relative_area = np.sum(img_binary) / img_binary.size

        # Calculate relative objects (number of connected components)
        _, num_objects = label(img_binary)

        results[threshold] = (relative_area, num_objects)

    return results


def structural_features(img):
    # Compute the grey-level co-occurrence matrix
    glcm = grey_level_co_occurrence_matrix(img)

    contrast_val = contrast(glcm)
    correlation_val = correlation(glcm)
    energy_val = energy(glcm)
    entropy_val = entropy(img)
    homogeneity_val = homogeneity(glcm)
    relative_areas_and_objects_val = relative_areas_and_objects(img)

    return {"Contrast": contrast_val, "Correlation": correlation_val, "Energy": energy_val,
            "Entropy": entropy_val, "Homogeneity": homogeneity_val, "Relative Areas and Objects": relative_areas_and_objects_val}

