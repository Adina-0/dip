# extract structural features from the pollen images to be used in the classification task
# proposed features: contrast, correlation, energy, entropy, homogeneity, relative areas and objects
# following the paper: M. del Pozo-Baños, et al., Features extraction techniques for pollen grain classification, Neurocomputing (2014), http://dx.doi.org/10.1016/j.neucom.2014.05.085i

import numpy as np
import cv2
import utils
from scipy.ndimage import label
from skimage import feature
import matplotlib.pyplot as plt


def grey_level_co_occurrence_matrix(img, mask):
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
            if i + 1 < rows and j + 1 < cols and mask[i, j] != 0 and mask[i + 1, j + 1] != 0:
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
    entropy_values = {}

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
        entropy_values[f"Entropy Channel {channel}"] = entropy

    # Return entropy of all channels as a dict
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


def lbp_features(img, radius=5, points=None, method="uniform", mask=None):
    """
    Compute Local Binary Pattern (LBP) features of an image.

    Parameters:
    - img: ndarray
        Input grayscale image.
    - radius: int
        Radius of the circle used for computing LBP (default: 4).
    - points: int
        Number of points sampled on the circle (default: 8 * radius).
    - method: str
        LBP computation method ('uniform', 'nri_uniform', 'default', etc.).
    - mask: ndarray (optional)
        Binary mask to specify the region of interest in the image.

    Returns:
    - hist: ndarray
        Normalized histogram of LBP features.
    """
    # Ensure the image is grayscale
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Set default number of points
    if points is None:
        points = 8 * radius

    # Compute LBP
    lbp = feature.local_binary_pattern(img_gray, points, radius, method=method)

    # Apply mask if provided
    if mask is not None:
        lbp = np.where(mask, lbp, np.nan)  # Masked regions set to NaN

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].axis("off")
    # ax[1].axis("off")
    # ax[0].imshow(img_gray, cmap="gray")
    # ax[1].imshow(lbp, cmap="gray")
    # plt.show()

    # Compute histogram, ignoring NaN values
    hist, _ = np.histogram(
        lbp[~np.isnan(lbp)],  # Only consider non-NaN values
        bins=np.arange(0, points + 3),  # Add 2 for the 'uniform' method range
        range=(0, points + 2)
    )

    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Avoid division by zero

    return hist


def gabor_mean(img, mask):
    # Ensure the image is grayscale
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Create a Gabor kernel
    gabor_kernel = cv2.getGaborKernel((31, 31), 10, np.pi / 4., 10.0, 0.5, 0, ktype=cv2.CV_32F)

    # Apply the Gabor filter
    filtered_image = cv2.filter2D(img_gray, cv2.CV_32F, gabor_kernel)
    filtered_image = np.where(mask, filtered_image, np.nan)  # Masked pixels are set to NaN

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].imshow(img_gray, cmap="gray")
    ax[1].imshow(filtered_image, cmap="gray")
    plt.show()

    # Calculate the mean, ignoring NaN values
    gabor_mean = np.nanmean(filtered_image)
    return gabor_mean


def fourier_mean(masked_img, mask):
    # Ensure grayscale
    if len(masked_img.shape) == 3:
        img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = masked_img

    # Apply FFT to the masked image
    f = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f)  # Shift zero-frequency component to the center
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    psd_spectrum = np.log(np.abs(f_shift) ** 2 + 1)

    # Apply a window (e.g., Gaussian) to reduce artifacts induced by the mask
    rows, cols = img_gray.shape
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    gaussian_window = np.exp(-(x ** 2 + y ** 2) / 0.5)
    windowed_mask = mask * gaussian_window
    windowed_image = img_gray * windowed_mask

    # Perform FFT on the windowed image
    f_windowed = np.fft.fft2(windowed_image)
    f_shift_windowed = np.fft.fftshift(f_windowed)
    magnitude_spectrum_windowed = 20 * np.log(np.abs(f_shift_windowed) + 1)
    psd_spectrum_windowed = np.log(np.abs(f_shift_windowed) ** 2 + 1)

    # Frequency axis values
    fx = np.fft.fftshift(np.fft.fftfreq(cols))  # Frequency range for x-axis
    fy = np.fft.fftshift(np.fft.fftfreq(rows))  # Frequency range for y-axis

    # Plot results
    plt.figure(figsize=(10, 10))

    # Plot masked image
    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Masked Image')
    plt.axis('off')  # Remove axis labels

    # Plot PSD spectrum
    plt.subplot(2, 2, 2)
    plt.imshow(psd_spectrum, cmap='gray', extent=[fx[0], fx[-1], fy[-1], fy[0]])
    plt.title('PSD Spectrum of Masked Image')
    plt.xlabel('Frequency (fx)')
    plt.ylabel('Frequency (fy)')

    # Plot windowed image
    plt.subplot(2, 2, 3)
    plt.imshow(windowed_image, cmap='gray')
    plt.title('Windowed Masked Image')
    plt.axis('off')  # Remove axis labels

    # Plot PSD spectrum of windowed image
    plt.subplot(2, 2, 4)
    plt.imshow(psd_spectrum_windowed, cmap='gray', extent=[fx[0], fx[-1], fy[-1], fy[0]])
    plt.title('PSD Spectrum of Windowed Image')
    plt.xlabel('Frequency (fx)')
    plt.ylabel('Frequency (fy)')

    plt.tight_layout()
    plt.show()

    # Calculate the mean, ignore NaN values
    fft_mean = np.nanmean(psd_spectrum_windowed)

    return fft_mean



def structural_features(img, mask):
    # Compute the grey-level co-occurrence matrix
    glcm = grey_level_co_occurrence_matrix(img, mask)

    contrast_val = contrast(glcm)
    correlation_val = correlation(glcm)
    energy_val = energy(glcm)
    entropy_val = entropy(img)
    homogeneity_val = homogeneity(glcm)
    relative_areas_and_objects_val = relative_areas_and_objects(img)
    gabor_mean_val = gabor_mean(img, mask)
    fourier_mean_val = fourier_mean(img, mask)

    lpb_hist = lbp_features(img, mask=mask)
    lpb_descriptors = {**{f"LBP {key}": value for key, value in utils.central_tendency(lpb_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.dispersion(lpb_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.distribution_shape(lpb_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.range_values(lpb_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.entropy(lpb_hist).items()}}

    # excluded for now: "Relative Areas and Objects": relative_areas_and_objects_val
    return {"Contrast": contrast_val, "Correlation": correlation_val, "Energy": energy_val,
            "Homogeneity": homogeneity_val, "Gabor Mean": gabor_mean_val,
            "Fourier Mean": fourier_mean_val} | entropy_val | lpb_descriptors
