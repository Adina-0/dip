# extract structural features from the pollen images to be used in the classification task
# proposed features: contrast, correlation, energy, entropy, homogeneity, relative areas and objects
# choice of features following the paper: M. del Pozo-Baños, et al., Features extraction techniques for pollen grain classification, Neurocomputing (2014), http://dx.doi.org/10.1016/j.neucom.2014.05.085i

import numpy as np
import cv2
import utils
from scipy.ndimage import label
from skimage import feature
import matplotlib.pyplot as plt

def grey_level_co_occurrence_matrix(img, mask, grey_levels=128):
    # Convert the image to grayscale
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Scale the image to the specified grey levels
    img_scaled = (img_gray * (grey_levels - 1) / 255).astype(np.uint8)

    # Mask the image to exclude background pixels (both adjacent pixels must be non-zero)
    valid_mask = (mask[:-1, :-1] != 0) & (mask[1:, 1:] != 0)
    row_vals = img_scaled[:-1, :-1][valid_mask]
    col_vals = img_scaled[1:, 1:][valid_mask]

    # Initialize the co-occurrence matrix
    glcm = np.zeros((grey_levels, grey_levels), dtype=np.float64)

    # Use NumPy's advanced indexing to calculate the co-occurrence matrix
    for row, col in zip(row_vals, col_vals):
        glcm[row, col] += 1

    # Normalize the co-occurrence matrix
    if glcm.sum() > 0:
        glcm /= glcm.sum()

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


def entropy(image, mask=None, channels=None):
    """
    Calculate entropy for specific channels of an image, considering a mask.

    Parameters:
    image (ndarray): Input image in BGR format.
    mask (ndarray): Binary mask to specify the region of interest (same dimensions as the image).
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

        # Apply mask
        if mask is not None:
            selected_channel = selected_channel[mask > 0]

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


def objects_and_areas(img, mask=None):
    """
    Calculate number of objects, mean and std of their area for a given greyscale image.

    Parameters:
    image (ndarray): gray scale image.
    mask (ndarray): binary mask of area to be considered.

    Returns:
    number of objects, mean area of objects, variance of area of objects
    """
    obj_and_areas = {}

    img[np.isnan(img)] = 0 # set NaN values to 0
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) # normalize image

    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        _, img_binary = cv2.threshold(1-img, threshold, 1, cv2.THRESH_BINARY)
        img_binary = np.where(mask, img_binary, 0)

        # plt.imshow(img, cmap="gray")
        # plt.show()
        #
        # plt.imshow(img_binary, cmap="gray")
        # plt.show()

        # Calculate relative objects (number of connected components)
        _, num_objects = label(img_binary)

        # Calculate mean area of objects (pixels with value 1)
        area_mean = np.sum(img_binary) / num_objects

        obj_and_areas[f"LBP: Objects (thresh={threshold})"] = num_objects
        obj_and_areas[f"LBP: Mean Area (thresh={threshold})"] = area_mean

    return obj_and_areas


def lbp_features(img, radius=3, points=None, method="uniform", mask=None, plot=False):
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

    # # Adjust the mask to account for valid neighbors
    # if mask is not None:
    #     # Initialize the valid mask with the shape of the input image
    #     valid_mask = np.ones_like(mask, dtype=bool)
    #
    #     # Iterate over every pixel in the image
    #     for i in range(radius, img_gray.shape[0] - radius):
    #         for j in range(radius, img_gray.shape[1] - radius):
    #             # Get the neighborhood of the current pixel within the radius
    #             neighborhood = mask[i - radius:i + radius + 1, j - radius:j + radius + 1]
    #
    #             # Check if all neighboring pixels are valid (i.e., part of the mask)
    #             if np.all(neighborhood):  # All neighbors should be valid (True)
    #                 valid_mask[i, j] = True
    #             else:
    #                 valid_mask[i, j] = False
    #
    #     # Apply the valid mask to LBP (invalid regions set to NaN)
    #     lbp = np.where(valid_mask, lbp, np.nan)

    lbp = np.where(mask, lbp, np.nan)

    # Plot the results
    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(img_gray, cmap="gray")
        ax[1].imshow(lbp, cmap="gray")
        plt.show()

    # Compute histogram, ignoring NaN values
    hist, _ = np.histogram(
        lbp[~np.isnan(lbp)],  # Only consider non-NaN values
        bins=np.arange(0, points + 3),  # Add 2 for the 'uniform' method range
        range=(0, points + 2)
    )

    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Avoid division by zero

    return lbp, hist


def gabor_mean(img, mask, plot=False):
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

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(img_gray, cmap="gray")
        ax[1].imshow(filtered_image, cmap="gray")
        plt.show()

    # Calculate the mean, ignoring NaN values
    gabor_mean = np.nanmean(filtered_image)
    return gabor_mean


def fourier_mean(masked_img, mask, plot=False):
    # Ensure the input is grayscale
    if masked_img.ndim == 3:
        img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = masked_img

    # Apply a Gaussian window to reduce edge artifacts
    rows, cols = img_gray.shape
    x, y = np.meshgrid(
        np.linspace(-1, 1, cols), np.linspace(-1, 1, rows), indexing="xy"
    )
    gaussian_window = np.exp(-(x ** 2 + y ** 2) / 0.1)
    windowed_image = img_gray * mask * gaussian_window

    # Normalize the windowed image by the area of the mask
    # Avoid division by zero by adding a small epsilon
    mask_area = np.sum(mask)  # Total number of 1s in the mask
    windowed_image_normalized = windowed_image / (mask_area + 1e-7)

    # Perform FFT on the masked image
    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))  # Shift zero-frequency component to the center
    psd_spectrum = np.log(np.abs(f_shift) ** 2)

    # Perform FFT on the windowed image
    f_windowed = np.fft.fftshift(np.fft.fft2(windowed_image_normalized))
    psd_spectrum_windowed = np.log1p(np.abs(f_windowed) ** 2)

    # Frequency axis values
    fx = np.fft.fftshift(np.fft.fftfreq(cols))  # Frequency range for x-axis
    fy = np.fft.fftshift(np.fft.fftfreq(rows))  # Frequency range for y-axis

    # Plot results
    if plot is True:
        plot_fourier(img_gray, windowed_image, psd_spectrum, psd_spectrum_windowed, fx, fy)

    # Calculate the mean, ignore NaN values
    fft_mean = np.nanmean(psd_spectrum_windowed)

    return fft_mean



def structural_features(img, mask, plot_bool=False):

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grayscale, (5, 5), 0)
    enhanced = cv2.equalizeHist(denoised)

    if plot_bool:
        plot_preprocessing(grayscale, denoised, enhanced)
    # Compute the grey-level co-occurrence matrix
    glcm = grey_level_co_occurrence_matrix(grayscale, mask)

    contrast_val = contrast(glcm)
    correlation_val = correlation(glcm)
    energy_val = energy(glcm)
    entropy_val = entropy(img, mask)
    homogeneity_val = homogeneity(glcm)

    gabor_mean_val = gabor_mean(enhanced, mask, plot=plot_bool)
    fourier_mean_val = fourier_mean(enhanced, mask, plot=plot_bool)

    lbp_img, lbp_hist = lbp_features(enhanced, mask=mask, plot=plot_bool)
    lpb_descriptors = {**{f"LBP {key}": value for key, value in utils.central_tendency(lbp_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.dispersion(lbp_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.distribution_shape(lbp_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.range_values(lbp_hist).items()},
                       **{f"LBP {key}": value for key, value in utils.entropy(lbp_hist).items()}}

    lbp_obj_and_areas = objects_and_areas(lbp_img, mask) # future work: include variance of area (needs more complex calculation)

    return {"Contrast": contrast_val, "Correlation": correlation_val, "Energy": energy_val,
            "Homogeneity": homogeneity_val, "Gabor Mean": gabor_mean_val,
            "Fourier Mean": fourier_mean_val} | entropy_val | lpb_descriptors



def plot_fourier(img_gray, windowed_image, psd_spectrum, psd_spectrum_windowed, fx, fy):
    plt.figure(figsize=(10, 10))

    # Plot masked image
    plt.subplot(2, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Masked Image')
    plt.axis('off')

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


def plot_preprocessing(grayscale, denoised, enhanced):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
