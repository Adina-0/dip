import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "./Data/1.Ageratum/Image-9_2024-03-06_grain_0.png"

def convert_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# decorrelation stretching -- doesn't seem to work as intended
def decorrelation_stretch(img):
    # Reshape image to an MxN = 3 array
    reshaped = img.reshape(-1, 3).astype(np.float64)

    # Compute covariance matrix and eigenanalysis
    mean = reshaped.mean(axis=0)
    reshaped -= mean
    cov_matrix = np.cov(reshaped, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Normalize eigenvalues
    normalization_matrix = np.diag(1.0 / np.sqrt(eigvals))

    # Apply decorrelation stretching
    decorrelated = np.dot(np.dot(reshaped, eigvecs), normalization_matrix)
    decorrelated = np.dot(decorrelated, eigvecs.T) + mean

    # Clip to valid range
    decorrelated = np.clip(decorrelated, 0, 255).astype(np.uint8)

    return decorrelated.reshape(img.shape)

# alternative to decorrelation stretching, enhancing contrast
def apply_clahe_per_channel(image):
    # Split the channels
    channels = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_channels = [clahe.apply(ch) for ch in channels]

    # Merge the channels back
    enhanced_image = cv2.merge(enhanced_channels)
    return enhanced_image

# CLAHE on LAB (Lightness):
# The L channel in the LAB color space represents lightness, which can be enhanced independently without affecting color saturation.
def apply_clahe_on_lab(image):
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to the L (lightness) channel
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))

    # Convert back to RGB
    enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return enhanced_image


# use Gaussian of laplacian to find edges - no-no for now
def preprocess_findedges(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 224x224 (common size for deep learning models)
    image_resized = cv2.resize(image, (224, 224))

    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    image_blurred = cv2.GaussianBlur(image_gray, (3, 3), sigmaX=2)

    # Apply Laplacian operator for edge detection
    laplacian = cv2.Laplacian(image_blurred, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian)

    # Apply Canny edge detection
    edges = cv2.Canny(laplacian_abs.astype(np.uint8), threshold1=50, threshold2=100)

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax = ax.ravel()

    titles = ['Blurred Image', 'Laplacian', 'Canny Edges']
    images = [image_blurred, laplacian_abs, edges]

    for i in range(3):
        ax[i].imshow(images[i], cmap='gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()
    return edges


# best one so far, using the pipeline proposed by M. del Pozo-Banos et al. (M. del Pozo-Baños, et al., Features extraction techniques for pollen grain classification, Neurocomputing (2014), http://dx.doi.org/10.1016/j.neucom.2014.05.085i)
def preprocess_pipeline(image_path):
    image = cv2.imread(image_path)
    enhanced_image = apply_clahe_on_lab(image)
    enhanced_image2 = apply_clahe_per_channel(image) # results were not as good...

    # Step 3: Convert to HSV and extract the saturation (S) channel
    hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2HSV)
    s_channel = hsv_image[:, :, 1]

    # Step 4: Equalize the histogram of the saturation channel
    equalized_s = cv2.equalizeHist(s_channel)

    # Step 5: Binarize the saturation channel
    threshold = 0.75 * 255  # Since OpenCV uses values between 0-255
    threshold = 0.90 * 255  # Since OpenCV uses values between 0-255
    _, binary_mask = cv2.threshold(equalized_s, threshold, 255, cv2.THRESH_BINARY)

    # Step 6: Eliminate noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask_close = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=6)
    cleaned_mask = cv2.morphologyEx(cleaned_mask_close, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 7: Apply the mask to the original image
    final_result = cv2.bitwise_and(enhanced_image, enhanced_image, mask=cleaned_mask)


    # Visualization
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.ravel()

    titles = [
        'Original', 'Enhanced: CLAHE on RGB', 'HSV image', 'Equalized S channel', 'Binary Mask', 'Final Result'
    ]

    images = [
        image, enhanced_image, hsv_image, equalized_s, binary_mask, final_result
    ]

    for i in range(len(images)):
        if len(images[i].shape) == 3:  # Convert BGR to RGB for display
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax[i].imshow(images[i], cmap='gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()
    return final_result, binary_mask


def preprocessing_pipeline1(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize image to 224x224 (common size for deep learning models)
    image_resized = cv2.resize(image, (224, 224))

    # Convert to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur for noise reduction
    image_denoised = cv2.GaussianBlur(image_gray, (3, 3), sigmaX=2)
    image_denoised2 = cv2.edgePreservingFilter(image_gray, flags=cv2.CV_64F)

    # Histogram equalization (contrast enhancement)
    image_equalized = cv2.equalizeHist(image_denoised2.astype(np.uint8))

    # Sobel edge detection (gradient-based edge detection)
    sobel_x = cv2.Sobel(image_equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_equalized, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)

    # Otsu's thresholding for segmentation
    _, image_binary = cv2.threshold(image_equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Morphological Operations: Pollen grains can sometimes be connected or fragmented, and morphological operations can help clean up the image (e.g., removing small artifacts or closing gaps).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Elliptical kernel to better match pollen grain shapes

    # # Erosion: Removes small noise by eroding the white regions
    # image_eroded = cv2.erode(image_binary, kernel, iterations=5)
    #
    # # Dilation: Expands white regions, useful after erosion to restore object size
    # image_dilated = cv2.dilate(image_eroded, kernel, iterations=5)

    # Opening: Erosion followed by dilation, removes small noise
    image_opened = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel, iterations=6)

    # Closing: Dilation followed by erosion, closes small gaps
    image_closed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, kernel, iterations=5)

    _, edges2 = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
    removed_background = (255 - image_closed) / 255.0 * edges2

    # find holes in the pollen grains




    removed_background = cv2.bitwise_and(image_resized, image_resized, mask=(255 - image_closed).astype(np.uint8))

    # Visualization
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    ax = ax.ravel()

    titles = [
        'Original Image', 'Denoised', 'Histogram Equalized', 'Edges', 'Closed', 'Removed Background'
    ]
    images = [
        image, image_denoised, image_equalized, edges,
        image_closed, removed_background
    ]

    for i in range(len(images)):
        if len(images[i].shape) == 3:  # Convert BGR to RGB for display
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax[i].imshow(images[i], cmap='gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    binary_mask = cv2.bitwise_not(image_closed)

    plt.tight_layout()
    plt.show()
    return removed_background, binary_mask


def preprocessing_pipeline2(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Resize to a consistent size (224x224 for deep learning or appropriate size for features)
    image_resized = cv2.resize(image, (224, 224))

    # Step 3: Convert to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Step 4: Noise reduction using bilateral filter (preserves edges better than Gaussian blur)
    image_denoised = cv2.bilateralFilter(image_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 5: Contrast enhancement using CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_contrast = clahe.apply(image_denoised)

    # Step 6: Edge enhancement with Laplacian of Gaussian (LoG)
    # log_edges = cv2.Laplacian(image_contrast, cv2.CV_64F, ksize=5)
    # edges = cv2.convertScaleAbs(log_edges)

    # Step 6: (as Laplacian of Gaussian (LoG) edges didn't work well --> Canny
    edges = cv2.Canny(image_contrast, threshold1=80, threshold2=110)


    # Step 7: Binarization using adaptive thresholding (handles uneven lighting)
    image_binary = cv2.adaptiveThreshold(
        image_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )

    # Step 8: Morphological operations (Opening to clean noise, Closing to fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Shape suited for pollen grains
    image_opened = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel, iterations=2)
    image_closed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 9: Background removal by masking edges
    edges_mask = cv2.bitwise_and(image_closed, edges)
    background_removed = cv2.bitwise_and(image_contrast, image_contrast, mask=edges_mask)

    # Visualization
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    ax = ax.ravel()

    titles = [
        'Original Image', 'Grayscale', 'Denoised', 'Contrast Enhanced (CLAHE)', 'Edges (Canny)',
        'Binary (Adaptive)'
    ]
    images = [
        image, image_gray, image_denoised, image_contrast, edges,
        image_binary
    ]

    for i in range(len(images)):
        if len(images[i].shape) == 3:  # Convert BGR to RGB for display
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax[i].imshow(images[i], cmap='gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

    return background_removed


# still problems with 'sceletonizing'
def preprocessing_pipeline3(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Resize for consistency
    image_resized = cv2.resize(image, (224, 224))

    # Step 3: Convert to grayscale
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply median filtering to suppress fine details while preserving edges
    image_filtered = cv2.medianBlur(image_gray, 5)

    # Step 5: Enhance contrast globally (simple histogram equalization)
    image_contrast = cv2.equalizeHist(image_filtered)

    # Step 6: Large-scale smoothing to suppress small features
    large_kernel = (15, 15)  # Larger kernel to focus on overall structure
    image_smoothed = cv2.GaussianBlur(image_contrast, large_kernel, sigmaX=10)

    # Step 7: Edge detection using Canny with tuned thresholds
    edges = cv2.Canny(image_smoothed, threshold1=50, threshold2=100)

    # Step 8: Morphological closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Step 9: Simplify edges using skeletonization
    skeleton = cv2.ximgproc.thinning(edges_closed)

    # Step 10: Optional inversion to highlight main structures
    output = 255 - skeleton

    # Visualization
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.ravel()

    titles = [
        'Original Image', 'Grayscale', 'Median Filtered', 'Contrast Enhanced',
        'Edges (Canny)', 'Skeletonized Output'
    ]
    images = [
        image, image_gray, image_filtered, image_contrast, edges_closed, output
    ]

    for i in range(len(images)):
        if len(images[i].shape) == 3:  # Convert BGR to RGB for display
            ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax[i].imshow(images[i], cmap='gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

