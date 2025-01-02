"""


Descriptors featured:
mean intensity, intensity variation, mean R, mean G, mean B, mean brightness, brightness variation,
mean density, density variation, mean saturation, hue typical, hue variation


"""

import cv2
import numpy as np

def calculate_descriptors(img):
    img = img.astype(np.float32)

    # Split channels
    B, G, R = cv2.split(img)

    # Intensity (I = 1/3 * (R + G + B))
    intensity = (R + G + B) / 3
    mean_intensity = np.mean(intensity)
    intensity_variation = np.std(intensity)

    # Mean Red, Green, Blue
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)

    # Brightness (Br = 0.299R + 0.587G + 0.114B)
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    mean_brightness = np.mean(brightness)
    brightness_variation = np.std(brightness)

    # Density (D = (R + G + B) / (3 * max_value))
    density = (R + G + B) / 3  # Already normalized to [0, 1]
    mean_density = np.mean(density)
    density_variation = np.std(density)

    # Saturation Index (SI = 1 - min(R, G, B) / I, if I != 0)
    min_rgb = np.minimum(np.minimum(R, G), B)
    saturation_index = np.where(intensity > 0, 1 - min_rgb / intensity, 0)
    mean_saturation = np.mean(saturation_index)

    # Convert to HSV for Hue calculations
    hsv_image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)
    H = H.astype(np.float32)  # OpenCV Hue range is [0, 180]

    # Hue Typical and Hue Variation
    hue_typical = np.bincount(H.ravel().astype(int)).argmax()  # Most frequent hue
    hue_variation = np.std(H)

    # Results
    return {
        "mean_intensity": mean_intensity,
        "intensity_variation": intensity_variation,
        "mean_R": mean_R,
        "mean_G": mean_G,
        "mean_B": mean_B,
        "mean_brightness": mean_brightness,
        "brightness_variation": brightness_variation,
        "mean_density": mean_density,
        "density_variation": density_variation,
        "mean_saturation": mean_saturation,
        "hue_typical": hue_typical,
        "hue_variation": hue_variation,
    }

def color_descriptors(img, mask):
    # Ensure the mask is binary (1 for foreground, 0 for background)
    mask = mask.astype(bool)

    # Split channels and convert to float for accurate calculations
    B, G, R = cv2.split(img.astype(np.float32))

    # Intensity (I = 1/3 * (R + G + B))
    intensity = (R + G + B) / 3.0
    masked_intensity = intensity[mask]
    mean_intensity = np.mean(masked_intensity)
    intensity_variation = np.std(masked_intensity)

    # Mean Red, Green, Blue
    mean_R = np.mean(R[mask])
    mean_G = np.mean(G[mask])
    mean_B = np.mean(B[mask])

    # Brightness (Br = 0.299R + 0.587G + 0.114B)
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    masked_brightness = brightness[mask]
    mean_brightness = np.mean(masked_brightness)
    brightness_variation = np.std(masked_brightness)

    # Density (D = (R + G + B) / (3 * max_value))
    max_value = 255.0  # Assumes unnormalized image
    density = (R + G + B) / (3.0 * max_value)
    masked_density = density[mask]
    mean_density = np.mean(masked_density)
    density_variation = np.std(masked_density)

    # Saturation Index (SI = 1 - min(R, G, B) / I, if I != 0)
    min_rgb = np.minimum(np.minimum(R, G), B)
    epsilon = 1e-6  # To avoid division by zero
    saturation_index = np.where(intensity > epsilon, 1.0 - min_rgb / intensity, 0)
    masked_saturation = saturation_index[mask]
    mean_saturation = np.mean(masked_saturation)

    # Convert to HSV for Hue calculations
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)
    H = H.astype(np.float32)

    # Apply mask to Hue
    masked_hue = H[mask]

    # Hue Typical and Hue Variation
    if masked_hue.size > 0:  # Avoid errors if no foreground pixels
        hue_typical = np.bincount(masked_hue.astype(int)).argmax()  # Most frequent hue
        hue_variation = np.std(masked_hue)
    else:
        hue_typical = 0
        hue_variation = 0

    # Results
    return {
        "mean_intensity": mean_intensity,
        "intensity_variation": intensity_variation,
        "mean_R": mean_R,
        "mean_G": mean_G,
        "mean_B": mean_B,
        "mean_brightness": mean_brightness,
        "brightness_variation": brightness_variation,
        "mean_density": mean_density,
        "density_variation": density_variation,
        "mean_saturation": mean_saturation,
        "hue_typical": hue_typical,
        "hue_variation": hue_variation,
    }
