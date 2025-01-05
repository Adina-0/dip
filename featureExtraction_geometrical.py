import cv2 as cv
import numpy as np
import math


def geometric_features(masked_image, mask, largest_contour, binary_image):
    # Area
    area = cv.contourArea(largest_contour)

    # bounding box
    x, y, width, height = cv.boundingRect(largest_contour)
    area_b = width * height
    bounding_boxes = [(x, y, width, height)]

    # extent
    extent = area / area_b

    # aspect ratio
    aspect_ratio = width / height

    # centroid
    white_pixels = np.argwhere(binary_image == 1)
    C_x = np.mean(white_pixels[:, 1])  # x-coordinate (horizontal)
    C_y = np.mean(white_pixels[:, 0])
    centroid = (C_x, C_y)

    # ellipse_shape
    ellipse = cv.fitEllipse(largest_contour)
    center, axes, orientation = ellipse

    # Major axis length
    majoraxis_length = max(axes)

    # Minor axis length
    minoraxis_length = min(axes)

    # eccentricity
    eccentricity = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)

    # convex_area
    hull = cv.convexHull(largest_contour)
    cv.fillConvexPoly(binary_image, hull, 1)
    convex_area = np.count_nonzero(binary_image == 1)

    # The Equivalent diameter
    equivalent_diameter = 2 * math.sqrt(area / math.pi)

    # Solidity
    solidity = area / convex_area

    # Perimeter
    perimeter = cv.arcLength(largest_contour, closed=True)

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Thickness
    kernel_t = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    thickness = 0
    eroded = binary_image.copy()

    while cv.countNonZero(eroded) > 0:  # Check if there are still white pixels
        eroded = cv.erode(eroded, kernel_t, iterations=1)
        thickness += 1

    # Weighted Centroid
    intensity_image = masked_image.copy()
    intensity_image = cv.cvtColor(intensity_image, cv.COLOR_BGR2GRAY)
    # Calculate pixel coordinates and intensities in the masked region
    coords = np.column_stack(np.where(mask > 0))
    x_coords, y_coords = coords[:, 1], coords[:, 0]
    intensities = intensity_image[mask > 0]
    # Compute the Weighted Centroid
    weighted_x = np.sum(x_coords * intensities) / np.sum(intensities)
    weighted_y = np.sum(y_coords * intensities) / np.sum(intensities)
    weighted_centroid = (weighted_x, weighted_y)

    minoraxis_length = min(axes)
    # return [area, bounding_boxes, extent, aspect_ratio, centroid, majoraxis_length, minoraxis_length, eccentricity,
    #         convex_area, equivalent_diameter, solidity, perimeter, circularity, thickness, weighted_centroid]

    return {
        "Area": area,
        # "Bounding Boxes": bounding_boxes,
        "Extent": extent,
        "Aspect Ratio": aspect_ratio,
        # "Centroid": centroid,
        "Major Axis Length": majoraxis_length,
        "Minor Axis Length": minoraxis_length,
        "Eccentricity": eccentricity,
        "Convex Area": convex_area,
        "Equivalent Diameter": equivalent_diameter,
        "Solidity": solidity,
        "Perimeter": perimeter,
        "Circularity": circularity,
        "Thickness": thickness,
        # "Weighted Centroid": weighted_centroid
    }