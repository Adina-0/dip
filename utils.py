import numpy as np
import scipy.stats as stats

def central_tendency(data):
    mean = np.mean(data)  # Mean
    median = np.median(data)  # Median
    mode = stats.mode(data)[0]  # Mode
    return {"Mean": mean, "Median": median, "Mode": mode}

def dispersion(data):
    std_dev = np.std(data)  # Standard deviation
    iqr = stats.iqr(data)  # Interquartile range (IQR)
    return {"Standard Deviation": std_dev, "IQR": iqr}

def distribution_shape(data):
    skewness = stats.skew(data)  # Skewness
    kurtosis = stats.kurtosis(data)  # Kurtosis
    return {"Skewness": skewness, "Kurtosis": kurtosis}

def range_values(data):
    minimum = np.min(data)  # Minimum
    maximum = np.max(data)  # Maximum
    return {"Minimum": minimum, "Maximum": maximum}

def entropy(data):
    entropy = stats.entropy(np.histogram(data, bins='auto')[0])  # Entropy
    return {"Entropy": entropy}

def flatten_vector(vector):
    flattened_vector = []
    for item in vector:
        if isinstance(item, dict):
            values = []
            for value in item.values():
                if isinstance(value, (list, tuple)):
                    values.extend(value)
            flattened_vector.extend(values)
        elif isinstance(item, tuple) or isinstance(item, list):
            flattened_vector.extend(item)  # Add tuple or list elements
        elif isinstance(item, np.ndarray):
            flattened_vector.extend(item.ravel()) # Add flattened numpy array
        else:
            flattened_vector.append(item)

    return flattened_vector


def calculate_descriptor_stats(all_data):
    descriptor_stats = {}  # {class_name: {descriptor_name: {'mean': value, 'std': value}}}

    for class_name, descriptors in all_data.items():
        descriptor_stats[class_name] = {}
        for descriptor_name, values in descriptors.items():
            mean = np.mean(values)
            std = np.std(values)
            descriptor_stats[class_name][descriptor_name] = {'mean': mean, 'std': std}

    return descriptor_stats