import numpy as np
import csv
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


def write_csv_stats(data, output_file):
    """
    Writes a CSV file from a dictionary with the structure {class_name: {descriptor_name: {"mean": value, "std": value}}, {descriptor_name2: {"mean": value2, "std": value2}, ...}.

    Args:
        data (dict): Dictionary containing the data.
        output_file (str): Path to the output CSV file.
    """
    # Collect all unique descriptor names
    descriptors = set()
    for class_data in data.values():
        descriptors.update(class_data.keys())
    descriptors = sorted(descriptors)  # sort descriptors alphabetically

    # Write the CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        header = ['']
        for class_name in data.keys():
            header.extend([class_name])
            header.extend([''])
        writer.writerow(header)

        # Write the subheader row (mean and std for each class)
        subheader = ['Descriptors'] + ['mean', 'std'] * len(data)
        writer.writerow(subheader)

        # Write each descriptor row
        for descriptor in descriptors:
            row = [descriptor]
            for class_name in data:
                # Get the mean and std for the current descriptor, default to None if not present
                values = data[class_name].get(descriptor, {})
                mean = f"{values.get('mean', ''):.2f}" if 'mean' in values else ''
                std = f"{values.get('std', ''):.2f}" if 'std' in values else ''
                row.extend([mean, std])
            writer.writerow(row)


def write_csv_all_data(data, file_name):
    """
    Writes a CSV file from a dictionary with the structure {class_name: {descriptor_name1: [], descriptor_name2: [], ...}}.

    Args:
        data (dict): Dictionary containing the data.
        file_name (str): Path to the output CSV file.
    """
    # Initialize the CSV header
    header = ['Class']

    # Collect all descriptor names from the data
    descriptor_names = sorted(
        {descriptor_name for descriptors in data.values() for descriptor_name in descriptors.keys()})

    # Add descriptor names to the header
    header.extend(descriptor_names)

    # Prepare the rows for each class
    rows = []

    # Process each class
    for class_name, descriptors in data.items():
        # Assume all descriptors have the same length
        num_values = len(next(iter(descriptors.values())))  # Get the length from any descriptor

        # Create rows for each value in the descriptors
        for i in range(num_values):
            row = [class_name]
            for descriptor_name in descriptor_names:
                row.append(descriptors.get(descriptor_name, [])[i])
            rows.append(row)

    # Write data to CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header
        writer.writerows(rows)  # Write rows

