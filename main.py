# Pipeline for pollen image classification
# Requires: path to the input folders containing pollen data, each folder containing images of a single species

import utils
import time
import os
import warnings
import pandas as pd

import feature_performance as fp
import pre_processing_old as pp
import featureExtraction_structural as fs
import featureExtraction_color as fc
import featureExtraction_geometrical as fg
import classification_features as cf


# Path to input folders containing pollen training data
data_path = "./Data/"
output_identifier = "test"

# Features to include
include_color = True
include_geometry = True
include_structure = True

inlcude_randomForest = True
include_FCNN = True

############################################################################
# Output paths, set to None if no storage is needed
output_allFeatures = f"./Results/{output_identifier}_allFeatures.csv"
output_featureStats = f"./Results/{output_identifier}_featureStats.csv"
output_model = f"./Results/{output_identifier}_RF"  # creates three files: '{output_model}_model.joblib', '{output_model}_metadata.joblib' and '{output_model}_label-encoder.joblib'

# Flags for plotting and padding
plot_bool = False
pad_bool = True

############################################################################

# Suppress the specific runtime warning (saturation index)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in divide.*")
begin = time.time()

print("Find global max dim")
max_dim = pp.find_global_max_dimension(data_path)
print("done")

# Paths to folders containing pollen images
img_folders = os.listdir(data_path)
n_classes = len(img_folders)

# Dictionary to store all data
all_data = {}  # {class_name: {descriptor_name: [values per image]}}
df_allData = pd.DataFrame()


for class_index, img_folder in enumerate(img_folders):
    if img_folder[0] == ".":  # Skip hidden files
        n_classes -= 1
        continue

    # Initialize the class in the dictionary
    class_name = img_folder  # folder name as class name
    all_data[class_name] = {}

    print(f"\nProcessing folder: {img_folder}")
    img_folder = os.path.join(data_path, img_folder + "/")

    # Iterate over all images in the folder
    for img_path in os.listdir(img_folder):
        if img_path[0] == ".":  # Skip hidden files
            continue

        print(img_path)
        path = os.path.join(img_folder, img_path)

        img_preprocessed, binary_mask, largest_contour, binary_image = pp.process_image_to_black_background(path, max_dim, pad=pad_bool)

        # cv2.imshow("Test", img_preprocessed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Extract features
        features_structure = {}
        features_geometric = {}
        features_color = {}

        if include_structure:
            features_structure = fs.structural_features(img_preprocessed, binary_mask, plot_bool=plot_bool)
        if include_color:
            features_color = fc.color_descriptors(img_preprocessed, binary_mask)
        if include_geometry:
            features_geometric = fg.geometric_features(img_preprocessed, binary_mask, largest_contour, binary_image)

        feature_dict = {**features_structure, **features_color, **features_geometric}

        # Update all_data dictionary
        for descriptor_name, descriptor_value in feature_dict.items():
            if descriptor_name not in all_data[class_name]:
                all_data[class_name][descriptor_name] = []
            all_data[class_name][descriptor_name].append(descriptor_value)

        # Convert feature_dict to a DataFrame
        df_image = pd.DataFrame([feature_dict])
        df_image['Class'] = class_name
        df_image['Image'] = img_path
        df_image.set_index('Image', inplace=True)

        # Reorder columns: move 'Class' to the first position
        columns = ['Class'] + [col for col in df_image.columns if col != 'Class']
        df_image = df_image[columns]

        df_allData = pd.concat([df_allData, df_image], ignore_index=False)


descriptor_stats = utils.calculate_descriptor_stats(all_data)
utils.write_csv_stats(descriptor_stats, output_featureStats)

df_allData.to_csv(output_allFeatures) # Write all features to a CSV file

# for further processing, drop all rows with nan values
df_cleaned = df_allData.dropna()
rows_deleted = df_allData.shape[0] - df_cleaned.shape[0]
print(f"Deleted {rows_deleted} rows with NaN values")
df_allData = df_cleaned

end = time.time()
print(f"Total time to extract features: {(end - begin) / 60:.2f} minutes")


if inlcude_randomForest:
    begin = time.time()

    print("Random Forest:")
    rf = fp.analyze_feature_performance(df_allData, output_model) # Classify using Random Forest

    end = time.time()
    print(f"Total time for Random Forest: {(end - begin) / 60:.2f} minutes")

if include_FCNN:
    begin = time.time()

    print("FCNN:")
    # train and evaluate the classification model -
    features = df_allData.drop(columns=['Class'])
    X = df_allData['Class']
    feature_names = features.columns.tolist()  # Save feature names as a list for plots

    # get number of columns
    num_features = (features.shape[1],)

    # Create and train the FCNN model
    model = cf.create_model(num_features, n_classes)
    trained_model, X_val, y_val = cf.train_model(model, features, X, n_classes)

    end = time.time()
    print(f"Total time for include_FCNN training: {(end - begin) / 60:.2f} minutes")

    cf.evaluate_model(trained_model, X_val, y_val, feature_names)

    end = time.time()
    print(f"Total time for include_FCNN including SHAP analysis: {(end - begin) / 60:.2f} minutes")
