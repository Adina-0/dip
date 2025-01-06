# Pipeline for pollen image classification
# Requires: path to the input folders containing pollen data, each folder containing images of a single species

import utils
import time
import feature_performance as fp
import pre_processing as ppM
import featureExtraction_structural as fs
import featureExtraction_color as fc
import featureExtraction_geometrical as fg
import pandas as pd
import classification_features as cf
import os
import warnings

# Suppress the specific runtime warning (saturation index)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in divide.*")

begin = time.time()

# Path to input folders containing pollen training data
data_path = "./Data_short/"
output_allFeatures = "./Results/allFeatures_test.csv"
output_featureStats = "./Results/featureStats_test.csv"
plot_bool = False
pad_bool = True

print("Find global max dim")
max_dim = ppM.find_global_max_dimension(data_path)
print("done")

# Paths to folders containing pollen images
img_folders = os.listdir(data_path)
n_classes = len(img_folders)

features = ()
X = ()

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

        img_preprocessed, binary_mask, largest_contour, binary_image = ppM.process_image_to_black_background(path, max_dim, pad=pad_bool)
        # cv2.imshow("Test", img_preprocessed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Extract features
        features_structure = fs.structural_features(img_preprocessed, binary_mask, plot_bool=plot_bool)
        features_color = fc.color_descriptors(img_preprocessed, binary_mask)
        features_geometric = fg.geometric_features(img_preprocessed, binary_mask, largest_contour, binary_image)

        # # Combine structural, color and geometric features into a single vector for CNN
        # feature_vector = list(features_structure.values()) + list(features_color.values()) + list(features_geometric.values())
        # flattened_feature_vector = utils.flatten_vector(feature_vector)
        # # Add the processed feature vector to features
        # features += (flattened_feature_vector,)
        # X += (class_index,)

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
fp.analyze_feature_performance(df_allData) # Classify using Random Forest

end = time.time()
print(f"Total time: {end - begin:.2f} seconds = {(end - begin) / 60:.2f} minutes")

# # train and evaluate the classification model -
# num_features = (len(features[0]),)
# model = cf.create_model(num_features, n_classes)
# trained_model, X_val, y_val = cf.train_model(model, features, X, n_classes)
# cf.evaluate_model(trained_model, X_val, y_val)

