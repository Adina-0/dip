import preprocessing as pp
import utils
import feature_performance as fp
import pre_processing as ppM
import featureExtraction_structural as fs
import featureExtraction_color as fc
import classification_features as cf
import os

# Path to input folders containing pollen training data
data_path = "./Data/"
# Paths to folders containing pollen images
img_folders = os.listdir(data_path)

features = ()
X = ()
n_classes = len(img_folders)

# Dictionary to store all data
all_data = {}  # {class_name: {descriptor_name: [values per image]}}

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

        # Uncomment below for alternate preprocessing pipeline
        # img_preprocessed, binary_mask = pp.preprocessing_pipeline1(path)
        # img_preprocessed_gray = pp.convert_to_gray(img_preprocessed)

        img_preprocessed, binary_mask = ppM.process_image_to_black_background(path)

        # Extract structural features
        features_structure = fs.structural_features(img_preprocessed, binary_mask)
        # print("\nStructural features:")
        # for key, value in features_structure.items():
        #     print(f"\t{key}: {value}")

        # Uncomment below for an alternative set of color descriptors
        # features_color = fc.calculate_descriptors(img_preprocessed)
        # for key, value in features_color.items():
        #     print(f"{key}: {value}")

        features_color2 = fc.color_descriptors(img_preprocessed, binary_mask)
        # print("\nColor features:")
        # for key, value in features_color2.items():
        #     print(f"\t{key}: {value}")

        # Combine structural and color features into a single vector
        feature_vector = list(features_structure.values()) + list(features_color2.values())
        flattened_feature_vector = utils.flatten_vector(feature_vector)

        # Add the processed feature vector to features
        features += (flattened_feature_vector,)
        X += (class_index,)

        # Update all_data dictionary
        for descriptor_name, descriptor_value in features_structure.items():
            if descriptor_name not in all_data[class_name]:
                all_data[class_name][descriptor_name] = []
            all_data[class_name][descriptor_name].append(descriptor_value)

        for descriptor_name, descriptor_value in features_color2.items():
            if descriptor_name not in all_data[class_name]:
                all_data[class_name][descriptor_name] = []
            all_data[class_name][descriptor_name].append(descriptor_value)


descriptor_stats = utils.calculate_descriptor_stats(all_data)
# descriptor_evaluation = desc_eval.evaluate_descriptors(all_data)
# print("Mutual Information:", descriptor_evaluation['mutual_info'])

fp.analyze_feature_performance(all_data)


# train and evaluate the classification model
# num_features = (len(features[0]),)
# model = cf.create_model(num_features, n_classes)
# model = cf.train_model(model, features, X, n_classes)


# TODO: apply the following
"""
Data Augmentation (for Training)
Why: Data augmentation helps improve model generalization by artificially increasing the size of the training set.
How: Apply random transformations like rotation, flipping, zooming, and shifting to the images.
Example: 
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(image_dataset)

Feature Extraction (Optional for Classic ML)
Why: For traditional machine learning models (e.g., SVM, Random Forest), feature extraction can help convert image data into more manageable representations.
How: Extract features using techniques like Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), or Scale-Invariant Feature Transform (SIFT).
Example:
    from skimage.feature import hog
    features, hog_image = hog(image_equalized, block_norm='L2-Hys', visualize=True)
"""


