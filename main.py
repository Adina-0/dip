import preprocessing as pp
import featureExtraction_structural as fs
import featureExtraction_color as fc
import classification_features as cf
import os

# path to folder containing pollen images
img_folder = f"./Data/1.Ageratum/"
features = ()
X = ()


# iterate over all images in the folder
for img_path in os.listdir(img_folder):
    print(f"\n\nProcessing image: {img_path}")
    path = os.path.join(img_folder, img_path)
    img_preprocessed, binary_mask = pp.preprocessing_pipeline1(path)
    # img_preprocessed_gray = pp.convert_to_gray(img_preprocessed)

    features_structure = fs.structural_features(img_preprocessed)
    print("\nStructural features:")
    for key, value in features_structure.items():
        print(f"\t{key}: {value}")

    # features_color = fc.calculate_descriptors(img_preprocessed)
    # for key, value in features_color.items():
    #     print(f"{key}: {value}")

    features_color2 = fc.color_descriptors(img_preprocessed, binary_mask)
    print("\nColor features:")
    for key, value in features_color2.items():
        print(f"\t{key}: {value}")

    feature_vector = list(features_structure.values()) + list(features_color2.values())
    features += (feature_vector,)
    X += ("Ageratum",)
    num_features = (len(features),)

model = cf.create_model(num_features, 1)
model = cf.train_model(model, features, X)


# path = f"./Data/1.Ageratum/Image-9_2024-03-06_grain_0.png"
# preprocessing.preprocess_pipeline(path) # preprocess_pipeline (after Pozo-Banos) works best for now, others to try out: preprocessing_pipeline1, 2 and 3
#


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


