import pandas as pd
from Results import pre_processing as pp
import featureExtraction_structural as fs
import featureExtraction_geometrical as fg
import featureExtraction_color as fc
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from sklearn.preprocessing import LabelEncoder


# path to the image to be classified
image_path = './Test/ageratum.png'
real_species_name = ''  # real species name if for testing purposes

# path to the Random Forest Model and Metadata
model_path = './Results/20250109_onlyStrucutre_RF'

# -----------------------------------------------------------------------------------------------------------------------

rf = model_path + '_model.joblib'
le = model_path + '_label-encoder.joblib'
md = model_path + '_metadata.joblib'

def calculate_features(img_path: str, species: str):
    # test the model on a new image
    (img_preprocessed,
     binary_mask,
     largest_contour,
     binary_image) = pp.process_image_to_black_background(img_path)

    # Extract features
    features_structure = fs.structural_features(img_preprocessed, binary_mask)
    features_color = fc.color_descriptors(img_preprocessed, binary_mask)
    features_geometric = fg.geometric_features(img_preprocessed, binary_mask, largest_contour, binary_image)

    feature_dict = {**features_structure, **features_color, **features_geometric}

    # Convert feature_dict to a DataFrame
    df = pd.DataFrame([feature_dict])
    df['Image'] = img_path
    df.set_index('Image', inplace=True)
    df['Class'] = species

    columns = ['Class'] + [col for col in df.columns if col != 'Class']
    df = df[columns]
    return df


def classify(trained_model: str | RandomForestClassifier, label_encoder: str | LabelEncoder, metadata: str | dict, test_data: pd.DataFrame):

    if isinstance(trained_model, str):
        trained_model = load(trained_model)
    elif not isinstance(trained_model, RandomForestClassifier):
        raise ValueError("Invalid model type. Please provide a valid model.")

    if isinstance(label_encoder, str):
        label_encoder = load(label_encoder)
    elif not isinstance(label_encoder, LabelEncoder):
        raise ValueError("Invalid label encoder type. Please provide the path or a label encoder object.")

    if isinstance(metadata, str):
        metadata = load(metadata)
    elif not isinstance(metadata, dict):
        raise ValueError("Invalid metadata type. Please provide the path or a metadata dictionary.")

    # Ensure correct feature order
    feature_order = metadata['features']
    ordered_instance = test_data[feature_order]

    # Predict class
    prediction_encoded = trained_model.predict(ordered_instance)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return prediction_label


feature_df = calculate_features(image_path, real_species_name)
predicted_species = classify(rf, le, md, feature_df)

print(f"Real species: {real_species_name}")
print(f"Predicted species: {predicted_species}")
