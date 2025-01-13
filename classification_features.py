import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# Define the fully connected model for feature analysis
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X, y, n_classes):
    X = np.array(X)
    y = np.array(y)

    # Encode Class Labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = tf.keras.utils.to_categorical(y, num_classes=n_classes)

    # Create a dictionary with index -> class name mapping
    class_index_mapping = {index: class_name for index, class_name in enumerate(le.classes_)}

    # split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply StandardScaler to the features, else problems with regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)  # Use same scaler for validation data

    # Implement Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # compute class weights to handle imbalanced classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y.argmax(axis=1)), y=y.argmax(axis=1))
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], class_weight=class_weight_dict)
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Plot training vs. validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return model, X_val, y_val, class_index_mapping

# Feature importance from weights
def analyze_weights(model):
    first_layer_weights = model.layers[1].get_weights()[0]
    feature_importance = np.abs(first_layer_weights).sum(axis=1)
    return feature_importance

# Wrapper to make the Keras model compatible with permutation_importance
class KerasModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X).argmax(axis=1)

# Permutation feature importance
def permutation_importance_analysis(model, X_val, y_val):
    """
    Compute permutation feature importance using a custom scoring function.
    """
    y_val_classes = y_val.argmax(axis=1)  # Convert one-hot labels back to classes

    # Define a custom scoring function
    def score_func(X):
        y_pred = model.predict(X).argmax(axis=1)
        return accuracy_score(y_val_classes, y_pred)

    # Compute the base accuracy
    base_score = score_func(X_val)

    # Initialize storage for feature importance
    n_features = X_val.shape[1]
    importances = np.zeros(n_features)

    # Permute each feature in turn and compute the change in accuracy
    for i in range(n_features):
        X_val_permuted = X_val.copy()
        np.random.shuffle(X_val_permuted[:, i])
        permuted_score = score_func(X_val_permuted)
        importances[i] = base_score - permuted_score

    return importances

# SHAP analysis
def shap_analysis(model, X_val, feature_names, class_index_mapping=None):
    # # for debugging only!
    # predictions = model.predict(X_val)
    # print(f"Predictions shape: {predictions.shape}")

    # reduce data size using K-means clustering
    n_samples = X_val.shape[0]
    n_clusters = min(50, n_samples)  # Use max 100 clusters

    # Reduce data size using K-means clustering
    X_val_summary = shap.kmeans(X_val, n_clusters)
    reduced_X_val = X_val[:n_clusters]  # Use the same number of samples

    # Explainer with dynamic sample size for SHAP
    explainer = shap.KernelExplainer(model.predict, X_val_summary)
    nsamples = min(50, n_samples)  # Use max 50 samples
    shap_values = explainer.shap_values(reduced_X_val, nsamples=nsamples)

    ## for debugging
    # for i in range(shap_values.shape[2]):  # Loop through the class dimension (axis 2)
    #     class_shap_values = shap_values[:, :, i]  # Get SHAP values for the i-th class
    #     print(f"Class {i} SHAP values shape: {class_shap_values.shape}")

    # Aggregate SHAP values across classes
    shap_values_mean = np.mean(np.abs(shap_values), axis=2) # mean across classes

    # # debugging: Ensure shapes match
    # print(f"Aggregated SHAP values shape: {shap_values_mean.shape}")
    # print(f"X_val shape: {X_val.shape}")
    # assert shap_values_mean.shape == X_val.shape, "Aggregated SHAP values shape must match X_val."

    # Generate summary plot
    shap.summary_plot(shap_values_mean, reduced_X_val, feature_names=feature_names)

    # # Generate individual SHAP plots for each class
    # num_classes = shap_values.shape[2]
    # for i in range(num_classes):
    #     class_shap_values = shap_values[:, :, i]  # Get SHAP values for the i-th class
    #     plt.figure(figsize=(10, 6))
    #     shap.summary_plot(class_shap_values, reduced_X_val, feature_names=feature_names)
    #     plt.title(f'SHAP Summary Plot for Class {class_index_mapping[i]}')
    #     plt.tight_layout()
    #     plt.show()



# Evaluate the model and visualize feature importance
def evaluate_model(trained_model, X_val, y_val, feature_names, class_index_mapping):

    ## model is not linear, so weights-based feature importance is not meaningful
    # # Weight-based feature importance
    # weights_importance = analyze_weights(trained_model)
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(weights_importance)), weights_importance)
    # plt.title('Feature Importance from Weights')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Importance')
    #
    # # Replace indices with feature names
    # plt.xticks(range(len(weights_importance)), feature_names, rotation=90)
    # plt.tight_layout()
    # plt.show()

    # Permutation importance
    perm_importance = permutation_importance_analysis(trained_model, X_val, y_val)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(perm_importance)), perm_importance)
    plt.title('Permutation Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')

    # Replace indices with feature names
    plt.xticks(range(len(perm_importance)), feature_names, rotation=90)
    plt.tight_layout()
    plt.show()

    # SHAP analysis
    shap_analysis(trained_model, X_val, feature_names, class_index_mapping)
