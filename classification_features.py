import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import shap

# Define the fully connected model for feature analysis
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X, y, n_classes):
    X = np.array(X)
    y = np.array(y)

    y = tf.keras.utils.to_categorical(y, num_classes=n_classes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return model, X_val, y_val

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
def shap_analysis(model, X_val):
    explainer = shap.KernelExplainer(model.predict, X_val)
    shap_values = explainer.shap_values(X_val, nsamples=100)
    shap.summary_plot(shap_values, X_val)

# Evaluate the model and visualize feature importance
def evaluate_model(trained_model, X_val, y_val):
    # Weight-based feature importance
    weights_importance = analyze_weights(trained_model)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights_importance)), weights_importance)
    plt.title('Feature Importance from Weights')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()

    # Permutation importance
    perm_importance = permutation_importance_analysis(trained_model, X_val, y_val)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(perm_importance)), perm_importance)
    plt.title('Permutation Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()

    # SHAP analysis
    shap_analysis(trained_model, X_val)
