import tensorflow as tf
from sklearn.model_selection import train_test_split


# Define the fully connected model for extracted features
def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.5),
        # output layer for multi-class classification
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='Adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Model Summary: {model.summary()}")
    return model


def train_model(model, X, y):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # print model information
    print(f"Model summary after training: {model.summary()}")

    return model





