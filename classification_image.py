# code after:
# Pollen Grain Recognition Using Convolutional Neural Network
# Natalia Khanzhina1, Evgeny Putin1, Andrey Filchenkov1 and Elena Zamyatina2,3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout


# Define the CNN architecture
def create_model(input_shape):
    model = Sequential([
        # First convolutional layer
        Conv2D(6, kernel_size=(7, 7), activation='relu', input_shape=input_shape),

        # Second convolutional layer
        Conv2D(16, kernel_size=(5, 5), activation='relu'),

        # Third convolutional layer
        Conv2D(32, kernel_size=(5, 5), activation='relu'),

        # Flatten the output
        Flatten(),

        # Fully connected dense layers
        Dense(100, activation='relu'),
        Dropout(0.5),  # Add dropout to reduce overfitting
        Dense(50, activation='relu'),

        # Output layer (assuming binary classification; adjust units/activation for your task)
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='Adadelta',
                  loss='binary_crossentropy',  # Adjust loss for multi-class tasks
                  metrics=['accuracy'])
    return model


# Example input shape (adjust as needed, e.g., (height, width, channels))
input_shape = (64, 64, 3)  # Replace with your actual input shape
model = create_model(input_shape)

# Print model summary
model.summary()
