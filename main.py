# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define the path to the dataset (update this to your dataset path)
dataset_path = 'path/to/sign_language_dataset'

# Function to load and preprocess images from the dataset
def load_images_from_folder(folder):
    """
    Load images from a folder where each subfolder represents a sign.
    Returns images and their corresponding labels as NumPy arrays.
    """
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                # Read the image
                image = cv2.imread(image_path)
                # Resize to 64x64 pixels
                image = cv2.resize(image, (64, 64))
                # Normalize pixel values to [0, 1]
                image = image / 255.0
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load the dataset
print("Loading dataset...")
images, labels = load_images_from_folder(dataset_path)
print(f"Loaded {len(images)} images with {len(set(labels))} unique labels.")

# Encode labels (convert categorical labels to numerical values)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print("Labels encoded.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} images, Test set: {len(X_test)} images")

# Build the CNN model
model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    # Second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Third convolutional layer
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Flatten the output for dense layers
    layers.Flatten(),
    # Dense layer
    layers.Dense(128, activation='relu'),
    # Output layer (number of classes = number of unique signs)
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(
