import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("traffic_sign_classifier.h5")

# Define test directory and class labels
test_dir = "dataset/test"
class_labels = sorted(os.listdir(test_dir))  # Automatically get class names from folder names

# Function to predict and display images
def predict_images(test_dir):
    for label in class_labels:
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue  # Skip if not a directory

        print(f"\nTesting images for class: {label}\n")
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)

            # Load and preprocess image
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict class
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            # Display image with prediction
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_class}\nActual: {label}")
            plt.axis("off")
            plt.show()

# Run predictions on test images
predict_images(test_dir)
