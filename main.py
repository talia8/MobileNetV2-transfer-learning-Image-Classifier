import tensorflow as tf
import numpy as np
import cv2
from model import preprocess_image

# Function to make a prediction
def predict_image(image, model):
    """
    Uses the trained model to make a prediction on a single image.
    
    Args:
        image: Image to be tested.
        model: The trained Keras model.
    
    Returns:
        A string indicating if the image is of a dog or a cat.
    """
    # Make a prediction using the model
    prediction = model.predict(image)

    # Since it's a binary classification (dog vs cat), the output will be a single value
    # If the output is >= 0, it's a dog; otherwise, it's a cat
    if prediction >= 0:
        return "Dog"
    else:
        return "Cat"

# Load the saved model
model = tf.keras.models.load_model('dogs_vs_cats.h5')

# Path to the image you want to test
image_path = <path_to_your_image>  # Replace with the actual path to your image

image = cv2.imread(image_path)

processed_image = preprocess_image(image)

# Make a prediction
result = predict_image(processed_image, model)

# Print the result
print(f"The image is a: {result}")
