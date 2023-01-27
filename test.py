from tensorflow import keras
from PIL import Image
import numpy as np

# Load the saved model
model = keras.models.load_model('cifar10_model.h5')

# Open and preprocess the image
img = Image.open('path to your image') #replace 'plane.jpg' with the path to your image
img = img.resize((32, 32))  # Resize the image to (32, 32) as expected by the model
img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch size

# Make a prediction
predictions = model.predict(img_array)

# Decode the predictions
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
predicted_class = class_names[np.argmax(predictions[0])]

print("The image is classified as:", predicted_class)
