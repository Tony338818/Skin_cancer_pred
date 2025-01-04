from flask import Flask, request, render_template, jsonify
from tensorflow.keras import models
import cv2 as cv
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = models.load_model('Skin_Cancer_model.h5')

# Define the image size used for the model
IMG_SIZE = (128, 128)



@app.route('/')
def index():
    # Render the HTML upload form
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Get the uploaded image
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    image = cv.imread(file_path)
    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    image = cv.resize(image, IMG_SIZE)  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)[0][0]

    # Interpret the prediction
    result = "Malignant" if prediction > 0.5 else "Benign"

    # Clean up the uploaded file
    os.remove(file_path)
    return 

    return jsonify({'result': result})


if __name__ == '__main__':
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
