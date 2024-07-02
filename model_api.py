from flask import Flask, request, jsonify
from joblib import load
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
import numpy as np


app = Flask(__name__)

cropRecommendationModel = load('rf_crop_recommendation_model.joblib')  # Adjust path as needed

fertilizerRecommendationModel = load('rf_fertilizer_recommendation_model.joblib')  # Adjust path as needed

diseasePredictionModel = tf.keras.models.load_model('cnn_model.h5')


crop_label_mapping = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
                 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
                 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
                 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 
                 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}

# Define the dictionary mapping integers to strings for fertilizer types
fertilizer_label_mapping = {
    0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 
    5: 'DAP', 6: 'DAP-MOP', 7: 'DAP-SSP', 8: 'DAP-Urea', 9: 'MOP', 
    10: 'MOP-SSP', 11: 'MOP-Urea', 12: 'SSP', 13: 'SSP-DAP', 
    14: 'SSP-DAP-Urea', 15: 'SSP-MOP', 16: 'SSP-Urea', 17: 'Urea', 
    18: 'Urea-DAP', 19: 'Urea-DAP-MOP', 20: 'Urea-SSP', 21: 'Urea-SSP-MOP'
}

disease_class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

@app.route('/predictCrop', methods=['POST'])
def predictCrop():
    data = request.get_json(force=True)
    prediction = cropRecommendationModel.predict([data['features']])
    # Map the numeric prediction to the corresponding string
    predicted_labels = [crop_label_mapping[pred] for pred in prediction.tolist()]
    return jsonify(predicted_labels)

@app.route('/predictFertilizer', methods=['POST'])
def predictFertilizer():
    data = request.get_json(force=True)
    prediction = fertilizerRecommendationModel.predict([data['features']])
    # Map the numeric prediction to the corresponding string
    predicted_labels = [fertilizer_label_mapping[pred] for pred in prediction.tolist()]
    return jsonify(predicted_labels)


@app.route('/predict-disease', methods=['POST'])
def predictDisease():
    data = request.get_json()
    image_b64 = data['image']
    decoded = base64.b64decode(image_b64)
    image = Image.open(BytesIO(decoded))
    image = image.resize((256, 256))  # assuming the model expects 256x256 images
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # model expects a batch of images

    predictions = diseasePredictionModel.predict(image_array)
    predicted_class = disease_class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return jsonify({'predicted_class': str(predicted_class), 'confidence':str(confidence)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
