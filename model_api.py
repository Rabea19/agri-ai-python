from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('rf_model.joblib')  # Adjust path as needed

label_mapping = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
                 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
                 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
                 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 
                 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    # Map the numeric prediction to the corresponding string
    predicted_labels = [label_mapping[pred] for pred in prediction.tolist()]
    return jsonify(predicted_labels)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
