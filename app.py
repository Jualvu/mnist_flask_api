from flask import Flask, request, jsonify
from flask_cors import CORS
from neural_network_numpy.Model_numpy import Model_numpy
from neural_network_plain_python.Model_plain_python import Model_plain_python

app = Flask(__name__)
CORS(app)  # Enable CORS so your React app can talk to this API

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # get json data from request
    image = data.get("image_sample") # get 28x28 int matrix from the json request
    model_to_use = data.get("model_to_use") # get 28x28 int matrix from the json request

    if image is None or len(image) != 784: # check for a complete image
        return jsonify({'error': 'Invalid image format. Expected [783] array.'}), 400

    if model_to_use not in ('nn_numpy', 'nn_plain_python'):
        return jsonify({'error': 'Invalid model selection. Expected a string value equal to: "nn_numpy" or "nn_plain_python".'}), 400

    print(type(image))
    print(f"x: {len(image)}\n")


    try:
        if model_to_use == 'nn_numpy':
            predicted_num, prediction_confidence = model_numpy.predict(image_sample=image)
        elif model_to_use == 'nn_plain_python':
            predicted_num, prediction_confidence = model_plain_python.predict(image_sample=image)

        print(f"num: {predicted_num}")
        print(f"prediction_confidence: {prediction_confidence}")

        return jsonify({'prediction': int(predicted_num), 'confidence': int(prediction_confidence * 100)})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    model_numpy = Model_numpy()
    model_plain_python = Model_plain_python()

    app.run(debug=True)
