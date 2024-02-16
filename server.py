from flask import Flask
import numpy as np
import joblib
import json

app = Flask(__name__)

from flask import request

@app.route('/predict', methods=['GET'])
def predict():
    # Example of handling a GET request with query parameters
    input_values = request.args.get('input')
    if input_values:
        try:
            iris_input = [float(x) for x in input_values.split(',')]
            input_array = np.array(iris_input).reshape(1, -1)
            model = joblib.load('iris_model.pkl')
            prediction = model.predict(input_array)
            # Manually serialize data to JSON and set Content-Type
            response_body = json.dumps({'prediction': prediction.tolist()})
            return response_body, 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({'error': str(e)}), 400, {'Content-Type': 'application/json'}
    else:
        return json.dumps({'error': 'No input provided'}), 400, {'Content-Type': 'application/json'}
    # # Get the input array from the request
    # get_json = request.get_json()
    # iris_input = get_json['input']
    
    # # TODO: Import trained model
    # model = joblib.load('iris_model.pkl')
    
    # # TODO: Make prediction using the model 
    # # HINT: use np.array().reshape(1, -1) to convert input to 2D array
    # input_array = np.array(iris_input).reshape(1,-1)
    # prediction = model.predict(input_array)
    
    # # TODO: Return the prediction as a response

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
