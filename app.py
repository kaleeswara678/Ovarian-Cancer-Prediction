import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model and feature names
model = pickle.load(open('model.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the user input
            user_input = [float(request.form[f'feature{i+1}']) for i in range(len(feature_names))]

            # Prepare the user input as a numpy array
            user_input_arr = np.array(user_input).reshape(1, -1)

            # Predict the result
            prediction = model.predict(user_input_arr)
            prediction_proba = model.predict_proba(user_input_arr)

            # Prepare the prediction result
            result = f"Predicted type: {prediction[0]} with probability {prediction_proba[0][model.classes_.tolist().index(prediction[0])]:.2f}"

            return render_template('index.html', features=feature_names, prediction=result)

        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
