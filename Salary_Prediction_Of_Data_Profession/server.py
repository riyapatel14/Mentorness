from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = joblib.load('salary_prediction_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_data = pd.DataFrame(data, index=[0])
    new_data_processed = preprocessor.transform(new_data)
    prediction = model.predict(new_data_processed)[0]
    return jsonify({'predicted_salary': prediction})

if __name__ == '__main__':
    app.run(debug=True)
