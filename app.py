from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your model
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Log the incoming data for debugging
        app.logger.info("Received data: %s", data)
        
        # Get input values from the front-end request
        venue = data.get('venue')
        batting_team = data.get('batting_team')
        bowling_team = data.get('bowling_team')
        striker = data.get('striker')
        bowler = data.get('bowler')

        # Ensure all inputs are provided
        if not all([venue, batting_team, bowling_team, striker, bowler]):
            return jsonify({'error': 'All input fields are required!'}), 400

        # Transform inputs and make prediction
        # Convert categorical variables to numerical if required by your model
        input_data = np.array([[venue, batting_team, bowling_team, striker, bowler]])
        prediction = model.predict(input_data)

        return jsonify({'predicted_score': int(prediction[0])})

    except Exception as e:
        app.logger.error("Error occurred: %s", str(e))  # Log the error message
        return jsonify({'error': str(e)}), 500  # Return the error message in the response

if __name__ == '__main__':
    app.run(port=5000, debug=True)
