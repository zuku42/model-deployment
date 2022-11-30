import pickle
import pandas as pd

from flask import Flask, request, jsonify

from custom_transformers import FeatureNormalizer, FeatureDropper, CustomImputer

app = Flask('draft_prediction')

# Create an API end point
@app.route('/predict', methods=['POST'])
def get_prediction():
    # Load pickled model file
    model = pickle.load(open('models/binary_model.sav', 'rb'))
    # Get player data
    player = pd.DataFrame(request.get_json())
    # Predict the probability of getting drafted using the model
    predicted_probability = model.predict_proba(player)[0][1]
    # Return a json object containing the features and prediction
    return jsonify(predicted_probability)

if __name__ == '__main__':
    # Run the app at 0.0.0.0:9696
    app.run(debug=True, port=9696, host='0.0.0.0')