# model-deployment

Flask app running a machine learning model predicting the probability of a given basketball player getting drafted by an NBA team, based on their performance in their final season in the NCAA.

Files in this repository:
- train.py - training of the estimator on a transformed dataset.
- custom_transformers.py - auxiliary script with class definitions used in training.
- data/ncaa_players.csv - clean and transformed data used for training. Examples from this dataset can be used to test the application.
- binary_model.sav - final estimator used by the app to make predictions.
- app.py - Flask app responsible for handling requests and estimating the probability of getting drafted.
- Dockerfile - Docker file
- requirements.txt - python libraries required to run the application.
- example_request.ipynb - demostration of how to post a request to the Flask app to obtain probability of getting drafted. Data from data/ncaa_players.csv can be used to test the app on different examples.

Problem description, data collection, data exploration, data transformation, model development and full reasoning behind model selection is presented here: https://github.com/zuku42/nba-draft-prediction.


