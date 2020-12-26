# Disaster Response Pipeline Project
### Introduction:
This project creates a NLP powered app that can automatically classify disaster messages sent by people in distress into 36 categories.
With this the aid agencies can triage and filter the relevant messages and respond swiftly.

### Project Components:
The project has 3 main components
1. ETL pipeline
The ETL script does the following:
- Load messages and categories datasets
- Merge and clean the data
- Save the cleaned data in a SQLite database

2. ML pipeline
THE ML script does the following:
- Load the data from the SQLite database
- Split it into training and test sets
- Creates a pipeline of transformers and estimator, performs grid search over a few of the pipeline parameters and builds and trains a model
- Evaluates the model and outputs the F1 score, precision, recall and accuracy for individual categories
- Saves the model as a pickle file

3. Inference Pipeline
The app does the inference as follows:
- Loads the model for prediction
- Takes user input for a random message via the web app and predicts the classes
- The predicted classes are highlighted and rendered in the web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
