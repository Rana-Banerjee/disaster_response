import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    This function loads the data from the SQLite database and returns the X, Y and categories
    Inputs:
    database_filepath: Path of the SQLite database
    Returns:
    X: The list of messages as a pandas series object
    Y: The ground truth values of the categories of the messages
    Y.columns: List of names of the categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    con = engine.connect()
    df = pd.read_sql('select * from MESSAGES', con)
    X = df['message'] 
    Y = df.drop(columns=['id','message','original','genre'])
    return X, Y, Y.columns

def tokenize(text):
    '''
    This function preprocesses the messages and tokenizes them
    Inputs:
    text: The text string that needs to be tokenized
    Returns:
    clean_tokens: The list of cleaned token for the given text string
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function builds a pipeline of transformer and estimator, performs grid search on the pipeline parameters and builds a model
    Inputs:
    None
    Returns:
    cv: The Grid search pipeline object that can be trained
    '''
    pipeline = Pipeline([('tfidf_vect',TfidfVectorizer(tokenizer=tokenize))
                     ,('cls', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
    'cls__estimator__n_estimators': [50],
    'cls__estimator__min_samples_split': [2]

    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model and outputs the F1 score, precision, recall and accuracy for individual categories
    Inputs:
    model: The model that needs to be evaluated
    X_test: The test data on which the model needs to be evaluated
    Y_test: The ground truth values against which the model is evaluated
    category_names: The list of names of the categories
    Returns:
    Prints F1, Precision and Recall for each category
    Prints overall accuracy, precision, recall across all categories
    '''
    accuracy=0
    precision=0
    recall = 0
    n1='\n'
    y_pred = model.predict(X_test)
    for feat in range(y_pred.shape[1]):
        accuracy+=(y_pred[:,feat]==Y_test.iloc[:,feat]).mean()
        cr = classification_report(y_pred[:,feat],Y_test.iloc[:,feat])
        f1 = cr.split()[-2]
        pr, rc = cr.split()[-4:-2]
        precision+= float(pr)
        recall+=float(rc)
        print(f'Output category: {category_names[feat]}{n1}  F1: {f1}, Precision: {pr}, Recall:{rc}')

    accuracy/= y_pred.shape[1]
    precision/= y_pred.shape[1]
    recall/= y_pred.shape[1]
    print(f'Overall accuracy, precision, recall across all categories is {round(accuracy,4)}, {round(precision,4)}, {round(recall,4)}')

def save_model(model, model_filepath):
    '''
    This function saves the model to the given filepath as a pickle file
    Input:
    model: The model to be saved
    model_filepath: The path to which the model needs to be saved to
    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()