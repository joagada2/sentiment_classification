#library importation
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, f1_score, \
    roc_auc_score, precision_score, recall_score
import numpy as np
from datetime import datetime, timedelta
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
import warnings
warnings.filterwarnings(action="ignore")
import joblib
import os
import mlflow
from dagshub import DAGsHubLogger

#load dataset
@task
def get_data():
    a = pd.read_csv('../data/IMDB Dataset.csv')
    return a

@task
def clean_data(b):
    b['review'] = str(b['review']).lower() #lower case
    b['review'] = re.sub(r"@\S+ ", r' ', b['review'])  # remove all mentions and replace with a single empty space
    b['review'] = re.sub('https://.*', '', b['review'])  # remove all urls
    b['review'] = re.sub("\s+", ' ', b['review'])  # remove multiple spaces or tabs and replace with a single space
    b['review'] = re.sub("\n+", ' ', b['review'])  # remove multiple empty lines
    b['review'] = re.sub("[^a-zA-Z]", ' ', b['review'])  # take ontly text and ignore other non text characters
    return b
@task
def model_training(c):
    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
    # definition of dependent variable
    y = c.sentiment
    # definition of independent variable
    X = vectorizer.fit_transform(c.review)
    # split dataset into training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # fit the Multinomial Naive Mayes Model
    model = naive_bayes.MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, '../model/agada_sentiment_classifier')
    joblib.dump(vectorizer, '../model/agada_sentiment_vectorizer')
    X_test.to_csv('../data/x_test.csv')
    y_test.to_csv('../data/y_test.csv')
    return model

@task
def model_eval(d):
    #link up to dagshub MLFlow environment
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/joe88data/loan-default-prediction-model.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'joe88data'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'e94114ca328c75772401898d749decb6dbcbeb21'
    with mlflow.start_run():
        # Load data and model
        X_test = pd.read_csv('../data/final/X_test.csv')
        y_test = pd.read_csv('../data/final/y_test.csv')
        d = joblib.load('../model/agada_sentiment_classifier')
        # Get metrices
        f1 = f1_score(y_test,d.predict_proba(X_test)[:,1])
        precision = precision_score(y_test,d.predict_proba(X_test)[:,1])
        recall = recall_score(y_test,d.predict_proba(X_test)[:,1])
        balanced_accuracy = balanced_accuracy_score(y_test,d.predict_proba(X_test)[:,1])
        Area_under_ROC = roc_auc_score(y_test, d.predict_proba(X_test)[:, 1])
        # Get metrics =

        print(f"F1 Score of this model is {f1}.")
        print(f"Accuracy Score of this model is {balanced_accuracy}.")
        print(f"Area Under ROC is {Area_under_ROC}.")
        print(f"Precision of this model is {precision}.")
        print(f"Recall for this model is {recall}.")

        # helper class for logging model and metrics
        class BaseLogger:
            def __init__(self):
                self.logger = DAGsHubLogger()

            def log_metrics(self, metrics: dict):
                mlflow.log_metrics(metrics)
                self.logger.log_metrics(metrics)

            def log_params(self, params: dict):
                mlflow.log_params(params)
                self.logger.log_hyperparams(params)
        logger = BaseLogger()
        # function to log parameters to dagshub and mlflow
        def log_params(c: naive_bayes.MultinomialNB):
            logger.log_params({"model_class": type(c).__name__})
            model_params = c.get_params()

            for arg, value in model_params.items():
                logger.log_params({arg: value})

        # function to log metrics to dagshub and mlflow
        def log_metrics(**metrics: dict):
            logger.log_metrics(metrics)
        # log metrics to remote server (dagshub)
        log_params(d)
        log_metrics(f1_score=f1, accuracy_score=balanced_accuracy, area_Under_ROC=Area_under_ROC, precision=precision,
                recall=recall)
            # log metrics to local mlflow
            # mlflow.sklearn.log_model(model, "model")
            # mlflow.log_metric('f1_score', f1)
            # mlflow.log_metric('accuracy_score', accuracy)
            # mlflow.log_metric('area_under_roc', area_under_roc)
            # mlflow.log_metric('precision', precision)
            # mlflow.log_metric('recall', recall)
