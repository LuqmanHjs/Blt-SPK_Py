import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import datetime
from bs4 import ResultSet
import sys
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import numpy as np
import pickle
import xlsxwriter
import io
import joblib
import altair as alt
from pandas import ExcelWriter
from pandas import ExcelFile
from jcopml.plot import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import plost
from sklearn.pipeline import Pipeline
import mysql.connector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe, cat_pipe
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer   
from streamlit_option_menu import option_menu
from login import validate_login, show_login_page,LoggedOut_Clicked

buffer = io.BytesIO()
def accuracy_score(y_true, y_pred):
        return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)
# NB 1
def pre_processing(df):
        
        X = df.drop([df.columns[-1]], axis = 1)
        y = df[df.columns[-1]]
        return X, y

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}
        self.likelihood = {}
        self.evidence = {}
        self.denominator={}

    def calculate_likelihood(self, feature, target, value, cls):
        numerator = len(feature[(target == cls) & (feature == value)])
        denominator = len(feature[target == cls])
        return numerator / denominator

    def fit(self, X, y):
        unique_classes = np.unique(y)

        # Calculate prior probabilities
        for cls in unique_classes:
            self.prior[cls] = len(y[y == cls]) / len(y)

        # Calculate likelihood probabilities for each feature and class
        for feature in X.columns:
            self.likelihood[feature] = {}
            self.evidence[feature] = {}
            unique_values = np.unique(X[feature])
            for cls in unique_classes:
                self.likelihood[feature][cls] = {}
                self.evidence[feature][cls] = {}
                for value in unique_values:
                    likelihood_value = self.calculate_likelihood(X[feature], y, value, cls)
                    self.likelihood[feature][cls][value] = likelihood_value
                    self.evidence[feature][cls][value] = len(X[X[feature] == value]) / len(y)

    def predict(self, X):
        predictions = []
        rules = []
        for _, row in X.iterrows():
            posterior = {}
            rule = {}
            for cls in self.prior.keys():
                posterior[cls] = self.prior[cls]
                rule[cls] = []
                for feature in X.columns:
                    likelihood_value = self.likelihood[feature][cls][row[feature]]
                    evidence_value = self.evidence[feature][cls][row[feature]]
                    rule[cls].append(f"P({feature}={row[feature]}|{cls}) = {likelihood_value}")
                for feature in X.columns:
                    evidence_value = self.evidence[feature][cls][row[feature]]
                    rule[cls].append(f"Evidence({feature}={row[feature]}) = {evidence_value}")
                rule[cls].append(f"P({cls}) = {self.prior[cls]}")
              # Calculate posterior probability
                posterior[cls] = np.prod([self.likelihood[feature][cls][row[feature]] for feature in X.columns]) * self.prior[cls] /np.prod([self.evidence[feature][cls][row[feature]] for feature in X.columns])

            
            predicted_class = max(posterior, key=posterior.get)
            predictions.append(predicted_class)
            rules.append(rule)

        X['Predicted Class'] = predictions
        X['Rules'] = rules
        return X
    
   
    



#        
# Data TRAINING
d1 = pd.read_excel("D:/TA/XCEL Over/Over30_50.xlsx")
df = pd.DataFrame(d1, columns=["jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","klasifikasi"])
df2 = df.loc[:,'jenis_pkj':'klasifikasi']
# st.write(df2)
X,y  = pre_processing(df2)

# Instantiate NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier()

# Fit the classifier to training data
nb_classifier.fit(X,y)


#  Uji Data
d1 = pd.read_excel("D:/TA/XCEL Over/OverUJI_30_50.xlsx")
df = pd.DataFrame(d1, columns=["jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","klasifikasi"])
df2 = df.loc[:,'jenis_pkj':'klasifikasi']
X_test,y_test  = pre_processing(df2)
# Make predictions on test data
result = nb_classifier.predict(X_test)

# Print the result
st.write("Result:")
st.dataframe(result)
r = nb_classifier.likelihood
st.dataframe(r)
st.dataframe(nb_classifier.evidence)

