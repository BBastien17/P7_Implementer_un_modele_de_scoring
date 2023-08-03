#Importation des librairies
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
import requests
import math
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
#from sklearn.metrics import classification_report, confusion_matrix
import mlflow.sklearn
#Librairie pour XGBoostClassifier
from xgboost import XGBClassifier
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
import pickle

#Importation des fonctions contenants les pages
from page_prospect import page_p
from page_client import page_c


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

#def main():

#Création d'un side bar pour choisir entre une page pour les clients et les prospects
page = st.sidebar.selectbox('Page Navigation', ["Prospects", "Clients"])
st.sidebar.markdown("""---""")
st.sidebar.write("Created by Bastien B")
#st.sidebar.image("images/logo.png", width=100)
st.sidebar.image("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/blob/67f1e5e3069cb689852317b5cef495492e04b8fa/images/logo.png", width=100)

path = 'Projet_7/'

#Import des données clients
#data_work = pd.read_csv("C:/Users/Bastien/Projet_7/data_work.csv")
data_work = pd.read_csv("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/blob/67f1e5e3069cb689852317b5cef495492e04b8fa/data_work.csv")
#data_target = pd.read_csv("C:/Users/Bastien/Projet_7/data_target.csv")
data_target = pd.read_csv("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/blob/67f1e5e3069cb689852317b5cef495492e04b8fa/data_target.csv")
data_complete = data_work.copy()
data_complete["Target"] = data_target

#Import du modèle XGBClassifier
xgb = XGBClassifier()
#model = load_model('xgb_model_final/model')
#model = mlflow.sklearn.load_model('xgb_model_final/')
model = mlflow.sklearn.load_model("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/tree/67f1e5e3069cb689852317b5cef495492e04b8fa/xgb_model_final/")
#load_joblib_model = joblib.load('pipeline_scoring.joblib')
#st.write(load_joblib_model)
#st.write("Temp")
#st.write(model)
#model = mlflow.sklearn.load_model('xgb_model_final/model')

#Appel de la page correspondante en fonction du choix réalisé dans le menu déroulant
if page == "Prospects": 
    page_p(data_work, data_target, data_complete)
    
if page == "Clients":
    page_c(data_work, data_target, data_complete)
    
