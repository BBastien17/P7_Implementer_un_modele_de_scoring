#from crypt import methods
from flask import Flask, render_template, redirect, request, url_for
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
import requests
import math
import seaborn as sns
import os
import requests
import streamlit as st
import unittest
import urllib
import pickle
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
#Librairies pour faire les tests avec unittest
import unittest
from unittest.mock import Mock



#Importation du modèle mlflow
#path = 'Projet_7/'
#model = mlflow.sklearn.load_model('xgb_model_final/')
##model = mlflow.sklearn.load_model("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/tree/02846cde70faec8112fc7a9f407882a6210886f9/xgb_model_final")
#Importation des infos clients
#data_work_complet = pd.read_csv("C:/Users/Bastien/Projet_7/data_work.csv")
##data_work_complet = pd.read_csv("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/blob/02846cde70faec8112fc7a9f407882a6210886f9/data_work.csv")
##print(data_work_complet.head())
#data_target_complet = pd.read_csv("C:/Users/Bastien/Projet_7/data_target.csv")
##data_target_complet = pd.read_csv("https://github.com/BBastien17/P7_Implementer_un_modele_de_scoring/blob/02846cde70faec8112fc7a9f407882a6210886f9/data_target.csv")


#Fonction pour calculer le score prédictproba du client.
def calc_score_predictproba (ref_client, data_work_complet):
    #Création d'un dataframe avec les information du client sélectionné :
    #data_work_client = pd.DataFrame(data_work_complet,index=[ref_client])
    data_work_client = pd.DataFrame(data_work_complet,index=[ref_client])
    print(data_work_client)
    #A vérifier si ligne en dessous utile ou non car sert justepour avoir la variable target
    #data_work_target_client = pd.DataFrame(data_target,index=[ref_client])
    #Création du dictionnaire où l'on stocke les résultats
    list_result_work = {'Type_de_pret':data_work_client['Type_de_pret'], 'Genre':data_work_client['Genre'],
                        'Age':data_work_client['Age'], 'Niveau_d_etudes':data_work_client['Niveau_d_etudes'],
                        'Regime_matrimonial':data_work_client['Regime_matrimonial'],
                        'Nb_enfants': data_work_client['Nb_enfants'],
                        'Nb_membre_famille':data_work_client['Nb_membre_famille'], 
                        'Montant_des_revenus':data_work_client['Montant_des_revenus'],
                        'Note_region_client':data_work_client['Note_region_client'],
                        'Nb_demande_client':data_work_client['Nb_demande_client'],
                        'Montants_du_pret':data_work_client['Montants_du_pret'],
                        'Montant_des_annuites':data_work_client['Montant_des_annuites'],
                        'Nb_jours_credits':data_work_client['Nb_jours_credits'],                       
                        'Montant_anticipation_pret':data_work_client['Montant_anticipation_pret'],
                        'Delai_anticipation_pret':data_work_client['Delai_anticipation_pret']}
    
    #On transforme ensuite le dictionnaire en dataframe
    data_work_list_result = pd.DataFrame(data=list_result_work)
    #On oublie pas de préparer la transformation des variables catégorielles en variables numériques plus tard
    transf_data_work_categ = {'Prêts de trésorerie': 0, 'Prêts renouvelables': 1,
                              'autre': 0, 'célibataire': 1, 'marié(e)': 2,
                              'M': 1, 'F': 0,
                              '3 à 4': 0, '5 à 8': 1}    
    #Découpage des datasets en dataset de train et de test (proportion 80/20)
    X_train, X_test, y_train, y_test = train_test_split(data_work_complet, data_target_complet, test_size = 0.2, random_state=42)

    #Librairie pour encoder des variables catégorielles
    encoder = LabelEncoder()
    #On remet la variable concernant une période sous un format positif
    X_test['Delai_anticipation_pret'] = X_test['Delai_anticipation_pret'].mul(-1)
    #On remet la variable concernant une période sous un format positif
    X_train['Delai_anticipation_pret'] = X_train['Delai_anticipation_pret'].mul(-1)
    #Création d'une variable avec la liste des colonnes catégorielles du dataset features
    data_categ = list(data_work_complet.select_dtypes(exclude=["number"]).columns)
    #Encodage des variables catégorielles
    for col in data_categ:
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.fit_transform(X_test[col])
    #Entrainement du modèle
    model.fit(X_train, y_train)

    #On transforme les variables catégorielles en variables numériques
    data_work_list_result_transf = data_work_list_result.replace(transf_data_work_categ)
    print(data_work_list_result_transf)
    #Prédiction du résultat
    score = model.predict_proba(data_work_list_result_transf)
    return score

class UneClasseDeTest(unittest.TestCase):

    def test_is_sourcefile_when_sourcefile(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".py"

        resultat = is_sourcefile(path)

        self.assertTrue(resultat)
        path.is_file.assert_called()

    def test_is_sourcefile_when_file_does_not_exist(self):
        path = Mock()
        path.is_file.return_value = False

        with self.assertRaises(Exception):
            is_sourcefile(path)

        path.is_file.assert_called()

    def test_is_sourcefile_when_not_expected_suffix(self):
        path = Mock()
        path.is_file.return_value = True
        path.suffix = ".txt"

        resultat = is_sourcefile(path)

        self.assertFalse(resultat)
        path.is_file.assert_called()


app = Flask(__name__)

#@app.route("/", methods=['GET', 'POST'])
#def index():
#    if request.method == "POST":
       # getting input with name = fname in HTML form
#       id_client_choice = request.form.get("id_client")
#       print(id_client_choice)
       #score_client = calc_score_predictproba(id_client_choice)
       #return "L'ID client est  " + id_client_choice
       #return "L'ID client est  " + score_client
    #score_client = calc_score_predictproba(id_client_choice)
    #print(score_client)
#    return render_template('dashboard.html', todos = todos)

#todos = 0
todos = {}

@app.get("/")
def index():
    #score_client = calc_score_predictproba(todos)
    #print(todos)
    
    return render_template('dashboard.html', todos=todos)
    #return render_template('dashboard.html')



@app.route('/add', methods = ['GET', 'POST'])
def add():
    if request.method == 'POST':
        todos.clear()
        index = len(todos) + 1
        #todos = todos[0]
        todos[index] = request.form.get("id_client")
        #todos = request.form.get("id_client")
        print(todos)
        return redirect(url_for('client_description'))
    return render_template('add.html')#, todos=todos)

@app.route('/client_description', methods = ['GET', 'POST'])
def client_description():
    dict_key_select = list(todos)[-1]
    ref_client = todos[dict_key_select]
    ref_client = int(ref_client)
    print(ref_client)
    score_client = calc_score_predictproba(ref_client, data_work_complet)
    print(score_client)
    score_client_accept = round(score_client[0][0], 3)
    #Ligne suivante deja commentée
    #score_client_accept = score_client
    print(score_client_accept)
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('client_description.html', value=ref_client, score=score_client_accept)
    #return render_template('client_description.html')

#@app.route("/")#, methods=['GET', 'POST'])
#def index():
#    score_client = calc_score_predictproba(id_client)
#    return render_template('dashboard.html', value = score_client)

#@app.route("/id_client", methods=["GET", "POST"])
#def recup_id_client():
#    for key, value in request.form.items():
#        print("key: {0}, value: {1}".format(key, value))

#@app.route("/action1", methods=["POST"])
#def action1 ():
#    print("Coucou")
#    return redirect('http://192.168.1.12:8501', code=302)
    

if __name__ == '__main__':
    
    #Pour les test unitaires (paramétré pour fonctionner sur la version locale)
    #unittest.main()
    
    
    #app.run(https://projet7bboyer.azurewebsites.net/)
    #app.run(host="20.105.232.31", port=8080)

    #Pour simulation en local
    #app.run(host="0.0.0.0", port=8080)

    #app.run(host="0.0.0.0", port=5000)
    #app.run()
    #app.run(host='0.0.0.0', port=80, debug=True)
    print("connexion OK")
    #Pour le déploiement en ligne
    port = os.environ.get("Port", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)
    #app.run()
