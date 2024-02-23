# Import des librairies
import streamlit as st
import shap
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Create a FastAPI instance
app = st

# Loading the model and data
model = pickle.load(open('model/model.pkl', 'rb'))
data = pd.read_csv('test_df_sample.csv')
data_train = pd.read_csv('train_df_sample.csv')

cols = data.select_dtypes(['float64']).columns
data_scaled = data.copy()
data_scaled[cols] = StandardScaler().fit_transform(data[cols])
cols = data_train.select_dtypes(['float64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[cols] = StandardScaler().fit_transform(data_train[cols])

explainer = shap.TreeExplainer(model['classifier'])

# Functions
def welcome():
    """
    Welcome message.
    :param: None
    :return: Message (string).
    """
    return 'Welcome to the App'

def check_client_id(client_id: int):
    """
    Customer search in the database
    :param: client_id (int)
    :return: message (string).
    """
    if client_id in list(data['SK_ID_CURR']):
        return True
    else:
        return False

def get_prediction(client_id: int):
    """
    Calculates the probability of default for a client.
    :param: client_id (int)
    :return: probability of default (float).
    """
    client_data = data[data['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model.predict_proba(info_client)[0][1]
    return prediction

def get_data_voisins(client_id: int):
    """ Calcul les plus proches voisins du client_id et retourne le dataframe de ces derniers.
    :param: client_id (int)
    :return: dataframe de clients similaires (json).
    """
    features = list(data_train_scaled.columns)
    features.remove('SK_ID_CURR')
    features.remove('TARGET')

    # Création d'une instance de NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')

    # Entraînement du modèle sur les données
    nn.fit(data_train_scaled[features])
    reference_id = client_id
    reference_observation = data_scaled[data_scaled['SK_ID_CURR'] == reference_id][features].values
    indices = nn.kneighbors(reference_observation, return_distance=False)
    df_voisins = data_train.iloc[indices[0], :]

    return df_voisins.to_json()

def shap_values_local(client_id: int):
    """ Calcul les shap values pour un client.
        :param: client_id (int)
        :return: shap values du client (json).
        """
    client_data = data_scaled[data_scaled['SK_ID_CURR'] == client_id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    shap_val = explainer(client_data)[0][:, 1]

    return {'shap_values': shap_val.values.tolist(),
            'base_value': shap_val.base_values,
            'data': client_data.values.tolist(),
            'feature_names': client_data.columns.tolist()}

def shap_values():
    """ Calcul les shap values de l'ensemble du jeu de données
    :param:
    :return: shap values
    """
    shap_val = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    return {'shap_values_0': shap_val[0].tolist(),
            'shap_values_1': shap_val[1].tolist()}

# Titre de la page
st.set_page_config(page_title="Dashboard Prêt à dépenser", layout="wide")

# Sidebar
with st.sidebar:
    # Page selection
    page = st.selectbox('Navigation', ["Home", "Information du client", "Interprétation locale",
                                       "Interprétation globale"])

    # ID Selection
    st.markdown("""---""")

    list_id_client = list(data['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    id_client_dash = st.selectbox("ID Client", list_id_client)
    st.write('Vous avez choisi le client ID : ' + str(id_client_dash))

    st.markdown("""---""")

if page == "Home":
    st.title("Dashboard Prêt à dépenser - Home Page")
    st.markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les raisons\n"
                "d'approbation ou refus de leur demande de crédit.\n"

                "\nLes prédictions sont calculées à partir d'un algorithme d'apprentissage automatique, "
                "préalablement entraîné. Il s'agit d'un modèle *Light GBM* (Light Gradient Boosting Machine). "
                "Les données utilisées sont disponibles [ici](https://www.kaggle.com/c/home-credit-default-risk/data). "
                "Lors du déploiement, un échantillon de ces données a été utilisé.\n"

                "\nLe dashboard est composé de plusieurs pages :\n"
                "- **Information du client**: Vous pouvez y retrouver toutes les informations relatives au client "
                "selectionné dans la colonne de gauche, ainsi que le résultat de sa demande de crédit. "
                "Je vous invite à accéder à cette page afin de commencer.\n"
                "- **Interprétation locale**: Vous pouvez y retrouver quelles caractéristiques du client ont le plus "
                "influencé le choix d'approbation ou refus de la demande de crédit.\n"
                "- **Intérprétation globale**: Vous pouvez y retrouver notamment des comparaisons du client avec "
                "les autres clients de la base de données ainsi qu'avec des clients similaires.")

if page == "Information du client":
    st.title("Dashboard Prêt à dépenser - Page Information du client")

    st.write("Cliquez sur le bouton ci-dessous pour commencer l'analyse de la demande :")
    button_start = st.button("Statut de la demande")
    if button_start:
        if id_client_dash != '<Select>':
            # Affichage des informations clients
            st.write("Information du client sélectionné :")
            info_client = data[data['SK_ID_CURR'] == id_client_dash].drop('SK_ID_CURR', axis=1)
            st.write(info_client)

            # Prédiction
            st.write("Résultat de la prédiction :")
            prediction = model.predict_proba(info_client)[0][1]
            st.write(f"Probabilité de défaut : {prediction:.2%}")

            # Comparaison avec les voisins
            st.write("Comparaison avec les clients similaires :")
            voisins = json.loads(get_data_voisins(id_client_dash))
            st.write(pd.DataFrame(voisins).drop('SK_ID_CURR', axis=1))

if page == "Interprétation locale":
    st.title("Dashboard Prêt à dépenser - Page Interprétation locale")
    st.write("Cliquez sur le bouton ci-dessous pour voir l'interprétation locale :")
    button_local = st.button("Voir l'interprétation locale")
    if button_local:
        if id_client_dash != '<Select>':
            # Affichage des SHAP values
            st.write("SHAP values du client sélectionné :")
            shap_local = shap_values_local(id_client_dash)
            shap_df = pd.DataFrame(shap_local['data'], columns=shap_local['feature_names'])
            shap_df['SHAP Value'] = shap_local['shap_values']
            st.write(shap_df)

            # Graphique des SHAP values
            st.write("Graphique des SHAP values :")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_local['shap_values'], shap_df, show=False)
            st.pyplot(fig)

if page == "Interprétation globale":
    st.title("Dashboard Prêt à dépenser - Page Interprétation globale")
    st.write("Cliquez sur le bouton ci-dessous pour voir l'interprétation globale :")
    button_global = st.button("Voir l'interprétation globale")
    if button_global:
        # Affichage des SHAP values globales
        st.write("SHAP values de l'ensemble du jeu de données :")
        shap_global = shap_values()
        shap_df_global = data_scaled.drop('SK_ID_CURR', axis=1)
        shap_df_global['SHAP Value (class 0)'] = shap_global['shap_values_0']
        shap_df_global['SHAP Value (class 1)'] = shap_global['shap_values_1']
        st.write(shap_df_global)
