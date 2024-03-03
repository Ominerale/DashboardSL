# Library imports
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import uvicorn
import shap

# Create a FastAPI instance
app = FastAPI()

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
@app.get('/')
def welcome():
    """
    Welcome message.
    :param: None
    :return: Message (string).
    """
    return 'Welcome to the API'


@app.get('/{client_id}')
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


@app.get('/prediction/{client_id}')
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


@app.get('/clients_similaires/{client_id}')
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


@app.get('/shaplocal/{client_id}')
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

@app.get('/shap/')
def shap_values():
    """ Calcul les shap values de l'ensemble du jeu de données
    :param:
    :return: shap values
    """
    # explainer = shap.TreeExplainer(model['classifier'])
    shap_val = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    return {'shap_values_0': shap_val[0].tolist(),
            'shap_values_1': shap_val[1].tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    # uvicorn.run(app, host='127.0.0.1', port=8000)