import streamlit as st
import pandas as pd
import requests
import json
import boto3

st.set_page_config('Dashboard',layout="wide")

API_URL = st.secrets.api.api_url
S3_ACCESS_ID = st.secrets.s3.access_id
S3_BUCKET_NAME = st.secrets.s3.bucket_name
S3_ACCESS_KEY = st.secrets.s3.access_key
S3_REGION = st.secrets.s3.region
SRC_DATA = 'test_df.csv'

@st.cache_data
def load_data(bucket,object_key):

    s3 = boto3.client(
        's3',
        aws_access_key_id=S3_ACCESS_ID,
        aws_secret_access_key=S3_ACCESS_KEY,
        region_name=S3_REGION
    )

    s3.download_file(bucket, object_key, object_key)
    df = pd.read_csv(object_key, sep=';')
    return df

@st.cache_data
def load_model():
    url = API_URL + '/load_model_by_name'
    json = {
        "name": "sk-learn-xgboost-model"
    }
    response = requests.post(url,json=json)
    if response.status_code == 200:
        print("Modèle chargé")

@st.cache_data
def get_client_prediction(client_features):
    url = API_URL + '/predict'
    payload = {"features": client_features}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return None


# Chargement du modèle
load_model()

# Chargement des données
st.session_state['df'] = load_data(S3_BUCKET_NAME,SRC_DATA)
df = st.session_state['df']


# Récupération des données du client
id_client = st.sidebar.number_input(
    'ID du client',
    min_value=int(st.session_state['df']['SK_ID_CURR'].min()),
    max_value=int(st.session_state['df']['SK_ID_CURR'].max()),
    value=None,
    key='id_client'
)

if id_client in st.session_state['df']['SK_ID_CURR'].to_list():
    st.session_state['features'] = df.loc[df['SK_ID_CURR'] == id_client].drop(columns=['SK_ID_CURR']).iloc[0].to_dict()
    st.session_state['prediction_client'] = get_client_prediction(st.session_state['features'])
    st.sidebar.success('Données du client chargé avec succès')
else:
    if 'prediction_client' in st.session_state:
        st.session_state.pop('prediction_client')
    st.sidebar.error("L'ID renseigné n'existe pas, merci de rentrer un identifiant valide")


# Explication des features principales
with open('features_descriptions.json', 'r') as f:
    features_descriptions = json.load(f)

with st.sidebar.expander("Explication des variables"):
    feature = st.selectbox("Choisissez une feature", list(features_descriptions.keys()))
    st.write(features_descriptions[feature])


# Création des différentes pages de l'application et du menu de navigation
informations = st.Page("informations.py",title="Informations du client",icon="👤",default=True)
score = st.Page("score.py",title="Score et interprétation",icon="🏆")
comparaison = st.Page("comparaison.py",title="Analyse et comparaison des variables",icon="📊")

pg = st.navigation([informations,score,comparaison])
pg.run()