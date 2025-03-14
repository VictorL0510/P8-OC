import streamlit as st
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import pickle
import boto3

S3_ACCESS_ID = st.secrets.s3.access_id
S3_BUCKET_NAME = st.secrets.s3.bucket_name
S3_ACCESS_KEY = st.secrets.s3.access_key
S3_REGION = st.secrets.s3.region
SRC_DATA = 'shap_values.pkl'

@st.cache_data
def get_shap_values(bucket, object_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=S3_ACCESS_ID,
        aws_secret_access_key=S3_ACCESS_KEY,
        region_name=S3_REGION
    )

    s3.download_file(bucket, object_key, object_key)
    with open(object_key, "rb") as f:
        shap_values = pickle.load(f)
    return shap_values

if 'prediction_client' not in st.session_state:
    st.error('Veuillez saisir un ID Client')

else:
    st.title('Score et interprétation')

    THRESHOLD = st.session_state.prediction_client['threshold']

    st.sidebar.markdown('---')

    st.sidebar.metric(
            "Seuil de Décision",
            f"{THRESHOLD:.1%}"
        )

    if st.session_state.prediction_client["prediction"] == 0:
        st.success("✅ Crédit recommandé")
    else:
        st.error("❌ Crédit non recommandé")

    columns = st.columns(2)

    with columns[0]:
        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = st.session_state.prediction_client['default_probability']*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilité de défaut du client", 'font': {'size': 24}},
        number={'suffix': "%", 'valueformat': ".2f"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, THRESHOLD*100], 'color': 'green'},
                {'range': [THRESHOLD*100, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': THRESHOLD*100}}))
        
        st.plotly_chart(fig, key="iris", on_select="rerun")

    df = st.session_state['df']

    shap_values = get_shap_values(S3_BUCKET_NAME,SRC_DATA)

    index_client = df.loc[df['SK_ID_CURR'] == st.session_state['id_client']].index[0]
    client_shap_values = shap_values[index_client]
    client_data = df.drop(columns=['SK_ID_CURR']).iloc[index_client]

    shap.initjs()

    with columns[1]:
        st.header('Contribution des variables au score obtenu')
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(client_shap_values, show=False)
        st.pyplot(fig)
        plt.clf()

    st.markdown('---')

    st.header('Comparaison avec les contributions globales')
    fig = plt.figure(figsize=(2, 2))
    shap.summary_plot(shap_values, df.drop(columns=['SK_ID_CURR']), show=False)
    st.pyplot(fig)