import streamlit as st
import pandas as pd

if 'prediction_client' not in st.session_state:
    st.error('Veuillez saisir un ID Client')

else:
    client_features = st.session_state["features"]

    # 1. Présentation synthétique via des KPI
    st.subheader("Informations principales")

    kpi1, kpi2, kpi3 = st.columns(3)

    # 1) Âge
    age_in_years = round(abs(client_features.get("DAYS_BIRTH", 0)) / 365, 1)
    kpi1.metric("Âge", f"{age_in_years} ans")

    # 2) Revenu total
    income = client_features.get("AMT_INCOME_TOTAL", 0)
    kpi2.metric("Revenu total", f"{income:,.0f} €")

    # 3) Montant du crédit
    credit = client_features.get("AMT_CREDIT", 0)
    kpi3.metric("Montant du crédit", f"{credit:,.0f} €")


    # 2. Récapitulatif des données clients
    st.subheader("Détails des variables du client")

    features_selected = st.multiselect('Sélectionner les variables dont on souhaite afficher la valeur',client_features.keys())

    cols = st.columns(4)

    count = 0
    for feature in features_selected:
        cols[count % 4].metric(label = feature,value = client_features[feature])
        count +=1