import streamlit as st
import plotly.express as px
import json

if 'prediction_client' not in st.session_state:
    st.error('Veuillez saisir un ID Client')

else:
    df = st.session_state["df"]

    # Analyse bi-variée
    st.header('Analyse bi-variée')
    with open('features_descriptions.json', 'r') as f:
        features_descriptions = json.load(f)

    columns = list(features_descriptions.keys())

    seuil = 10

    def is_categorical(col):
        return df[col].nunique() <= seuil

    cols = st.columns(2)

    with cols[0]:
        var1 = st.selectbox('Sélectionner une première variable', columns)

    with cols[1]:
        # Exclure la variable sélectionnée pour var1
        options2 = [col for col in columns if col != var1]
        var2 = st.selectbox('Sélectionner une seconde variable', options2)

    # Déterminer si les variables doivent être traitées comme catégorielles
    is_cat1 = is_categorical(var1)
    is_cat2 = is_categorical(var2)

    if not is_cat1 and not is_cat2:
        # Deux variables numériques : scatter plot
        fig = px.scatter(df, x=var1, y=var2, title="Nuage de points")
    elif is_cat1 and not is_cat2:
        fig = px.box(df, x=var1, y=var2, title=f"Boxplot de {var2} par {var1}")
    elif not is_cat1 and is_cat2:
        fig = px.box(df, x=var2, y=var1, title=f"Boxplot de {var1} par {var2}")
    else:
        contingency = df.groupby([var1, var2]).size().reset_index(name='counts')
        fig = px.bar(contingency, x=var1, y='counts', color=var2,
                    title=f"Répartition de {var1} et {var2}", barmode='group')

    st.plotly_chart(fig)

    st.markdown('---')


    # Comparaison avec les autres clients
    st.header('Comparaison avec les autres clients')

    feature_selected = st.selectbox("Sélectionnez une feature", list(features_descriptions.keys()))
    client_features = st.session_state.get('features', {})
    client_value = client_features.get(feature_selected)

    col1, col2 = st.columns(2)
        
    # Box Plot avec la valeur du client indiquée
    fig_box = px.box(df, y=feature_selected, title=f"Box Plot de {feature_selected}")
    fig_box.add_scatter(
        x=[1],
        y=[client_value],
        mode="markers",
        marker=dict(color="red", size=12),
        name="Votre valeur"
    )
    col1.plotly_chart(fig_box, use_container_width=True)

    # Histogramme avec ligne verticale indiquant la valeur du client
    fig_hist = px.histogram(
        df, 
        x=feature_selected, 
        nbins=100, 
        title=f"Distribution de {feature_selected} parmi les clients"
    )
    fig_hist.add_vline(
        x=client_value, 
        line_width=3, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Votre valeur: {client_value}",
        annotation_position="top right"
    )
    col2.plotly_chart(fig_hist, use_container_width=True)