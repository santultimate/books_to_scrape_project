import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Fonction pour charger les données à partir d'un lien Wikipedia
def load_data_from_wikipedia(url):
    try:
        tables = pd.read_html(url)
        return tables[0]  # On prend la première table de la page
    except Exception as e:
        st.error(f"Erreur lors du chargement des données depuis Wikipedia : {e}")
        return None

# Fonction pour charger les données locales
def load_data_local(file):
    data = pd.read_csv(file, parse_dates=True, index_col=0)
    return data

# Modèle ARIMA
def model_arima(data, p=5, d=1, q=0):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

# Modèle SARIMA
def model_sarima(data, p=1, d=1, q=1, P=1, D=1, Q=1, s=12):
    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    model_fit = model.fit()
    return model_fit

# Modèle Prophet
def model_prophet(data):
    data.reset_index(inplace=True)
    data.columns = ['ds', 'y']
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    return forecast, model

# Interface Streamlit
st.title("Comparaison de Modèles Prédictifs")

# Choix de la source de données
upload_option = st.radio("Comment souhaitez-vous charger vos données ?", ('Lien Wikipedia', 'Télécharger un fichier'))

data = None
if upload_option == 'Lien Wikipedia':
    url = st.text_input("Entrez le lien Wikipedia contenant les données")
    if url:
        data = load_data_from_wikipedia(url)
elif upload_option == 'Télécharger un fichier':
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    if uploaded_file:
        data = load_data_local(uploaded_file)

# Si les données sont chargées, affichons un aperçu
if data is not None:
    st.write("Aperçu des données :", data.head())

    # Sélection du modèle à utiliser
    model_choice = st.selectbox("Choisissez un modèle de prédiction", ["ARIMA", "SARIMA", "Prophet"])

    if model_choice == "ARIMA":
        p = st.number_input("Paramètre p", value=5)
        d = st.number_input("Paramètre d", value=1)
        q = st.number_input("Paramètre q", value=0)
        steps = st.number_input("Pas de prédiction", value=10)

        if st.button("Lancer le modèle ARIMA"):
            model_fit = model_arima(data['value'], p, d, q)
            st.write("Résumé du modèle ARIMA :", model_fit.summary())
            forecast = model_fit.forecast(steps=steps)
            st.write(f"Prédiction sur {steps} pas :")
            st.write(forecast)
            plt.plot(data.index, data['value'], label="Données historiques")
            plt.plot(pd.date_range(data.index[-1], periods=steps+1, freq='D')[1:], forecast, label="Prédiction")
            plt.legend()
            st.pyplot(plt)

    elif model_choice == "SARIMA":
        p = st.number_input("Paramètre p (saison)", value=1)
        d = st.number_input("Paramètre d (différenciation)", value=1)
        q = st.number_input("Paramètre q (erreur)", value=1)
        P = st.number_input("Paramètre P (saisonnier)", value=1)
        D = st.number_input("Paramètre D (différenciation saisonnière)", value=1)
        Q = st.number_input("Paramètre Q (erreur saisonnière)", value=1)
        s = st.number_input("Saisonnalité (s)", value=12)
        steps = st.number_input("Pas de prédiction", value=10)

        if st.button("Lancer le modèle SARIMA"):
            model_fit = model_sarima(data['value'], p, d, q, P, D, Q, s)
            st.write("Résumé du modèle SARIMA :", model_fit.summary())
            forecast = model_fit.forecast(steps=steps)
            st.write(f"Prédiction sur {steps} pas :")
            st.write(forecast)
            plt.plot(data.index, data['value'], label="Données historiques")
            plt.plot(pd.date_range(data.index[-1], periods=steps+1, freq='D')[1:], forecast, label="Prédiction")
            plt.legend()
            st.pyplot(plt)

    elif model_choice == "Prophet":
        steps = st.number_input("Pas de prédiction (mois)", value=12)

        if st.button("Lancer le modèle Prophet"):
            forecast, model = model_prophet(data)
            st.write("Prévision avec Prophet :", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            fig = model.plot(forecast)
            st.pyplot(fig)

else:
    st.write("Veuillez charger des données pour continuer.")
