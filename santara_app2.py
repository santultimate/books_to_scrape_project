import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['views'] = scaler.fit_transform(data[['views']])

# Liste des articles Wikipédia
articles = ['Python_(programming_language)', 'Data_science', 'Machine_learning', 
            'Artificial_intelligence', 'Blockchain', 'Deep_learning', 
            'Cloud_computing', 'Big_data', 'Internet_of_things', 'Cybersecurity']

headers = {'User-Agent': 'MyStreamliApp/0.1(https://myapp.example.com)'  }

# Fonction pour récupérer les données de consultation d'une page Wikipédia via l'API de Wikimedia
def get_wikipedia_data(article, start_date, end_date):
    url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start_date}/{end_date}'
    response = requests.get(url)
        
    if response.status_code == 200:
        data = response.json()
        items = data['items']
        dates = [item['timestamp'][:8] for item in items]
        views = [item['views'] for item in items]
        df = pd.DataFrame({'date': dates, 'views': views})
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        st.error(f"Erreur lors de la récupération des données (Code {response.status_code}): {response.text}")
        return None

# Fonction pour afficher les modèles et prédictions
def arima_model(data):
    model = ARIMA(data['views'], order=(5, 1, 0))
    model_fit = model.fit()
    st.write(model_fit.summary())
    data['arima_pred'] = model_fit.predict(start=0, end=len(data)-1)
    plt.plot(data['date'], data['views'], label='Données réelles')
    plt.plot(data['date'], data['arima_pred'], label='Prédictions ARIMA', color='red')
    plt.legend()
    st.pyplot()

def prophet_model(data):
    data_prophet = data.rename(columns={'date': 'ds', 'views': 'y'})
    model = Prophet()
    model.fit(data_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig = model.plot(forecast)
    st.pyplot(fig)

# Interface Streamlit
st.title('Comparaison des Modèles Prédictifs sur les Vues Wikipédia')
st.markdown("Sélectionnez un article Wikipédia et une plage de dates pour extraire les données de consultation.")

# Sélection d'un article dans la liste
article = st.selectbox('Sélectionnez un article Wikipédia', articles)

# Entrée des dates de début et de fin
start_date = st.date_input('Date de début', datetime(2022, 1, 1))
end_date = st.date_input('Date de fin', datetime(2023, 1, 1))

# Afficher le bouton pour récupérer les données
if st.button('Récupérer les données'):
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    # Récupérer les données de l'article sélectionné
    data = get_wikipedia_data(article, start_date_str, end_date_str)
    
    if data is not None:
        st.write("Données récupérées :")
        st.dataframe(data)
        
        # Comparer les modèles
        st.write("### Modèle ARIMA")
        arima_model(data)
        
        st.write("### Modèle Prophet")
        prophet_model(data)
