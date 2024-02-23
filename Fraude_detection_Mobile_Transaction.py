# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:10:12 2023

@author: a902744
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:06:26 2023

@author: a902744
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
#import numpy as np
#from visualization import viz_data
import seaborn as sns
import matplotlib.pylab as plt 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from numerize.numerize import numerize
from streamlit_extras.metric_cards import style_metric_cards



import random
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import confusion_matrix
import plotly.express as px 
import numpy as np


import duckdb
import plotly.graph_objects as go

import joblib
from sklearn.metrics import r2_score

#from pandas_profiling import profile_report
#from ydata_profiling import ProfileReport

#from streamlit_pandas_profiling import st_profile_report
#from sklearn.model_selection import train_test_splitfrom sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#from sklearn.metrics import precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")


# load Style css
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


df = pd.read_csv('data.csv')

#st.dataframe(df.head()) 

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("Ce projet s'inscrit dans un contexte de controle des transactions. L'objectif est de prédire si une transaction est fraudulause ou pas à partir de ses caractéristique.")
    
    
    st.write("Ce jeu de données contient des transactions de mobile money générées avec le simulateur PaySim. La simulation était basée sur un échantillon de transactions réelles recueillies par une entreprise qui est le fournisseur du service financier mobile actuellement opérationnel dans plus de 14 pays à travers le monde. Les données sont un ensemble de journaux financiers d'un mois d'un service d'argent mobile mis en œuvre dans un pays africain.")
    
    st.write("Le jeu de données contient (suivant l'exemple ci-dessus) : step - correspond à une unité de temps dans le monde réel. Dans ce cas, 1 étape représente 1 heure de temps. Nombre total d'étapes : 744 (simulation sur 30 jours).")
    
    if st.checkbox("Explication des variables") :
        
        st.write("type - CASH-IN, CASH-OUT, DEBIT, PAYMENT et TRANSFER.")
        st.write("amount - montant de la transaction en monnaie locale.")
        st.write ("nameOrig - client ayant initié la transaction.")
        st.write("oldbalanceOrg - solde initial avant la transaction.")
        st.write("newbalanceOrig - nouveau solde après la transaction.")
        st.write("nameDest - client destinataire de la transaction.")
        st.write("oldbalanceDest - solde initial du destinataire avant la transaction. Notez qu'il n'y a pas d'information pour les clients dont le nom commence par M (commerçants).")
        st.write("newbalanceDest - nouveau solde du destinataire après la transaction. Notez qu'il n'y a pas d'information pour les clients dont le nom commence par M (commerçants).")
        st.write("isFraud - Il s'agit des transactions effectuées par des agents frauduleux dans la simulation. Dans ce jeu de données spécifique, le comportement frauduleux des agents vise à tirer profit en prenant le contrôle des comptes clients et en essayant de vider les fonds en les transférant vers un autre compte, puis en les retirant du système.")
        st.write("isFlaggedFraud - Le modèle commercial vise à contrôler les transferts massifs d'un compte à un autre et signale les tentatives illégales. Une tentative illégale dans ce jeu de données est une tentative de transfert de plus de 200 000 dans une seule transaction.")
    #st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire si une transaction est fraudulause ou pas.")
    
    #st.image("fraude_detection.jpg")


elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    #st.write(df.info)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
        
    if st.checkbox("Afficher l'ensemble des variable"):
        st.write(df.columns)
    
        
    if st.checkbox("Afficher les différents type de transaction "):
       st.write((df["type"]).unique())
       st.write(df['type'].value_counts())
       
       
    if st.checkbox("Afficher les distributions des type de transaction"):
        # The classes are heavily skewed we need to solve this issue later.
        st.write('CASH_OUT', round(df['type'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
        st.write('PAYMENT', round(df['type'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
        st.write('CASH_IN', round(df['type'].value_counts()[2]/len(df) * 100,2), '% of the dataset')
        st.write('TRANSFER', round(df['type'].value_counts()[3]/len(df) * 100,2), '% of the dataset')
        st.write('DEBIT', round(df['type'].value_counts()[4]/len(df) * 100,2), '% of the dataset')

    if st.checkbox("Distribution des transactions frauduleuse et non frauduleuse  ") :
        
        st.write('Non Frauduleuse', round(df['isFraud'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
        st.write('Frauduleuse', round(df['isFraud'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
    
           
 
elif page == pages[2]:
    st.write("### Analyse de données")
    
   
    # Montant moyen par type de transaction 
    rentabilité_transaction = duckdb.sql("""SELECT 
    type,
    AVG(amount) AS montant_moyen_type_transaction
    FROM df
    GROUP BY type""").to_df()
    #rentabilité_transaction
    
    # fraude par type de transaction
    
    #ratio de fraude par type de transaction
    fraud_par_type_transaction= duckdb.sql("""SELECT 
    type,
    round(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100/ COUNT(*),3) AS fraud_ratio_by_type
    FROM df
    GROUP BY type""").to_df()

    # distribution des type de transaction 
    
    distribution_type_transaction = duckdb.sql("""SELECT 
    type,
    COUNT(*) AS nombre_transactions,
    round(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pourcentage_total
    FROM df
    GROUP BY type
    """).to_df()
    
    #Évolution du Solde Moyen des Comptes Clients
    
    evolution_solde_compte_client = duckdb.sql(""" SELECT 
    nameOrig,
    avg( newbalanceOrig - oldbalanceOrg) AS solde
    FROM df
    GROUP BY nameOrig
    ORDER BY AVG(newbalanceOrig - oldbalanceOrg) ASC""").to_df()
    
    fraud_type_transaction = duckdb.sql("""SELECT 
    type, sum(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as nombre_fraude_par_type_transaction,
    round(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100/ COUNT(*),3) AS fraud_ratio_by_type
    FROM df
    GROUP BY type""").to_df()
    
    
    #switcher

    
    
    
    def Home():
        #with st.expander("VIEW EXCEL DATASET"):
            #showData=st.multiselect('Filter: ',df.columns,default=["Policy","Expiry","Location","State","Region","Investment","Construction","BusinessType","Earthquake","Flood","Rating"])
           # st.dataframe(df[showData],use_container_width=True)
        #compute top analytics
        total_montant = (df['amount']).sum()
        investment_min = (df['amount']).min()
        investment_mean = (df['amount']).mean()
        investment_max = (df['amount']).max() 
        


        total1,total2,total3,total4 =st.columns(4,gap='small')
        with total1:
            st.info('Somme du montant',icon="💰")
            st.metric(label="",value=f"{total_montant:,.0f}")
    
        with total2:
            st.info('Montant minimal',icon="💰")
            st.metric(label="",value=f"{investment_min:,.0f}")
    
        with total3:
            st.info('Average montant',icon="💰")
            st.metric(label="",value=f"{investment_mean:,.0f}")
    
        with total4:
            st.info('Montant maximal',icon="💰")
            st.metric(label="",value=f"{investment_max:,.0f}")
        style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow="#F71938" , border_size_px = 0.7)
            
    Home()
    
    # Etude sur la variable 

    if st.checkbox("distribution des types de transaction"):
    
        col1, col2 = st.columns(2)
    
     
        with col1:
            st.subheader('distribution des type de transaction')
            st.write(distribution_type_transaction)
        
        # Affichage des résultats DuckDB dans la deuxième colonne
        with col2:
            st.subheader('ratio de fraude par type de transaction')
        st.write(fraud_par_type_transaction)
    
    
    
    
    st.write("Distribution des différentes type de transaction")  
    labels = distribution_type_transaction["type"]
    values = distribution_type_transaction["nombre_transactions"]
    
    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    st.write(fig)
    st.write("Distribution des fraudes suivant les différents type de transactions")
    labels = fraud_type_transaction["type"]
    values = fraud_type_transaction["nombre_fraude_par_type_transaction"]
    
    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    st.write(fig)
    
    # Assuming `st` is a Streamlit object used for displaying the plot
    #labels1 = distribution_type_transaction["type"]
    #values1 = distribution_type_transaction["nombre_transactions"]
    #labels2 = fraud_type_transaction["type"]
    #values2 = fraud_type_transaction["nombre_fraude_par_type_transaction"]
    
    
    # Create subplots: use 'domain' type for Pie subplot
    #fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]])
    #fig.add_trace(go.Pie(labels=labels1, values=values1, name="Distribution des différents types de transaction"),
        #          1, 1)
    #fig.add_trace(go.Pie(labels=labels2, values=values2, name="Distribution des fraudes suivant les différents types de transaction"),
           #       1, 2)
    
   # fig.update_traces(hole=0.4, hoverinfo="labels+percent+name")
    
    #fig.update_layout(
        #title_text="Distribution des Transactions et Fraudes par Type",
        # Add annotations in the center of the donut pies.
       # annotations=[dict(text='Transactions', x=0.18, y=0.5, font_size=20, showarrow=False),
           #          dict(text='Fraudes', x=0.82, y=0.5, font_size=20, showarrow=False)])
    
    # Assuming you want to display the figure in Streamlit
    #st.plotly_chart(fig)
    
    
    # etude sur la variable nameOrig et nameDest
    
    dt = df.copy()
    
    dt['first_letter_nameOrig'] = dt['nameOrig'].str[0]
    dt['first_letter_nameDest'] = dt['nameDest'].str[0]
    
    st.write("Distribution des clients et marchands")
    
    labels = dt["first_letter_nameDest"].index
    values = dt["first_letter_nameDest"].value_counts()
    
    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3 ,textinfo='label+percent',
                             insidetextorientation='radial')])
    st.write(fig)
    
    # Rentabilité par type de transaction 
    rentabilité_transaction = duckdb.sql("""SELECT 
    type,
    sum(amount) AS montant_type_transaction
    FROM df
    GROUP BY type""").to_df()
    
    st.write("Rentabilité par type de transaction")
    #st.write(rentabilité_transaction)
    fig = px.histogram(rentabilité_transaction , x = "type" , y = "montant_type_transaction" , text_auto=True)
    st.write(fig)
       
       
    if st.checkbox("montant moyen par type transaction "):
  
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader('Montant moyen par type de transaction')
            st.write(rentabilité_transaction)
        
        # Affichage des résultats DuckDB dans la deuxième colonne
        with col4:
            st.subheader('Évolution du Solde Moyen des Comptes Clients')
            st.write(evolution_solde_compte_client)
        # Créer une courbe avec les données filtrées
        
        
        fig = px.scatter(evolution_solde_compte_client, x="nameOrig", y="solde")
        st.write(fig)
        
    
    
    # Supprime les espaces en trop dans les noms de colonnes
    var_catégorielle = ['type', 'nameOrig', 'nameDest']

    # Assure-toi que les noms de colonnes existent dans ton dataframe
    colonnes_existantes = df.columns.tolist()
    colonnes_a_supprimer = [colonne for colonne in var_catégorielle if colonne in colonnes_existantes]
    
    # Supprime les colonnes catégorielles si elles existent dans le dataframe
    df_num = df.drop(columns=colonnes_a_supprimer)
   
    # matrice de correlation
    mat_cor = df_num.corr().round(decimals=2)
    st.write(mat_cor)
   
    fig3, ax = plt.subplots()
    sns.heatmap(df_num.corr(), ax=ax)
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig3)
    
    # Etude sur la variable isflaggegfraud 
   
    # quells sonts les clients  qui font des transferts massifs d'un compte à un autre # remarquer pour ce cas de figure si le montant est supérieur à 200 000 on signale cet transaction comme étant illégale 
    Client_trans_massive  = duckdb.query(("select nameOrig , amount from df where isFlaggedFraud == 1"))
    #st.write(Client_trans_massive)
    # create columns
    
    # Taux de Signalement des Transactions Massives 
    taux_signalisation_transaction_massive =  duckdb.sql("""SELECT 
    round(SUM(CASE WHEN amount > 200000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) AS flagged_transactions_rate
    FROM df""").to_df()

    def transfer_massive():
        #with st.expander("VIEW EXCEL DATASET"):
            #showData=st.multiselect('Filter: ',df.columns,default=["Policy","Expiry","Location","State","Region","Investment","Construction","BusinessType","Earthquake","Flood","Rating"])
           # st.dataframe(df[showData],use_container_width=True)
        #compute top analytics
        taux_signalisation_transaction_massive =  duckdb.sql("""SELECT 
        round(SUM(CASE WHEN amount > 200000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) AS flagged_transactions_rate
        FROM df""").to_df()
        #Client_trans_massive  = duckdb.query(("select nameOrig , amount from df where isFlaggedFraud == 1"))
        montant_massive_min = Client_trans_massive['amount'].min()
        montant_massive_max = Client_trans_massive['amount'].max()
        nombre_isflagged_fraud_massive = len(Client_trans_massive) 
        


        total1,total2,total3,total4 =st.columns(4,gap='small')
        with total1:
            st.info('Taux de signalisation',icon="💰")
            

            st.metric(label="",value=f"{ taux_signalisation_transaction_massive:,.0f}")
    
        with total2:
            st.info('Montant minimal',icon="💰")
            st.metric(label="",value=f"{montant_massive_min:,.0f}")
    
        with total3:
            st.info('Average montant',icon="💰")
            st.metric(label="",value=f"{montant_massive_max:,.0f}")
    
        with total4:
            st.info('Montant maximal',icon="💰")
            st.metric(label="",value=f"{nombre_isflagged_fraud_massive:,.0f}")
        style_metric_cards(background_color="#FFFFFF",border_left_color="#686664",border_color="#000000",box_shadow="#F71938" , border_size_px = 1)
        
            
   # transfer_massive()

    st.write("Les clients ayant effectué des transactions massives")
    # les clients qui font des transaction massive 
    Client_trans_massive  = duckdb.query("select nameOrig , amount from df where isFlaggedFraud == 1").to_df()   
    fig = px.scatter(Client_trans_massive, x="nameOrig", y="amount")
    st.write(fig)
    
    
    
    data = df.copy()
    
    # ajouter une nouvelle variable time 
    
    # Si vous connaissez la date de début des données, utilisez-la. Sinon, supposons une date de début fictive.
    date_debut = pd.to_datetime('2023-02-01')  # Remplacez par la date réelle si elle est connue
    
    # Convertir step en heures en ajoutant la date de début
    data['timestamp'] = date_debut + pd.to_timedelta(data['step'], unit='h')
    
    # Extraire le jour de la semaine et ajouter une colonne 'jour_de_la_semaine'
    data['jour_de_la_semaine'] = data['timestamp'].dt.day_name()
        
    import matplotlib.pyplot as plt

    # Assurez-vous que 'timestamp' est au format datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    st.write(data)
    # KPI : Répartition des montants de transactions par timestamp avec visualisation graphique
    #plt.figure(figsize=(14, 8))
    #fig1 = px.scatter(data['timestamp'], data['amount'], alpha=0.5)
    #fig2 = px.scatter(data, x="timestamp", y="amount")
    #plt.title('Répartition des montants de transactions par timestamp')
    #plt.xlabel('Timestamp')
    #plt.ylabel('Montant de la transaction')
    #plt.show()
    #st.write(fig2)
    
    
    
    ##################################################################
    # Exemple de données fictives
    data = df
    
    labels2 = ['Fraud', 'Non-Fraud']
    values2 = [20, 80]
    
    # Créer la figure avec des sous-graphiques
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'bar'}, {'type':'pie'}]])
    
    # Ajouter le premier sous-graphique (barplot)
    fig.add_trace(go.Bar(x=data['type'], y=data['amount'], marker_color=data['isFraud'], text=data['isFraud'], hoverinfo='y+text', showlegend=False),
                  row=1, col=1)
    
    # Ajouter le deuxième sous-graphique (pie chart)
    fig.add_trace(go.Pie(labels=labels2, values=values2, name="Distribution des fraudes suivant les différents types de transaction"),
                  row=1, col=2)
    
    # Mettre à jour la mise en page et les annotations
    fig.update_layout(
        title_text="Distribution des Transactions et Fraudes par Type",
        annotations=[
            dict(text='Transactions', x=0.18, y=0.5, font_size=20, showarrow=False),
            dict(text='Fraudes', x=0.82, y=0.5, font_size=20, showarrow=False)
        ]
    )
    
    # Afficher la figure dans Streamlit
    st.plotly_chart(fig)
 
elif page == pages[3]:
    
    data_t = df.copy()
    
    data_cash_trans = data_t.query('type == "TRANSFER" or type == "CASH_OUT"')
    
    var_catégorielle = ['nameOrig', 'nameDest', 'isFlaggedFraud']

    colonnes_existantes = df.columns.tolist()
    colonnes_a_supprimer = [colonne for colonne in var_catégorielle if colonne in colonnes_existantes]

    data_cash_trans = df.drop(columns=colonnes_a_supprimer)
    
    for label, content in data_cash_trans.items():
        if not pd.api.types.is_numeric_dtype(content):
            data_cash_trans[label] = pd.Categorical(content).codes+1
    
    X = data_cash_trans.drop('isFraud', axis=1)
    y = data_cash_trans['isFraud']
    
    
    np.random.seed(37)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # test avec une methode de machine Learning 
    
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    #Definition du modèle
    model = XGBClassifier()
    model.fit(X_train,y_train)
    #Prediction et affichage de la matrice de confusion
    predict_train = model.predict(X_train)
    c_train = confusion_matrix(y_train, predict_train)
    predict_test = model.predict(X_test)
    c_test = confusion_matrix(y_test, predict_test)
    
    ## load model 
    
    #loading in the model to predict on the data 
    #pickle_in = open('classifier.pkl', 'rb') 
    #classifier = pickle.load(pickle_in) 
    import streamlit as st
    import numpy as np
    
    # Supposons que votre modèle soit déjà chargé et stocké dans la variable 'model'
    # Vous devrez remplacer la ligne suivante par le chargement réel de votre modèle.
    # model = ...
    
    def make_prediction(step, type_transaction, amount, newbalanceOrg, oldbalanceOrig, newbalanceDest, oldbalanceDest, isflaggegfraude):
        # Convertir les entrées en nombres (assurez-vous que toutes les valeurs sont numériques)
        inputs = np.array([step, type_transaction, amount, newbalanceOrg, oldbalanceOrig, newbalanceDest, oldbalanceDest]).astype(float)
    
        # Effectuer la prédiction
        prediction = model.predict([inputs])
    
        return prediction
    
    def main():
       # st.title("Prédiction de Fraude")
    
        html_temp = """
        <div style ="background-color:yellow;padding:13px">
        <h1 style ="color:black;text-align:center;">Fraud detection in mobile money transfer </h1>
        </div>
        """
    
        st.markdown(html_temp, unsafe_allow_html=True)
    
        step = st.text_input("Heure de transaction", "Type Here")
        type_transaction = st.text_input("Type de transaction", "Type Here")
        amount = st.text_input("Montant", "Type Here")
        newbalanceOrg = st.text_input("Solde avant la transaction ", "Type Here")
        oldbalanceOrig = st.text_input("Solde après la transaction", "Type Here")
        newbalanceDest = st.text_input("Solde avant la transaction du destinataire", "Type Here")
        oldbalanceDest = st.text_input("Solde après la transaction du Destinataire", "Type Here")
        #isflaggegfraude = st.text_input("Est-il une transaction massive ", "Type Here")
    
        result = ""
    
        if st.button("Predict"):
            result = make_prediction(step, type_transaction, amount, newbalanceOrg, oldbalanceOrig, newbalanceDest, oldbalanceDest)
            # st.success('Le résultat est {}'.format(result))
            if result == 0:
                st.success('Transaction non frauduleuse')
            elif result == 1:
                st.error('Transaction frauduleuse')
                
        from sklearn.metrics import accuracy_score
        
        # Calcul de l'accuracy (précision)
        if 'X_test' in locals() and 'y_test' in locals():
        # Supposons que X_test soit une liste de listes représentant les entrées de test
            y_pred = [make_prediction(*x) for x in X_test]
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")
            
        # Connexion à la base de données PostgreSQL
        DATABASE_URL = "postgresql://your_username:your_password@localhost:5432/your_database_name"
        engine = create_engine(DATABASE_URL)
        
        # Création d'une table (si elle n'existe pas déjà)
        TABLE_NAME = "transactions"
        CREATE_TABLE_QUERY = """
        CREATE TABLE IF NOT EXISTS transactions (
            step INT,
            type_transaction VARCHAR(50),
            amount FLOAT,
            newbalanceOrg FLOAT,
            oldbalanceOrig FLOAT,
            newbalanceDest FLOAT,
            oldbalanceDest FLOAT,
            isflaggedfraud BOOLEAN
        );
        """
        engine.execute(CREATE_TABLE_QUERY)
        
        # Collecte des données à partir des widgets Streamlit
        step = st.text_input("Heure de la transaction", "")
        type_transaction = st.radio("Sélectionner le type de transaction:", ("Cash_out", "Transfer", "Other"))
        amount = st.text_input("Montant", "")
        newbalanceOrg = st.text_input("Solde avant la transaction", "")
        oldbalanceOrig = st.text_input("Solde après la transaction", "")
        newbalanceDest = st.text_input("Solde avant la transaction du destinataire", "")
        oldbalanceDest = st.text_input("Solde après la transaction du Destinataire", "")
        isflaggedfraud = st.checkbox("Transaction massive (frauduleuse)?")
        isfraud = 
        
        # Création d'un DataFrame avec les données
        data = {
            "step": [step],
            "type_transaction": [type_transaction],
            "amount": [amount],
            "newbalanceOrg": [newbalanceOrg],
            "oldbalanceOrig": [oldbalanceOrig],
            "newbalanceDest": [newbalanceDest],
            "oldbalanceDest": [oldbalanceDest],
            "isflaggedfraud": [isflaggedfraud]
        }
        df = pd.DataFrame(data)
        
        # Enregistrement des données dans la base de données
        df.to_sql(TABLE_NAME, engine, if_exists='append', index=False)
        
        # Affichage pour vérifier que les données ont bien été enregistrées
        st.write("Données enregistrées dans la base de données.")
    
    if __name__ == "__main__":
        main()

        
        
    
    
    
    
               