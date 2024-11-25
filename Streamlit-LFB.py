import streamlit as st  # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import plotly.express as px # type: ignore
import folium
#from streamlit_folium import st_folium # type: ignore
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from scipy.stats import uniform, randint
import warnings
warnings.simplefilter("ignore")
import joblib
import pickle


st.title("Temps de réponse de la brigade des pompiers de Londres")
st.sidebar.image("London_Fire_Brigade_logo.svg.png")
st.sidebar.title("Sommaire")
pages = ["Le projet", "Exploration des données", "Quelques visualisations", "Préparation des données", "Modélisation", "Machine Learning", "Conclusion", "Rapport de performance"]
page = st.sidebar.radio("Aller vers", pages)

st.sidebar.title("Auteurs")
st.sidebar.write("Paul DEVIN")
st.sidebar.write("Thomas GARRIGUE")
st.sidebar.write("Marie HERBERT")
st.sidebar.write("Namita KALA")

# Mise en cache des données
@st.cache_data
def load_incident_data():
    return pd.read_csv(r"C:\Users\devin\OneDrive\Bureau\Data_Pompier\streamlit final\incident_data.csv")

@st.cache_data
def load_mobilization_data():
    return pd.read_csv(r"C:\Users\devin\OneDrive\Bureau\Data_Pompier\streamlit final\mobilization_data.csv")

@st.cache_data
def load_df_data():
    return pd.read_excel(r"C:\Users\devin\OneDrive\Bureau\Data_Pompier\streamlit final\df_soutenance.xlsx")

# Introduction
if page == pages[0] :
    st.header("Le projet")

    st.markdown("""
               Ce projet a été réalisé dans le cadre de notre formation Data Analyst via l'organisme [Datascientest](https://datascientest.com/en/).

               L'objectif principal est de prédire le temps d'arrivée sur les lieux de la brigade des pompiers de Londres.
               """)
    st.write("Ce streamlit présente notre démarche.")

    st.write("")
    st.write("")

    st.subheader("Contexte du projet")
    st.markdown("""
            La brigade des pompiers de Londres est le service d'incendie et de sauvetage le plus actif du Royaume-Uni et **l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde**.

            L'effectif se compose en 2015 de 5 992 personnes dont 5 096 pompiers opérationnels (officiers compris), tous professionnels.
            """)
    st.write("Cette brigade peut intervenir sur des incidents variés comme le secours à la personne, les accidents de la route, dans l'eau et les incendies.")

    st.write("")
    st.write("")

    st.subheader("Leurs indicateurs")
    st.markdown("""
            La LFB dispose de différents indicateurs à atteindre, dont :
               - Temps moyen mensuel d'arrivée du premier équipage : **6 min**
               - Temps moyen mensuel d'arrivée du second équipage : **8 min**
               - Premier équipage arrivé en moins de 10 min (mensuel) : **90 % des cas**
               - Premier équipage arrivé en moins de 12 min (mensuel) : **95 % des cas**
                """)

    st.write("")
    st.write("")

    st.subheader("La problématique de notre projet")
    st.markdown("""
                Comment prédire et optimiser le temps d'arrivée sur les lieux des incidents des pompiers de Londres en fonction des effectifs matériels, des effectifs humains mis en œuvre,
                du type d'incidents, du lieu d'incidents et des données temporelles ?
                """)

    st.write("")
    st.write("")

    st.subheader("Les objectifs de notre projet pour la brigade des pompiers de Londres")
    st.write("**Contexte technique :**")
    st.markdown("""
                - Permettre d'aider les équipiers pompiers à améliorer leur temps de réponse aux incidents, c-a-d le temps d'arrivée sur les lieux
                - Prévoir le bon nombre d'équipes déployées selon le type d'incidents et le lieu d'incidents.
                """)

    st.write("**Contexte économique :**")
    st.markdown("""
                Réduire le temps d'arrivée sur les lieux pour :
                - limiter le nombre d'équipes déployées
                - limiter les dommages matériels et humains,
                - et donc le coût global d'intervention.
                """)

    st.write("")
    st.write("")

    st.subheader("L'objectif de notre projet")
    st.markdown("""
                **Prédire le temps de réponse  des interventions de la brigade des pompiers de Londres, fonction :**
                - **du nombre d'équipes déployées,**
                - **du type d'incidents**
                - **et du lieu de l'incident.**
                """)

# Exploration des données
if page == pages[1]: 
    st.header("Exploration des données")

    # Mise en cache des dataframes
    @st.cache_data
    def load_incident_data():
        # Chargement de la table incidents
        return pd.read_csv(r"C:\Users\devin\OneDrive\Bureau\Data_Pompier\streamlit final\incident_data.csv")  # Remplacez le chemin d'accès par le vôtre

    @st.cache_data
    def load_mobilization_data():
        # Chargement de la table mobilisations
        return pd.read_csv(r"C:\Users\devin\OneDrive\Bureau\Data_Pompier\streamlit final\mobilization_data.csv")  # Remplacez le chemin d'accès par le vôtre

    incident_data = load_incident_data()
    mobilization_data = load_mobilization_data()

    st.subheader("Infos sur les incidents")

    if st.checkbox("Afficher quelques lignes de la table incidents"):
        st.dataframe(incident_data.head(10))

    if st.checkbox("Afficher les statistiques des variables numériques de la table incidents"):
        @st.cache_data
        def describe_incidents():
            return incident_data.describe()
        
        st.dataframe(describe_incidents())

    if st.checkbox("Afficher les valeurs manquantes de la table incidents"):
        @st.cache_data
        def na_counts_incidents():
            return incident_data.isna().sum()
        
        st.dataframe(na_counts_incidents())

    st.subheader("Infos sur les mobilisations")

    if st.checkbox("Afficher quelques lignes de la table mobilisations"):
        st.dataframe(mobilization_data.head(10))

    if st.checkbox("Afficher les statistiques des variables numériques de la table mobilisations"):
        @st.cache_data
        def describe_mobilizations():
            return mobilization_data.describe()
        
        st.dataframe(describe_mobilizations())

    if st.checkbox("Afficher les valeurs manquantes de la table mobilisations"):
        @st.cache_data
        def na_counts_mobilizations():
            return mobilization_data.isna().sum()
        
        st.dataframe(na_counts_mobilizations())
    
    # Merge des jeux de données
    st.subheader("Merge des jeux de données")
    st.write("Afin de traiter les données, nous avons regrouper les jeux de données en un seul DataFrame, mergées selon les variables indiquées dans le schéma ci-dessous.")
    st.image("Jeux de données.png", caption = "Les 2 tables et leurs relations")
    st.header("Zoom sur les données")

# Pertinence des variables
    st.subheader("Pertinence des variables")
    st.write("Nous pouvons regrouper nos variables en catégories pour l’élaboration de nos modèles de machine learning :")

# Catégories de variables
    st.markdown("""
    - **Identification et temporalité des incidents**:
      Existe-t-il une temporalité spécifique pour chaque type d’incident ?
    - **Types et propriétés des incidents**:
      En quoi le type d'incident influence-t-il le temps d'intervention ?
    - **Données géographiques des incidents**:
     La caserne sollicitée est-elle la plus proche du lieu de l'incident ?
     Certaines zones géographiques sont-elles plus sujettes aux incidents ? Si oui, quels types d’incidents ?
    - **Délais de prise en charge**:
    Permet de séquencer les temps d'intervention pour identifier les facteurs influents et améliorer la prédiction de ces délais.
    """)

# Image du mapping des catégories de variables
    st.image("Mapping_catégories_variables.png", caption="Mapping des catégories de variables")

# Variable cible
    st.subheader("Variable cible")
    st.write("La variable cible que nous allons chercher à prédire est le **temps d'arrivée sur les lieux de l'incident** (AttendanceTimeSeconds).")

# Segmentation du temps d’intervention
    st.write("**Segmentation du temps d’intervention**")
    st.image("Variable_cible.png", caption="Segmentation du temps d’intervention")

    
# Quelques visualisations
if page == pages[2]:
    df = load_df_data()
    st.header("Quelques visualisations")

    # Fonctions de création de chaque figurepour mise en cache
    @st.cache_data
    def plot_barplot_time_intervention():
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x="CalYear", y="AttendanceTimeSeconds", hue="IncidentGroup", data=df)
        plt.xlabel("Année")
        plt.ylabel("Temps d'intervention total en secondes")
        plt.title("Temps d'intervention total par année et par type d'incidents")
        return fig

    @st.cache_data
    def plot_boxplot_london_regions():
        fig = plt.figure(figsize=(14, 7))
        sns.boxplot(data=df, x="IncGeo_BoroughName", y="AttendanceTimeSeconds")
        plt.xticks(rotation=90)
        plt.title("Distribution des temps d'intervention par région de Londres")
        plt.xlabel("Région")
        plt.ylabel("Temps d'intervention (secondes)")
        return fig

    @st.cache_data
    def plot_boxplot_year():
        fig = plt.figure(figsize=(12, 6))
        sns.boxplot(x="CalYear", y="AttendanceTimeSeconds", data=df)
        plt.xlabel("Année")
        plt.ylabel("Temps d'intervention total en secondes")
        plt.title("Boxplot du temps d'intervention total par année")
        return fig

    @st.cache_data
    def plot_hist_attendance_time():
        fig = plt.figure(figsize=(12, 6))
        sns.histplot(df["AttendanceTimeSeconds"], bins=50, kde=True)
        plt.title("Distribution des temps d'intervention")
        plt.xlabel("Temps d'intervention (secondes)")
        plt.ylabel("Fréquence")
        return fig

    @st.cache_data
    def plot_hist_turnout_time():
        fig = plt.figure(figsize=(12, 6))
        sns.histplot(df["TurnoutTimeSeconds"], bins=50, kde=True)
        plt.title("Distribution des temps de mobilisation")
        plt.xlabel("Temps de mobilisation (secondes)")
        plt.ylabel("Fréquence")
        return fig

    @st.cache_data
    def plot_hist_travel_time():
        fig = plt.figure(figsize=(12, 6))
        sns.histplot(df["TravelTimeSeconds"], bins=50, kde=True)
        plt.title("Distribution des temps de déplacement")
        plt.xlabel("Temps de déplacement (secondes)")
        plt.ylabel("Fréquence")
        return fig

    @st.cache_data
    def plot_count_borough():
        fig = plt.figure(figsize=(14, 7))
        sns.countplot(data=df, y='IncGeo_BoroughName', order=df['IncGeo_BoroughName'].value_counts().index)
        plt.xticks(rotation=90)
        plt.xlabel("Nombre d'interventions")
        plt.ylabel("Arrondissement")
        plt.title("Distribution du nombre d'interventions par arrondissement")
        return fig

    @st.cache_data
    def plot_hist_incidents_by_year():
        fig = plt.figure(figsize=(14, 7))
        sns.histplot(data=df, x="CalYear", hue="IncidentGroup", multiple="stack", stat="count", palette="Set2", discrete=True, shrink=.8)
        plt.xlabel("Années")
        plt.ylabel("Volume")
        return fig

    @st.cache_data
    def plot_pie_incident_types():
        fig = plt.figure(figsize=(12, 6))
        plt.pie(df.IncidentGroup.value_counts().values, labels=df.IncidentGroup.value_counts().index, autopct="%1.1f%%", labeldistance=1.1)
        return fig

    @st.cache_data
    def plot_count_by_hour():
        df["Hour_mobilised"] = pd.to_datetime(df["DateAndTimeMobilised"]).dt.hour
        fig = plt.figure(figsize=(10, 7))
        sns.countplot(data=df, x='Hour_mobilised')
        plt.ylabel("Nombre d'interventions")
        plt.xlabel("Heure")
        return fig

    @st.cache_data
    def plot_count_by_month():
        fig = plt.figure(figsize=(10, 7))
        sns.countplot(x="Month", data=df, order=df["Month"].value_counts().index)
        plt.ylabel("Nombre d'incidents")
        plt.xlabel("Mois")
        return fig

    @st.cache_data
    def plot_count_by_weekday():
        df["Weekday_mobilised"] = pd.to_datetime(df["DateAndTimeMobilised"]).dt.weekday
        fig = plt.figure(figsize=(10, 7))
        sns.countplot(x=df["Weekday_mobilised"])
        plt.xticks([0, 1, 2, 3, 4, 5, 6], ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'])
        plt.ylabel("Nombre d'interventions")
        plt.xlabel("Jour de la semaine")
        return fig

    @st.cache_data
    def plot_count_by_time_of_day():
        fig = plt.figure(figsize=(10, 7))
        sns.countplot(x="TimeOfDay", data=df, order=["Night", "Early Morning", "Morning", "Afternoon", "Evening", "Late Evening"])
        plt.ylabel("Nombre d'incidents")
        plt.xlabel("Moment de la journée")
        return fig

    @st.cache_data
    def plot_count_by_season():
        fig = plt.figure(figsize=(10, 7))
        sns.countplot(x="Season", data=df, order=["Winter", "Spring", "Summer", "Fall"])
        plt.ylabel("Nombre d'incidents")
        plt.xlabel("Saisons")
        return fig

    @st.cache_data
    def sunburst():
        fig = px.sunburst(df, path=['IncidentGroup', 'StopCodeDescription'], height = 700, title="Catégorie principale et catégorie détaillée de l'incident")
        return fig


    # Affichage des graphiques
    st.subheader("Analyse du temps d'intervention", divider=True)
    tab1, tab2, tab3 = st.tabs(["Barplot temps d'intervention", "Boxplot interventions Londres", "Boxplot temps d'intervention par année"])
    with tab1: st.pyplot(plot_barplot_time_intervention())
    with tab2: st.pyplot(plot_boxplot_london_regions())
    with tab3: st.pyplot(plot_boxplot_year())

    st.subheader("Analyse de la séquence de temps", divider=True)
    tab4, tab5, tab6 = st.tabs(["Temps d'intervention", "Temps de mobilisation", "Temps de déplacement"])
    with tab4: st.pyplot(plot_hist_attendance_time())
    with tab5: st.pyplot(plot_hist_turnout_time())
    with tab6: st.pyplot(plot_hist_travel_time())

    st.subheader("Analyse du lieu d'intervention", divider=True)
    tab7, tab8 = st.tabs(["Mapping des interventions", "Interventions par arrondissement"])
    with tab7: st.image("Carte-interactive-Incidents.png", caption="Carte de densité des incidents de la ville de Londres")
    with tab8: st.pyplot(plot_count_borough())

    st.subheader("Analyse du type d'interventions", divider=True)
    tab9, tab10, tab11 = st.tabs(["Types d'incidents", "Répartition des incidents", "Sunburst types d'incident"])
    with tab9: st.pyplot(plot_hist_incidents_by_year())
    with tab10: st.pyplot(plot_pie_incident_types())
    with tab11: st.plotly_chart(sunburst())

    st.subheader("Analyse du volume d'interventions par temporalité", divider=True)
    tab12, tab13, tab14, tab15, tab16 = st.tabs(["Heures", "Mois", "Jours de la semaine", "Moments de la journée", "Saisons"])
    with tab12: st.pyplot(plot_count_by_hour())
    with tab13: st.pyplot(plot_count_by_month())
    with tab14: st.pyplot(plot_count_by_weekday())
    with tab15: st.pyplot(plot_count_by_time_of_day())
    with tab16: st.pyplot(plot_count_by_season())
  

# Pré-processing
if page == pages[3]:
    st.header("Nettoyage et préparation des données")

    st.subheader("A. Etapes détaillées Nettoyage des données", divider=True)
    tab1, tab2, tab3 = st.tabs(["1. Nettoyage des données", "2. Identification des variables à conserver", "3. Séparation des types de variables"])

    with tab1:
        tab1.subheader("*1. Nettoyage des données*")
        st.write("**a. Identification des doublons**")
        st.write("Nous avons identifié les doublons.")
        code = '''inci.duplicated.sum(), mobi.duplicated.sum()'''
        st.code(code, language="python")
        st.write("Il n'y a aucun doublon dans le jeu de données *Incidents*.")
        st.write("Dans le jeu de données *Mobilisations*, la variable *IncidentNumber* contient des doublons en raison de plusieurs équipes mobilisées pour le même incident.")
        
        st.write("**b. Suppression des NaN**")
        st.write("Nous avons supprimé les valeurs manquantes (NaN).")
        code = '''inci.isna.sum(), mobi.isna.sum()'''
        st.code(code, language="python")
        st.write("Les NaN ont été traités par diverses méthodes.")

    with tab2:
        tab2.subheader("*2. Identification des variables à conserver*")
        st.write("Nous avons sélectionné 31 variables explicatives.")
        
        @st.cache_data
        def load_var_exp_inci():
            return pd.DataFrame({
                "nom": ["TimeOfCall", "IncidentGroup", "StopCodeDescription", "SpecialServiceType", "PropertyCategory", 
                        "Postcode_district", "IncGeo_BoroughName", "IncidentStationGround", 
                        "FirstPumpArriving_AttendanceTime", "FirstPumpArriving_DeployedFromStation", 
                        "SecondPumpArriving_AttendanceTime", "SecondPumpArriving_DeployedFromStation", 
                        "NumStationsWithPumpsAttending", "NumPumpsAttending", "PumpCount", "PumpMinutesRounded", 
                        "NumCalls"],
                "desc": ["Heure d'appel de l'incident", "Catégorie principale de l'incident", "Catégorie détaillée de l'incident", 
                         "Type de service spécial", "Catégorie de la propriété (résidence, structure, etc.)", 
                         "Code postal du quartier", "Nom de l'arrondissement", "Nom de la caserne qui est intervenue", 
                         "Temps d'arrivée de la première équipe de secours (en seconde)", 
                         "Caserne qui a déployé la première équipe de secours", 
                         "Temps d'arrivée de la seconde équipe de secours (en seconde)", 
                         "Caserne qui a déployé la seconde équipe de secours", 
                         "Nombre de casernes avec secours déployés", "Nombre d'équipes avec secours déployés réels", 
                         "Nombre réel de pompes déployées", "Arrondi du nombre de minutes d'utilisation des pompes", 
                         "Nombre d'appels"]
            })
        
        var_exp_inci = load_var_exp_inci()
        st.dataframe(var_exp_inci, column_config={"nom": "Nom des variables", "desc": "Description"}, hide_index=True)

    with tab3:
        tab3.subheader("*3. Séparation des types de variables*")
        st.write("**a. Variables numériques**")
        code_var_num = '''num_feats = ["HourOfCall", "FirstPumpArriving_AttendanceTime", "NumStationsWithPumpsAttending", 
                                         "NumPumpsAttending", "PumpCount", "PumpMinutesRounded", "NumCalls", 
                                         "TurnoutTimeSeconds", "TravelTimeSeconds", "SecondPumpArriving_AttendanceTime", 
                                         "PumpOrder"]'''
        st.code(code_var_num, language="python")

        st.write("**b. Variables géographiques**")
        st.markdown("""
                    Nous avons créé un fichier *GPS station.csv* pour inclure les coordonnées géographiques, la catégorie géographique et la zone géographique.
                    """)
        st.image("gps_station.png", caption="Extrait du fichier GPS_Station.csv")

        st.write("Nous avons ajouté de nouvelles variables comme la distance entre l'incident et la station, le temps moyen de réponse par station et le nombre d'incidents par zone géographique.")
        code_geopy = '''
        def calculate_distance(row):
            station_coords = (row["Latitude_station_deploy"], row["Longitude_station_deploy"])
            incident_coords = (row["Latitude_incident"], row["Longitude_incident"])
            if not np.isnan(station_coords).any() and not np.isnan(incident_coords).any():
                return geodesic(station_coords, incident_coords).kilometers
            else:
                return np.nan

        df["DistanceToIncident_km"] = df.apply(calculate_distance, axis=1)
        '''
        st.code(code_geopy, language="python")

        st.write("**c. Variables temporelles**")
        var_temp = pd.DataFrame(
            {"nom" : ["HourOfCall","DateAndTimeMobilised", "Month","DayOfWeek","Hour_mobilised","Weekday_mobilised", "TimeOfDay",
                    "IsWeekend", "Season"], 
            "desc" : ["Heure de l'appel", "Date et heure de mobilisation", "Mois d'intervention", "Jour de la semaine", "Heure de mobilisation","Jour de la semaine mobilisé", "Moment de la journée", 
            "Est-ce un jour de weekend?", "Saison d'intervention"]}
                       )
        st.dataframe(var_temp, column_config ={"nom" : "Nom des variables","desc" : "Description"},hide_index=True)

        st.write("")
        st.write("") 

        st.write("**d. Variables catégorielles**")
        code_var_cat = '''cat_feats = ["TimeOfCall","IncidentGroup", "StopCodeDescription", "PropertyCategory","PropertyType", "IncGeo_BoroughName",
        "FirstPumpArriving_DeployedFromStation", "SecondPumpArriving_DeployedFromStation","PerformanceReporting", "DeployedFromStation_Name", "PlusCode_Code", 
        "SpecialServiceType", "Postcode_district", "IncidentStationGround", "DelayCode_Description", "DeployedFromLocation"]'''
        st.code(code_var_cat, language = "python")
        st.write("")
        st.write("")

    st.subheader("B. Etapes détaillées Transformation des données", divider=True)
    tab4, tab5 = st.tabs(["4. Transformation des données", "5. Premières itérations du modèle"])

    with tab4:
        tab4.subheader("*4. Transformation des données*")
        st.write("*Tableau de transformation des données Incidents*")
        st.image("Tableau_Transfo données Incidents.png", caption="Tableau de transformation des données Incidents")
        st.write("*Tableau de transformation des données Mobilisations*")
        st.image("Tableau_Transfo données Mobilisations.png", caption="Tableau de transformation des données Mobilisations")

    with tab5:
        tab5.subheader("*5. Premières itérations du modèle*")

        st.write("Il s'agit d'un jeu de données avec modèle à apprentissage supervisé car les données sont labellisées et on cherche à prédire une variable cible identifiée.")
        st.write("La variable cible **temps d'arrivée sur les lieux de l'incident** étant une variable quantitative, nous sommes dans un problème de régression.")

        st.markdown("""
            Nous avons tout d'abord lancer des premières itérations de modèle par une approche classique.

             - Création de Pipelines par type de données
             - Recherche des hyper-paramètres des modèles
             - Tests de différents modèles avec différentes variables explicatives intégrées dans le dataframe :\n  
                 a) Régression de Lasso \n 
                 b) Régression linéaire\n 
                 c) Régression de Ridge\n 
                 d) ElasticNet\n 
                 e) Random Forest\n 
                 f) Gradient Boost Regressor \n 
                 g) Decision Tree Regressor\n 
                 h) SVR model\n 
            - Comparaison de leurs métriques : R², RMSE, MAE, MSE et MedAE, des résidus et des erreurs de prédictions
                  """)

        st.markdown("""
            Nous avons fait varier : 
               - le nombre de variables explicatives, 
               - la façon de gérer les NaN des données numériques par la médiane ou la moyenne.

            Voici les comparaisons des métriques des différents modèles testés avec nos premières hypothèses.
            """)
      
        st.write("**a.Avec les variables *TurnoutTimeSeconds*, *TravelTimeSeconds*, *NumCalls*, *PumpMinutesRounded***")
        st.image("5a_Avec variables.png")

        st.write("**b.SANS les variables hyper-corrélées à la variable cible : *TurnoutTimeSeconds*, *TravelTimeSeconds***")
        st.image("5b_Sansvariables.png")

    st.subheader("C. Conclusion de cette étape")
    st.write("La première approche n'a pas produit des résultats satisfaisants en raison de certaines variables trop corrélées avec la cible.")

# Page : Modélisation
if page == pages[4]:
    st.header("Modélisation")
    st.write("Nous avons cherché à améliorer les métriques, notamment **R² : entre 0,7 et 0,9** et **MAE : entre 30 et 60**.")
    
    st.subheader("Étapes détaillées des optimisations complémentaires et évaluation", divider=True)

    tab6, tab7, tab8, tab9 = st.tabs([
        "Feature Engineering", 
        "Revue du preprocessing", 
        "Encodage cyclique des variables temporelles",
        "Évaluation des métriques de nos modèles"
    ])

    with tab6:
        tab6.subheader("*Feature Engineering*")
        st.write("Heatmap des corrélations entre les colonnes numériques")
        st.image("Heatmap optimisation.png")
        st.write("Figure 31 - Graphique en barres des 10 features les plus importantes")
        st.image("Top10-features_ELASTIC-NET.png")

    with tab7:
        tab7.subheader("*Revue du preprocessing*")
        st.markdown("""
         Puis, nous avons supprimé les outliers des valeurs temporelles (variable cible *AttendanceTimeSeconds* et des variables composantes de celles-ci *TurnOutTimeSeconds* et *TravelTimeSeconds*),
         dans le dataframe initial juste avant la transformation en un dataframe de modélisation. 
         
         Cela a permis notamment d'éliminer des lignes incohérentes (ie. des cas où le temps de préparation était supérieur au temps d'intervention), qui ne représentaient pas un gros volume de lignes.

         Nous avons pu observer de nettes améliorations de performances du modèle après ce pré-processing. C'est pourquoi nous avons conservé cette transformation.
        """)

        st.write("Voici les métriques des différents modèles après gestion des outliers *AttendanceTimeSeconds*, *TurnOutTimeSeconds* et *TravelTimeSeconds*:") 
        st.image("Metriques après outliers.png", caption = "Métriques après gestion des outliers des variables temporelles")
        st.write("")
        st.write("")

    with tab8:
        tab8.subheader("*Test d'optimisation complémentaires*")
        var_temp_cycl = pd.DataFrame(
            {
                "nom": ["HourOfCall_sin", "HourOfCall_cos", "DayOfWeek_sin", "DayOfWeek_cos", "Month_sin", "Month_cos"],
                "desc": [
                    "Encodage cyclique - sinus et cosinus de l'heure, du jour de la semaine et du mois pour capturer la cyclicité temporelle."
                ] * 6  # Répète la même description pour chaque entrée
            }
        )
        st.dataframe(var_temp_cycl, column_config={"nom": "Nom des variables", "desc": "Description"}, hide_index=True)

    with tab9:
        tab9.subheader("*Évaluation des métriques de nos modèles*")
        data_metrics = pd.DataFrame({
            "nom_metric": ["Accuracy Score (R²)", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Median Absolute Error (MedAE)"],
            "descr_metric": [
                "Mesure de précision R² proche de 1 signifie une bonne corrélation.",
                "MAE : moyenne des erreurs absolues (unités de la cible).",
                "MSE : moyenne des erreurs au carré, pour vérifier les grandes erreurs.",
                "RMSE : racine carrée du MSE, en unités de la cible.",
                "MedAE : médiane des erreurs absolues, résistante aux outliers."
            ]
        })
        st.data_editor(data_metrics, column_config={"nom_metric": "Métrique", "descr_metric": "Définition des métriques"}, hide_index=True)

    st.subheader("Conclusion de cette étape")
    st.write("Les optimisations retenues sont : réduction des variables explicatives, gestion des outliers, encodage cyclique de certaines variables et une sélection aléatoire de 200 000 lignes.")


# Page Machine Learning
if page == pages[5]:
    st.header("Machine Learning")

    st.subheader("Présentation des modèles testés")
    st.write("Nous avons choisi de tester 3 modèles de Machine Learning qui nous semblaient les plus appropriés à notre problématique.")

    st.write("**a. Random Forest Regression**")
    st.markdown("""
        Ce modèle est populaire auprès des Data Scientists pour sa facilité d'interprétation et sa stabilité. 
        Il fonctionne avec des arbres de décision et peut être utilisé pour des tâches de régression ou de classification, couvrant ainsi une large gamme de problèmes.
    """)

    st.write("**b. Lasso**")
    st.markdown("""
        LASSO (Least Absolute Shrinkage and Selection Operator) est une méthode de régularisation qui effectue à la fois la sélection de variables 
        et la régularisation pour améliorer la précision de prédiction et l'interprétabilité du modèle.
    """)

    st.write("**c. Elastic Net**")
    st.write("""
        La régularisation Elastic Net combine les techniques Ridge et Lasso, permettant de conserver des variables fortement corrélées 
        et d'éviter une sélectivité excessive.
    """)

    st.subheader("Tests des modèles")

    info_model = ['Random Forest', 'Lasso', 'Elastic Net']
    model_choisi = st.selectbox("Sélection de l'algorithme de Machine Learning", options=info_model)

    st.write("Vous avez sélectionné:", model_choisi)

    if model_choisi == "Random Forest":
        st.subheader("Analyse du modèle choisi", divider=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Métriques de régression", "Valeurs réelles vs prédites", "Erreurs", "Résidus"])

        with tab1:
            tab1.subheader("Métriques de régression")
            st.write("**Meilleurs paramètres pour Random Forest :**")
            col6rf, col7rf = st.columns(2)
            col6rf.metric(label="min_samples_split", value=2)
            col7rf.metric(label="max_features", value="sqrt")

            st.write("**Métriques**")
            R2_rf = 0.841
            RMSE_rf = 47.91
            MSE_rf = 2295.27
            MAE_rf = 32.40
            MedAE_rf = 21.70

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("R²", round(R2_rf, 2))
            col2.metric("MAE", round(MAE_rf, 2))
            col3.metric("MSE", round(MSE_rf, 2))
            col4.metric("RMSE", round(RMSE_rf, 2))
            col5.metric("MedAE", round(MedAE_rf, 2))

        with tab2:
            tab2.subheader("Comparaison des valeurs réelles vs prédites")
            st.image("Comparaison-réelles-prédites_RF.png", caption="Comparaison des valeurs réelles vs prédites modèle Random Forest")

        with tab3:
            tab3.subheader("Distribution des erreurs (résidus)")
            st.image("Distribution-erreurs_RF.png", caption="Distribution des erreurs (résidus) modèle Random Forest")

        with tab4:
            tab4.subheader("Analyse des résidus")
            st.image("Analyse-résidus_RF.png", caption="Analyse des résidus modèle Random Forest")
            st.subheader("Courbe des résidus cumulés")
            st.image("Courbe-résidus-cumulés_RF.png", caption="Courbe des résidus cumulés modèle Random Forest")

    if model_choisi == "Lasso":
        st.subheader("Analyse du modèle choisi", divider=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Métriques de régression", "Valeurs réelles vs prédites", "Erreurs", "Résidus"])

        with tab1:
            tab1.subheader("Métriques de régression")
            alpha_lasso = 0.336
            st.metric("alpha", round(alpha_lasso, 2))

            st.write("**Métriques**")
            R2_lasso = 0.879
            RMSE_lasso = 40.623
            MSE_lasso = 1650.261
            MAE_lasso = 27.941
            MedAE_lasso = 15.877

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("R²", round(R2_lasso, 2))
            col2.metric("MAE", round(MAE_lasso, 2))
            col3.metric("MSE", round(MSE_lasso, 2))
            col4.metric("RMSE", round(RMSE_lasso, 2))
            col5.metric("MedAE", round(MedAE_lasso, 2))

        with tab2:
            tab2.subheader("Comparaison des valeurs réelles vs prédites")
            st.image("Comparaison-réelles-prédites_LASSO.png", caption="Comparaison des valeurs réelles vs prédites modèle Lasso")

        with tab3:
            tab3.subheader("Distribution des erreurs (résidus)")
            st.image("Distribution-erreurs_LASSO.png", caption="Distribution des erreurs (résidus) modèle Lasso")

        with tab4:
            tab4.subheader("Analyse des résidus")
            st.image("Analyse-résidus_LASSO.png", caption="Analyse des résidus modèle Lasso")
            st.subheader("Courbe des résidus cumulés")
            st.image("Courbe-résidus -cumulés_LASSO.png", caption="Courbe des résidus cumulés modèle Lasso")

    if model_choisi == "Elastic Net":
        st.subheader("Analyse du modèle choisi", divider=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Métriques de régression", "Valeurs réelles vs prédites", "Erreurs", "Résidus"])

        with tab1:
            tab1.subheader("Métriques de régression")
            col6, col7 = st.columns(2)
            col6.metric("l1_ratio", 0.9)
            col7.metric("alpha", round(0.234, 3))

            st.write("**Métriques**")
            R2_en = 0.878
            RMSE_en = 40.738
            MSE_en = 1659.550
            MAE_en = 26.103
            MedAE_en = 17.470

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("R²", round(R2_en, 2))
            col2.metric("MAE", round(MAE_en, 2))
            col3.metric("MSE", round(MSE_en, 2))
            col4.metric("RMSE", round(RMSE_en, 2))
            col5.metric("MedAE", round(MedAE_en, 2))

        with tab2:
            tab2.subheader("Comparaison des valeurs réelles vs prédites")
            st.image("Comparaison-réelles-prédites_ELASTICNET.png", caption="Comparaison des valeurs réelles vs prédites modèle Elastic Net")

        with tab3:
            tab3.subheader("Distribution des erreurs (résidus)")
            st.image("Distribution-erreurs_ELASTIC-NET.png", caption="Distribution des erreurs (résidus) modèle Elastic Net")

        with tab4:
            tab4.subheader("Analyse des résidus")
            st.image("Analyse-résidus_ELASTIC-NET.png", caption="Analyse des résidus modèle Elastic Net")
            st.subheader("Courbe des résidus cumulés")
            st.image("Courbe-résidus-cumulés_ELASTIC-NET.png", caption="Courbe des résidus cumulés modèle Elastic Net")

    st.subheader("Modèle final retenu")
    st.markdown("""
        Le modèle **Elastic Net** présente les meilleures performances :
        - Hyperparamètres : l1_ratio = 0.9, alpha = 0.233.
        - Évaluation :
            - RMSE: 40.3051
            - R²: 0.8872
            - MAE: 26.0744
            - MSE: 1624.5013
            - MedAE: 18.1165
    """)

# Conclusion
if page == pages[6]:
    st.header("Conclusion")
    
    st.subheader("Rappel des objectifs du Projet")
    st.write("**Objectif principal** : Prédire le temps de réponse aux interventions de la brigade des pompiers de Londres, en fonction de :")
    st.markdown("""
    - Nombre d'équipes déployées  
    - Type d’incidents  
    - Lieu de l’incident  
    """)

    st.subheader("Résultats et observations")
    st.markdown("""
    - **Précision des prédictions** : Nous sommes capables de prédire le temps d'intervention avec un intervalle de certitude de **26 secondes**.
    - **Analyse des facteurs impactant le temps d'intervention** :  
        Sélection et importance des variables explicatives montrent les principaux facteurs de rallongement des temps d'intervention :
        - **Zone géographique** :  
            - Les zones urbaines très fréquentées allongent les temps d'intervention en raison de la circulation.
            - En périphérie de Londres, les temps d'intervention ont tendance à augmenter.
        - **Concentration des incidents** :  
            - Forte concentration d'incidents dans l'hyper-centre, mais seulement trois casernes peuvent trianguler efficacement les interventions dans ce périmètre.
        - **Fausses alarmes** :  
            - Nombre élevé de fausses alarmes, surtout pour les incidents critiques, mobilisant inutilement des pompes qui pourraient être nécessaires ailleurs.
        - **Heure de mobilisation** :  
            - Le temps d'intervention augmente entre **10h et 18h**, période de forte circulation.
        - **Motifs de retard** :  
            - Trois motifs principaux : le trafic routier, les travaux urbains et les adresses incomplètes ou incorrectes, ce dernier étant un point d'action possible pour les équipes.
    """)

    st.subheader("Impact des métriques de modélisation")
    st.markdown("""
    - **Utilité pour la brigade des pompiers de Londres** :
        - Les métriques de notre modèle peuvent améliorer le processus de réponse aux appels en permettant à la plateforme d'appel de :
            - Prédire le temps d’arrivée avec précision.
            - Rassurer les citoyens en donnant une estimation du temps d’intervention.
            - Optimiser l'organisation des interventions entre les différentes équipes.
    """)
# Reporting pour la LFB
if page == pages[7]:
    st.header("Reporting pour la LFB")
    st.subheader("Page d'accueil du Reporting")
    st.image("Power BI Paul\Sommaire.png")
    st.subheader("Présentation des tables")
    st.image("Power BI Paul\Tables.png")
    st.subheader("Incidents report")
    st.image("Power BI Paul\Incident.png")
    st.subheader("Stations performance")
    st.image("Power BI Paul\Station performance.png")
    st.subheader("Time function")
    st.image("Power BI Paul\Timing.png")
    st.subheader("Geographic by Stations")
    st.image("Power BI Paul\Geographic.png")