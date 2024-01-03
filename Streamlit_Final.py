import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import calendar


st.set_page_config(layout="wide") 

#TITRE & SOMMAIRE

st.title("Projet Feux de Forêt")
st.sidebar.title("Sommaire")
pages=["Introduction","Analyse", "Datavisualisation", "Prédiction", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

#Création des DF - Partie Analyses
df_init_head = pd.read_csv("df_init_head.csv")
df_init_head = df_init_head.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
df_clean_head = pd.read_csv("df_clean_head.csv")
df_clean_head = df_clean_head.drop("Unnamed: 0", axis=1)
df_na_init = pd.read_csv("df_na_init.csv")
df_na_init = df_na_init.drop("Unnamed: 0", axis=1)
df_na_clean = pd.read_csv("df_na_clean.csv")
df_na_clean = df_na_clean.drop("Unnamed: 0", axis=1)
df_initial = pd.read_csv("df_init_red.csv")

# function to highlight rows based on average sale
def highlight_average_sale(s, pourcentage = 5):
    r = pd.Series(data = False,index = s.index)
    r['Colonnes'] = s.loc['Pourcentage de valeurs manquantes'] > pourcentage

    return ['background-color: orange' if r.any() else '' for v in r]



# apply the formatting
df_na_init = df_na_init.style.format({'Pourcentage de valeurs manquantes':'{:.0%}'}).apply(highlight_average_sale, pourcentage = 0.3, axis = 1).set_caption('Nombre de valeurs manquantes par colonne <br> dans le dataset initial')
df_na_clean = df_na_clean.style.format({'Pourcentage de valeurs manquantes':'{:.0%}'}).apply(highlight_average_sale, pourcentage = 0.3, axis = 1).set_caption('Nombre de valeurs manquantes par colonne <br> dans le dataset nettoyé')


#PAGE INTRODUCTION :
if page == pages[0]:

    # Texte en bas de la barre latérale avec des sauts de ligne
    texte_multiligne = " <br> <br> <br> <br> <br> <br> <br> MARS 2023- DA: <br> Nora CHELABI WALTZ<br>Pierre GARRIGUES<br>Thomas KIENY<br>Betül TUNCER"
    st.sidebar.markdown(texte_multiligne, unsafe_allow_html=True)
    st.title("Introduction")
    st.image("Fd.jpeg", caption="Feux de fôret", use_column_width=True)
    import streamlit as st

    encadre_texte = """
    <div style="border: 2px solid #000; padding: 10px; margin: 20px;">
        <p>Les feux de forêt aux États-Unis ont connu une évolution dramatique au cours des 
        dernières décennies, et cette transformation est en partie liée au dérèglement climatique 
        qui sévit à l'échelle mondiale. Au cours des 30 dernières années, ces incendies ont pris 
        une ampleur sans précédent, mettant en péril des écosystèmes fragiles, des communautés entières, 
        et suscitant de profondes préoccupations quant à l'avenir de nos forêts et de notre planète.<br> <br>
        Dans cette optique, nous avons choisi de participer à ce projet afin de proposer un moyen 
        de prévenir les incendies forestiers en fonction d'un certain nombre de facteurs à partir de 
        données relevées sur l'ensemble des États-Unis de 1992 à 2015. Cette étude nous amènera à nous 
        questionner sur l'ensemble des facteurs influant sur le déclenchement d'un feu de forêt, 
        leur impact respectif, mais aussi les limites au déploiement d'un tel système de prévention.<p>
    </div>
    """

    st.markdown(encadre_texte, unsafe_allow_html=True)

if page == pages[1]:
    st.title("Partie 1 : Analyse des données")

    st.header("**Création des datasets**")
    st.subheader("Le dataset original")
    st.dataframe(df_init_head)
    st.write(f"Ce [dataset](https://www.kaggle.com/rtatman/188-million-us-wildfires) présente les données de {df_initial.shape[0]} feux, déclarés sur le territoire américain entre 1992 et 2015.")
    st.dataframe(df_na_init)

    st.write("Le dataset original contient un grand nombre de valeurs manquantes. Nous le nettoyons en supprimant entre autres les (rares) doublons, afin d'en tirer un dataset propre et plus léger.")
     
    
    #DATASET NETTOYE
    st.subheader("Le dataset nettoyé")
    st.write("Le [dataset Kaggle original](https://www.kaggle.com/rtatman/188-million-us-wildfires) disposait d'une quarantaine de variables. Nous avons réduit ce chiffre à une dizaine de colonnes, en supprimant les variables redondantes.  \n Pour la suite du processus d'analyse, nous avons choisi de conserver la variable des comtés, malgré son nombre important de valeurs inconnues.")
    st.dataframe(df_clean_head)      
    st.dataframe(df_na_clean)

    #DATASET ENRICHI
    st.subheader("Le dataset enrichi")
    df_enrichi = pd.read_csv("df_trans.csv")
    df_enrichi = df_enrichi.drop("Unnamed: 0", axis=1)
    st.write("Nous avons ajouté à ces données originelles de nouvelles datas, tirées du [National Centers for Environmental Information américain](https://www.ncei.noaa.gov/). Il s'agit de données météorologiques, contenant l'évolution des températures moyennes et maximales ainsi que celle des précipitations par mois et par comté.")
    
    st.dataframe(df_enrichi.head())

    st.write("**Pour cela, nous n'avons conservé dans ce nouveau DataFrame que les feux dont était précisé le comté, soit 1.2 million environ.**")
    st.write("**Nous avons également dû supprimer les lignes concernant les feux hawaiens et portoricains, ces deux Etats n'étant pas pris en compte dans les données du NCEI.**")

    

    st.write("Nous gardons en tête que la plupart des lignes supprimées concernaient **des feux plus anciens**.")
    st.write("*Ces graphiques présentent le nombre de feux dont les comtés n'étaient pas renseignés, par année et par état :*")
    st.image("NAsEtatsAnnée.jpg")   


    #ANALYSES
    st.header("Premières analyses")
    st.write("Nous pouvons d'ores et déjà tirer quelques conclusions de ces données, avant de les modéliser.")
    st.write("***NOTE : Nous utilisons ici les données nettoyées, mais pas enrichies, et travaillons donc sur l'ensemble des feux.***")
    st.subheader("Analyse géographique")
    #NOMBRE DE FEUX
    
    st.write("Les Etats présentant le plus grand nombre de feux sont, dans l'ordre, la Californie, la Géorgie, le Texas, la Caroline du Nord et la Floride.")
    st.write("En revanche, la plupart des mégafeux se déclarent en Alaska : causés souvent par la foudre, ils y sont plus difficiles d'accès pour les pompiers. Ils s'étendent donc sans interruption.")
    
    #SLIDER
    year_range = st.slider("Sélectionnez une période : ", 1992, 2015, (1992, 2015))
    st.write("Période : ", year_range)
    
    df_initial_slider = df_initial.loc[(df_initial['FIRE_YEAR'] >= year_range[0]) & (df_initial['FIRE_YEAR'] <= year_range[1])]

    #MEGAFEUX
    if st.checkbox("Voir les données adaptées aux Mégafeux") :
        st.markdown("*Note : Les Mégafeux englobent tous les incendies de plus de 10 000 m².*")
        df_initial_MF = df_initial_slider.loc[df_initial_slider['FIRE_SIZE']>24710]
        df_init_MF_group = df_initial_MF.groupby(['STATE']).agg({'OBJECTID':'count','FIRE_SIZE':'sum'}).rename(columns={'FIRE_SIZE':'Etendue des feux (en km²)','OBJECTID':'Nombre de feux','STATE':'Etats'}).sort_values(by='Nombre de feux', ascending=False)
        df_init_MF_group['Etendue des feux (en km²)'] = round(df_init_MF_group['Etendue des feux (en km²)'].div(247,105),2)
        st.dataframe(df_init_MF_group)
    else:
        df_init_group = df_initial_slider.groupby(['STATE']).agg({'OBJECTID':'count','FIRE_SIZE':'sum'}).rename(columns={'FIRE_SIZE':'Etendue des feux (en km²)','OBJECTID':'Nombre de feux','STATE':'Etats'}).sort_values(by='Nombre de feux', ascending=False)
        df_init_group['Etendue des feux (en km²)'] = round(df_init_group['Etendue des feux (en km²)'].div(247,105),2)
        st.dataframe(df_init_group)
    
    #ANALYSE TEMPORELLE
    st.subheader("Analyse temporelle")

    df_c = pd.read_csv('df_c.csv')

    df_c_group = df_c.groupby(['Année'],as_index=False).agg({'Nombre de feux':'sum','Etendue totale des feux':'sum'})
    df_c_group['Etendue totale des feux'] = round(df_c_group['Etendue totale des feux'].div(247,105),2)
    
    st.dataframe(df_c_group)
    Moyenne_1 = df_c_group.loc[df_c_group['Année'] < 2004]['Nombre de feux'].mean()
    Moyenne_2 = df_c_group.loc[df_c_group['Année'] >= 2004]['Nombre de feux'].mean()
    Max_1 = df_c_group.loc[df_c_group['Année'] < 2004]['Nombre de feux'].max()
    Max_2 = df_c_group.loc[df_c_group['Année'] >= 2004]['Nombre de feux'].max()
    Moyenne_Area1 = df_c_group.loc[df_c_group['Année'] < 2004]['Etendue totale des feux'].mean()
    Moyenne_Area2 = df_c_group.loc[df_c_group['Année'] >= 2004]['Etendue totale des feux'].mean()
    st.write(f"Entre 1992 et 2003, {int(Moyenne_1)} feux de forêt se déclaraient en moyenne annuellement, avec un pic à {Max_1} .  \nEntre 2004 et 2015, cette moyenne annuelle passe à {int(Moyenne_2)} feux, avec un pic à {Max_2}.")
    st.write(f"Cette tendance se répercute sur l'étendue totale brûlée, avec {int(Moyenne_Area1)} km² incendiés en moyenne chaque année entre 1992 et 2003, contre {int(Moyenne_Area2)}km² entre 2004 et 2015.")
    
    st.write("La datavizualisation nous permet de creuser ces résultats.")

if page == pages[2]:
    st.title("Partie 2 : Datavisualisation")
    
    with st.container():
        #Dataframes utilisés pour les graphiques
        df_etat_viz= pd.read_csv("df_etat_viz.csv")
        df_pie_viz = pd.read_csv("df_pie_viz_en.csv")
        df = pd.read_csv("dfstreamlit.csv")
        dfmeteo = pd.read_csv("df_meteo.csv")
        df_map = pd.read_csv("df_map.csv")

        #Conversion Fahrenheit to Celsius
        def F_to_C(x):
            y = (x - 32) * 5/9
            return y
        dfmeteo["TAVG"] = dfmeteo["TAVG"].apply(F_to_C)
        dfmeteo["TMAX"] = dfmeteo["TMAX"].apply(F_to_C)
        df_etat_viz["TAVG"] = df_etat_viz["TAVG"].apply(F_to_C)

        #Conversion Précipitation en mm
        def inchtomm(x):
            y = x * 25.4
            return y

        dfmeteo["PCPN"] = dfmeteo["PCPN"].apply(inchtomm)
        df_etat_viz["PCPN"] = dfmeteo["PCPN"].apply(inchtomm)

        #Mapping Etat:
        Liste_ID = ['AL','AK','AZ','AR','CA','NC','SC','CO','CT','ND','SD','DE',
                    'FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
                    'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NY','NM',
                    'OH','OK','OR','PA','RI','TN','TX','UT','VT','VA','WV','WA','WI','WY']
        Liste_Etats = ['Alabama','Alaska','Arizona','Arkansas','California','North Carolina',
                    'South Carolina','Colorado','Connecticut','North Dakota','South Dakota',
                    'Delaware','Florida','Georgia','Hawaï','Idaho','Illinois','Indiana','Iowa',
                    'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan',
                    'Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada',
                    'New Hampshire','New Jersey','New York','New Mexico','Ohio','Oklahoma','Oregon',
                    'Pennsylvania','Rhode Island','Tennessee','Texas','Utah','Vermont','Virginia',
                    'West Virginia','Washington','Wisconsin','Wyoming']

        mapping_etat = pd.DataFrame(zip(Liste_ID, Liste_Etats),columns=["ID_value","Etat"])
        df["STATE_NAME"] = df["STATE"].map(mapping_etat.set_index("ID_value")["Etat"])
        dfmeteo["State"] = dfmeteo["state"].map(mapping_etat.set_index("ID_value")["Etat"])



        ## Introduction
        causes_feux_comment = """
        Les incendies de forêt sont souvent le résultat de plusieurs facteurs, tels que les conditions météorologiques, la géographie, la végétation, la densité de population, et d'autres.
        Il est essentiel de souligner que les variables dans notre jeu de données initial ne présentent pas de corrélations significatives les unes avec les autres.
        Nous pouvons observer certains phénomènes, mais il est important de noter que ces tendances ne sont pas uniformes à travers tous les États.
        """
    
        st.markdown(causes_feux_comment)
        ####1 - GRAPHIQUE GENERAL 
        st.subheader("Visualisations à partir des données initiales")



        ### Carte USA nombre /cause /étendu
        #DATA : 
        df_cause = pd.read_csv("df_cause.csv")

        #Boîte de sélection pour le choix de la carte à afficher (nombre de feux, causes, étendu)
        color_by = st.selectbox('Afficher la carte des USA selon:', ['Nombre de feux', 'Causes des feux','Etendue des feux'])
        #Choix des années
        start_year = st.number_input("Année de début", min_value=df_cause['FIRE_YEAR'].min(), max_value=df_cause['FIRE_YEAR'].max(), value=df_cause['FIRE_YEAR'].min())
        end_year = st.number_input("Année de fin", min_value=df_cause['FIRE_YEAR'].min(), max_value=df_cause['FIRE_YEAR'].max(), value=df_cause['FIRE_YEAR'].max())
        # Filtrer les données en fonction des années sélectionnées
        filtered_data_year = df_cause[(df_cause['FIRE_YEAR'] >= start_year) & (df_cause['FIRE_YEAR'] <= end_year)]
        seasons = filtered_data_year['SEASON'].unique()
        #Choix de la saison
        selected_seasons = st.selectbox('Sélectionnez une saison',['Toutes les saisons']+ list(seasons))
        # Filtrer les données en fonction des saisons sélectionnées
        if selected_seasons and not all(selected_seasons):
            filtered_data = filtered_data_year[filtered_data_year['SEASON'].isin(selected_seasons)]
        else:
            filtered_data = filtered_data_year  # Sélectionnez toutes les saisons par défaut
        # Grouper les données par État et identifier la cause principale pour les saisons sélectionnées
        total_fire = filtered_data.groupby("STATE").size().reset_index(name="Nombre de feux")
        main_cause_by_state = filtered_data.groupby("STATE")["CAUSE_CATEG"].value_counts().reset_index(name="Nbre")
        main_cause_by_state = main_cause_by_state.loc[main_cause_by_state.groupby("STATE")["Nbre"].idxmax()]
        # Calculer l'étendue totale des feux par État et cause
        total_fire_size_by_state = filtered_data.groupby(['STATE'])['FIRE_SIZE'].sum().reset_index()
        


        # Carte des États-Unis pour les 3 choix :
        if color_by == 'Nombre de feux':
            fig = px.choropleth(total_fire, 
                                locations="STATE",
                                locationmode="USA-states",
                                color="Nombre de feux",
                                title=f"Nombre de feux par État entre {start_year} et {end_year} ({(selected_seasons)})",
                                labels={"Nombre de feux": "Nombre de feux"},
                                scope="usa",
                                color_continuous_scale="Viridis")
        elif color_by == 'Etendue des feux':
            fig = px.choropleth(total_fire_size_by_state, 
                                locations="STATE",
                                locationmode="USA-states",
                                color="FIRE_SIZE",
                                title=f"Etendue total des feux par État entre {start_year} et {end_year} ({(selected_seasons)})",
                                labels={"FIRE_SIZE": "Etendue des feux"},
                                scope="usa",
                                color_continuous_scale="Inferno")
        else:
            fig = px.choropleth(main_cause_by_state, 
                                locations="STATE",
                                locationmode="USA-states",
                                color="CAUSE_CATEG",
                                title=f"Principale cause des feux par État entre {start_year} et {end_year} ({(selected_seasons)})",
                                labels={"CAUSE_CATEG": "Cause principale"},
                                scope="usa",
                                color_discrete_map={"Criminel": "red", "Naturel": "green", "Indefinie": "gray", "Humain-involontaire": "blue"})
        st.plotly_chart(fig)

        #### 2 - VISUALISATION PAR ETAT 

        st.header("Datavisualisation pour un Etat")
        
        liste_etats = ['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Idaho', 'Illinois',
        'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
        'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
        'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Alaska']
        liste_etats = np.sort(liste_etats)
        selected_state = st.selectbox("Sélectionner un État:", liste_etats)

        # Filtrer les données en fonction de l'Etat choisi
        filtered_df_pie = df_pie_viz[(df_pie_viz['Etat']==selected_state) & (df_pie_viz['FIRE_YEAR']>= start_year) & (df_pie_viz['FIRE_YEAR']<= end_year)]
        dfmeteo_state = dfmeteo[(dfmeteo['state']==selected_state)& (dfmeteo['Année']>= start_year) & (dfmeteo['Année']<= end_year)]
        df_state = df[(df['STATE_NAME']==selected_state)& (df['FIRE_YEAR']>= start_year) & (df['FIRE_YEAR']<= end_year)]

        # Groupement des données par année
        yearly_counts = df_state.groupby("FIRE_YEAR").size().reset_index(name="Nombre de feux")
        yearly_counts_meteo = dfmeteo_state.groupby("Année").size().reset_index(name="Nombre de feux")

        # Grouper les données par année et compter les causes des feux
        yearly_cause_counts = df_state.groupby(["FIRE_YEAR", "CAUSE_CATEG"]).size().reset_index(name="Nombre de feux")

        ### a -graphique en histogramme empilé
        couleurs = {"Criminel": "red","Naturel": "green","Indefinie": "gray", "Humain-involontaire": "blue"}
        fig5 = px.bar(yearly_cause_counts, x="FIRE_YEAR", y="Nombre de feux", color="CAUSE_CATEG",color_discrete_map=couleurs, title=f"Évolution des causes des feux en {selected_state} entre {start_year} et {end_year}")
        st.plotly_chart(fig5)
    
        cause_counts = df_state['CAUSE_CATEG'].value_counts()
        main_cause = cause_counts.idxmax()
        total_fires = len(df_state)  
        st.write(f"En {selected_state}, entre {start_year} et {end_year}, {total_fires} feux ont été relevés. La principale cause de ces feux est \"{main_cause}\".")


        ### b - pie chart catégorie de feux
        # Calculer les comptages des catégories de taille de feu
        size_counts = filtered_df_pie['FIRE_SIZE_CLASS'].value_counts().sort_index(ascending=True)
        nouvel_ordre = ['A', 'B', 'C', 'D', 'Autre']
        size_counts = size_counts.reindex(nouvel_ordre, fill_value=0)
        labels = ['A : <= 0.25 acres', 'B : 0.26 - 9.9 acres', 'C : 10 - 99.9 acres', 'D : 100 - 299 acres', 'Autre : >= 300 acres']

        # Créer un DataFrame pour le camembert - Ne marche pas pour BT mais Thomas oui 
        pie_data_size = pd.DataFrame({'size_class': size_counts.index, 'count': size_counts.values, 'labels': labels})
        # Créer un camembert avec Plotly Express
        fig = px.pie(pie_data_size, values='count', names='labels', title=f'Catégorie des feux de forêt en {selected_state} entre {start_year} et {end_year}', category_orders={"labels": ['A : <= 0.25 acres', 'B : 0.26 - 9.9 acres','C : 10 - 99.9 acres','D : 100 - 299 acres','Autre : >= 300 acres']})
        # Afficher le camembert avec Streamlit
        st.plotly_chart(fig)
        pie_categorie =(
            f"Nous constatons à l'aide du pie chart ci-dessus que les incendies de catégorie A et B sont les plus fréquents, soulignant la prédominance des feux de petite et moyenne envergure.")
        st.write(pie_categorie)

        #### 3- GRAPHIQUES AVEC DONNEES METEO:
        st.write('') 
        st.write('') 
        st.subheader("Visualisations à partir des données enrichies")

        #Curseur pour sélectionner les années début et fin
        year_range = st.slider(f'Sélectionnez une période pour l\'Etat: {selected_state}', min_value=1992, max_value=2015, value=(1992, 2015))
        # Filtrer le DataFrame météo en fonction de la plage d'années sélectionnée
        dfmeteo_state_year = dfmeteo[(dfmeteo['State'] == selected_state) & (dfmeteo['Année'] >= year_range[0]) & (dfmeteo['Année'] <= year_range[1])]
        filtered_df_state = df_etat_viz[(df_etat_viz['Etat'] == selected_state) & (df_etat_viz['Année'] >= year_range[0]) & (df_etat_viz['Année'] <= year_range[1])]

        #Warning NA dans données météo 
        st.write(" En raison de l'absence d'information sur les comtés dans le dataframe initial nous n'avons pas pu récupérer toutes les données météos. Ci-dessous vous trouverez le % de valeurs manquantes sur l'Etat sélectionné. Les coupures dans les graphiques sont liées à ces NA.")
        na_percentage = (df_state['COUNTY'].isna().mean()) * 100 
        st.warning(f"Pourcentage de valeurs manquantes pour {selected_state} : {na_percentage:.2f}%")

        if na_percentage >= 50:
            st.warning("Le pourcentage de valeurs manquantes pour votre sélection est supérieur à 50%. De fait, il ne nous paraît pas pertinent d'afficher des graphiques qui pourraient vous induire en erreur.")
        else:
            with st.container():
                ### a- Boites à moustaches 
                fig = plt.figure(figsize=(14, 5))
                # Description
                min_TAVG = dfmeteo_state_year['TAVG'].min()
                max_TAVG = dfmeteo_state_year['TAVG'].max()
                min_TMAX = dfmeteo_state_year['TMAX'].min()
                max_TMAX = dfmeteo_state_year['TMAX'].max()
                # Premier sous-graphique pour TAVG
                plt.subplot(141)
                sns.boxplot(data=dfmeteo_state_year, y='TAVG')
                plt.title("Température moyenne")  
                # Deuxième sous-graphique pour TMAX partageant l'axe des y avec le premier sous-graphique
                plt.subplot(142)
                sns.boxplot(data=dfmeteo_state_year, y='TMAX')
                plt.title("Température maximale")
                # Troisième sous-graphique pour PCPN
                plt.subplot(143)
                sns.boxplot(data=dfmeteo_state_year , y='PCPN')
                plt.title("Précipitation")
                plt.subplot(144)
                sns.boxplot(data=dfmeteo_state_year , y='Etendue totale des feux')
                plt.title("Etendue totale des feux")
                # Ajuster les espacements entre les sous-graphiques
                plt.tight_layout()
                st.pyplot(plt)

                description =(
                f"Les températures moyennes en {selected_state} varient entre {min_TAVG:.2f}°C et {max_TAVG:.2f}°C. "
                f"Les températures maximales enregistrées entre {year_range[0]} et {year_range[1]} varient entre {min_TMAX:.2f}°C et {max_TMAX:.2f}°C.")
                st.write(description)

                ### b - Créer un graphique du nombre de feux par année avec une courbe de l'évolution des températures moyennes
                fig = plt.figure(figsize=(14,6))
                fig = px.bar(filtered_df_state, x='Date', y='Nombre de feux', title=f"Températures moyennes mensuelles en {selected_state} entre {year_range[0]} et {year_range[1]}", color_discrete_sequence=['orange'])
                fig.add_scatter(x=filtered_df_state['Date'], y=filtered_df_state['TAVG'], mode='lines', name='Température moyenne (TAVG)', yaxis='y2', line=dict(color='red'))
                fig.update_layout(width = 1000, barmode='overlay', xaxis_title='Année', yaxis=dict(title='Nombre de feux', titlefont=dict(color='orange'), tickfont=dict(color='orange')), yaxis2=dict(title='TAVG (en °C)', overlaying='y', side='right', titlefont=dict(color='red'), tickfont=dict(color='red')))
                st.plotly_chart(fig)

                graph_b =(
                f"Ce graphique nous permet de constater que les pics de nombre de feux arrivent généralement à la fin du printemps et en été lors des fortes chaleurs.")
                st.write(graph_b)

                ### c - Créer un graphique du nombre de feux par année avec une courbe de l'évolution des précipitations cumulées
                fig = plt.figure(figsize=(14,6))
                fig = px.bar(filtered_df_state, x='Date', y='Nombre de feux', title=f"Précipitations cumulées en {selected_state} entre {year_range[0]} et {year_range[1]}", color_discrete_sequence=['orange'])
                fig.add_scatter(x=filtered_df_state['Date'], y=filtered_df_state['PCPN'], mode='lines', name='Précipitations cumulées (PCPN)', yaxis='y2', line=dict(color='blue'))
                fig.update_layout(width = 1000, barmode='overlay', xaxis_title='Année', yaxis=dict(title='Nombre de feux', titlefont=dict(color='orange'), tickfont=dict(color='orange')), yaxis2=dict(title='PCPN (en mm)', overlaying='y', side='right', titlefont=dict(color='blue'), tickfont=dict(color='blue')))
                st.plotly_chart(fig)

                graph_c =(
                f"Ce graphique nous permet de constater que lorsque les précipitations sont importantes au cours d'une année, le nombre de feux dans l'état est minimisé.")
                st.write(graph_c)


        #### 3 - COMPARAISON ENTRE DEUX ETATS :
        st.write('') 
        st.write('') 
        st.subheader("Comparaison entre deux états")

        # Boîtes de sélection pour choisir les éléments à comparer
        etats = [str(etat) for etat in df['STATE_NAME'].unique()]
        etats = sorted(etats)
        selected_element = st.selectbox("État 1", etats)
        selected_element_to_compare = st.selectbox("État 2", etats)
        # Filtrer le DataFrame en fonction de la plage d'années sélectionnée
        filtered_df = df[(df['STATE_NAME'] == selected_element) | (df['STATE_NAME'] == selected_element_to_compare)]
        filtered_dfmeteo=dfmeteo[(dfmeteo['State'] == selected_element) | (dfmeteo['State'] == selected_element_to_compare)]

        
        
        # Filtrer le DataFrame en fonction de l'élément sélectionné
        selected_df = filtered_df[filtered_df['STATE_NAME'] == selected_element]
        # Filtrer le DataFrame en fonction de l'élément à comparer
        compare_df = filtered_df[filtered_df['STATE_NAME'] == selected_element_to_compare]
        #Compter le nombre de feux par année pour chaque État
        selected_counts = selected_df['FIRE_YEAR'].value_counts().sort_index().sum()
        compare_counts = compare_df['FIRE_YEAR'].value_counts().sort_index().sum()

        ### a - Graphique cause pour chacun des états 
        # Calculer la répartition des causes des feux en pourcentage
        causes_counts_selected = (selected_df['CAUSE_CATEG'].value_counts() / selected_df.shape[0]) * 100
        causes_counts_compare = (compare_df['CAUSE_CATEG'].value_counts() / compare_df.shape[0]) * 100
        # Liste des catégories de causes des feux
        categories = causes_counts_selected.index
        bar_height = 0.25
        # Position des barres pour les deux États
        index = range(len(categories))
        # Créez un seul graphique pour afficher les deux États côte à côte
        plt.figure(figsize=(12, 6))
        plt.barh(index, causes_counts_selected.values, bar_height, label=selected_element, color='lightblue')
        plt.barh([i + bar_height for i in index], causes_counts_compare.values, bar_height, label=selected_element_to_compare, color='orange')
        # Étiquettes des catégories
        plt.title('Causes des Feux en %')
        plt.yticks([i + bar_height / 2 for i in index], categories)
        plt.legend()
        st.pyplot(plt)

        st.markdown(" ")
        st.markdown(" ")

        ### b- Camembert des saisons avec deux sous-graphiques
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        selected_seasons = selected_df['SEASON'].value_counts()
        compare_seasons = compare_df['SEASON'].value_counts()
        labels = selected_seasons.index
        sizes1 = selected_seasons.values
        sizes2 = compare_seasons.values
        axes[0].pie(sizes1, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title(f'{selected_element}')
        axes[1].pie(sizes2, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1].set_title(f'{selected_element_to_compare}')
        plt.suptitle('Distribution des feux par saison')
        st.pyplot(plt)

        ### c- Courbe évolution TAVG et PCPN par mois 
        
        # Filtrer le DataFrame
        selected_dfmeteo = filtered_dfmeteo[filtered_dfmeteo['State'] == selected_element]
        compare_dfmeteo = filtered_dfmeteo[filtered_dfmeteo['State'] == selected_element_to_compare]
        # calcul de la moyenne des TAVG par mois 
        tavg_selected = selected_dfmeteo.groupby('Mois')['TAVG'].mean()
        tavg_compare = compare_dfmeteo.groupby('Mois')['TAVG'].mean()
        # calcul de la moyenne des PCPN par mois
        pcpn_selected = selected_dfmeteo.groupby('Mois')['PCPN'].mean()
        pcpn_compare = compare_dfmeteo.groupby('Mois')['PCPN'].mean()
        # Obtenir les noms des mois correspondants
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        # figure avec deux sous-graphiques
        plt.figure(figsize=(16, 6))
        # Premier sous-graphique pour l'évolution des TAVG
        plt.subplot(1, 2, 1)
        plt.plot(month_names, tavg_selected, label=selected_element,linewidth=2)
        plt.plot(month_names, tavg_compare, label=selected_element_to_compare,linewidth=2)
        plt.xlabel('Mois')
        plt.ylabel('TAVG')
        plt.title('Évolution des TAVG par mois')
        plt.legend()
        plt.xticks(rotation=45)
        # Deuxième sous-graphique pour l'évolution des PCPN
        plt.subplot(1, 2, 2)
        plt.plot(month_names, pcpn_selected, label=selected_element,linewidth=2)
        plt.plot(month_names, pcpn_compare, label=selected_element_to_compare,linewidth=2)
        plt.xlabel('Mois')
        plt.ylabel('PCPN')
        plt.title('Évolution des PCPN par mois')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.xticks(rotation=45)
        st.pyplot(plt)


        comparaison_comment="""      
        
        Ces graphiques présentent les variations mensuelles des températures et des précipitations dans les États sélectionnés. 
        Ils offrent un moyen de comprendre, dans certains cas, la fréquence des feux selon les saisons d'un État à l'autre. 
        Il semble y avoir une tendance à moins d'incendies lorsque les températures sont plus basses et les précipitations plus élevées.
        """

        st.markdown(comparaison_comment) 

        ## Conclusion partie
   
        cl_comment = """
        Nous avons utiliser notre dataset d'origine ainsi que des données enrichies avec les données météorologiques.
        Malgré l'enrichissement de notre ensemble de données avec des informations météorologiques, nous n'observons toujours pas de corrélations significatives entre les variables. 
        
        Cependant, nous pouvons tirer quelques observations importantes :
    
        * Les feux d'origine humaine sont plus fréquents en été et au printemps. 
        * Les feux d'origine naturelle, tels que ceux provoqués par la foudre, semblent plus répandus dans la région de l'Ouest, où les saisons de sécheresse et d'orage peuvent être plus prononcées.
        * La région du Sud semble particulièrement touchée par les incendies d'origine humaine, notamment les incendies criminels et involontaires.
        * il semble y avoir beaucoup moins de feux lorsque les températures sont basses.
        * plus les précipitations sont importantes moins il y a de feux.
        """


        st.subheader("Conclusion Datavisualisation")
        st.markdown(cl_comment)



if page == pages[3]:
    st.title("Partie 3 : Prédiction du nombre de feux")

    intro_pred = """
    Notre modèle est basé sur un RandomForestRegressor, modèle qui s'est montré le plus performant \
    lors de nos différents essais au cours du projet. Le modèle permet donc de prédire le nombre de feux \
    en fonction de l'Etat et du comté choisi, de la période de l'année ainsi que \
    des données météorologiques sélectionnées. Le résultat est un intervalle de nombre de feux basé sur \
    la MAE du modèle."""

    st.markdown(intro_pred)
    st.write("**Note: un modèle est entraîné pour chaque Etat des Etats Unis.**")
             
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVC
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    
    df = pd.read_csv('df_trans.csv')
    #df.Etat = df.Etat.apply(lambda x: str(x).zfill(2))
    df.Comté = df.Comté.apply(lambda x: str(x).zfill(3))

    #Mapping Etat:
    Liste_ID = [1,2,3,4,5,6,7,8,9,
            10, 11,12,13,14,15,16,17,18,19, 20,
            21,22,23,24,25,26,27,28,29, 30,
            31,32,33,34,35,36,37,38,39, 40,
            41,42,43,44,45,46,47,48,50]
    Liste_ID2 = ['01','02','03','04','05','06','07','08','09',
            '10', '11','12','13','14','15','16','17','18','19', '20',
            '21','22','23','24','25','26','27','28','29', '30',
            '31','32','33','34','35','36','37','38','39', '40',
            '41','42','43','44','45','46','47','48','50']
    Liste_Etats = ['Alabama','Arizona','Arkansas','California','Colorado','Connecticut','Delaware',
                   'Florida','Georgia','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky',
                   'Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi',
                   'Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico',
                   'New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania',
                   'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah',
                   'Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming','Alaska']

    mapping_etat = pd.DataFrame(zip(Liste_ID,Liste_ID2, Liste_Etats),columns=["ID_value","ID_text","Etat"])

    #Paramétrage des prédictions souhaitées
    df_month = pd.DataFrame(['Janvier','Février','Mars','Avril',
                                                      'Mai','Juin','Juillet','Août','Septembre',
                                                      'Octobre','Novembre','Décembre'],columns=['MoisTxt'])
    
    Monthlist = [1,2,3,4,5,6,7,8,9,10,11,12]
    Mois_choisi = st.selectbox('Mois', df_month['MoisTxt'])
    Mois_choisi_ID = df_month[df_month['MoisTxt'] == Mois_choisi].index.values + 1
    Mois_choisi_ID = Mois_choisi_ID[0]

    #Récupération des paramétrages du user
    Statelist_name = mapping_etat['Etat'].unique()
    Statelist_IDvalue = mapping_etat['ID_value'].unique()
    Statelist_IDtext = mapping_etat['ID_text'].unique()
    Etat_choisi = st.selectbox('Etat',Statelist_name)
    
    #Définition de la variable Etat sous plusieurs format:
    for state in Statelist_name:
        if state == Etat_choisi:
            Etat_choisi_IDvalue = list(mapping_etat.loc[mapping_etat['Etat']==Etat_choisi,'ID_value'])[0]
            Etat_choisi_IDtext = list(mapping_etat.loc[mapping_etat['Etat']==Etat_choisi,'ID_text'])[0]

    #Définition de la variable comté
    Countylist = df['Comté'][df['Etat'] == Etat_choisi_IDvalue].unique()
    Comté_choisi = st.selectbox('Comté',Countylist)
    
    #Conversion Fahrenheit to Celsius
    def F_to_C(x):
        y = (x - 32) * 5/9
        return y
    def C_to_F(x):
        y = (x * 9/5) + 32
        return y

    def inchtomm(x):
        y = x * 25.4
        return y
    
    def mmtoinch(x):
        y = x / 25.4
        return y
    
    #Variable Températures et Précipitations
    TAVG = st.slider(label='Température moyenne mensuelle en degrés Celcius',step=1.,min_value=0.,
                           max_value=50.,value=25.,format="%.2f")
    TAVG = C_to_F(TAVG)
    TMAX = st.slider(label="Température maximum mensuelle en degrés Celcius",step=1.,min_value=0.,
                           max_value=50.,value=40.,format="%.2f")
    TMAX = C_to_F(TMAX)
    PCP = st.slider(label="Précipitations mensuelles en mm",step=10,min_value=0,max_value=1000,
                          value=100)   
    PCP = mmtoinch(PCP)

    #Calcul des moyennes pour référence
    st.write("A titre de référence, vous trouverez ci-dessous la moyenne des des températures moyennes "
            "(TAVG) et la médiane des températures maximales (TMAX) et des précipitations (PCPN) "
            "sur la période, l'Etat et le comté choisis.")
    groupby = df[(df['Etat'] == Etat_choisi_IDvalue) & 
                 (df['Comté'] == Comté_choisi) & 
                 (df['Mois'] == Mois_choisi_ID)].groupby(['Comté','Mois']). \
                    agg({'TAVG':'mean','TMAX':'median','PCPN':'median'})
    
    groupby['TAVG'] = groupby['TAVG'].apply(lambda x: np.round(F_to_C(x)),3)
    groupby['TMAX'] = groupby['TMAX'].apply(lambda x: np.round(F_to_C(x)),3)
    groupby['PCPN'] = groupby['PCPN'].apply(lambda x: np.round(inchtomm(x),3))
    st.dataframe(groupby)

    #Construction du df pour la prédiction:
    dico_state_county = {}
    for state in Statelist_IDtext:
        if state == Etat_choisi_IDtext:
            for comté in Countylist:
                if comté == Comté_choisi:
                    dico_state_county['State_County_'+f'{state}'+"-"+f'{comté}']=1
                else:
                    dico_state_county['State_County_'+f'{state}'+"-"+f'{comté}']=0
    
    dico_mois = {'Mois':Mois_choisi_ID}

    def Merge_dic(dict1, dict2, dict3,dict4):
        new_dico = {**dict1,**dict2,**dict3,**dict4}
        return new_dico
  
    inputs = {"TMAX":TMAX,
              "TAVG":TAVG,
              "PCPN":PCP}
    
    dico_année = {'Année':2023}

    X_test = pd.DataFrame(Merge_dic(dico_année,dico_mois,inputs,dico_state_county),index=[0])
    
    #st.dataframe(X_test)
    
    # load the saved model
    reg = joblib.load(f'{Etat_choisi_IDvalue}'+'model.joblib')

    # calculation of metrics
    df_resultats = pd.read_csv('df_resultats.csv',index_col=0)
    df_resultats = df_resultats[df_resultats['Etat'] == Etat_choisi_IDvalue]
    df_resultats = df_resultats.drop(columns=['Etat','Train_Score','Test_score'])
    df_resultats = df_resultats.apply(lambda x: np.round(x,2))
    MAE = np.round(df_resultats['Test_MAE'].values,0)

    # use the loaded model to make predictions
    predictions = reg.predict(X_test)
    predictions = np.round(predictions,0)

    st.write("D'après les paramètres sélectionnés, le modèle prédit entre " + str(predictions - MAE) + \
            "et " + str(predictions + MAE) + " feux sur la période choisie.")
    if df_resultats.iloc[0,5] < 0.4:
        st.warning("Attention le R2 de ce modèle est très bas et les résultats peuvent ne pas être fiables.")
    st.write("Voici les métriques associées au modèle sélectionné :")
    st.dataframe(df_resultats)


if page == pages[4]:
    st.write("Notre modèle pourrait être facilement amélioré par **l'accès à de nouvelles données**.")
    st.write("Sur ces deux graphiques, nous observons par exemple la relation entre **répartition des forces de pompiers, et nombre et étendue des feux**.")
    st.image("Pompiers_Feux.jpg")
    st.write("Le modèle que nous avons élaboré peut également revêtir **un rôle pédagogique** : il met ludiquement en lumière la relation claire entre\
             hausse des températures ou baisse des précipitations, et augmentation du nombre de feux.")