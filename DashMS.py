import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import math
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go
import shap
import lightgbm
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")


def main():
    SHAP_GENERAL = "feature-globale-important.png"

    st.set_page_config(page_title='Évaluation des demandes de prêt',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')

    # Display the title
    st.title('Évaluation des demandes de prêt ')

    @st.cache
    def load_data():
        PATH = 'dataP7/'

        df = pd.read_parquet(PATH + 'df_test.parquet')

        df_test = pd.read_parquet(PATH + 'application_test.parquet')

        df_train = pd.read_parquet(PATH + 'application_train.parquet')

        # description des features
        description = pd.read_csv(PATH + 'HomeCredit_columns_description.csv',
                                  usecols=['Row', 'Description'], \
                                  index_col=0, encoding='unicode_escape')

        return df, df_test, df_train, description

    @st.cache
    def load_model():
        return pickle.load(open('./LGBMClassifier.pkl', 'rb'))

    @st.cache
    def info_client_describe(data, id_client):
        info_client = data[data['SK_ID_CURR'] == int(id_client)]
        return info_client

    def barr(appliid, feature, valeur_client, title):
        if (not (math.isnan(valeur_client))):
            fig = plt.figure(figsize=(10, 4))

            t0 = appliid.loc[appliid['TARGET'] == 0]
            t1 = appliid.loc[appliid['TARGET'] == 1]

            if (feature == "DAYS_BIRTH"):
                sns.kdeplot((t0[feature] / -365).dropna(), label='Remboursé', color='g')
                sns.kdeplot((t1[feature] / -365).dropna(), label='non rembourse', color='r')
                plt.axvline(float(valeur_client / -365), \
                            color="blue", linestyle='--', label='Position Client')

            elif (feature == "DAYS_EMPLOYED"):
                sns.kdeplot((t0[feature] / 365).dropna(), label='Remboursé', color='g')
                sns.kdeplot((t1[feature] / 365).dropna(), label='non rembourse', color='r')
                plt.axvline(float(valeur_client / 365), color="blue", \
                            linestyle='--', label='Position Client')
            else:
                sns.kdeplot(t0[feature].dropna(), label='Remboursé', color='g')
                sns.kdeplot(t1[feature].dropna(), label='non rembourse', color='r')
                plt.axvline(float(valeur_client), color="blue", \
                            linestyle='--', label='Position Client')

            plt.title(title, fontsize='20', fontweight='bold')

            # plt.ylabel('Nombre de clients')
            # plt.xlabel(fontsize='14')
            plt.legend()
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car cest  (NaN)")

    # @st.cache
    def catego(appliid, feature, valeur_client, \
               titre, ylog=False, label_rotation=False,
               horizontal_layout=True):
        if (valeur_client.iloc[0] != np.nan):
            temp = appliid[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index, 'nombre de contras': temp.values})

            categories = appliid[feature].unique()
            categories = list(categories)

            percg_cat = appliid[[feature, \
                                 'TARGET']].groupby([feature], as_index=False).mean()
            percg_cat["TARGET"] = percg_cat["TARGET"] * 100
            percg_cat.sort_values(by='TARGET', ascending=False, inplace=True)

            if (horizontal_layout):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 24))

            s = sns.countplot(ax=ax1,
                              x=feature,
                              data=appliid,
                              hue="TARGET",
                              order=percg_cat[feature],
                              palette=['g', 'r'])
            pos1 = percg_cat[feature].tolist().index(valeur_client.iloc[0])
            ax1.set(ylabel="Nombre de clients")
            ax1.set_title(titre, fontdict={'fontsize': 15, 'fontweight': 'bold'})
            ax1.axvline(int(pos1), color="blue", linestyle='--', label='Position Client')
            ax1.legend(['Position Client', 'Remboursé', 'non rembourse'])

            if ylog:
                ax1.set_yscale('log')
                ax1.set_ylabel("Count (log)", fontdict={'fontsize': 15, \
                                                        'fontweight': 'bold'})
            if (label_rotation):
                s.set_xticklabels(s.get_xticklabels(), rotation=90)

            s = sns.barplot(ax=ax2,
                            x=feature,
                            y='TARGET',
                            order=percg_cat[feature],
                            data=percg_cat,
                            palette='Set2')
            pos2 = percg_cat[feature].tolist().index(valeur_client.iloc[0])
            # st.write(pos2)
            if (label_rotation):
                s.set_xticklabels(s.get_xticklabels(), rotation=90)
            plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_title(titre + " (% Défaillants)", \
                          fontdict={'fontsize': 15, 'fontweight': 'bold'})
            ax2.axvline(int(pos2), color="blue", linestyle='--', label='Position Client')
            ax2.legend()
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car il y a (NaN)")
            # Chargement des données

    df, df_test, df_train, description = load_data()

    features_num = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    features_col = [col for col in df.columns if col not in features_num]

    # Chargement du modèle
    model = load_model()
    with st.sidebar:

        st.write("## ID Client")
        id_list = df["SK_ID_CURR"].tolist()
        id_client = st.selectbox("Sélectionner l'identifiant du client", id_list)
        st.write("## Actions à effectuer")
        show_credit_decision = st.checkbox("Afficher la décision de crédit")
        show_client_details = st.checkbox("Afficher les informations du client")
        show_client_comparison = st.checkbox("Comparer aux autres clients")
        show_bivariés= st.checkbox("afficher un garphe bivariés")
        shap_general = st.checkbox("Afficher la feature importance globale")

        if (st.checkbox("Aide description des features")):
            list_features = description.index.to_list()
            list_features = list(dict.fromkeys(list_features))
            feature = st.selectbox('Sélectionner une variable', \
                                   sorted(list_features))

            desc = description['Description'].loc[description.index == feature][:1]
            st.markdown('**{}**'.format(desc.iloc[0]))

    # Afficher l'ID Client sélectionné

    st.write("ID Client Sélectionné :", id_client)

    if (int(id_client) in id_list):
        info_client = info_client_describe(df_test, id_client)

        # -------------------------------------------------------
        # Afficher la décision de crédit
        # -------------------------------------------------------

        if (show_credit_decision):

            # Appel de l'API :
            API_url = "https://modele-scoring-api.herokuapp.com/app/" + str(id_client)

            with st.spinner('Chargement du score du client...'):

                json_url = urlopen(API_url)

                API_data = json.loads(json_url.read())
                classe_predite = API_data['prediction']
                if classe_predite == 1:
                    decision = ' (Crédit Refusé)'
                else:
                    decision = '(Crédit Accordé)'
                probabilite = 1 - API_data['probabilite']

                score_client = round(probabilite * 100, 2)
                left_column, right_column = st.columns((1, 2))

                left_column.markdown('Risque de défaut: **{}%**'.format(str(score_client)))
                left_column.markdown('Seuil par défaut du modèle: **50%**')

                if classe_predite == 1:
                    left_column.markdown(
                        'Décision: <span style="color:red">**{}**</span>'.format(decision), \
                        unsafe_allow_html=True)
                else:
                    left_column.markdown(
                        'Décision: <span style="color:green">**{}**</span>' \
                            .format(decision), \
                        unsafe_allow_html=True)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+delta+number",
                    title={'text': 'Pourcentage de risque de défaut'},
                    value=score_client,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'steps': [
                               {'range': [0, 25], 'color': "lightgreen"},
                               {'range': [25, 50], 'color': "lightyellow"},
                               {'range': [50, 75], 'color': "orange"},
                               {'range': [75, 100], 'color': "red"},
                           ],
                           'threshold': {
                               'line': {'color': "black", 'width': 10},
                               'thickness': 0.8,
                               'value': score_client},

                           'bar': {'color': "black", 'thickness': 0.2},
                           },
                ))

                gauge.update_layout(width=450, height=250,
                                    margin=dict(l=50, r=50, b=0, t=0, pad=4))

                right_column.plotly_chart(gauge)
            show_feature_importance_local = st.checkbox(
                "Afficher les variables ayant le plus contribué à la décision du modèle ?")
            if (show_feature_importance_local):
                shap.initjs()

                number = st.slider('Sélectionner le nombre de feautures à afficher ?', \
                                   2, 20, 8)

                X = df[df['SK_ID_CURR'] == int(id_client)]
                X = X[features_col]

                fig, ax = plt.subplots(figsize=(15, 15))
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                shap.summary_plot(shap_values[0], X, plot_type="bar", \
                                  max_display=number, color_bar=False, plot_size=(8, 8))

                st.pyplot(fig)
        # -------------------------------------------------------
        # Afficher les informations du client
        # -------------------------------------------------------

        information_personal_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT",
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",
        }

        default_list = \
            ["GENRE", "AGE", "STATUT FAMILIAL", "NB ENFANTS", "REVENUS", "MONTANT CREDIT"]
        numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
                              'AMT_ANNUITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

        rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
        horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

        if (show_client_details):
            st.header('Informations relatives au client')

            with st.spinner('Chargement des informations relatives au client...'):
                information_personal_df = info_client[list(information_personal_cols.keys())]
                # personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
                information_personal_df.rename(columns=information_personal_cols, inplace=True)

                information_personal_df["AGE"] = int(round(information_personal_df["AGE"] / 365 * (-1)))
                information_personal_df["NB ANNEES EMPLOI"] = \
                    int(round(information_personal_df["NB ANNEES EMPLOI"] / 365 * (-1)))

                filtered = st.multiselect("Choisir les informations à afficher", \
                                          options=list(information_personal_df.columns), \
                                          default=list(default_list))
                df_info = information_personal_df[filtered]
                df_info['SK_ID_CURR'] = info_client['SK_ID_CURR']
                df_info = df_info.set_index('SK_ID_CURR')
                st.table(df_info.astype(str).T)
                show_all_info = st \
                    .checkbox("Afficher toutes les informations (dataframe brute)")
                if (show_all_info):
                    st.dataframe(info_client)
        # -------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        # -------------------------------------------------------
        if (show_client_comparison):
            st.header('Comparaison aux autres clients')

            with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
                var = st.selectbox("Sélectionner une variable", \
                                   list(information_personal_cols.values()))
                feature = list(information_personal_cols.keys()) \
                    [list(information_personal_cols.values()).index(var)]
                if (feature in numerical_features):
                    barr(df_train, feature, info_client[feature], var)
                elif (feature in rotate_label):
                    catego(df_train, feature, \
                           info_client[feature], var, False, True)
                elif (feature in horizontal_layout):
                    catego(df_train, feature, \
                           info_client[feature], var, False, True, True)
                else:
                    catego(df_train, feature, info_client[feature], var)
        if (show_bivariés):
            st.header("Sélectinner les deux variables")

            var1 = st.selectbox("Sélectionner la première variable", options=df_train.columns)

            var2 = st.selectbox("Sélectionner la deuxième variable", options=df_train.columns)

            fig3 = px.scatter(df_train, x=var1, y=var2)
            st.plotly_chart(fig3)
            # -------------------------------------------------------
            # Afficher la feature importance globale

        if (shap_general):
            st.header('Feature importance globale')
            st.image('feature-globale-important.png')
    else:
        st.markdown("**Identifiant non reconnu**")


if __name__ == '__main__':
    main()