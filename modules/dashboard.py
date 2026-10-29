"""Page 1 : Dashboard — KPIs et synthèse générale."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.css import COULEURS_TARGET, render_hero


def render(df_raw: pd.DataFrame, artefacts: dict | None, model):
    render_hero(
        "🏦", "Vue d'ensemble", "Détection de la fraude bancaire",
        "Explorez les transactions, évaluez le modèle prédictif et testez une classification en temps réel.",
    )
    st.markdown(
        """
        Cette application permet d'explorer un jeu de transactions bancaires, d'évaluer un modèle
        prédictif de détection de fraude, et de tester la classification d'une nouvelle transaction
        en temps réel.

        La variable cible **`Target`** comporte 3 classes :
        - 🟢 **Normal** : transaction normale
        - 🟠 **Suspect** : transaction suspecte, à surveiller
        - 🔴 **Fraude** : transaction frauduleuse confirmée
        """
    )

    montant_total = df_raw["Montant"].sum()
    montant_moyen = df_raw["Montant"].mean()
    clients_uniques = df_raw["ID Clients"].nunique() if "ID Clients" in df_raw.columns else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{df_raw.shape[0]:,}".replace(",", " "))
    col2.metric("Fraudes recensées", int((df_raw["Target"] == "Fraude").sum()))
    col3.metric("Taux de fraude", f"{(df_raw['Target'] == 'Fraude').mean() * 100:.1f} %")
    col4.metric("Modèle retenu", artefacts.get("model_nom", type(model).__name__) if artefacts else "non entraîné")

    col1, col2, col3 = st.columns(3)
    col1.metric("Montant total", f"{montant_total:,.0f} FCFA".replace(",", " "))
    col2.metric("Montant moyen / transaction", f"{montant_moyen:,.0f} FCFA".replace(",", " "))
    if clients_uniques is not None:
        col3.metric("Clients uniques", f"{clients_uniques:,}".replace(",", " "))

    if artefacts:
        st.caption(
            f"Modèle sélectionné automatiquement parmi {len(artefacts.get('comparaison_modeles', []))} "
            f"modèles comparés (rappel Fraude = "
            f"{artefacts['metriques_test']['rapport_classification']['Fraude']['recall'] * 100:.1f}%), "
            f"puis optimisé par GridSearchCV. Voir la page *🤖 Entraînement ML* pour le détail."
        )

    st.markdown("---")

    # ---- Tendance temporelle + répartition ----
    if "Date" in df_raw.columns:
        df_tendance = df_raw.copy()
        df_tendance["Date"] = pd.to_datetime(df_tendance["Date"], errors="coerce")
        df_tendance["Jour"] = df_tendance["Date"].dt.date
        tendance = df_tendance.groupby(["Jour", "Target"]).size().reset_index(name="Nombre")

        col1, col2 = st.columns([1.6, 1])
        with col1:
            st.subheader("Tendance temporelle")
            fig = px.line(
                tendance, x="Jour", y="Nombre", color="Target",
                color_discrete_map=COULEURS_TARGET,
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.subheader("Répartition globale")
            counts = df_raw["Target"].value_counts().reindex(["Normal", "Suspect", "Fraude"])
            fig = px.pie(
                values=counts.values, names=counts.index,
                color=counts.index, color_discrete_map=COULEURS_TARGET,
                hole=0.55, template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Pages disponibles")
    st.markdown(
        """
        - **📂 Import & données** : chargez un CSV/Excel, générez des données de test, diagnostic qualité.
        - **📊 Exploration des données** : statistiques et graphiques interactifs, filtres globaux.
        - **🤖 Entraînement ML** : comparaison des modèles supervisés + détection non supervisée (Isolation Forest).
        - **📉 Évaluation** : matrice de confusion, courbe ROC, importance des variables, seuil de décision.
        - **🔮 Prédiction en temps réel** : saisissez une transaction et obtenez sa classification instantanée.
        - **🗂️ Historique & rapports** : suivi des prédictions, export CSV / PDF / Excel.
        """
    )
