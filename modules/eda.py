"""Pages 4-5 : Analyse exploratoire & Visualisations Plotly."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.css import COULEURS_TARGET, colorer_statut, render_hero


def render(df_raw: pd.DataFrame):
    render_hero(
        "📊", "Analyse exploratoire", "Exploration des données",
        "Statistiques et graphiques interactifs, avec filtres globaux sur l'ensemble des transactions.",
    )

    with st.expander("Aperçu des données brutes", expanded=False):
        st.dataframe(df_raw.head(20), width="stretch")
        st.caption(f"{df_raw.shape[0]} lignes × {df_raw.shape[1]} colonnes — "
                   f"{df_raw.isnull().sum().sum()} valeurs manquantes, {df_raw.duplicated().sum()} doublons.")

    # ---------------------------------------------------------------- Filtres globaux
    st.subheader("🔎 Filtres")
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if df["Date"].notna().any():
            date_min, date_max = df["Date"].min().date(), df["Date"].max().date()
            plage = st.date_input("Plage de dates", (date_min, date_max), min_value=date_min, max_value=date_max)
        else:
            plage = None
    with col2:
        montant_min, montant_max = float(df["Montant"].min()), float(df["Montant"].max())
        plage_montant = st.slider("Montant (FCFA)", montant_min, montant_max, (montant_min, montant_max))
    with col3:
        types_sel = st.multiselect(
            "Type de transaction", sorted(df["Type de transaction"].unique()),
            default=sorted(df["Type de transaction"].unique()),
        )
    with col4:
        statuts_sel = st.multiselect(
            "Statut (Target)", sorted(df["Target"].unique()), default=sorted(df["Target"].unique()),
        )

    if plage and isinstance(plage, tuple) and len(plage) == 2:
        df = df[(df["Date"].dt.date >= plage[0]) & (df["Date"].dt.date <= plage[1])]
    df = df[
        (df["Montant"] >= plage_montant[0]) & (df["Montant"] <= plage_montant[1])
        & (df["Type de transaction"].isin(types_sel))
        & (df["Target"].isin(statuts_sel))
    ]
    st.caption(f"{df.shape[0]} transaction(s) après filtrage sur {df_raw.shape[0]} au total.")

    if df.empty:
        st.warning("Aucune transaction ne correspond à ces filtres.")
        return

    st.markdown("---")

    # ---------------------------------------------------------------- Répartition cible
    st.subheader("Répartition de la variable cible")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        counts = df["Target"].value_counts().reindex(["Normal", "Suspect", "Fraude"]).fillna(0)
        df_counts = pd.DataFrame({
            "Statut": counts.index,
            "Nombre": counts.values.astype(int),
            "Pourcentage": (counts.values / max(counts.values.sum(), 1) * 100).round(1),
        })
        st.dataframe(
            df_counts.style.map(colorer_statut, subset=["Statut"]),
            hide_index=True, width="stretch",
        )
    with col2:
        fig = px.bar(
            counts, x=counts.index, y=counts.values,
            color=counts.index, color_discrete_map=COULEURS_TARGET,
            labels={"x": "Statut", "y": "Nombre de transactions"},
            title="Répartition des transactions par statut", template="plotly_dark",
        )
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    st.info(
        "⚠️ La classe **Fraude** est fortement minoritaire (~3,7% des transactions dans le jeu de "
        "référence), ce qui est pris en compte lors de l'entraînement du modèle (`class_weight='balanced'`)."
    )

    # ---------------------------------------------------------------- Type / statut opération
    st.subheader("Type de transaction et statut d'opération")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="Type de transaction", color="Target", barnorm="percent",
            color_discrete_map=COULEURS_TARGET, template="plotly_dark",
            title="Répartition (%) du statut selon le type de transaction",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.histogram(
            df, x="Status operation", color="Target", barnorm="percent",
            color_discrete_map=COULEURS_TARGET, template="plotly_dark",
            title="Répartition (%) du statut selon le statut d'opération",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------- Montant
    st.subheader("Distribution du montant selon le statut")
    fig = px.box(
        df, x="Target", y="Montant", color="Target",
        category_orders={"Target": ["Normal", "Suspect", "Fraude"]},
        color_discrete_map=COULEURS_TARGET, points=False, template="plotly_dark",
        title="Distribution du montant par statut (échelle log)",
    )
    fig.update_yaxes(type="log")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------- Localisation
    st.subheader("Taux de fraude / suspicion par localisation")
    nb_top = st.slider("Nombre de localisations à afficher", 5, 20, 10)
    top_loc = df["Localisation"].value_counts().head(nb_top).index
    sous_df = df[df["Localisation"].isin(top_loc)]
    if not sous_df.empty:
        ct = pd.crosstab(sous_df["Localisation"], sous_df["Target"], normalize="index").reindex(top_loc) * 100
        colonnes_dispo = [c for c in ["Fraude", "Suspect"] if c in ct.columns]
        fig = px.bar(
            ct, x=ct.index, y=colonnes_dispo, barmode="group",
            color_discrete_map=COULEURS_TARGET, template="plotly_dark",
            labels={"value": "Pourcentage (%)", "Localisation": "", "variable": "Statut"},
            title=f"Taux (%) de Fraude / Suspect - top {nb_top} localisations",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------- Treemap
    st.subheader("Répartition croisée localisation × type de transaction")
    df_treemap = df.groupby(["Localisation", "Type de transaction", "Target"]).size().reset_index(name="Nombre")
    fig = px.treemap(
        df_treemap, path=["Localisation", "Type de transaction", "Target"], values="Nombre",
        color="Target", color_discrete_map=COULEURS_TARGET, template="plotly_dark",
        title="Volume de transactions par localisation, type et statut",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------------- Heatmap de corrélation
    st.subheader("Corrélation entre variables numériques")
    df_num = df.copy()
    df_num["heure"] = pd.to_datetime(df_num["Date"], errors="coerce").dt.hour
    df_num["jour_semaine"] = pd.to_datetime(df_num["Date"], errors="coerce").dt.dayofweek
    colonnes_num = [c for c in ["Montant", "heure", "jour_semaine"] if c in df_num.columns]
    if len(colonnes_num) >= 2:
        corr = df_num[colonnes_num].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template="plotly_dark", title="Matrice de corrélation",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Pas assez de variables numériques disponibles pour une matrice de corrélation.")

    st.caption(
        "ℹ️ Aucune colonne de coordonnées (`lat`/`lon`) n'est présente dans ce jeu de données : "
        "la carte géographique n'est donc pas disponible ici. Le treemap ci-dessus offre une vue "
        "équivalente de la répartition par localisation."
    )
