"""Page 2 : Import & Gestion des données."""
import streamlit as st

from utils.css import render_hero
from utils.data_io import charger_fichier_transactions, diagnostiquer_dataframe, generer_donnees_synthetiques

COLONNES_ATTENDUES = [
    "ID Clients", "Type de transaction", "Status operation",
    "Localisation", "Date", "Montant", "Target",
]


def render():
    render_hero(
        "📂", "Source des données", "Import & gestion des données",
        "Chargez votre propre fichier, ou générez un jeu de données synthétique pour tester l'application.",
    )

    utilise_defaut = "df_actif" not in st.session_state or st.session_state["df_actif"] is None
    if utilise_defaut:
        st.info("ℹ️ L'application utilise actuellement le jeu de données par défaut "
                "(`data/Bank_transaction_scenario1.csv`).")
    else:
        st.success(f"✅ Jeu de données actif : **{st.session_state.get('df_actif_nom', 'personnalisé')}** "
                   f"({st.session_state['df_actif'].shape[0]} lignes).")

    st.caption(
        "⚠️ Le modèle prédictif déjà entraîné reste basé sur le schéma du jeu de données d'origine. "
        "Changer la source ici met à jour les pages *Dashboard* et *Exploration des données* ; "
        "pour ré-entraîner le modèle sur ces nouvelles données, relancez `python model/train_model.py` "
        "après avoir remplacé `data/Bank_transaction_scenario1.csv`."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Charger un fichier")
        fichier = st.file_uploader("CSV (séparateur `;`) ou Excel (.xlsx)", type=["csv", "xlsx", "xls"])
        if fichier is not None:
            try:
                df_upload = charger_fichier_transactions(fichier)
                colonnes_manquantes = [c for c in COLONNES_ATTENDUES if c not in df_upload.columns]
                if colonnes_manquantes:
                    st.error(f"Colonnes manquantes par rapport au schéma attendu : {colonnes_manquantes}")
                else:
                    if st.button("✅ Utiliser ce fichier", width="stretch"):
                        st.session_state["df_actif"] = df_upload
                        st.session_state["df_actif_nom"] = fichier.name
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur de lecture du fichier : {e}")

    with col2:
        st.subheader("🎲 Données synthétiques")
        st.caption("Génère un jeu de transactions fictives, avec le même schéma, pour tester l'application "
                   "sans fichier réel.")
        n_lignes = st.slider("Nombre de transactions à générer", 200, 5000, 2000, 100)
        if st.button("🎲 Générer un jeu de données synthétique", width="stretch"):
            st.session_state["df_actif"] = generer_donnees_synthetiques(n=n_lignes)
            st.session_state["df_actif_nom"] = f"Données synthétiques ({n_lignes} lignes)"
            st.rerun()

    if not utilise_defaut:
        if st.button("↩️ Revenir au jeu de données par défaut"):
            st.session_state["df_actif"] = None
            st.session_state["df_actif_nom"] = None
            st.rerun()

    st.markdown("---")
    st.subheader("🩺 Diagnostic du jeu de données actif")

    df_diag = st.session_state.get("df_actif")
    if df_diag is not None:
        diag = diagnostiquer_dataframe(df_diag)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Lignes", f"{diag['n_lignes']:,}".replace(",", " "))
        col2.metric("Colonnes", diag["n_colonnes"])
        col3.metric("Mémoire", f"{diag['memoire_mo']} Mo")
        col4.metric("Valeurs manquantes", diag["valeurs_manquantes"])
        col5.metric("Doublons", diag["doublons"])

        with st.expander("Types de colonnes"):
            st.json(diag["types"])

        with st.expander("Aperçu des données (20 premières lignes)"):
            st.dataframe(df_diag.head(20), width="stretch")
    else:
        st.caption("Le diagnostic complet du jeu de données par défaut est disponible sur la page "
                   "*📊 Exploration des données*.")
