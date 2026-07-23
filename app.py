"""
Application Streamlit - Détection de la fraude bancaire.

Point d'entrée : configuration de la page, chargement des données/du modèle (mis en cache),
navigation latérale, puis délégation à chaque module de `modules/` pour le rendu de la page active.

Lancement :
    streamlit run app.py
"""
import os
import sys

import joblib
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

from preprocessing import charger_donnees, encoder_features, nettoyer_et_enrichir, preparer_transaction
from utils.css import injecter_theme, render_sidebar_brand
from utils.history import charger_historique_persistant

from modules import dashboard, data_loader, eda, evaluation, ml_training, prediction, reporting

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "Bank_transaction_scenario1.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "predictions_history.csv")

st.set_page_config(page_title="Détection de fraude bancaire", page_icon="🏦", layout="wide")
injecter_theme()


# ----------------------------------------------------------------------------
# Chargement des données et du modèle (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_data
def _lire_fichier_defaut():
    if not os.path.exists(CSV_PATH):
        return None
    return charger_donnees(CSV_PATH)


def get_raw_data():
    """Retourne le jeu de données actif : celui chargé/généré depuis la page
    *Import & gestion des données* si présent, sinon le fichier par défaut."""
    df_actif = st.session_state.get("df_actif")
    if df_actif is not None:
        return df_actif
    return _lire_fichier_defaut()


@st.cache_resource
def get_model_artifacts():
    """Charge le modèle depuis le disque. S'il est absent (ex : premier déploiement sur
    Streamlit Community Cloud sans model/fraud_model.pkl commité), l'entraîne automatiquement
    à la volée à partir des données par défaut."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    if not os.path.exists(CSV_PATH):
        return None

    with st.spinner(
        "Aucun modèle entraîné trouvé — entraînement automatique en cours "
        "(quelques dizaines de secondes, une seule fois)..."
    ):
        import train_model  # noqa: E402
        train_model.main()

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def get_test_split(_top_localisations, _feature_columns):
    """Reconstruit le même split train/test que celui utilisé à l'entraînement (même
    random_state), à partir du fichier de données PAR DÉFAUT (le modèle entraîné y est lié)."""
    df = _lire_fichier_defaut()
    df_clean, _ = nettoyer_et_enrichir(df, top_localisations=_top_localisations)
    X, _ = encoder_features(df_clean, feature_columns=_feature_columns)

    artefacts = get_model_artifacts()
    y = artefacts["label_encoder"].transform(df_clean["Target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


# ----------------------------------------------------------------------------
# Barre latérale / navigation
# ----------------------------------------------------------------------------
render_sidebar_brand()
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📂 Import & données",
        "📊 Exploration des données",
        "🤖 Entraînement ML",
        "📉 Évaluation & seuil",
        "🔮 Prédiction en temps réel",
        "🗂️ Historique & rapports",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Application développée à partir du pipeline validé dans le notebook "
    "`detection_fraude_bancaire.ipynb`."
)

df_raw = get_raw_data()
artefacts = get_model_artifacts()

if "historique_predictions" not in st.session_state:
    st.session_state["historique_predictions"] = charger_historique_persistant(HISTORY_PATH).to_dict("records")

if df_raw is None:
    st.error("Fichier de données introuvable : `data/Bank_transaction_scenario1.csv`. "
             "Placez-le dans le dossier `data/`, ou générez un jeu de données synthétique "
             "depuis la page *📂 Import & données*.")
    st.stop()

if artefacts is None:
    st.warning(
        "Aucun modèle entraîné trouvé (`model/fraud_model.pkl`). "
        "Lancez d'abord le script d'entraînement :\n\n"
        "```bash\npython model/train_model.py\n```\n\n"
        "Les pages *Dashboard* et *Exploration des données* restent disponibles en attendant."
    )

model = artefacts["model"] if artefacts else None
scaler = artefacts["scaler"] if artefacts else None
le_target = artefacts["label_encoder"] if artefacts else None
feature_columns = artefacts["feature_columns"] if artefacts else None
top_localisations = artefacts["top_localisations"] if artefacts else None
best_threshold_f1 = artefacts["best_threshold_f1"] if artefacts else 0.5


# ----------------------------------------------------------------------------
# Dispatch vers le module de la page active
# ----------------------------------------------------------------------------
if page == "🏠 Dashboard":
    dashboard.render(df_raw, artefacts, model)

elif page == "📂 Import & données":
    data_loader.render()

elif page == "📊 Exploration des données":
    eda.render(df_raw)

elif page == "🤖 Entraînement ML":
    ml_training.render(artefacts, model)

elif page == "📉 Évaluation & seuil":
    evaluation.render(
        artefacts, model, feature_columns, top_localisations, scaler, le_target,
        best_threshold_f1, get_test_split,
    )

elif page == "🔮 Prédiction en temps réel":
    prediction.render(
        df_raw, artefacts, model, scaler, le_target, feature_columns, top_localisations,
        best_threshold_f1, preparer_transaction, HISTORY_PATH,
    )

elif page == "🗂️ Historique & rapports":
    reporting.render(HISTORY_PATH)
