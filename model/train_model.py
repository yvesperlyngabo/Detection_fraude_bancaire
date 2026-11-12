"""
Script d'entraînement du modèle de détection de fraude bancaire (version 2).

Ce qui change par rapport à la version 1 (voir l'évaluation par validation croisée) :
    - Variables CAUSALES uniquement : nb_transactions_client est exclu du modèle car il
      compte aussi les transactions FUTURES du client (information indisponible en
      production). Il est remplacé par nb_trans_passees ; les variables 7 jours sont conservées.
    - Sélection du modèle et réglage des hyperparamètres par validation croisée PAR CLIENT
      (les clients du pli de validation ne sont jamais vus à l'entraînement), avec comme
      critère l'AUC-PR de la classe « Fraude » (et non le rappel seul, qui favorise
      les modèles qui déclenchent trop d'alertes).
    - Seuil de décision calibré sur des prédictions hors échantillon, jamais sur le jeu de test.
    - Jeu de test = environ 20 % des clients, mis de côté avant toute sélection.

Pipeline :
    1. chargement et variables  2. jeu de test par clients  3. comparaison de 6 modèles
    4. Isolation Forest (non supervisé, à titre complémentaire)  5. réglage du meilleur
    6. seuil de décision  7. évaluation finale sur le jeu de test  8. sauvegarde

Usage (depuis la racine du projet) :
    python model/train_model.py

Génère `model/fraud_model.pkl`, utilisé ensuite par `app.py`.
"""
import os
import sys
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import CLASSES_CIBLE, charger_donnees, encoder_features, nettoyer_et_enrichir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "Bank_transaction_scenario1.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

GRAINE = 42
N_PLIS = 5

# Compte calculé sur tout le jeu de données (passé ET futur) : inutilisable en production.
VARIABLES_EXCLUES = ["nb_transactions_client"]

# ------------------------------------------------------------------------
# Modèles optionnels (installés via requirements.txt). Le script continue
# de fonctionner même s'ils sont absents, avec un avertissement.
# ------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False


# ------------------------------------------------------------------------
# Fabriques d'estimateurs "vierges" (réinstanciés à chaque pli et à chaque combinaison
# d'hyperparamètres)
# ------------------------------------------------------------------------
def get_estimateurs():
    estimateurs = {
        "Régression Logistique": lambda: LogisticRegression(max_iter=5000, random_state=GRAINE),
        "Arbre de décision": lambda: DecisionTreeClassifier(random_state=GRAINE),
        "Random Forest": lambda: RandomForestClassifier(random_state=GRAINE, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=GRAINE),
    }
    if XGBOOST_DISPONIBLE:
        estimateurs["XGBoost"] = lambda: XGBClassifier(eval_metric="mlogloss", random_state=GRAINE)
    if LIGHTGBM_DISPONIBLE:
        estimateurs["LightGBM"] = lambda: LGBMClassifier(random_state=GRAINE, verbosity=-1)
    return estimateurs


# Grilles d'hyperparamètres (volontairement compactes : chaque combinaison est évaluée
# sur 5 plis par client)
PARAM_GRIDS = {
    "Régression Logistique": {"C": [0.01, 0.1, 1, 10]},
    "Arbre de décision": {
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [200, 400],
        "max_depth": [8, None],
        "min_samples_leaf": [1, 3],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    },
    "LightGBM": {
        "n_estimators": [200, 400],
        "num_leaves": [15, 31],
        "learning_rate": [0.05, 0.1],
    },
}


# ------------------------------------------------------------------------
# Outils d'évaluation
# ------------------------------------------------------------------------
def plis_par_client(X, y, groupes, n_plis=N_PLIS, graine=GRAINE):
    """Plis stratifiés sur la cible, sans qu'aucun client soit à la fois dans l'entraînement et la validation."""
    sgkf = StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine)
    return list(sgkf.split(X, y, groupes))


def predictions_hors_echantillon(fabrique, X, y, groupes, label_fraude, params=None):
    """
    Probabilités prédites pour chaque ligne par un modèle qui n'a JAMAIS vu le client de cette
    ligne (validation croisée par client). Renvoie (probabilités, AUC-PR par pli).
    """
    proba = np.zeros((len(y), len(np.unique(y))))
    ap_plis = []
    for idx_tr, idx_va in plis_par_client(X, y, groupes):
        scaler = StandardScaler().fit(X[idx_tr])
        modele = fabrique()
        if params:
            modele.set_params(**params)
        modele.fit(scaler.transform(X[idx_tr]), y[idx_tr],
                   sample_weight=compute_sample_weight("balanced", y[idx_tr]))
        proba[idx_va] = modele.predict_proba(scaler.transform(X[idx_va]))
        y_bin = (y[idx_va] == label_fraude).astype(int)
        if y_bin.sum() > 0:
            ap_plis.append(average_precision_score(y_bin, proba[idx_va, label_fraude]))
    return proba, ap_plis


def choisir_seuil(y_bin, proba_fraude, beta=1.0):
    """Seuil qui maximise le F-beta de la classe Fraude (beta=1 : F1 ; beta=2 : privilégie le rappel)."""
    precisions, rappels, seuils = precision_recall_curve(y_bin, proba_fraude)
    if len(seuils) == 0:
        return 0.5
    p, r = precisions[:-1], rappels[:-1]
    denom = beta ** 2 * p + r
    fbeta = np.divide((1 + beta ** 2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(seuils[np.argmax(fbeta)])


def metriques_fraude(y_bin, proba_fraude, seuil):
    """Métriques de la classe Fraude pour un seuil donné sur P(Fraude)."""
    pred = (proba_fraude >= seuil).astype(int)
    return {
        "seuil": round(float(seuil), 4),
        "rappel": round(float(recall_score(y_bin, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_bin, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_bin, pred, zero_division=0)), 4),
        "auc_pr": round(float(average_precision_score(y_bin, proba_fraude)), 4),
        "n_alertes": int(pred.sum()),
        "n_fraudes": int(y_bin.sum()),
    }


def main():
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE DÉTECTION DE FRAUDE BANCAIRE (v2)")
    print("=" * 70)
    if not XGBOOST_DISPONIBLE:
        print("Avertissement : xgboost n'est pas installé (pip install xgboost) : modèle ignoré.")
    if not LIGHTGBM_DISPONIBLE:
        print("Avertissement : lightgbm n'est pas installé (pip install lightgbm) : modèle ignoré.")

    # ---------- 1. Chargement et variables ----------
    print("\n[1/8] Chargement des données et feature engineering...")
    df = charger_donnees(CSV_PATH)
    print(f"      {df.shape[0]} transactions, {df.shape[1]} colonnes brutes.")
    df_clean, top_localisations = nettoyer_et_enrichir(df)

    le_target = LabelEncoder()
    le_target.fit(CLASSES_CIBLE)
    y = le_target.transform(df_clean["Target"])
    label_fraude = int(le_target.transform(["Fraude"])[0])
    groupes = df_clean["ID Clients"].to_numpy()

    X_df, colonnes = encoder_features(df_clean)
    feature_columns = [c for c in colonnes if c not in VARIABLES_EXCLUES]
    X = X_df[feature_columns].to_numpy(dtype=float)
    print(f"      {len(feature_columns)} variables explicatives (exclues : {', '.join(VARIABLES_EXCLUES)}).")

    # ---------- 2. Jeu de test réservé (environ 20 % des clients) ----------
    print("\n[2/8] Mise de côté d'un jeu de test (clients jamais utilisés pour choisir le modèle)...")
    idx_dev, idx_test = plis_par_client(X, y, groupes)[0]
    X_dev, y_dev, g_dev = X[idx_dev], y[idx_dev], groupes[idx_dev]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"      Développement : {len(idx_dev)} lignes ({len(set(g_dev))} clients) | "
          f"Test : {len(idx_test)} lignes ({len(set(groupes[idx_test]))} clients)")
    print(f"      Fraudes : {(y_dev == label_fraude).sum()} en développement, "
          f"{(y_test == label_fraude).sum()} en test")

    # ---------- 3. Comparaison des modèles (validation croisée par client) ----------
    print(f"\n[3/8] Comparaison des modèles ({N_PLIS} plis par client, critère : AUC-PR de la classe Fraude)...")
    estimateurs = get_estimateurs()
    resultats = []
    resultats_cv = []
    y_dev_bin = (y_dev == label_fraude).astype(int)

    for nom, fabrique in estimateurs.items():
        proba, ap_plis = predictions_hors_echantillon(fabrique, X_dev, y_dev, g_dev, label_fraude)
        y_pred = proba.argmax(axis=1)
        ap = average_precision_score(y_dev_bin, proba[:, label_fraude])
        rappel_fraude = recall_score(y_dev_bin, (y_pred == label_fraude).astype(int), zero_division=0)

        resultats.append({
            "Modèle": nom,
            "Accuracy": round(accuracy_score(y_dev, y_pred), 4),
            "Précision (macro)": round(precision_score(y_dev, y_pred, average="macro", zero_division=0), 4),
            "Rappel (macro)": round(recall_score(y_dev, y_pred, average="macro", zero_division=0), 4),
            "F1-score (macro)": round(f1_score(y_dev, y_pred, average="macro", zero_division=0), 4),
            "Rappel classe Fraude": round(rappel_fraude, 4),
            "_ap": ap,
        })
        resultats_cv.append({
            "Modèle": nom,
            "AUC-PR Fraude (hors échantillon)": round(float(ap), 4),
            "AUC-PR par pli (moyenne)": round(float(np.mean(ap_plis)), 4),
            "AUC-PR par pli (écart-type)": round(float(np.std(ap_plis, ddof=1)) if len(ap_plis) > 1 else 0.0, 4),
            "Précision classe Fraude": round(precision_score(
                y_dev_bin, (y_pred == label_fraude).astype(int), zero_division=0), 4),
            "Rappel classe Fraude": round(rappel_fraude, 4),
        })
        print(f"      {nom:25s} AUC-PR={ap:.3f}  rappel(Fraude)={rappel_fraude:.3f}  "
              f"précision(Fraude)={resultats_cv[-1]['Précision classe Fraude']:.3f}")

    resultats.sort(key=lambda r: r["_ap"], reverse=True)
    resultats_cv.sort(key=lambda r: r["AUC-PR Fraude (hors échantillon)"], reverse=True)
    for r in resultats:
        r.pop("_ap")

    # ---------- 4. Détection non supervisée (Isolation Forest) ----------
    print("\n[4/8] Détection non supervisée avec Isolation Forest...")
    # Le scaler est ajusté sur un DataFrame (noms de colonnes conservés), comme l'attend l'application.
    X_dev_df = X_df[feature_columns].iloc[idx_dev]
    X_test_df = X_df[feature_columns].iloc[idx_test]
    scaler_dev = StandardScaler().fit(X_dev_df)
    X_dev_s, X_test_s = scaler_dev.transform(X_dev_df), scaler_dev.transform(X_test_df)
    contamination = float((y_dev == label_fraude).mean())
    iso_forest = IsolationForest(n_estimators=300, contamination=contamination, random_state=GRAINE)
    iso_forest.fit(X_dev_s)  # l'entraînement n'utilise pas les étiquettes ; seule la proportion de fraudes règle contamination

    pred_iso_bin = (iso_forest.predict(X_test_s) == -1).astype(int)   # 1 = anomalie
    y_test_bin = (y_test == label_fraude).astype(int)
    metriques_isolation_forest = {
        "contamination": round(contamination, 4),
        "precision": round(precision_score(y_test_bin, pred_iso_bin, zero_division=0), 4),
        "rappel": round(recall_score(y_test_bin, pred_iso_bin, zero_division=0), 4),
        "f1": round(f1_score(y_test_bin, pred_iso_bin, zero_division=0), 4),
    }
    print(f"      Isolation Forest (anomalie vs Fraude réelle, jeu de test) : "
          f"précision={metriques_isolation_forest['precision']:.3f}  "
          f"rappel={metriques_isolation_forest['rappel']:.3f}  f1={metriques_isolation_forest['f1']:.3f}")
    print("      Note : non supervisé, il détecte des anomalies statistiques, pas spécifiquement des fraudes ;"
          " ses résultats sont rapportés à titre complémentaire.")

    # ---------- 5. Réglage du meilleur modèle ----------
    meilleur_nom = resultats[0]["Modèle"]
    print(f"\n[5/8] Modèle sélectionné : {meilleur_nom} (meilleure AUC-PR hors échantillon). Réglage...")
    fabrique = estimateurs[meilleur_nom]
    grille = list(ParameterGrid(PARAM_GRIDS.get(meilleur_nom, {})))
    meilleur_ap, meilleurs_params, meilleure_proba = -1.0, {}, None
    for params in (grille or [{}]):
        proba, _ = predictions_hors_echantillon(fabrique, X_dev, y_dev, g_dev, label_fraude, params)
        ap = average_precision_score(y_dev_bin, proba[:, label_fraude])
        if ap > meilleur_ap:
            meilleur_ap, meilleurs_params, meilleure_proba = ap, params, proba
    print(f"      Meilleurs paramètres : {meilleurs_params}  (AUC-PR hors échantillon : {meilleur_ap:.3f})")

    # ---------- 6. Seuil de décision (prédictions hors échantillon du développement) ----------
    print("\n[6/8] Calibrage du seuil de décision sur les prédictions hors échantillon...")
    proba_oof_fraude = meilleure_proba[:, label_fraude]
    seuil_f1 = choisir_seuil(y_dev_bin, proba_oof_fraude, beta=1.0)
    seuil_f2 = choisir_seuil(y_dev_bin, proba_oof_fraude, beta=2.0)
    m_f1 = metriques_fraude(y_dev_bin, proba_oof_fraude, seuil_f1)
    m_f2 = metriques_fraude(y_dev_bin, proba_oof_fraude, seuil_f2)
    print(f"      Seuil F1 = {seuil_f1:.3f} -> rappel={m_f1['rappel']:.3f}, précision={m_f1['precision']:.3f}")
    print(f"      Seuil F2 = {seuil_f2:.3f} -> rappel={m_f2['rappel']:.3f}, précision={m_f2['precision']:.3f}"
          " (privilégie le rappel)")

    # ---------- 7. Évaluation finale sur le jeu de test ----------
    print("\n[7/8] Évaluation finale sur le jeu de test (clients jamais vus, jamais utilisés pour choisir)...")
    meilleur_modele = fabrique()
    if meilleurs_params:
        meilleur_modele.set_params(**meilleurs_params)
    meilleur_modele.fit(X_dev_s, y_dev, sample_weight=compute_sample_weight("balanced", y_dev))

    proba_test = meilleur_modele.predict_proba(X_test_s)
    y_pred_final = proba_test.argmax(axis=1)
    labels_ordre = le_target.inverse_transform(sorted(set(y))).tolist()
    rapport = classification_report(y_test, y_pred_final, target_names=labels_ordre, output_dict=True)
    print(classification_report(y_test, y_pred_final, target_names=labels_ordre))
    test_f1 = metriques_fraude(y_test_bin, proba_test[:, label_fraude], seuil_f1)
    print(f"      Classe Fraude au seuil F1 ({seuil_f1:.3f}) : rappel={test_f1['rappel']:.3f}, "
          f"précision={test_f1['precision']:.3f}, F1={test_f1['f1']:.3f}, AUC-PR={test_f1['auc_pr']:.3f} "
          f"({test_f1['n_fraudes']} fraudes, {test_f1['n_alertes']} alertes)")
    print(f"      Repère : un classement au hasard donnerait une AUC-PR d'environ {y_test_bin.mean():.3f}.")

    # ---------- 8. Sauvegarde des artefacts ----------
    print("\n[8/8] Sauvegarde du modèle et des artefacts...")
    metriques_test = {
        "accuracy": accuracy_score(y_test, y_pred_final),
        "rapport_classification": rapport,
        "matrice_confusion": confusion_matrix(y_test, y_pred_final).tolist(),
        "labels_ordre": labels_ordre,
        "n_test": int(len(y_test)),
    }

    artefacts = {
        # --- clés lues par l'application (inchangées par rapport à la version 1) ---
        "model": meilleur_modele,
        "model_nom": meilleur_nom,
        "scaler": scaler_dev,
        "label_encoder": le_target,
        "feature_columns": list(feature_columns),
        "top_localisations": list(top_localisations),
        "best_threshold_f1": seuil_f1,
        "metriques_test": metriques_test,
        "grid_search_best_params": meilleurs_params,
        "comparaison_modeles": resultats,
        "metriques_isolation_forest": metriques_isolation_forest,
        # --- nouveautés de la version 2 ---
        "best_threshold_f2": seuil_f2,
        "comparaison_modeles_cv": resultats_cv,
        "metriques_fraude_cv": {"seuil_f1": m_f1, "seuil_f2": m_f2},
        "metriques_fraude_test": {"seuil_f1": test_f1,
                                  "seuil_f2": metriques_fraude(y_test_bin, proba_test[:, label_fraude], seuil_f2)},
        "variables_exclues": list(VARIABLES_EXCLUES),
        "methode_evaluation": "validation croisée par client ; test = clients mis de côté",
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artefacts, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print("\nTerminé. Vous pouvez maintenant lancer l'application : streamlit run app.py")


if __name__ == "__main__":
    main()
