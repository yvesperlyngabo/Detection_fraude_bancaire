"""
Évaluation par validation croisée stratifiée répétée.

Pourquoi ce script ?
    Le jeu de test d'un découpage unique ne contient qu'environ 40 fraudes : le rappel
    varie de plusieurs points d'un tirage à l'autre, et deux entraînements successifs
    ne sont pas comparables. Ici, chaque modèle est évalué sur de nombreux découpages
    (k plis x répétitions), et on rapporte une moyenne et un écart-type.

Ce que le script compare :
    - 4 modèles supervisés (mêmes réglages par défaut que train_model.py)
    - 3 jeux de variables :
        complet          : toutes les variables
        sans_7j          : sans montant_moyen_7j ni nb_trans_7j
        sans_comport.    : sans les variables 7 jours ni nb_transactions_client
      Les mêmes plis sont utilisés pour les 3 jeux, ce qui permet une comparaison appariée.

Métriques (classe "Fraude") : rappel, précision, F1 et AUC-PR (average precision).

Option --par-client :
    Les plis sont alors construits par client (toutes les transactions d'un même client
    sont soit dans l'entraînement, soit dans le test). Cela mesure la capacité du modèle à
    reconnaître la fraude chez des clients JAMAIS VUS à l'entraînement, ce qui est la
    situation réelle d'un nouveau client. Si les performances s'effondrent par rapport au
    découpage par transaction, le modèle s'appuie sur l'identité des clients.

Usage (depuis la racine du projet) :
    python model/evaluate_cv.py
    python model/evaluate_cv.py --par-client
    python model/evaluate_cv.py --splits 5 --repeats 3
Génère model/cv_results.csv, ou model/cv_results_par_client.csv avec --par-client.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import CLASSES_CIBLE, charger_donnees, encoder_features, nettoyer_et_enrichir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "Bank_transaction_scenario1.csv")
SORTIE_CSV = os.path.join(BASE_DIR, "model", "cv_results.csv")
SORTIE_CSV_CLIENT = os.path.join(BASE_DIR, "model", "cv_results_par_client.csv")

VARIABLES_7J = ["montant_moyen_7j", "nb_trans_7j"]
METRIQUES = ["rappel", "precision", "f1", "auc_pr"]


def get_estimateurs():
    """Mêmes modèles et réglages par défaut que train_model.py (sans les modèles faibles)."""
    estimateurs = {
        "Random Forest": lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        estimateurs["XGBoost"] = lambda: XGBClassifier(eval_metric="mlogloss", random_state=42)
    except ImportError:
        print("xgboost absent : modèle ignoré.")
    try:
        from lightgbm import LGBMClassifier
        estimateurs["LightGBM"] = lambda: LGBMClassifier(random_state=42, verbosity=-1)
    except ImportError:
        print("lightgbm absent : modèle ignoré.")
    return estimateurs


def diagnostic_variables_7j(df_clean):
    """Montre si les variables 7 jours contiennent de l'information exploitable."""
    avec_historique = df_clean["nb_trans_7j"] > 0
    print(f"Part des transactions ayant un historique sur 7 jours : {avec_historique.mean():.1%}")
    print("Moyennes par classe :")
    print(df_clean.groupby("Target")[["nb_trans_7j", "montant_moyen_7j", "nb_transactions_client"]]
          .mean().round(2).to_string())
    print()


def diagnostic_clients(df_clean):
    """Mesure à quel point les clients sont identifiables et concentrent la fraude."""
    par_client = df_clean.groupby("ID Clients")
    n_clients = par_client.ngroups
    fraudes = (df_clean[df_clean["Target"] == "Fraude"]
               .groupby("ID Clients").size().sort_values(ascending=False))
    print(f"Clients : {n_clients} ; transactions par client : médiane {par_client.size().median():.0f}")
    if len(fraudes):
        n_top = max(1, int(round(0.10 * n_clients)))
        part_top = fraudes.head(n_top).sum() / fraudes.sum()
        print(f"Clients ayant au moins une fraude : {len(fraudes)} ({len(fraudes) / n_clients:.1%})")
        print(f"Part des fraudes portée par les 10 % de clients les plus touchés : {part_top:.1%}")
    n_valeurs = par_client["nb_transactions_client"].first().nunique()
    print(f"Valeurs distinctes de nb_transactions_client : {n_valeurs} pour {n_clients} clients "
          "(plus c'est proche du nombre de clients, plus la variable identifie le client)")
    print()


def generer_plis(X, y, groupes, n_splits, n_repeats, par_client):
    """Plis stratifiés répétés, par transaction ou par client."""
    if par_client:
        for r in range(n_repeats):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42 + r)
            yield from sgkf.split(X, y, groupes)
    else:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        yield from cv.split(X, y)


def evaluer(X, y, groupes, label_fraude, estimateurs, jeux, n_splits, n_repeats, par_client):
    lignes = []
    total = n_splits * n_repeats
    for i, (idx_tr, idx_te) in enumerate(generer_plis(X, y, groupes, n_splits, n_repeats, par_client), start=1):
        print(f"  pli {i}/{total}", flush=True)
        y_tr, y_te = y[idx_tr], y[idx_te]
        y_bin = (y_te == label_fraude).astype(int)
        if y_bin.sum() == 0:
            print("    (aucune fraude dans ce pli de test : pli ignoré)")
            continue
        poids = compute_sample_weight("balanced", y_tr)

        for nom_jeu, colonnes in jeux.items():
            X_tr, X_te = X.iloc[idx_tr][colonnes], X.iloc[idx_te][colonnes]
            scaler = StandardScaler().fit(X_tr)          # ajusté sur le pli d'entraînement seul
            X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

            for nom, fabrique in estimateurs.items():
                modele = fabrique()
                modele.fit(X_tr_s, y_tr, sample_weight=poids)
                proba = modele.predict_proba(X_te_s)[:, list(modele.classes_).index(label_fraude)]
                pred_bin = (modele.predict(X_te_s) == label_fraude).astype(int)
                lignes.append({
                    "pli": i, "modele": nom, "jeu": nom_jeu,
                    "rappel": recall_score(y_bin, pred_bin, zero_division=0),
                    "precision": precision_score(y_bin, pred_bin, zero_division=0),
                    "f1": f1_score(y_bin, pred_bin, zero_division=0),
                    "auc_pr": average_precision_score(y_bin, proba),
                })
    return pd.DataFrame(lignes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=int, default=5, help="nombre de plis (défaut : 5)")
    parser.add_argument("--repeats", type=int, default=2, help="nombre de répétitions (défaut : 2)")
    parser.add_argument("--csv", default=CSV_PATH, help="chemin du fichier de transactions")
    parser.add_argument("--par-client", action="store_true",
                        help="plis par client (clients jamais vus à l'entraînement)")
    args = parser.parse_args()

    print("Chargement et préparation des données...")
    df = charger_donnees(args.csv)
    df_clean, _ = nettoyer_et_enrichir(df)
    diagnostic_variables_7j(df_clean)
    diagnostic_clients(df_clean)

    le = LabelEncoder().fit(CLASSES_CIBLE)
    y = le.transform(df_clean["Target"])
    label_fraude = le.transform(["Fraude"])[0]
    X, colonnes = encoder_features(df_clean)
    X = X.reset_index(drop=True)

    jeux = {
        "complet": list(colonnes),
        "sans_7j": [c for c in colonnes if c not in VARIABLES_7J],
        "sans_comport.": [c for c in colonnes if c not in VARIABLES_7J + ["nb_transactions_client"]],
    }
    groupes = df_clean["ID Clients"].to_numpy()
    n_fraudes = int((y == label_fraude).sum())
    mode = "par client (clients jamais vus)" if args.par_client else "par transaction"
    print(f"{len(y)} transactions, {n_fraudes} fraudes ; "
          f"{args.splits} plis x {args.repeats} répétitions ; plis {mode}.\n")

    resultats = evaluer(X, y, groupes, label_fraude, get_estimateurs(), jeux,
                        args.splits, args.repeats, args.par_client)
    sortie = SORTIE_CSV_CLIENT if args.par_client else SORTIE_CSV
    resultats.to_csv(sortie, index=False)

    # --- Résumé : moyenne ± écart-type sur les plis ---
    print("\n" + "=" * 78)
    print("RÉSULTATS (classe Fraude) : moyenne ± écart-type sur les plis")
    print("=" * 78)
    resume = resultats.groupby(["modele", "jeu"])[METRIQUES].agg(["mean", "std"])
    for (modele, jeu), ligne in resume.iterrows():
        morceaux = [f"{m}={ligne[(m, 'mean')]:.3f}±{ligne[(m, 'std')]:.3f}" for m in METRIQUES]
        print(f"{modele:18s} {jeu:14s} " + "  ".join(morceaux))

    # --- Comparaison appariée : 'complet' contre les autres jeux, pli par pli ---
    print("\n" + "=" * 78)
    print("COMPARAISON APPARIÉE : 'complet' moins l'autre jeu (positif = 'complet' meilleur)")
    print("=" * 78)
    for autre in ["sans_7j", "sans_comport."]:
        print(f"\n'complet' contre '{autre}' :")
        for modele in resultats["modele"].unique():
            a = resultats[(resultats.modele == modele) & (resultats.jeu == "complet")].set_index("pli")
            b = resultats[(resultats.modele == modele) & (resultats.jeu == autre)].set_index("pli")
            ecart = (a[METRIQUES] - b[METRIQUES])
            gains = (ecart["auc_pr"] > 0).mean()
            print(f"  {modele:18s} écart auc_pr={ecart['auc_pr'].mean():+.3f}±{ecart['auc_pr'].std():.3f}  "
                  f"écart rappel={ecart['rappel'].mean():+.3f}  "
                  f"'complet' meilleur sur {gains:.0%} des plis (auc_pr)")

    print(f"\nRésultats détaillés : {sortie}")
    print("Lecture : un écart n'est convaincant que s'il dépasse nettement son écart-type "
          "et reste de même signe sur la plupart des plis.")


if __name__ == "__main__":
    main()
