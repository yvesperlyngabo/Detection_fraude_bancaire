# 🏦 Détection de la fraude bancaire

Application Streamlit de détection de fraude bancaire, construite à partir du pipeline de
machine learning développé et validé dans le notebook `detection_fraude_bancaire.ipynb`
(exploration des données, feature engineering, comparaison de modèles, optimisation par
GridSearchCV, calibrage d'un seuil de décision).

## 📁 Structure du projet

```
├── data/
│   └── Bank_transaction_scenario1.csv   # Jeu de données des transactions bancaires
│
├── model/
│   ├── preprocessing.py                 # Fonctions de prétraitement partagées (entraînement + app)
│   ├── train_model.py                   # Script d'entraînement (6 modèles + Isolation Forest)
│   └── fraud_model.pkl                  # Modèle sérialisé (généré)
│
├── utils/
│   ├── css.py                           # Thème visuel (palette, typographies, composants)
│   ├── data_io.py                       # Import CSV/Excel + générateur de données synthétiques
│   └── history.py                       # Historique des prédictions + exports CSV/PDF/Excel
│
├── modules/
│   ├── dashboard.py                     # Page 1 : KPIs & synthèse
│   ├── data_loader.py                   # Page 2 : import & santé des données
│   ├── eda.py                           # Pages 4-5 : exploration & visualisations Plotly
│   ├── ml_training.py                   # Pages 6-7 : comparaison des modèles + Isolation Forest
│   ├── evaluation.py                    # Pages 8-9 : matrice de confusion, ROC, seuil de décision
│   ├── prediction.py                    # Pages 10-12 : prédiction temps réel + jauge de risque
│   └── reporting.py                     # Page 13 : historique & rapports (CSV/PDF/Excel)
│
├── app.py                               # Point d'entrée : navigation & orchestration des modules
├── requirements.txt                     # Dépendances
├── Dockerfile                           # Image Docker de l'application
├── docker-compose.yml                   # Lancement simplifié via Docker Compose
├── .streamlit/config.toml               # Thème natif Streamlit (base sombre, couleurs)
├── .dockerignore
├── .gitignore
└── README.md
```

## 🎯 Fonctionnalités de l'application

- **🏠 Dashboard** : KPIs (transactions, fraudes, taux, montant total/moyen, clients uniques, modèle
  actif), tendance temporelle et répartition globale.
- **📂 Import & données** : upload CSV/Excel, génération d'un jeu de données synthétique pour tester
  l'app sans fichier réel, diagnostic qualité (lignes, colonnes, mémoire, valeurs manquantes, doublons).
- **📊 Exploration des données** : filtres globaux (dates, montant, type, statut), répartition des
  statuts, montants, treemap localisation × type, heatmap de corrélation.
- **🤖 Entraînement ML** : comparaison de 6 modèles supervisés (Régression Logistique, Arbre de
  décision, Random Forest, Gradient Boosting, XGBoost, LightGBM) + détection non supervisée
  complémentaire (Isolation Forest).
- **📉 Évaluation & seuil** : rapport de classification, matrice de confusion, importance des
  variables, courbes précision-rappel et ROC, curseur de seuil de décision interactif.
- **🔮 Prédiction en temps réel** : formulaire de saisie d'une transaction, verdict stylisé,
  probabilités par classe et **jauge de niveau de risque** (vert/orange/rouge).
- **🗂️ Historique & rapports** : suivi des prédictions effectuées, filtrage, export **CSV**, **PDF**
  et **Excel**.

## ⚙️ Installation

```bash
# Cloner le projet, puis se placer à la racine
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Entraîner le modèle

Le modèle sérialisé n'est pas versionné (voir `.gitignore`) : il doit être généré une première fois.

```bash
python model/train_model.py
```

Ce script :
1. charge `data/Bank_transaction_scenario1.csv` ;
2. applique le feature engineering (variables temporelles, montant en log, fréquence client,
   regroupement des localisations rares) ;
3. compare **6 modèles supervisés** : Régression Logistique, Arbre de décision, Random Forest,
   Gradient Boosting, **XGBoost**, **LightGBM** (tous pondérés `balanced` pour compenser la rareté
   de la classe Fraude) ;
4. entraîne en complément un modèle **non supervisé (Isolation Forest)**, qui détecte des anomalies
   statistiques sans jamais utiliser la variable cible — utile pour repérer des schémas de fraude
   inédits, jamais rencontrés à l'entraînement ;
5. **sélectionne automatiquement** le modèle supervisé ayant le meilleur rappel sur la classe Fraude,
   puis l'optimise par `GridSearchCV` (validation croisée à 5 plis, `scoring="recall_macro"`) ;
6. calibre un seuil de décision optimal (F1) pour la classe *Fraude* ;
7. sauvegarde le modèle, la comparaison complète des modèles et tous les artefacts nécessaires dans
   `model/fraud_model.pkl`.

> XGBoost et LightGBM sont optionnels : si l'un des deux n'est pas installé, le script continue avec
> un avertissement et compare uniquement les modèles disponibles.

### 2. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur (par défaut sur `http://localhost:8501`).

## 🐳 Utilisation avec Docker

### Construire et lancer avec Docker Compose (recommandé)

```bash
docker compose up --build
```

L'application est ensuite accessible sur **http://localhost:8501**.

Le modèle est entraîné automatiquement **pendant la construction de l'image** (voir `Dockerfile`) :
inutile de lancer `train_model.py` séparément. Pour ré-entraîner (par exemple après avoir modifié
`data/Bank_transaction_scenario1.csv`), il suffit de reconstruire l'image :

```bash
docker compose up --build --force-recreate
```

Arrêter le conteneur :

```bash
docker compose down
```

### Sans Docker Compose (commandes Docker classiques)

```bash
# Construire l'image
docker build -t fraud-detection-app .

# Lancer le conteneur
docker run -p 8501:8501 --name fraud_detection_app fraud-detection-app

# Arrêter / supprimer le conteneur
docker stop fraud_detection_app
docker rm fraud_detection_app
```

## 📤 Publier le projet sur GitHub

Depuis la racine du projet :

```bash
git init
git add .
git commit -m "Initial commit - Fraud detection app"
git branch -M main
git remote add origin https://github.com/<votre-utilisateur>/<votre-repo>.git
git push -u origin main
```

> `model/fraud_model.pkl` et l'environnement virtuel (`venv/`) sont exclus du dépôt via `.gitignore` —
> c'est volontaire : le modèle est régénéré à la demande (`python model/train_model.py` en local, ou
> automatiquement lors du `docker build`). Le fichier de données `data/Bank_transaction_scenario1.csv`
> reste, lui, versionné.

Pour les mises à jour suivantes :

```bash
git add .
git commit -m "Description des changements"
git push
```


Le fichier CSV utilise le **point-virgule** (`;`) comme séparateur et doit contenir au minimum les
colonnes suivantes :

| Colonne | Description |
|---|---|
| `ID Clients` | Identifiant du client |
| `Numero de compte` | Numéro de compte bancaire |
| `Identifiant operation` | Identifiant unique de la transaction |
| `Type de transaction` | ATM / Paiement en ligne / Paiement électronique |
| `Status operation` | Validé / Echoué / En attente |
| `Localisation` | Ville de la transaction |
| `Date` | Date et heure de la transaction |
| `Montant` | Montant de la transaction |
| `Target` | Normal / Suspect / Fraude (variable cible) |

## 🔧 Notes techniques

- Le module `model/preprocessing.py` centralise toute la logique de transformation des données
  (variables temporelles, log du montant, regroupement des localisations, encodage one-hot) afin de
  garantir qu'une nouvelle transaction saisie dans l'application soit traitée exactement de la même
  façon que les données d'entraînement.
- La classe **Fraude** étant fortement minoritaire (~3,7% des transactions), tous les modèles
  supervisés sont entraînés avec une pondération `balanced` des échantillons (via
  `compute_sample_weight`, compatible avec tous les modèles y compris ceux sans paramètre
  `class_weight` natif comme XGBoost), et le seuil de décision par défaut (0,5) est recalibré vers un
  seuil optimisant le F1-score sur cette classe.
- **Isolation Forest** est entraîné séparément, sans jamais utiliser la variable cible : c'est un
  modèle purement non supervisé qui repère des transactions statistiquement atypiques. Ses
  performances sont généralement inférieures à celles des modèles supervisés sur ce jeu de données
  (la fraude n'y est pas uniquement une anomalie statistique), mais il reste pertinent en pratique
  pour détecter des schémas de fraude nouveaux, absents des données d'entraînement.
- Pour ré-entraîner le modèle avec de nouvelles données, il suffit de remplacer
  `data/Bank_transaction_scenario1.csv` et de relancer `python model/train_model.py`.

## 📌 Pistes d'amélioration possibles

- Rééquilibrage supplémentaire de la classe Fraude (SMOTE).
- Autres modèles d'anomalie non supervisés (One-Class SVM, Local Outlier Factor, autoencodeur).
- Variables comportementales enrichies par client (montant moyen historique, écarts au comportement
  habituel, vélocité des transactions).
- Ré-entraînement automatique planifié (pipeline MLOps).
