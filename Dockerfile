# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Dépendances système :
# - build-essential : compilation de certaines dépendances Python
# - libgomp1        : requis par LightGBM/XGBoost (OpenMP)
# - curl            : utilisé par le HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python en premier (meilleur cache Docker :
# cette étape n'est reconstruite que si requirements.txt change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code (data/, model/, app.py, ...)
COPY . .

# Entraînement du modèle au moment du build : l'image est ainsi autonome
# et contient directement model/fraud_model.pkl, prêt à l'emploi.
RUN python model/train_model.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
