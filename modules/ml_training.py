"""Pages 6-7 : Entraînement supervisé (comparaison de modèles) + Isolation Forest."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.css import render_hero


def render(artefacts: dict, model):
    render_hero(
        "🤖", "Modélisation", "Entraînement & comparaison des modèles",
        "Comparaison des modèles supervisés testés, puis détection non supervisée complémentaire.",
    )

    if artefacts is None:
        st.error("Aucun modèle entraîné. Lancez `python model/train_model.py` puis rechargez la page.")
        st.stop()

    m = artefacts["metriques_test"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modèle retenu", artefacts.get("model_nom", type(model).__name__))
    col2.metric("Accuracy (test)", f"{m['accuracy'] * 100:.1f} %")
    col3.metric("Rappel - Fraude", f"{m['rapport_classification']['Fraude']['recall'] * 100:.1f} %")
    col4.metric("Précision - Fraude", f"{m['rapport_classification']['Fraude']['precision'] * 100:.1f} %")

    st.caption(f"Meilleurs hyperparamètres (GridSearchCV) : `{artefacts['grid_search_best_params']}` — "
               f"évalué sur {m['n_test']} transactions de test.")

    st.markdown("---")
    st.subheader("🏆 Comparaison des modèles supervisés testés")
    st.caption(
        "Tous les modèles sont entraînés avec une pondération 'balanced' des classes pour compenser "
        "la rareté de la classe Fraude. Le modèle avec le meilleur rappel sur la classe Fraude est "
        "automatiquement sélectionné puis optimisé par GridSearchCV."
    )
    if "comparaison_modeles" in artefacts:
        comp_df = pd.DataFrame(artefacts["comparaison_modeles"]).set_index("Modèle")
        comp_df_pct = (comp_df * 100).round(1)

        def surligner_meilleur(row):
            est_meilleur = row.name == artefacts.get("model_nom")
            return ["background-color: #1f4d33" if est_meilleur else "" for _ in row]

        st.dataframe(comp_df_pct.style.apply(surligner_meilleur, axis=1), use_container_width=True)

        fig = px.bar(
            comp_df.reset_index(), x="Modèle",
            y=["Accuracy", "Rappel (macro)", "F1-score (macro)", "Rappel classe Fraude"],
            barmode="group", template="plotly_dark",
            title="Comparaison des modèles (jeu de test)",
        )
        fig.update_layout(yaxis_title="Score", legend_title="Métrique",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Comparaison indisponible — réentraînez le modèle avec la dernière version de "
                "`train_model.py` pour l'obtenir.")

    st.markdown("---")
    st.subheader("🔍 Détection non supervisée complémentaire — Isolation Forest")
    st.caption(
        "Contrairement aux modèles ci-dessus, Isolation Forest n'utilise **jamais** la variable "
        "cible : il repère des transactions statistiquement atypiques (anomalies), sans avoir appris "
        "ce qu'est réellement une fraude. Il est donc moins précis ici, mais reste utile en pratique "
        "pour détecter des schémas de fraude inédits, jamais rencontrés à l'entraînement."
    )
    if "metriques_isolation_forest" in artefacts:
        iso = artefacts["metriques_isolation_forest"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Taux d'anomalies attendu", f"{iso['contamination'] * 100:.1f} %")
        col2.metric("Précision (vs Fraude réelle)", f"{iso['precision'] * 100:.1f} %")
        col3.metric("Rappel (vs Fraude réelle)", f"{iso['rappel'] * 100:.1f} %")
        col4.metric("F1-score", f"{iso['f1'] * 100:.1f} %")

        st.caption(
            "Lecture : la précision indique la part d'anomalies détectées qui sont réellement des "
            "fraudes ; le rappel indique la part des fraudes réelles effectivement repérées comme "
            "anomalies. Les faux positifs correspondent aux transactions normales signalées à tort."
        )
    else:
        st.info("Résultats Isolation Forest indisponibles — réentraînez le modèle avec la dernière "
                "version de `train_model.py` pour les obtenir.")
