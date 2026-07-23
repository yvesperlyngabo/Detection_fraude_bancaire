"""
Thème visuel de l'application : palette navy / or / teal, typographies
(Space Grotesk pour les titres, Inter pour l'interface, IBM Plex Mono pour les chiffres),
emblème radar animé, cartes de métriques, badges de statut, carte de verdict, boutons, etc.
"""
import streamlit as st

# Palette sémantique des statuts (utilisée par les graphiques Plotly ET les badges HTML)
COULEURS_TARGET = {"Normal": "#34D399", "Suspect": "#F5A524", "Fraude": "#FB4570"}
CLASSE_CSS = {"Normal": "badge-normal", "Suspect": "badge-suspect", "Fraude": "badge-fraude"}


def injecter_theme():
    """Injecte la feuille de style de l'application (à appeler une seule fois, au démarrage)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --bg: #0B1220;
            --panel: #131C31;
            --panel-alt: #1B2540;
            --line: #26314C;
            --gold: #E8B54D;
            --teal: #2DD4BF;
            --ink: #E8ECF6;
            --ink-muted: #93A0C2;
            --emerald: #34D399;
            --amber: #F5A524;
            --rose: #FB4570;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

        .stApp {
            background:
                radial-gradient(1100px 550px at 88% -8%, rgba(45,212,191,0.10), transparent),
                radial-gradient(900px 500px at -10% 110%, rgba(232,181,77,0.06), transparent),
                var(--bg);
        }

        /* ---------- Barre latérale ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1830 0%, #0B1220 100%);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 10px;
            padding: 8px 10px;
            margin-bottom: 2px;
            transition: background 0.15s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: var(--panel-alt);
        }

        .sidebar-emblem-wrap { display:flex; align-items:center; gap:12px; padding: 4px 0 20px 0; }
        .sidebar-emblem {
            position: relative; width:46px; height:46px; border-radius:50%;
            display:flex; align-items:center; justify-content:center; font-size:21px;
            background: var(--panel); border: 1px solid var(--line); flex-shrink:0;
        }
        .sidebar-emblem::before {
            content:""; position:absolute; inset:-6px; border-radius:50%;
            background: conic-gradient(from 0deg,
                transparent 0deg, var(--teal) 35deg, transparent 85deg,
                transparent 260deg, var(--gold) 315deg, transparent 360deg);
            animation: spin-radar 7s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
            .sidebar-emblem::before { animation: none; }
        }
        @keyframes spin-radar { to { transform: rotate(360deg); } }
        .sidebar-emblem span { position:relative; z-index:1; }
        .sidebar-brand-title {
            font-family:'Space Grotesk', sans-serif; font-weight:700;
            font-size:1.05rem; line-height:1.2; margin:0; color: var(--ink);
        }
        .sidebar-brand-sub {
            font-size:0.68rem; color: var(--ink-muted); text-transform:uppercase;
            letter-spacing:0.09em; margin:0;
        }

        /* ---------- En-tête de page (hero) ---------- */
        .hero { padding: 0 0 16px 0; margin-bottom: 10px; border-bottom: 1px solid var(--line); }
        .hero-eyebrow {
            font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--teal);
            letter-spacing:0.14em; text-transform:uppercase; margin:0 0 6px 0;
        }
        .hero-title {
            font-family:'Space Grotesk', sans-serif; font-size: 2rem; font-weight:700; margin:0;
            background: linear-gradient(90deg, var(--ink) 45%, var(--gold) 100%);
            -webkit-background-clip:text; background-clip:text; color:transparent;
            display:inline-block;
        }
        .hero-subtitle { color: var(--ink-muted); margin-top:8px; font-size:0.95rem; max-width: 820px; }

        /* ---------- Cartes de métriques ---------- */
        div[data-testid="stMetric"] {
            background: var(--panel); border:1px solid var(--line); border-left: 3px solid var(--gold);
            border-radius: 12px; padding: 14px 16px 10px 16px;
        }
        div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono', monospace; color: var(--ink); }
        div[data-testid="stMetricLabel"] { color: var(--ink-muted); }

        /* ---------- Badges de statut ---------- */
        .badge {
            display:inline-flex; align-items:center; gap:7px; padding:5px 12px;
            border-radius:999px; font-size:0.82rem; font-weight:600;
            font-family:'IBM Plex Mono', monospace;
        }
        .badge-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
        .badge-normal  { background: rgba(52,211,153,0.13); color:#34D399; }
        .badge-normal .badge-dot  { background:#34D399; }
        .badge-suspect { background: rgba(245,165,36,0.14); color:#F5A524; }
        .badge-suspect .badge-dot { background:#F5A524; }
        .badge-fraude  { background: rgba(251,69,112,0.15); color:#FB4570; }
        .badge-fraude .badge-dot  { background:#FB4570; }

        /* ---------- Carte de verdict (résultat de prédiction) ---------- */
        .verdict-card {
            border-radius: 14px; padding: 20px 22px; margin: 14px 0 18px 0;
            background: var(--panel); border: 1px solid var(--line); border-left-width: 4px;
        }
        .verdict-fraude  { border-left-color: var(--rose); }
        .verdict-suspect { border-left-color: var(--amber); }
        .verdict-normal  { border-left-color: var(--emerald); }
        .verdict-title { font-family:'Space Grotesk', sans-serif; font-size:1.15rem; font-weight:700; margin:0 0 4px 0; }
        .verdict-detail { color: var(--ink-muted); font-size:0.9rem; font-family:'IBM Plex Mono', monospace; }

        /* ---------- Boutons ---------- */
        .stButton>button, .stDownloadButton>button, button[kind="formSubmit"] {
            background: var(--gold); color:#241a05; border:none; border-radius:10px;
            font-weight:600; transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton>button:hover, .stDownloadButton>button:hover, button[kind="formSubmit"]:hover {
            transform: translateY(-1px); box-shadow: 0 6px 18px rgba(232,181,77,0.28);
            color:#241a05;
        }

        /* ---------- Tableaux et alertes ---------- */
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
        div[data-testid="stAlert"] { border-radius:10px; }

        /* Chiffres alignés proprement (montants, pourcentages) */
        div[data-testid="stMetricValue"], .stDataFrame { font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(icon: str, eyebrow: str, titre: str, sous_titre: str = ""):
    """Affiche l'en-tête stylisé d'une page (icône, eyebrow mono, titre en dégradé, sous-titre)."""
    sous_titre_html = f'<p class="hero-subtitle">{sous_titre}</p>' if sous_titre else ""
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-eyebrow">{eyebrow}</p>
            <h1 class="hero-title">{icon} {titre}</h1>
            {sous_titre_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand():
    """Affiche l'emblème radar + titre de marque dans la barre latérale."""
    st.sidebar.markdown(
        """
        <div class="sidebar-emblem-wrap">
            <div class="sidebar-emblem"><span>🏦</span></div>
            <div>
                <p class="sidebar-brand-title">Détection de fraude</p>
                <p class="sidebar-brand-sub">Surveillance bancaire</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_html(statut: str) -> str:
    """Retourne le HTML d'un badge coloré pour un statut (Normal / Suspect / Fraude)."""
    classe = CLASSE_CSS.get(statut, "badge-suspect")
    return f'<span class="badge {classe}"><span class="badge-dot"></span>{statut}</span>'


def render_verdict(statut: str, proba_fraude: float, seuil: float):
    """Affiche la carte de verdict stylisée (résultat de la classification d'une transaction)."""
    classe = {"Normal": "verdict-normal", "Suspect": "verdict-suspect", "Fraude": "verdict-fraude"}.get(
        statut, "verdict-suspect"
    )
    icone = {"Normal": "✅", "Suspect": "🟠", "Fraude": "🚨"}.get(statut, "🟠")
    libelle = "FRAUDE POTENTIELLE" if statut == "Fraude" else statut.upper()
    st.markdown(
        f"""
        <div class="verdict-card {classe}">
            <p class="verdict-title">{icone} Transaction classée : {libelle} {badge_html(statut)}</p>
            <p class="verdict-detail">Probabilité de fraude = {proba_fraude * 100:.1f} %  |  Seuil utilisé = {seuil:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def colorer_statut(val):
    """Fonction de style pandas : colore le texte d'une cellule selon le statut
    (Normal / Suspect / Fraude), pour prolonger la palette dans les tableaux natifs."""
    couleur = COULEURS_TARGET.get(val)
    if couleur:
        return f"color: {couleur}; font-weight: 600;"
    return ""
