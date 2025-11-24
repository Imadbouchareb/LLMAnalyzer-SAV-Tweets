from pathlib import Path

import streamlit as st

from lib.state import get_cfg
from lib.ui import inject_style, set_container_wide, inject_sticky_css

st.set_page_config(page_title="SAV Tweets — Hub", page_icon="💬", layout="wide")
inject_style()
set_container_wide()
inject_sticky_css()

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        .role-card {
            background-color: #1E293B;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.4rem;
            height: 100%;
            transition: all 0.2s ease-in-out;
        }
        .role-card:hover {
            transform: scale(1.03);
            border-color: rgba(70, 82, 255, 0.8);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        }
        .role-card .card-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: rgba(70, 82, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            margin-bottom: 0.8rem;
        }
        .role-card h3 {
            margin-top: 0;
            margin-bottom: 0.6rem;
        }
        .role-card p {
            font-size: 0.95rem;
            min-height: 72px;
            opacity: 0.85;
        }
        .role-card .card-button {
            display: inline-block;
            padding: 0.55rem 1rem;
            border-radius: 12px;
            background: #4652ff;
            color: #ffffff;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .role-card .card-button:hover {
            background-color: #3B82F6;
            transform: translateY(-2px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = get_cfg()

candidates = [
    Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
    Path(__file__).resolve().parents[1] / "tweets_scored_llm.csv",
]
auto_path = next((str(p) for p in candidates if p.exists()), cfg["paths"]["csv_path"])
csv_path = st.sidebar.text_input("Chemin CSV", value=auto_path, key="csv_path")
cfg["paths"]["csv_path"] = csv_path

if Path(csv_path).exists():
    st.sidebar.success(f"Fichier CSV détecté ✅\n{csv_path}")
else:
    st.sidebar.warning(
        "CSV introuvable.\n"
        "Assure-toi qu’il est disponible hors ligne (OneDrive → Toujours conserver sur cet appareil) "
        "ou indique un chemin absolu valide."
    )

st.markdown(
    """
    <div style="padding:2.5rem 2rem;background:linear-gradient(120deg,#1c1f3a,#121525);border-radius:18px;margin-bottom:2rem;">
        <h1 style="margin-bottom:0.5rem;">SAV Tweets — Maquettes</h1>
        <p style="font-size:1.05rem;max-width:640px;">
            Choisis la vue adaptée à ton rôle pour explorer les tweets scorés, suivre les alertes ou traiter les dossiers.
        </p>
        <div style="font-size:0.9rem;opacity:0.8;">
            Source courante&nbsp;: <code>{csv}</code>
        </div>
    </div>
    """.format(csv=csv_path),
    unsafe_allow_html=True,
)
cards = [
    {
        "titre": "Analyste",
        "emoji": "📊",
        "pitch": "Explorer les volumes, tendances, exports et analyses avancées.",
        "page": "/Analyste",
    },
    {
        "titre": "Manager",
        "emoji": "🧭",
        "pitch": "Piloter les KPI, monitorer les équipes et prioriser les urgences.",
        "page": "/Manager",
    },
    {
        "titre": "Agent SAV",
        "emoji": "🎧",
        "pitch": "Traiter la file d’attente, lire les résumés LLM et répondre plus vite.",
        "page": "/Agent_SAV",
    },
]

col_cards = st.columns(len(cards), gap="large")
for col, card in zip(col_cards, cards):
    with col:
        st.markdown(
            f"""
            <div class="role-card">
                <div class="card-icon">{card['emoji']}</div>
                <h3>{card['titre']}</h3>
                <p>{card['pitch']}</p>
                <div style="margin-top:1rem;">
                    <a class="card-button"
                       href="{card['page']}"
                       target="_self">
                        Ouvrir la vue
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption("Besoin d’un autre dataset ? Modifie le chemin CSV dans la barre latérale.")
