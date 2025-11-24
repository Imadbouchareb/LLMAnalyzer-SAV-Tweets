from pathlib import Path
import time
import shutil

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
        /* ==================== FOND GLOBAL ==================== */
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .main,
        section[data-testid="stMain"] > div,
        .stApp {
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #BFDBFE 100%) !important;
            background-attachment: fixed !important;
        }
        
        /* Pattern décoratif subtil */
        [data-testid="stAppViewContainer"]::before {
            content: "" !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background-image: 
                radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(37, 99, 235, 0.03) 0%, transparent 50%) !important;
            pointer-events: none !important;
            z-index: 0 !important;
        }

        /* ==================== TITRES ==================== */
        h1, h2, h3, h4, h5, h6 {
            color: #1E40AF !important;
            font-weight: 800 !important;
            margin-bottom: 0.75rem !important;
            text-shadow: 0 2px 4px rgba(30, 64, 175, 0.1) !important;
            letter-spacing: -0.5px !important;
        }
        
        h1 {
            font-size: 2.5rem !important;
            background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        /* ==================== TEXTE GÉNÉRAL ==================== */
        p, span, div, label, input, select, textarea, a {
            color: #1E293B !important;
        }

        /* Conteneurs Markdown */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] div {
            color: #1E293B !important;
        }

        /* ==================== ROLE CARDS (ACCUEIL) ==================== */
        .role-card {
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(10px) !important;
            border: 2px solid rgba(203, 213, 225, 0.5) !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            height: 100% !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 
                0 4px 6px rgba(15, 23, 42, 0.05),
                0 10px 15px rgba(15, 23, 42, 0.1) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .role-card::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 4px !important;
            background: linear-gradient(90deg, #3B82F6, #60A5FA) !important;
            opacity: 0 !important;
            transition: opacity 0.3s ease !important;
        }

        .role-card:hover {
            transform: translateY(-5px) scale(1.02) !important;
            border-color: #3B82F6 !important;
            box-shadow: 
                0 20px 25px rgba(59, 130, 246, 0.15),
                0 10px 10px rgba(59, 130, 246, 0.1) !important;
        }
        
        .role-card:hover::before {
            opacity: 1 !important;
        }

        .role-card .card-icon {
            width: 60px !important;
            height: 60px !important;
            border-radius: 16px !important;
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%) !important;
            color: #2563EB !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 2rem !important;
            margin-bottom: 1.5rem !important;
            box-shadow: inset 0 2px 4px rgba(255, 255, 255, 0.8) !important;
            border: 1px solid #BFDBFE !important;
        }

        .role-card h3 {
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
            color: #1E40AF !important;
            font-size: 1.5rem !important;
            font-weight: 800 !important;
        }

        .role-card p {
            font-size: 1rem !important;
            line-height: 1.6 !important;
            color: #475569 !important;
            margin-bottom: 2rem !important;
            min-height: 80px !important;
        }

        .role-card .card-button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 12px !important;
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            text-decoration: none !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
            width: 100% !important;
        }

        .role-card .card-button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3) !important;
        }

        /* ==================== BOUTONS GÉNÉRAUX ==================== */
        .stButton button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.85rem 1.75rem !important;
            font-weight: 700 !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35) !important;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%) !important;
            transform: translateY(-3px) !important;
        }

        /* Cacher la sidebar */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* ==================== FILE UPLOADER ==================== */
        [data-testid="stFileUploader"] {
            background-color: rgba(255, 255, 255, 0.6) !important;
            border-radius: 16px !important;
            padding: 1rem !important;
        }
        
        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        }
        
        [data-testid="stFileUploader"] section > div {
            color: #475569 !important;
        }
        
        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
        }
        
        [data-testid="stFileUploader"] .uploadedFile {
            background-color: #F1F5F9 !important;
            color: #1E293B !important;
            border-radius: 8px !important;
        }

        /* ==================== CODE BLOCKS ==================== */
        /* Ciblage large pour couvrir toutes les versions de Streamlit */
        [data-testid="stCodeBlock"],
        [data-testid="stCode"],
        .stCode,
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCode"] pre,
        .stCode pre {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        /* Ciblage du code à l'intérieur */
        [data-testid="stCodeBlock"] code,
        [data-testid="stCode"] code,
        .stCode code,
        [data-testid="stCodeBlock"] span,
        [data-testid="stCode"] span,
        .stCode span {
            color: #1E40AF !important;
            background-color: transparent !important;
            background: transparent !important;
            font-family: 'Consolas', 'Monaco', monospace !important;
            text-shadow: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = get_cfg()

# Vérifier si un fichier a été uploadé récemment (lire last_dataset.txt)
data_dir = Path(__file__).resolve().parents[1] / "data"
last_dataset_file = data_dir / "last_dataset.txt"

# Déterminer le chemin CSV actif
if last_dataset_file.exists():
    with open(last_dataset_file, "r", encoding="utf-8") as f:
        uploaded_csv_path = f.read().strip()
    if uploaded_csv_path and Path(uploaded_csv_path).exists():
        csv_path = uploaded_csv_path
    else:
        # Fallback aux candidats par défaut
        candidates = [
            Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
            Path(__file__).resolve().parents[1] / "tweets_scored_llm.csv",
        ]
        csv_path = next((str(p) for p in candidates if p.exists()), cfg["paths"]["csv_path"])
else:
    # Fallback aux candidats par défaut
    candidates = [
        Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
        Path(__file__).resolve().parents[1] / "tweets_scored_llm.csv",
    ]
    csv_path = next((str(p) for p in candidates if p.exists()), cfg["paths"]["csv_path"])

# Permettre la modification manuelle via sidebar
csv_path = st.sidebar.text_input("Chemin CSV", value=csv_path, key="csv_path")
cfg["paths"]["csv_path"] = csv_path

if Path(csv_path).exists():
    st.sidebar.success(f"Fichier CSV détecté ✅\n{csv_path}")
else:
    st.sidebar.warning(
        "CSV introuvable.\n"
        "Assure-toi qu'il est disponible hors ligne (OneDrive → Toujours conserver sur cet appareil) "
        "ou indique un chemin absolu valide."
    )

st.markdown(
    """
    <div style="padding:2.5rem 2rem;background:linear-gradient(120deg,#E0E7FF,#C7D2FE);border-radius:18px;margin-bottom:2rem;border:1px solid rgba(70, 82, 255, 0.2);">
        <h1 style="margin-bottom:0.5rem;color:#1E293B;">SAV Tweets — Maquettes</h1>
        <p style="font-size:1.05rem;max-width:640px;color:#334155;">
            Choisis la vue adaptée à ton rôle pour explorer les tweets scorés, suivre les alertes ou traiter les dossiers.
        </p>
        <div style="font-size:0.9rem;opacity:0.8;color:#475569;">
            Source courante&nbsp;: <code style="background:#F1F5F9;padding:0.2rem 0.5rem;border-radius:4px;color:#1E293B;">{csv}</code>
        </div>
    </div>
    """.format(csv=csv_path),
    unsafe_allow_html=True,
)

# Section Upload de fichiers (LLM-Tweet-Pipeline - Traitement LLM)
st.markdown("### 📤 Importer un nouveau fichier")
st.markdown("Uploadez le fichier CSV issu du traitement LLM (`LLM-Tweet-Pipeline`) pour mettre à jour les données.")

uploaded_file = st.file_uploader(
    "Sélectionnez votre fichier CSV",
    type=["csv"],
    help="Fichier CSV généré par l'application LLM-Tweet-Pipeline (dossier : C:\\Users\\hallo\\OneDrive\\Bureau\\IA Free Mobile\\LLM-Tweet-Pipeline)",
    key="csv_uploader"
)

if uploaded_file is not None:
    # Créer le dossier uploads s'il n'existe pas
    uploads_dir = Path(__file__).resolve().parents[1] / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Créer un nom de fichier unique avec timestamp
    timestamp = int(time.time())
    file_name = f"upload_{timestamp}_{uploaded_file.name}"
    file_path = uploads_dir / file_name
    
    # Sauvegarder le fichier
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Mettre à jour le chemin dans last_dataset.txt
    last_dataset_file = data_dir / "last_dataset.txt"
    with open(last_dataset_file, "w", encoding="utf-8") as f:
        f.write(str(file_path))
    
    st.success(f"✅ Fichier uploadé avec succès : `{file_name}`")
    st.info("✨ Le nouveau fichier est maintenant actif ! Vous pouvez accéder aux différentes vues ci-dessous.")
    
    # Afficher le chemin complet pour confirmation
    st.code(str(file_path), language="text")

st.divider()
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
