import streamlit as st

st.set_page_config(page_title="SAV Tweets — Bienvenue", page_icon="💬", layout="wide")

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

        /* ==================== BOUTONS ==================== */
        .stButton button,
        [data-testid="stButton"] button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 1rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            box-shadow: 
                0 6px 20px rgba(59, 130, 246, 0.35),
                0 3px 10px rgba(37, 99, 235, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 
                0 10px 30px rgba(37, 99, 235, 0.45),
                0 5px 15px rgba(30, 64, 175, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        }

        /* Force le texte blanc à l'intérieur du bouton */
        .stButton button p,
        .stButton button span,
        .stButton button div {
            color: #FFFFFF !important;
        }

        /* Cacher la sidebar */
        [data-testid="stSidebar"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 4rem 3rem;
        text-align: center;
        margin-top: 4rem;
        border: 2px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.15);
    ">
        <h1 style="font-size: 3.5rem; margin-bottom: 1.5rem; background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Bienvenue sur SAV Tweets
        </h1>
        <p style="font-size: 1.25rem; color: #475569; max-width: 700px; margin: 0 auto 2.5rem auto; line-height: 1.6; font-weight: 500;">
            Votre plateforme d'analyse et de gestion du support client propulsée par l'IA.<br>
            Analysez les sentiments, pilotez vos KPI et accélérez les réponses clients.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")
st.write("")

cols = st.columns([2, 3, 2])
with cols[1]:
    if st.button("Accéder à l’application", use_container_width=True, type="primary"):
        st.switch_page("pages/0_Accueil.py")

st.markdown(
    """
    <div style="position: fixed; bottom: 20px; width: 100%; text-align: center; color: #64748B; font-size: 0.9rem;">
        © 2024 SAV Tweets — Propulsé par Streamlit & IA
    </div>
    """,
    unsafe_allow_html=True
)
