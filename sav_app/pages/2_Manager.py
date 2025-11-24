# --- pages/2_Manager.py ------------------------------------------------------
# Tableau de bord Manager — Vue simplifiée (prod-ready)
# Lit le même fichier que l'Agent SAV via lib.data.load_df
# et retombe proprement en mode démo si le CSV / Excel renommé est illisible.
# ------------------------------------------------------------------------------

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
    AGGRID_OK = True
except ImportError:
    AGGRID_OK = False

# --------------------------------------------------------------------
# Thème Altair clair (fond blanc, axes gris, pas de bleu fluo)
# --------------------------------------------------------------------
def custom_light_theme():
    return {
        'config': {
            'view': {'continuousWidth': 400, 'continuousHeight': 300, 'stroke': None},
            'background': 'white',
            'axis': {
                'labelColor': '#1E293B',
                'titleColor': '#1E40AF',
                'gridColor': '#E2E8F0',
                'domainColor': '#CBD5E1',
                'tickColor': '#CBD5E1',
                'labelFontSize': 11,
                'titleFontSize': 12,
                'titleFontWeight': 600
            },
            'legend': {
                'labelColor': '#1E293B',
                'titleColor': '#1E40AF',
                'labelFontSize': 11,
                'titleFontSize': 12
            },
            'title': {
                'color': '#1E40AF',
                'fontSize': 14,
                'fontWeight': 700
            },
            'mark': {'tooltip': True}
        }
    }

alt.themes.register('custom_light', custom_light_theme)
alt.themes.enable('custom_light')
alt.data_transformers.disable_max_rows()

# ------------------------------------------------------------------------------
# Helper pour tableaux blancs (compatible toutes versions de pandas)
# ------------------------------------------------------------------------------
def show_white_table(df: pd.DataFrame, height: int = 320, key: str = None):
    """Affiche un DataFrame en mode tableau blanc (style Agent SAV) via AgGrid."""
    # Supprimer les colonnes dupliquées si elles existent
    df = df.loc[:, ~df.columns.duplicated()]
    if AGGRID_OK:
        # Configuration AgGrid pour matcher le style Agent
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            resizable=True, 
            wrapText=True, 
            autoHeight=True, 
            sortable=True, 
            filter=True
        )
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gb.configure_selection("single", use_checkbox=False)
        
        # Style ligne sélectionnée
        row_style = JsCode("""
        function(params) {
          if (params.node && params.node.isSelected()) {
            return { 'backgroundColor': 'rgba(255,255,255,0.07)' };
          }
          return {};
        }
        """)
        
        # Essayer de configurer getRowId si une colonne ID existe
        id_col = None
        for c in ["tweet_id", "ID", "id"]:
            if c in df.columns:
                id_col = c
                break
        
        try:
            if id_col:
                gb.configure_grid_options(getRowId=JsCode(f"function(p){{return p.data.{id_col};}}"), rowStyle=row_style)
            else:
                gb.configure_grid_options(rowStyle=row_style)
            allow_unsafe = True
        except Exception:
            allow_unsafe = False

        # Générer une clé unique si non fournie
        if not key:
            import uuid
            key = f"aggrid_{uuid.uuid4()}"

        AgGrid(
            df,
            gridOptions=gb.build(),
            height=height,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=allow_unsafe,
            theme='light',
            key=key,
        )
    else:
        # Fallback : st.dataframe standard (le CSS injecté s'occupera du style)
        st.dataframe(df, use_container_width=True, height=height)

# ------------------------------------------------------------------------------
# Rendre importable lib/* même si on lance DIRECT cette page
# (Ex: python -m streamlit run pages/2_Manager.py)
# ------------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]  # dossier racine (ex: sav_app)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from lib.data import (
    load_df,  # loader existant
    normalize_status_empty,
    filter_by_status_like_agent,
    count_open_like_agent,
    STATUS_OPTIONS,
    apply_edits,
)
from lib.state import get_cfg
from lib.ui import inject_style, set_container_wide, inject_sticky_css, hide_sidebar

# ------------------------------------------------------------------------------
# Config Streamlit (large + thème custom)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tableau de bord Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_sidebar()
inject_style()
set_container_wide()
inject_sticky_css()
cfg = get_cfg()

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
        
        /* Labels de formulaire */
        label[data-baseweb="label"],
        .stSelectbox label,
        .stTextInput label,
        .stDateInput label,
        .stSlider label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] p {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        /* ==================== ICÔNES ET SVG ==================== */
        svg,
        [data-testid^="stSelectbox"] svg,
        [data-testid^="stTextInput"] svg,
        [data-testid^="stSlider"] svg,
        [data-testid^="stDateInput"] svg,
        [data-testid^="stAlert"] svg,
        [data-testid^="stTab"] svg,
        [data-testid^="stDataFrame"] svg,
        [data-testid="stExpander"] svg {
            color: #475569 !important;
            fill: #475569 !important;
        }
        
        /* Icônes dans les boutons : blanc pour contraste */
        [data-testid^="stButton"] svg,
        .stButton button svg,
        button svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        /* Cartes personnalisées : design glassmorphism moderne */
        .ma-card {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            color: #1E293B !important;
            border: 2px solid rgba(203, 213, 225, 0.5) !important;
            border-radius: 20px !important;
            box-shadow: 
                0 10px 40px rgba(59, 130, 246, 0.1),
                0 4px 16px rgba(15, 23, 42, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
            padding: 2rem !important;
            margin-bottom: 1.5rem !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
        }
        
        .ma-card::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            height: 4px !important;
            background: linear-gradient(90deg, #3B82F6 0%, #60A5FA 50%, #3B82F6 100%) !important;
            background-size: 200% 100% !important;
        }
        
        .ma-card:hover {
            transform: translateY(-2px) !important;
            box-shadow: 
                0 15px 50px rgba(59, 130, 246, 0.15),
                0 6px 20px rgba(15, 23, 42, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
            border-color: rgba(59, 130, 246, 0.4) !important;
        }
        
        .ma-card h3 {
            color: #1E40AF !important;
            margin: 0 0 0.75rem 0 !important;
            font-weight: 800 !important;
            font-size: 1.25rem !important;
        }

        /* ==================== INPUTS ET SELECTBOX - Design unifié ==================== */
        /* Conteneurs des inputs uniquement */
        .stTextInput,
        .stDateInput,
        .stTextArea,
        .stSelectbox,
        [data-testid="stTextInput"],
        [data-testid="stDateInput"],
        [data-testid="stTextArea"],
        [data-testid="stSelectbox"] {
            background: transparent !important;
            border: none !important;
        }
        
        /* Inputs (TextInput, DateInput, TextArea) - Fond blanc comme les selectbox */
        .stTextInput input,
        .stDateInput input,
        .stTextArea textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTextArea"] textarea {
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(8px) !important;
            color: #1E293B !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1rem !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            min-height: 42px !important;
            box-shadow: 
                0 2px 8px rgba(59, 130, 246, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
            transition: all 0.3s ease !important;
            outline: none !important;
        }
        
        /* CRITIQUE - Enlever TOUTES les bordures noires du DateInput et TextInput */
        [data-testid="stDateInput"] > div,
        [data-testid="stDateInput"] > div > div,
        [data-testid="stTextInput"] > div,
        [data-testid="stTextInput"] > div > div,
        .stDateInput > div,
        .stDateInput > div > div,
        .stTextInput > div,
        .stTextInput > div > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Selectbox - Design unifié avec les inputs */
        .stSelectbox div[data-baseweb="select"],
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        div[data-baseweb="select"] {
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(8px) !important;
            border: none !important;
            border-radius: 12px !important;
            min-height: 42px !important;
            box-shadow: 
                0 2px 8px rgba(59, 130, 246, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
            transition: all 0.3s ease !important;
        }
        
        /* Conteneur interne du selectbox - SANS bordures ni barres */
        div[data-baseweb="select"] > div {
            background: transparent !important;
            border: none !important;
            border-bottom: none !important;
            border-top: none !important;
            box-shadow: none !important;
            outline: none !important;
            padding: 0.6rem 1rem !important;
            min-height: auto !important;
        }
        
        /* Enlever TOUS les pseudo-éléments et bordures internes */
        /* MODIFICATION: On ne cache plus les pseudo-éléments du select simple, on enlève juste les bordures */
        div[data-baseweb="select"] *::before,
        div[data-baseweb="select"] *::after {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        /* Pour le multiselect, on reste agressif pour la barre noire */
        [data-testid="stMultiSelect"] *::before,
        [data-testid="stMultiSelect"] *::after {
            display: none !important;
            content: none !important;
            border: none !important;
        }
        
        /* Enlever TOUTES les bordures des multiselect - CORRECTION BARRE NOIRE */
        [data-testid="stMultiSelect"] > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stMultiSelect"] div[data-baseweb] {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        
        /* Sécurité supplémentaire pour la barre noire verticale (curseur) - UNIQUEMENT MULTISELECT */
        [data-testid="stMultiSelect"] input,
        [data-testid="stMultiSelect"] input:focus,
        [data-testid="stMultiSelect"] [role="combobox"] {
            border: none !important;
            border-left: none !important;
            border-right: none !important;
            border-width: 0 !important;
            box-shadow: none !important;
            outline: none !important;
            caret-color: transparent !important; /* Cache le curseur noir */
            background: transparent !important;
        }
        
        /* Pour le Selectbox simple, on garde le curseur normal pour éviter de cacher le texte si lié, 
           mais on enlève les bordures */
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] input:focus,
        [data-testid="stSelectbox"] [role="combobox"],
        div[data-baseweb="select"] input {
            border: none !important;
            border-left: none !important;
            border-right: none !important;
            border-width: 0 !important;
            box-shadow: none !important;
            outline: none !important;
            background: transparent !important;
            /* On n'impose PAS caret-color transparent ici pour tester si ça bloque le texte */
        }

        /* SUPPRESSION RADICALE DE TOUTES LES BORDURES INTERNES */
        [data-testid="stMultiSelect"] > div > div *,
        [data-testid="stMultiSelect"] > div > div *::before,
        [data-testid="stMultiSelect"] > div > div *::after {
            border-color: transparent !important;
            border-width: 0 !important;
            border-style: none !important;
        }

        /* Supprimer tout artefact de bordure sur les conteneurs internes */
        [data-testid="stMultiSelect"] div[class*="control"],
        [data-testid="stMultiSelect"] div[class*="value-container"] {
            border: none !important;
            box-shadow: none !important;
        }

        /* FORCER L'AFFICHAGE DE LA VALEUR SÉLECTIONNÉE (2024, 2025...) */
        div[data-baseweb="select"] > div:first-child {
            color: #1E293B !important;
            -webkit-text-fill-color: #1E293B !important;
            opacity: 1 !important;
            visibility: visible !important;
            z-index: 100 !important;
            display: flex !important;
        }
        
        /* CIBLAGE SPÉCIFIQUE DU TEXTE DANS LE SELECT - FIX ULTIME */
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[value] {
            color: #1E293B !important;
            -webkit-text-fill-color: #1E293B !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* CIBLAGE PAR ATTRIBUT VALUE (comme vu dans le DOM) */
        div[data-baseweb="select"] div[value] {
            color: #1E293B !important;
            -webkit-text-fill-color: #1E293B !important;
            opacity: 1 !important;
            visibility: visible !important;
            z-index: 1002 !important;
            position: relative !important;
            text-shadow: none !important;
        }

        /* ==================== CORRECTIF FINAL V3 (FORCE VISIBILITY) ==================== */
        /* 1. On cache le curseur et le texte de l'input (pour éviter doublon ou curseur) */
        [data-testid="stSelectbox"] input {
            caret-color: transparent !important;
            color: transparent !important;
            -webkit-text-fill-color: transparent !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        /* 2. On FORCE l'affichage de la div qui contient la valeur (2024, 2025...) */
        /* Streamlit cache parfois cette div quand l'input a le focus, on l'empêche */
        div[data-baseweb="select"] div[class*="singleValue"],
        div[data-baseweb="select"] div[class*="SingleValue"],
        div[data-baseweb="select"] div[value] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            color: #1E293B !important;
            -webkit-text-fill-color: #1E293B !important;
            z-index: 1005 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            bottom: 0 !important;
            right: 0 !important;
            align-items: center !important;
            padding-left: 1rem !important; /* Alignement texte */
            pointer-events: none !important; /* Clics passent au travers vers l'input */
        }

        /* 3. On s'assure que le conteneur parent est en relative pour le positionnement absolute */
        div[data-baseweb="select"] > div:first-child {
            position: relative !important;
        }

        /* 4. Multiselect (Sentiment) - On garde ce qui marche */
        [data-testid="stMultiSelect"] input {
            caret-color: transparent !important;
        }
        
        /* ==================== BOUTONS ==================== */
        .stButton button,
        [data-testid="stButton"] button,
        button[kind="primary"],
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"],
        .stDownloadButton button {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.85rem 1.75rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            box-shadow: 
                0 6px 20px rgba(59, 130, 246, 0.35),
                0 3px 10px rgba(37, 99, 235, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-transform: none !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .stButton button::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
            transition: left 0.5s !important;
        }
        
        .stButton button:hover::before {
            left: 100% !important;
        }
        
        .stButton button:hover,
        [data-testid="stButton"] button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover,
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%) !important;
            box-shadow: 
                0 10px 30px rgba(37, 99, 235, 0.45),
                0 5px 15px rgba(30, 64, 175, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-3px) scale(1.02) !important;
        }
        
        /* Bouton actif/cliqué */
        .stButton button:active,
        [data-testid="stButton"] button:active,
        button[kind="primary"]:active,
        button[kind="secondary"]:active {
            background: linear-gradient(135deg, #1E40AF 0%, #1E3A8A 100%) !important;
            transform: translateY(-1px) scale(0.98) !important;
            box-shadow: 
                0 3px 10px rgba(37, 99, 235, 0.4),
                inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Boutons désactivés (bouton actif/sélectionné) */
        .stButton button:disabled,
        [data-testid="stButton"] button:disabled,
        button[kind="primary"]:disabled,
        button[kind="secondary"]:disabled {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            color: #1E40AF !important;
            opacity: 1 !important;
            box-shadow: 
                0 4px 16px rgba(30, 64, 175, 0.2),
                inset 0 0 0 2px #3B82F6 !important;
            cursor: default !important;
            transform: none !important;
            border: 2px solid #3B82F6 !important;
        }
        
        /* Texte dans les boutons désactivés (actifs) */
        .stButton button:disabled p,
        .stButton button:disabled span,
        .stButton button:disabled div,
        button:disabled p,
        button:disabled span {
            color: #1E40AF !important;
            font-weight: 800 !important;
            text-shadow: none !important;
        }
        
        /* Texte dans les boutons normaux */
        .stButton button p,
        .stButton button span,
        .stButton button div,
        button p,
        button span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        }

        /* ==================== TABS ==================== */
        [data-testid="stTab"] {
            background-color: #FFFFFF !important;
            border-bottom: 2px solid #CBD5E1 !important;
            border-radius: 12px 12px 0 0 !important;
        }
        [data-testid="stTab"] button {
            color: #64748B !important;
            background-color: transparent !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stTab"] button:hover {
            color: #475569 !important;
            background-color: #F1F5F9 !important;
        }
        [data-testid="stTab"] button[data-selected="true"] {
            color: #2563EB !important;
            border-bottom: 3px solid #2563EB !important;
            background-color: #EFF6FF !important;
        }

        /* ==================== ALERTES ==================== */
        [data-testid^="stAlert"] {
            background: linear-gradient(135deg, rgba(239, 246, 255, 0.9), rgba(219, 234, 254, 0.8)) !important;
            backdrop-filter: blur(8px) !important;
            color: #1E293B !important;
            border: 2px solid rgba(147, 197, 253, 0.5) !important;
            border-left: 6px solid #3B82F6 !important;
            border-radius: 14px !important;
            padding: 1.25rem 1.5rem !important;
            box-shadow: 
                0 4px 16px rgba(59, 130, 246, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        [data-testid^="stAlert"]::before {
            content: "" !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.05), transparent 70%) !important;
            pointer-events: none !important;
        }
        
        [data-testid^="stAlert"] svg {
            color: #3B82F6 !important;
            fill: #3B82F6 !important;
            filter: drop-shadow(0 2px 4px rgba(59, 130, 246, 0.3)) !important;
        }
        [data-testid="stAlert"] p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }

        /* ==================== TABLEAUX/DATAFRAMES ==================== */
        [data-testid="stDataFrame"],
        .stDataFrame {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            color: #1E293B !important;
            border: 2px solid rgba(203, 213, 225, 0.5) !important;
            border-radius: 16px !important;
            box-shadow: 
                0 8px 32px rgba(59, 130, 246, 0.12),
                0 4px 16px rgba(15, 23, 42, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
            overflow: hidden !important;
        }
        [data-testid="stDataFrame"] th,
        .stDataFrame th {
            background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%) !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            border-bottom: 3px solid #60A5FA !important;
            padding: 1rem 0.75rem !important;
            font-size: 0.85rem !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        }
        [data-testid="stDataFrame"] td,
        .stDataFrame td {
            border-bottom: 1px solid rgba(226, 232, 240, 0.6) !important;
            padding: 0.85rem 0.75rem !important;
            color: #1E293B !important;
            font-weight: 500 !important;
            transition: background-color 0.2s ease !important;
        }
        [data-testid="stDataFrame"] tr:hover,
        .stDataFrame tr:hover {
            background: linear-gradient(90deg, rgba(239, 246, 255, 0.8), rgba(219, 234, 254, 0.6)) !important;
            transform: scale(1.001) !important;
        }
        [data-testid="stDataFrame"] tbody tr:nth-child(even),
        .stDataFrame tbody tr:nth-child(even) {
            background-color: rgba(248, 250, 252, 0.5) !important;
        }

        /* ==================== SLIDERS ==================== */
        [data-testid="stSlider"],
        .stSlider {
            background-color: transparent !important;
            padding: 0.5rem 0 !important;
        }
        
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] p,
        [data-testid="stSlider"] span,
        [data-testid="stSlider"] div,
        [data-testid="stSlider"] div,
        .stSlider label,
        .stSlider p,
        .stSlider span,
        .stSlider div {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSlider"] .st-emotion-cache-1gulkj5,
        .st-emotion-cache-1gulkj5 {
            color: #1E293B !important;
        }
        
        /* Track actif */
        [data-testid="stSlider"] .rc-slider-track,
        .stSlider .rc-slider-track {
            background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%) !important;
            height: 6px !important;
        }
        
        /* Poignée */
        [data-testid="stSlider"] .rc-slider-handle,
        .stSlider .rc-slider-handle {
            border: 3px solid #2563EB !important;
            background-color: #FFFFFF !important;
            width: 20px !important;
            height: 20px !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
        }
        
        [data-testid="stSlider"] .rc-slider-handle:hover,
        .stSlider .rc-slider-handle:hover {
            box-shadow: 0 3px 12px rgba(37, 99, 235, 0.4) !important;
        }
        
        /* Rail de fond */
        [data-testid="stSlider"] .rc-slider-rail,
        .stSlider .rc-slider-rail {
            background-color: #CBD5E1 !important;
            height: 6px !important;
        }
        
        /* Marques de valeur */
        [data-testid="stSlider"] .rc-slider-mark-text,
        .stSlider .rc-slider-mark-text {
            color: #475569 !important;
            font-weight: 500 !important;
        }

        /* ==================== EXPANDER ==================== */
        [data-testid="stExpander"],
        .streamlit-expanderHeader {
            background: linear-gradient(to bottom, #FFFFFF, #F8FAFC) !important;
            border: 2px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        }
        
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span,
        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span {
            color: #0F172A !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            padding: 1rem !important;
        }

        /* ==================== SIDEBAR ==================== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
            border-right: 2px solid #CBD5E1 !important;
        }
        
        /* ==================== AgGrid ==================== */
        .ag-theme-streamlit,
        .ag-theme-alpine,
        .ag-theme-material {
            --ag-background-color: #FFFFFF !important;
            --ag-foreground-color: #1E293B !important;
            --ag-header-background-color: #FFFFFF !important;
            --ag-header-foreground-color: #1E40AF !important;
            --ag-odd-row-background-color: #F8FAFC !important;
            --ag-row-hover-color: #EFF6FF !important;
            --ag-selected-row-background-color: #DBEAFE !important;
            --ag-border-color: #CBD5E1 !important;
            --ag-secondary-border-color: #E2E8F0 !important;
            background: #FFFFFF !important;
            backdrop-filter: blur(10px) !important;
            color: #1E293B !important;
            border: 2px solid rgba(203, 213, 225, 0.5) !important;
            border-radius: 16px !important;
            box-shadow: 
                0 10px 40px rgba(59, 130, 246, 0.15),
                0 4px 20px rgba(15, 23, 42, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
            overflow: hidden !important;
        }
        
        /* Forcer le fond blanc pour TOUS les éléments du tableau */
        .ag-theme-streamlit .ag-root-wrapper,
        .ag-theme-alpine .ag-root-wrapper,
        .ag-theme-material .ag-root-wrapper,
        .ag-theme-streamlit .ag-root,
        .ag-theme-alpine .ag-root,
        .ag-theme-material .ag-root,
        .ag-theme-streamlit .ag-body-viewport,
        .ag-theme-alpine .ag-body-viewport,
        .ag-theme-material .ag-body-viewport,
        .ag-theme-streamlit .ag-center-cols-viewport,
        .ag-theme-alpine .ag-center-cols-viewport,
        .ag-theme-material .ag-center-cols-viewport,
        .ag-theme-streamlit .ag-center-cols-container,
        .ag-theme-alpine .ag-center-cols-container,
        .ag-theme-material .ag-center-cols-container {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
        }
        
        .ag-theme-streamlit .ag-header,
        .ag-theme-alpine .ag-header,
        .ag-theme-material .ag-header {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(239, 246, 255, 0.95)) !important;
            color: #1E40AF !important;
            font-weight: 800 !important;
            border-bottom: 3px solid #3B82F6 !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
        }
        
        .ag-theme-streamlit .ag-header-cell-text,
        .ag-theme-alpine .ag-header-cell-text,
        .ag-theme-material .ag-header-cell-text {
            color: #1E40AF !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.8px !important;
            text-shadow: none !important;
        }
        
        .ag-theme-streamlit .ag-cell,
        .ag-theme-alpine .ag-cell,
        .ag-theme-material .ag-cell {
            color: #1E293B !important;
            font-weight: 500 !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        
        .ag-theme-streamlit .ag-row,
        .ag-theme-alpine .ag-row,
        .ag-theme-material .ag-row {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            border-color: #E2E8F0 !important;
        }
        
        .ag-theme-streamlit .ag-row-even,
        .ag-theme-alpine .ag-row-even,
        .ag-theme-material .ag-row-even {
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
        }
        
        .ag-theme-streamlit .ag-row-odd,
        .ag-theme-alpine .ag-row-odd,
        .ag-theme-material .ag-row-odd {
            background: #F8FAFC !important;
            background-color: #F8FAFC !important;
        }
        
        .ag-theme-streamlit .ag-row:hover,
        .ag-theme-alpine .ag-row:hover,
        .ag-theme-material .ag-row:hover {
            background: linear-gradient(90deg, rgba(239, 246, 255, 0.9), rgba(219, 234, 254, 0.7)) !important;
            transform: scale(1.001) !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1) !important;
        }
        
        .ag-theme-streamlit .ag-row-selected,
        .ag-theme-alpine .ag-row-selected,
        .ag-theme-material .ag-row-selected {
            background: linear-gradient(90deg, rgba(219, 234, 254, 0.8), rgba(191, 219, 254, 0.6)) !important;
            border-left: 4px solid #3B82F6 !important;
        }
        
        .ag-theme-streamlit .ag-icon,
        .ag-theme-alpine .ag-icon,
        .ag-theme-material .ag-icon {
            color: #475569 !important;
        }
        
        .ag-theme-streamlit .ag-header .ag-icon,
        .ag-theme-alpine .ag-header .ag-icon,
        .ag-theme-material .ag-header .ag-icon {
            color: #3B82F6 !important;
        }
        
        /* ==================== GRAPHIQUES ALTAIR ET VISUELS ==================== */
        /* FORCER fond blanc/transparent pour TOUS les éléments Vega */
        .vega-embed,
        .vega-embed canvas,
        .vega-embed svg,
        .vega-embed .vega-actions,
        [data-testid="stVegaLiteChart"],
        [data-testid="stVegaLiteChart"] canvas,
        [data-testid="stVegaLiteChart"] svg,
        .marks,
        svg.marks,
        g.mark-group,
        g.background,
        rect.background,
        .vega-embed .background,
        svg rect.background,
        svg > rect:first-child {
            background-color: transparent !important;
            background: transparent !important;
            fill: transparent !important;
        }
        
        /* Fond blanc pour les conteneurs de graphiques */
        [data-testid="stVegaLiteChart"] {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            box-shadow: 
                0 4px 16px rgba(59, 130, 246, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
        }
        
        /* Empêcher les graphiques de déborder */
        [data-testid="stVegaLiteChart"] .vega-embed,
        [data-testid="stVegaLiteChart"] canvas,
        [data-testid="stVegaLiteChart"] svg {
            max-width: 100% !important;
            width: 100% !important;
            height: auto !important;
        }
        
        /* FORCER tous les SVG à ne pas avoir de fond noir */
        [data-testid="stVegaLiteChart"] svg,
        .vega-embed svg {
            background: none !important;
            background-color: transparent !important;
        }
        
        /* Cibler spécifiquement le rectangle de fond Vega */
        svg > g > rect:first-child,
        svg g[class*="background"] rect,
        svg rect[fill="#333333"],
        svg rect[fill="#1f1f1f"],
        svg rect[fill="black"],
        svg rect[fill="#000000"] {
            fill: transparent !important;
            fill-opacity: 0 !important;
        }
        
        /* Texte et axes des graphiques en couleur visible */
        .vega-embed text,
        [data-testid="stVegaLiteChart"] text,
        .vega-embed .role-axis-label,
        .vega-embed .role-axis-title,
        .vega-embed .role-legend-label,
        .vega-embed .role-legend-title {
            fill: #1E293B !important;
            color: #1E293B !important;
        }
        
        /* Grilles des graphiques */
        .vega-embed .role-axis-grid,
        [data-testid="stVegaLiteChart"] .role-axis-grid {
            stroke: #E2E8F0 !important;
        }
        
        /* Lignes d'axes */
        .vega-embed .role-axis-domain,
        [data-testid="stVegaLiteChart"] .role-axis-domain {
            stroke: #CBD5E1 !important;
        }
        
        /* Actions Vega */
        .vega-embed .vega-actions a {
            color: #3B82F6 !important;
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 6px !important;
            padding: 0.25rem 0.5rem !important;
        }
        
        .vega-embed .vega-actions a:hover {
            background: rgba(59, 130, 246, 0.1) !important;
            border-color: #3B82F6 !important;
        }
        
        /* ==================== CORRECTIONS SUPPLÉMENTAIRES ==================== */
        
        /* Tous les éléments Streamlit avec classes emotion-cache */
        [class*="st-emotion-cache"] {
            color: #1E293B !important;
        }
        
        /* Widget Labels spécifiques */
        .st-emotion-cache-1gulkj5,
        .st-emotion-cache-16txtl3,
        .st-emotion-cache-1y4p8pa {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        
        /* Container et colonnes */
        .element-container,
        .row-widget,
        .stHorizontalBlock {
            background-color: transparent !important;
        }
        
        /* Métriques */
        [data-testid="stMetric"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #1E40AF !important;
            font-weight: 800 !important;
        }
        
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(239, 246, 255, 0.8)) !important;
            backdrop-filter: blur(10px) !important;
            border: 2px solid rgba(147, 197, 253, 0.4) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            box-shadow: 
                0 6px 24px rgba(59, 130, 246, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
            position: relative !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stMetric"]::before {
            content: "" !important;
            position: absolute !important;
            top: -50% !important;
            right: -50% !important;
            width: 200% !important;
            height: 200% !important;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.08), transparent 50%) !important;
            pointer-events: none !important;
        }
        
        [data-testid="stMetric"]:hover {
            transform: translateY(-4px) !important;
            box-shadow: 
                0 10px 32px rgba(59, 130, 246, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 1) !important;
            border-color: rgba(59, 130, 246, 0.6) !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            background: linear-gradient(135deg, #1E40AF, #3B82F6) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        
        /* Info, Warning, Error boxes */
        .stAlert,
        [data-baseweb="notification"] {
            background: linear-gradient(to right, #EFF6FF, #DBEAFE) !important;
            color: #1E293B !important;
            border: 2px solid #93C5FD !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
        }
        
        /* DateInput Calendar - FORCER BLANC PARTOUT */
        [data-baseweb="calendar"],
        [data-baseweb="calendar"] *,
        [data-baseweb="calendar"] > div,
        [data-baseweb="calendar"] div,
        [data-baseweb="calendar"] div *,
        [data-baseweb="calendar"] table,
        [data-baseweb="calendar"] tbody,
        [data-baseweb="calendar"] tr,
        [data-baseweb="calendar"] td,
        [data-baseweb="calendar"] td *,
        [data-baseweb="calendar"] td > *,
        [data-baseweb="calendar"] td div,
        .stDateInput [data-baseweb="calendar"],
        .stDateInput [data-baseweb="calendar"] *,
        [data-testid="stDateInput"] [data-baseweb="calendar"],
        [data-testid="stDateInput"] [data-baseweb="calendar"] * {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            background-image: none !important;
            color: #1E293B !important;
        }
        
        /* Supprimer les pseudo-éléments noirs */
        [data-baseweb="calendar"] *::before,
        [data-baseweb="calendar"] *::after {
            background-color: transparent !important;
            background: transparent !important;
            display: none !important;
        }
        
        [data-baseweb="calendar"] {
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.15) !important;
            padding: 1rem !important;
        }
        
        /* FORCER fond blanc sur TOUS les conteneurs internes */
        [data-baseweb="calendar"] > div,
        [data-baseweb="calendar"] > div > div,
        [data-baseweb="calendar"] [class*="CalendarContainer"],
        [data-baseweb="calendar"] [class*="CalendarHeader"],
        [data-baseweb="calendar"] [class*="Month"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
        }
        
        /* En-tête du calendrier (mois/année) */
        [data-baseweb="calendar"] header,
        [data-baseweb="calendar"] [role="heading"],
        [data-baseweb="calendar"] header *,
        [data-baseweb="calendar"] [role="heading"] * {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #1E40AF !important;
            font-weight: 700 !important;
        }
        
        /* Boutons de navigation du calendrier (flèches) - TOUJOURS VISIBLES */
        [data-baseweb="calendar"] header button,
        [data-baseweb="calendar"] header button *,
        [data-baseweb="calendar"] header button[aria-label],
        [data-baseweb="calendar"] button[aria-label*="previous"],
        [data-baseweb="calendar"] button[aria-label*="next"],
        [data-baseweb="calendar"] button[aria-label*="précédent"],
        [data-baseweb="calendar"] button[aria-label*="suivant"],
        [data-baseweb="calendar"] button[aria-label*="Previous"],
        [data-baseweb="calendar"] button[aria-label*="Next"] {
            background-color: #DBEAFE !important;
            background: #DBEAFE !important;
            color: #1E40AF !important;
            border: 2px solid #3B82F6 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: all 0.2s ease !important;
            padding: 0.5rem !important;
            min-width: 36px !important;
            min-height: 36px !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        
        /* SVG des flèches - les rendre TRÈS visibles */
        [data-baseweb="calendar"] header button svg,
        [data-baseweb="calendar"] header button svg *,
        [data-baseweb="calendar"] header button svg path,
        [data-baseweb="calendar"] button[aria-label*="previous"] svg,
        [data-baseweb="calendar"] button[aria-label*="next"] svg,
        [data-baseweb="calendar"] button[aria-label*="précédent"] svg,
        [data-baseweb="calendar"] button[aria-label*="suivant"] svg,
        [data-baseweb="calendar"] button[aria-label*="Previous"] svg,
        [data-baseweb="calendar"] button[aria-label*="Next"] svg,
        [data-baseweb="calendar"] button[aria-label*="previous"] svg *,
        [data-baseweb="calendar"] button[aria-label*="next"] svg *,
        [data-baseweb="calendar"] button[aria-label*="précédent"] svg *,
        [data-baseweb="calendar"] button[aria-label*="suivant"] svg *,
        [data-baseweb="calendar"] button[aria-label*="Previous"] svg *,
        [data-baseweb="calendar"] button[aria-label*="Next"] svg * {
            color: #1E40AF !important;
            fill: #1E40AF !important;
            stroke: #1E40AF !important;
            opacity: 1 !important;
            display: block !important;
            visibility: visible !important;
            width: 20px !important;
            height: 20px !important;
        }
        
        [data-baseweb="calendar"] header button:hover,
        [data-baseweb="calendar"] button[aria-label*="previous"]:hover,
        [data-baseweb="calendar"] button[aria-label*="next"]:hover,
        [data-baseweb="calendar"] button[aria-label*="précédent"]:hover,
        [data-baseweb="calendar"] button[aria-label*="suivant"]:hover,
        [data-baseweb="calendar"] button[aria-label*="Previous"]:hover,
        [data-baseweb="calendar"] button[aria-label*="Next"]:hover {
            background-color: #93C5FD !important;
            background: #93C5FD !important;
            color: #1E40AF !important;
            border-color: #2563EB !important;
            transform: scale(1.05) !important;
        }
        
        /* Boutons de dates - État normal */
        [data-baseweb="calendar"] td button,
        [data-baseweb="calendar"] td button *,
        [data-baseweb="calendar"] td button div,
        [data-baseweb="calendar"] [role="gridcell"] button,
        [data-baseweb="calendar"] [role="gridcell"] button *,
        [data-baseweb="calendar"] [role="gridcell"] button div {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #1E293B !important;
            font-weight: 500 !important;
            border: none !important;
            border-radius: 8px !important;
            width: 36px !important;
            height: 36px !important;
            transition: all 0.2s ease !important;
        }
        
        /* Dates au survol - FORCER fond bleu clair ABSOLUMENT PARTOUT */
        [data-baseweb="calendar"] td:has(button):hover,
        [data-baseweb="calendar"] td:has(button):hover *,
        [data-baseweb="calendar"] td:has(button):hover > *,
        [data-baseweb="calendar"] td:has(button):hover div,
        [data-baseweb="calendar"] td:has(button):hover button,
        [data-baseweb="calendar"] td:has(button):hover button *,
        [data-baseweb="calendar"] td:has(button):hover button div,
        [data-baseweb="calendar"] td button:hover,
        [data-baseweb="calendar"] td button:hover *,
        [data-baseweb="calendar"] td button:hover > *,
        [data-baseweb="calendar"] td button:hover div,
        [data-baseweb="calendar"] [role="gridcell"]:hover,
        [data-baseweb="calendar"] [role="gridcell"]:hover *,
        [data-baseweb="calendar"] [role="gridcell"] button:hover,
        [data-baseweb="calendar"] [role="gridcell"] button:hover *,
        [data-baseweb="calendar"] [role="gridcell"] button:hover div,
        [data-baseweb="calendar"] tbody tr:hover td,
        [data-baseweb="calendar"] tbody tr td:hover {
            background-color: #DBEAFE !important;
            background: #DBEAFE !important;
            background-image: none !important;
            color: #1E40AF !important;
        }
        
        /* Appliquer le style uniquement sur le bouton au survol */
        [data-baseweb="calendar"] td button:hover {
            border: 1px solid #93C5FD !important;
            transform: scale(1.05) !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Date sélectionnée */
        [data-baseweb="calendar"] td button[aria-pressed="true"],
        [data-baseweb="calendar"] [role="gridcell"] button[aria-pressed="true"],
        [data-baseweb="calendar"] td button[aria-selected="true"],
        [data-baseweb="calendar"] [role="gridcell"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4) !important;
        }
        
        /* Date aujourd'hui */
        [data-baseweb="calendar"] td button[aria-current="date"],
        [data-baseweb="calendar"] [role="gridcell"] button[aria-current="date"] {
            border: 2px solid #3B82F6 !important;
            font-weight: 700 !important;
            color: #1E40AF !important;
        }
        
        /* Dates désactivées (autres mois) */
        [data-baseweb="calendar"] td button:disabled,
        [data-baseweb="calendar"] [role="gridcell"] button:disabled {
            background-color: #F8FAFC !important;
            color: #CBD5E1 !important;
            opacity: 0.5 !important;
        }
        
        /* Sélecteurs de mois/année */
        [data-baseweb="calendar"] select,
        [data-baseweb="calendar"] select *,
        [data-baseweb="month"],
        [data-baseweb="month"] *,
        [data-baseweb="year"],
        [data-baseweb="year"] * {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #1E293B !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem !important;
            font-weight: 600 !important;
        }
        
        /* Texte dans les labels du calendrier */
        [data-baseweb="calendar"] label,
        [data-baseweb="calendar"] label *,
        [data-baseweb="calendar"] span,
        [data-baseweb="calendar"] p {
            background: transparent !important;
            color: #1E293B !important;
        }
        
        /* Checkbox et Radio */
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
            color: #334155 !important;
        }
        
        /* Correction finale pour tous les textes */
        .stApp [class*="st-"] {
            color: #1E293B !important;
        }
        
        /* Forcer la visibilité du texte des menus */
        [role="listbox"] li,
        [role="option"],
        [role="menuitem"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            font-weight: 500 !important;
        }
        
        [role="listbox"] li:hover,
        [role="option"]:hover,
        [role="menuitem"]:hover {
            background-color: #EFF6FF !important;
            color: #0F172A !important;
            font-weight: 600 !important;
        }
        
        /* ==================== DESIGN MODERNE GÉNÉRAL ==================== */
        
        /* Ombres et élévations */
        .element-container {
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }
        
        /* Scrollbar personnalisée */
        ::-webkit-scrollbar {
            width: 12px !important;
            height: 12px !important;
        }
        
        ::-webkit-scrollbar-track {
            background: #E2E8F0 !important;
            border-radius: 6px !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #CBD5E1 0%, #94A3B8 100%) !important;
            border-radius: 6px !important;
            border: 2px solid #E2E8F0 !important;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #94A3B8 0%, #64748B 100%) !important;
        }

        /* FORCER VISIBILITÉ DU TEXTE DANS LE SELECTBOX - APPROCHE GLOBALE */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            color: #1E293B !important;
            opacity: 1 !important;
        }
        
        /* Cible le texte directement */
        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stSelectbox"] div[data-baseweb="select"] div {
            color: #1E293B !important;
            visibility: visible !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Versioning des filtres pour reset robuste (clé utilisée dans les widgets ci‑dessous)
st.session_state.setdefault("m_filters_ver", 1)
ver = int(st.session_state["m_filters_ver"])

def _reset_manager_filters():
    import re
    # Supprimer les clés non versionnées courantes (au cas où)
    for k in [
        "m_tone_simple",
        "m_theme_simple",
        "m_year_simple",
        f"manager_status_{ver}",
        "m_period_simple",
    ]:
        if k in st.session_state:
            del st.session_state[k]

    # Purge défensive : supprimer d'anciennes clés versionnées (ex: m_tone_simple_1, m_period_simple_2, ...)
    for k in list(st.session_state.keys()):
        if re.match(r"^m_(tone_simple|theme_simple|year_simple|period_simple)_\d+$", k):
            del st.session_state[k]

    # Forcer de nouvelles clés au prochain rendu
    st.session_state["m_filters_ver"] = int(st.session_state.get("m_filters_ver", 1)) + 1
    st.rerun()

# ------------------------------------------------------------------------------
# Utils parsing date / nettoyage colonnes
# ------------------------------------------------------------------------------

def _parse_dt(series: pd.Series) -> pd.Series:
    """
    Rend les dates propres (timezone, formats variées).
    """
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().sum() == 0:
        # deuxième tentative style '27/10/2025 14:33'
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # drop timezone => naive
    if pd.api.types.is_datetime64tz_dtype(s):
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            try:
                s = s.dt.tz_convert(None)
            except Exception:
                pass
    return s


def _ensure_theme_list(raw):
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass
    if isinstance(raw, str) and raw.strip():
        import ast, json, re
        cleaned = raw.strip()
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(cleaned)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        matches = re.findall(r"['\"]([^'\"]+)['\"]", cleaned)
        if matches:
            return [m.strip() for m in matches if m.strip()]
        tokens = [tok.strip(" '\"") for tok in cleaned.strip("[]").split(",") if tok.strip(" '\"")]
        return tokens or [cleaned]
    return []


def _prepare_manager_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un DataFrame "brut" lu depuis tweets_scored_llm.csv
    pour qu'il ait toujours les colonnes dont le dashboard a besoin.
    """
    df = df.copy()

    # 1. ID tweet
    if "tweet_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "tweet_id"})
        else:
            df["tweet_id"] = df.index.astype(str)
    df["tweet_id"] = df["tweet_id"].astype(str)

    # 2. Texte affichable
    text_candidates = [
        "text_raw",
        "text_raw",
        "tweet",
        "text",
        "content",
        "body",
    ]
    txt_src = next((c for c in text_candidates if c in df.columns), None)
    if txt_src is not None:
        df["text_raw"] = df[txt_src].astype(str)
    else:
        df["text_raw"] = ""

    # 3. Date / timestamp
    date_candidates = [
        "created_at_dt",
        "created_at",
        "date",
        "datetime",
        "timestamp",
        "time",
        "posted_at",
        "tweet_created_at",
        "tweet_date",
        "createdAt",
        "created_at_utc",
        "date_utc",
        "date_time",
    ]
    src_date = next((c for c in date_candidates if c in df.columns), None)
    if src_date:
        df["created_at_dt"] = _parse_dt(df[src_date])
    else:
        df["created_at_dt"] = pd.Series(pd.NaT, index=df.index)

    # 4. Thèmes
    theme_src = next(
        (c for c in ("themes_list", "liste_thèmes", "liste_themes", "themes", "topics", "labels") if c in df.columns),
        None,
    )
    if theme_src:
        df["themes_list"] = df[theme_src].apply(_ensure_theme_list)
    else:
        df["themes_list"] = [[] for _ in range(len(df))]

    if "primary_label" in df.columns:
        df["theme_primary"] = df["primary_label"].fillna("").astype(str)
    else:
        alt_primary = next((c for c in ("theme", "main_theme", "main_label") if c in df.columns), None)
        if alt_primary:
            df["theme_primary"] = df[alt_primary].fillna("").astype(str)
        else:
            df["theme_primary"] = df["themes_list"].apply(
                lambda L: str(L[0]) if isinstance(L, list) and len(L) > 0 else ""
            )

    # 5. Sentiment
    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].fillna("").astype(str)
    else:
        alt_sent = next((c for c in ("llm_sentiment", "sentiment") if c in df.columns), None)
        if alt_sent:
            df["sentiment_label"] = df[alt_sent].fillna("").astype(str)
        else:
            df["sentiment_label"] = ""

    # 6. Urgence / Sévérité
    for c in ("llm_urgency_0_3", "llm_severity_0_3"):
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
        else:
            series = pd.Series([0.0] * len(df), index=df.index, dtype=float)
        df[c] = series.fillna(0.0)

    # 7. Statut SAV
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).fillna("Ouvert")
    else:
        alt_status = next((c for c in ("statut", "state") if c in df.columns), None)
        if alt_status:
            df["status"] = df[alt_status].astype(str).fillna("Ouvert")
        else:
            df["status"] = "Ouvert"

    # 8. Intent / Équipe pour l'onglet "Équipe & SLA"
    if "routing_team" not in df.columns and "intent_primary" not in df.columns:
        df["intent_primary"] = "SAV"

    # 9. Auteur
    author_src = next((c for c in ("author", "screen_name", "user", "username", "auteur") if c in df.columns), None)
    if author_src:
        df["author"] = df[author_src].fillna("").astype(str)
    else:
        df["author"] = ""

    return df


def _sample_manager_df() -> pd.DataFrame:
    """
    Petit jeu de 30 tweets factices pour l'affichage démo si aucune donnée réelle chargée.
    """
    base = pd.Timestamp.today().normalize()
    topics = [
        ["facturation"],
        ["réseau"],
        ["connexion"],
        ["application"],
        ["offre"],
        ["facturation", "réseau"],
    ]
    sentiments = ["negatif", "neutre", "positif"]
    statuses = ["Ouvert", "En attente client", "Clôturé"]

    rows = []
    for i in range(30):
        rows.append(
            {
                "tweet_id": f"SAMPLE-{i+1:03d}",
                "created_at_dt": base - pd.Timedelta(days=i),
                "sentiment_label": sentiments[i % len(sentiments)],
                "themes_list": topics[i % len(topics)],
                "llm_urgency_0_3": float(i % 4),
                "llm_severity_0_3": float((i + 1) % 4),
                "status": statuses[i % len(statuses)],
                "author": f"client_{i%6:02d}",
                "text_raw": "Tweet d’exemple généré pour la démo.",
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Chargement robuste des données réelles
# ------------------------------------------------------------------------------

def load_main_df_with_debug(dataset_key: str) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Essaie d'abord load_df(dataset_key), puis plusieurs chemins plausibles.
    Renvoie: (df_normalisé, used_path(str ou ""), debug_notes[list[str]]).
    """
    candidate_paths = [
        Path(r"C:\projetrncp\tweets_scored_llm.csv"),
        APP_ROOT / "tweets_scored_llm.csv",
        APP_ROOT / "data" / "tweets_scored_llm.csv",
        Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\tweets_scored_llm.csv"),
        Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
        Path.home() / "OneDrive" / "Bureau" / "IA Free Mobile" / "tweets_scored_llm.csv",
        Path.home() / "OneDrive" / "Bureau" / "IA Free Mobile" / "IA Free Mobile" / "tweets_scored_llm.csv",
    ]

    notes: list[str] = []
    
    # PRIORITÉ 1 : Lire depuis last_dataset.txt (fichier uploadé)
    data_dir = APP_ROOT / "data"
    last_dataset_file = data_dir / "last_dataset.txt"
    
    if last_dataset_file.exists():
        try:
            with open(last_dataset_file, "r", encoding="utf-8") as f:
                uploaded_csv_path = f.read().strip()
            if uploaded_csv_path and Path(uploaded_csv_path).exists():
                notes.append(f"last_dataset.txt: {uploaded_csv_path}")
                data = load_df(uploaded_csv_path)
                if data is not None and len(data) > 0:
                    notes.append(f"Fichier uploadé chargé: OK ({len(data)} lignes)")
                    return _prepare_manager_df(data), uploaded_csv_path, notes
        except Exception as exc:
            notes.append(f"last_dataset.txt: erreur -> {exc}")
    
    # PRIORITÉ 2 : Essayer load_df avec dataset_key
    try:
        direct_df = load_df(dataset_key)
        if direct_df is not None and len(direct_df) > 0:
            notes.append(f"load_df('{dataset_key}') : OK ({len(direct_df)} lignes)")
            return _prepare_manager_df(direct_df), f"load_df('{dataset_key}')", notes
        notes.append(f"load_df('{dataset_key}') : vide")
    except Exception as e:
        notes.append(f"load_df('{dataset_key}') : échec -> {e}")

    # PRIORITÉ 3 : Fallback sur les chemins candidats
    for p in candidate_paths:
        try:
            if not p.exists():
                notes.append(f"{p} : n'existe pas / pas trouvé")
                continue

            notes.append(f"{p} : trouvé, tentative lecture via load_df()")
            try:
                raw_df = load_df(str(p))
                if raw_df is not None and len(raw_df) > 0:
                    df_ready = _prepare_manager_df(raw_df)
                    notes.append(f"{p} : OK ({len(raw_df)} lignes)")
                    return df_ready, str(p), notes
                else:
                    notes.append(f"{p} : vide ou 0 lignes")
            except Exception as e:
                notes.append(f"{p} : échec lecture -> {e}")

        except Exception as e:
            notes.append(f"{p} : crash inattendu -> {e}")

    # Rien chargé
    return pd.DataFrame(), "", notes


# ------------------------------------------------------------------------------
# RÉEL : on charge le CSV/Excel. Sinon fallback démo.
# ------------------------------------------------------------------------------
dataset_key = cfg.get("manager_dataset_key", "tweets_scored_llm")
df_real, used_path, debug_notes = load_main_df_with_debug(dataset_key)
if df_real.empty:
    df = _sample_manager_df()
    data_loaded_ok = False
else:
    df = df_real
    data_loaded_ok = True

# RESET statuts : on ignore le CSV et on repart d'une colonne vide
df = normalize_status_empty(df)

# Appliquer les edits persistés (si présents)
df = apply_edits(df)

# ------------------------------------------------------------------------------
# Bandeau du haut : bouton retour (gauche) + info écran (droite)
# ------------------------------------------------------------------------------
top_left, top_right = st.columns([0.18, 0.82])

with top_left:
    if st.button("⬅️ Retour à l'accueil", use_container_width=True, key="manager_back_main"):
        st.switch_page("pages/0_Accueil.py")

with top_right:
    st.markdown(
        """
        <div style="
            background-color:#FFFFFF;
            border:1px solid #E5E7EB;
            border-radius:8px;
            padding:0.6rem 0.8rem;
            font-size:0.9rem;
            color:#0F172A;
            box-shadow:0 2px 6px rgba(15,23,42,0.06);
        ">
            <span style="opacity:0.8;">Vous êtes sur l'écran :</span>
            <strong>&nbsp;Manager</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# <-- juste au-dessus de "### 🔍 Filtres Manager" -->
st.button("🔄 Réinitialiser filtres", use_container_width=True, on_click=_reset_manager_filters, key="m_reset_btn")

st.markdown("### 🔍 Filtres Manager")

# ------------------------------------------------------------------------------
# Préparation des listes / valeurs par défaut pour les filtres
# ------------------------------------------------------------------------------
if df["created_at_dt"].notna().any():
    min_d_raw = df["created_at_dt"].min().date()
    max_d_raw = df["created_at_dt"].max().date()
    extended_min = (pd.Timestamp(min_d_raw) - pd.Timedelta(days=540)).date()
    extended_max = (pd.Timestamp(max_d_raw) + pd.Timedelta(days=180)).date()
else:
    min_d_raw = max_d_raw = None
    extended_min = extended_max = None

tone_opts = ["(Tous)"] + sorted(df["sentiment_label"].dropna().unique().tolist())

theme_values = (
    df["theme_primary"]
    .fillna("")
    .astype(str)
    .str.strip()
)
unique_themes = sorted([t for t in theme_values.unique().tolist() if t])
theme_opts = ["(Tous)"] + unique_themes

if df["created_at_dt"].notna().any():
    available_years = (
        df["created_at_dt"]
        .dropna()
        .dt.year
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
else:
    available_years = []
year_opts = ["(Toutes)"] + [str(y) for y in available_years]
# Par défaut : afficher "(Toutes)" (index 0)
default_year_index = 0

# ------------------------------------------------------------------------------
# Ligne de filtres (Période / Sentiment / Statut / Thème / Année)
# ------------------------------------------------------------------------------
col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([2, 1, 1, 1, 1])

with col_f1:
    if min_d_raw and max_d_raw:
        date_from, date_to = st.date_input(
            "Période",
            value=(min_d_raw, max_d_raw),
            min_value=extended_min,
            max_value=extended_max,
            key=f"m_period_simple_{ver}",
        )
    else:
        date_from = date_to = None
        st.caption("Dates indisponibles dans le fichier.")

with col_f2:
    tone_sel = st.selectbox("Sentiment", tone_opts, key=f"m_tone_simple_{ver}")

with col_f3:
    status_filter = st.multiselect("Statut", options=STATUS_OPTIONS, key=f"manager_status_{ver}")

with col_f4:
    theme_sel = st.selectbox("Thème", theme_opts, key=f"m_theme_simple_{ver}")

with col_f5:
    year_sel = st.selectbox(
        "Année (graphiques)",
        year_opts,
        index=default_year_index,
        key=f"m_year_simple_{ver}",
    )

# Infos source + diagnostic
if data_loaded_ok:
    st.markdown(
        f"""
        <div style="
            margin-top: 10px;
            padding: 12px 16px;
            background: linear-gradient(to right, rgba(240, 253, 244, 0.9), rgba(220, 252, 231, 0.8));
            backdrop-filter: blur(8px);
            border: 1px solid #86efac;
            border-left: 5px solid #22c55e;
            border-radius: 8px;
            color: #14532d;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        ">
            <span style="margin-right: 12px; font-size: 1.25rem;">✅</span>
            <div style="display: flex; flex-direction: column;">
                <span style="font-weight: 700; margin-bottom: 2px;">Source de données active</span>
                <span style="opacity: 0.9;">
                    {len(df):,} lignes chargées depuis : <span style="font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85em; background: rgba(255,255,255,0.6); padding: 2px 6px; border-radius: 4px; color: #14532d;">{used_path}</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("Mode démo : pas de données réelles trouvées", icon="⚠️")

with st.expander("Diagnostic import (avancé)"):
    for note in debug_notes:
        st.text(note)
    if not data_loaded_ok:
        st.markdown(
            """
1. Vérifie que le fichier existe bien au chemin indiqué.
2. Vérifie qu'il n'est **pas** ouvert dans Excel (Excel peut le locker).
3. Vérifie que c'est un vrai CSV ou un export Excel.
"""
        )

# ------------------------------------------------------------------------------
# Titre principal
# ------------------------------------------------------------------------------
st.title("Tableau de bord Manager")

# ------------------------------------------------------------------------------
# Application des filtres aux données
# ------------------------------------------------------------------------------
flt = df.copy()

if date_from and date_to:
    # st.date_input renvoie des datetime.date -> on compare sur .dt.date
    flt = flt[
        (flt["created_at_dt"].dt.date >= pd.to_datetime(date_from).date())
        & (flt["created_at_dt"].dt.date <= pd.to_datetime(date_to).date())
    ]

if tone_sel != "(Tous)":
    flt = flt[
        flt["sentiment_label"]
        .str.lower()
        .str.startswith(tone_sel[:3].lower())
    ]

if theme_sel != "(Tous)":
    flt = flt[flt["theme_primary"] == theme_sel]

# Appliquer filtrage Statut via helper
flt = filter_by_status_like_agent(flt, status_filter)

# Filtre année pour les graphiques
flt_graph = flt.copy()
if year_sel != "(Toutes)":
    try:
        year_int = int(year_sel)
        flt_graph = flt_graph[flt_graph["created_at_dt"].dt.year == year_int]
    except (ValueError, TypeError):
        # Si conversion échoue, garder toutes les données
        pass

# ------------------------------------------------------------------------------
# Construction des agrégats pour les graphes et KPIs
# ------------------------------------------------------------------------------
if flt.empty:
    st.info("Aucune donnée pour la sélection courante (filtres trop restrictifs ?).")
elif flt_graph.empty:
    st.warning(f"Aucune donnée pour l'année {year_sel}. Essayez une autre année ou sélectionnez '(Toutes)'.")
else:
    NAV_TABS = ["Vue globale", "Alertes", "Équipe", "Paramètres"]
    if "manager_view_tab" not in st.session_state:
        st.session_state["manager_view_tab"] = NAV_TABS[0]
    nav_cols = st.columns(len(NAV_TABS))
    for idx, label in enumerate(NAV_TABS):
        is_active = st.session_state["manager_view_tab"] == label
        if nav_cols[idx].button(("👉 " if is_active else "") + label,
                                use_container_width=True,
                                key=f"manager_nav_{idx}"):
            st.session_state["manager_view_tab"] = label
    active_tab = st.session_state["manager_view_tab"]

    # Agrégations
    daily = (
        flt_graph.assign(date=flt_graph["created_at_dt"].dt.date)
        .groupby("date", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )

    cumulative = (
        daily.assign(cumul=daily["count"].cumsum())
        if not daily.empty
        else pd.DataFrame(columns=["date", "count", "cumul"])
    )

    sent_dist = (
        flt_graph["sentiment_label"]
        .str.capitalize()
        .replace("", "Non précisé")
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    theme_series = (
        flt_graph["theme_primary"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    theme_series = theme_series[theme_series != ""]
    if not theme_series.empty:
        theme_df = (
            theme_series.value_counts()
            .rename_axis("theme")
            .reset_index(name="count")
            .head(10)
        )
    else:
        theme_df = pd.DataFrame(columns=["theme", "count"])

    author_df = (
        flt_graph["author"]
        .replace("", "Non précisé")
        .value_counts()
        .rename_axis("author")
        .reset_index(name="count")
        .head(10)
    )

    status_df = (
        flt_graph["status"]
        .replace("", "Non précisé")
        .str.capitalize()
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )

    heatmap = (
        flt_graph.assign(hour=flt_graph["created_at_dt"].dt.hour,
                   dow=flt_graph["created_at_dt"].dt.dayofweek)
        .groupby(["dow", "hour"])
        .size()
        .reset_index(name="count")
        if flt_graph["created_at_dt"].notna().any()
        else pd.DataFrame(columns=["dow", "hour", "count"])
    )
    if not heatmap.empty:
        dow_labels = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}
        heatmap["Jour"] = heatmap["dow"].map(dow_labels)

    scatter_df = flt_graph[
        ["llm_urgency_0_3", "llm_severity_0_3", "sentiment_label", "tweet_id", "author"]
    ].copy()
    scatter_df["sentiment_label"] = scatter_df["sentiment_label"].replace("", "Non précisé")

    tone_timeline = (
        flt_graph.assign(
            date=flt_graph["created_at_dt"].dt.date,
            sentiment=flt_graph["sentiment_label"]
            .str.capitalize()
            .replace("", "Non précisé"),
        )
        .groupby(["date", "sentiment"])
        .size()
        .reset_index(name="count")
        if flt_graph["created_at_dt"].notna().any()
        else pd.DataFrame(columns=["date", "sentiment", "count"])
    )

    hourly = (
        flt_graph.assign(hour=flt_graph["created_at_dt"].dt.hour)
        .groupby("hour")
        .size()
        .reset_index(name="count")
        .sort_values("hour")
        if flt_graph["created_at_dt"].notna().any()
        else pd.DataFrame(columns=["hour", "count"])
    )

    team_col = next((c for c in ("routing_team", "intent_primary") if c in flt_graph.columns), None)
    team_volume = (
        flt_graph.groupby(team_col)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if team_col
        else pd.DataFrame()
    )
    team_scores = (
        flt_graph.groupby(team_col)[["llm_urgency_0_3", "llm_severity_0_3"]]
        .mean()
        .reset_index()
        if team_col
        else pd.DataFrame()
    )

    urgent_table = flt_graph.sort_values(
        ["llm_urgency_0_3", "llm_severity_0_3"], ascending=False
    )[
        [
            "tweet_id",
            "author",
            "sentiment_label",
            "theme_primary",
            "llm_urgency_0_3",
            "llm_severity_0_3",
            "status",
            "text_raw",
        ]
    ]

    # helper tendances thèmes
    def _theme_counts_slice(df_slice: pd.DataFrame) -> pd.Series:
        vals = (
            df_slice["theme_primary"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        vals = vals[vals != ""]
        if vals.empty:
            return pd.Series(dtype=int)
        return vals.value_counts()

    if flt_graph["created_at_dt"].notna().any():
        valid_dates = flt_graph["created_at_dt"].dropna()
    else:
        valid_dates = pd.Series([], dtype="datetime64[ns]")

    theme_trends = pd.DataFrame(columns=["theme", "recent", "previous", "delta"])
    if not valid_dates.empty:
        max_ts = valid_dates.max()
        recent_start = max_ts - pd.Timedelta(days=6)
        past_start = recent_start - pd.Timedelta(days=7)

        recent_counts = _theme_counts_slice(flt_graph[flt_graph["created_at_dt"] >= recent_start])
        past_counts = _theme_counts_slice(
            flt_graph[
                (flt_graph["created_at_dt"] < recent_start)
                & (flt_graph["created_at_dt"] >= past_start)
            ]
        )
        if not recent_counts.empty or not past_counts.empty:
            all_idx = recent_counts.index.union(past_counts.index)
            theme_trends = pd.DataFrame(
                {
                    "theme": all_idx,
                    "recent": recent_counts.reindex(all_idx, fill_value=0).astype(int),
                    "previous": past_counts.reindex(all_idx, fill_value=0).astype(int),
                }
            )
            theme_trends["delta"] = theme_trends["recent"] - theme_trends["previous"]
            theme_trends = theme_trends.sort_values("delta", ascending=False).head(10)

    # split urgent / pas urgent
    urgency_split = pd.DataFrame(columns=["niveau", "count"])
    if not flt_graph.empty:
        urgency_split = (
            pd.Series(
                np.where(
                    flt_graph["llm_urgency_0_3"] >= 2,
                    "Urgent ≥ 2",
                    "Urgent < 2",
                )
            )
            .value_counts()
            .rename_axis("niveau")
            .reset_index(name="count")
        )

    # Heures critiques : % négatif par heure
    if flt_graph["created_at_dt"].notna().any():
        hourly_sentiment = (
            flt_graph.assign(
                hour=flt_graph["created_at_dt"].dt.hour,
                neg=flt_graph["sentiment_label"]
                .fillna("")
                .str.lower()
                .str.startswith(("neg", "nég")),
            )
            .groupby("hour")["neg"]
            .mean()
            .reset_index(name="neg_ratio")
            .sort_values("hour")
        )
        hourly_sentiment["neg_rate_pct"] = hourly_sentiment["neg_ratio"] * 100
    else:
        hourly_sentiment = pd.DataFrame(columns=["hour", "neg_rate_pct"])

    # --------------------------------------------------------------------------
    # Affichage selon l'onglet
    # --------------------------------------------------------------------------
    if active_tab == "Vue globale":
        # KPIs haut de page
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Tweets", f"{len(flt_graph):,}")
        col2.metric("Pourcentage urgent", f"{100 * (flt_graph['llm_urgency_0_3'] >= 2).mean():.1f} %")
        
        col3.metric("Pourcentage négatif", f"{100 * (flt_graph['sentiment_label'].fillna('').str.lower().str.startswith(('neg', 'nég'))).mean():.1f} %")
        col4.metric("Auteurs uniques", f"{flt_graph['author'].nunique():,}")
        col5.metric("Urgence moyenne", f"{flt_graph['llm_urgency_0_3'].mean():.2f}")

        # KPI demandé : Ouverts / À traiter
        st.metric("Ouverts / À traiter", f"{count_open_like_agent(flt_graph):,}")

        # Export (seulement les ouverts / à traiter)
        open_view = flt_graph[flt_graph["status"].astype(str).str.strip() != ""].copy()
        
        # Dédupliquer les colonnes si nécessaire (évite l'erreur orient='records')
        if open_view.columns.duplicated().any():
            open_view = open_view.loc[:, ~open_view.columns.duplicated()]
        
        st.caption(f"{len(open_view):,} ticket(s) ouverts / à traiter dans la vue actuelle.")
        if open_view.empty:
            st.info("Aucun ticket ouvert / à traiter à exporter avec les filtres actuels.")
        else:
            st.download_button(
                "⬇️ Exporter les tickets OUVERTS / À TRAITER (CSV)",
                data=open_view.to_csv(index=False).encode("utf-8"),
                file_name="tweets_manager_ouverts.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Exporter les tickets OUVERTS / À TRAITER (JSON)",
                data=open_view.to_json(orient="records", force_ascii=False).encode("utf-8"),
                file_name="tweets_manager_ouverts.json",
                use_container_width=True,
            )

        # === Sous-onglet Activité ===
        with st.container():
            tabs = st.tabs(
                [
                    "Activité",
                    "Qualité & expérience",
                    "Opérations",
                    "Équipe & SLA",
                    "Détails",
                    "Insights avancés",
                    "Tous les tweets",
                ]
            )

            # ---------- Onglet 0 : Activité ----------
            with tabs[0]:
                # ligne 1 : volume quotidien / volume cumulatif / auteurs
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.subheader("Volume quotidien")
                    if not daily.empty:
                        st.altair_chart(
                            alt.Chart(daily)
                            .mark_line(point=True)
                            .encode(
                                x="date:T",
                                y="count:Q",
                                tooltip=["date:T", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucune donnée temporelle disponible.")

                with col_a2:
                    st.subheader("Volume cumulatif")
                    if not cumulative.empty:
                        st.altair_chart(
                            alt.Chart(cumulative)
                            .mark_area(line=True, point=True, opacity=0.55)
                            .encode(
                                x="date:T",
                                y=alt.Y("cumul:Q", title="Total cumulé"),
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucun thème recensé.")

                with col_a3:
                    st.subheader("Auteurs les plus actifs")
                    if not author_df.empty:
                        st.altair_chart(
                            alt.Chart(author_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("count:Q", title="Volume"),
                                y=alt.Y("author:N", sort="-x", title="Auteur"),
                                tooltip=["author:N", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucun auteur recensé.")

            # ---------- Onglet 1 : Qualité & expérience ----------
            with tabs[1]:
                col_b1, col_b2 = st.columns(2)

                with col_b1:
                    st.subheader("Répartition des sentiments")
                    if not sent_dist.empty:
                        st.altair_chart(
                            alt.Chart(sent_dist)
                            .mark_bar()
                            .encode(
                                x=alt.X("sentiment:N", title="Sentiment"),
                                y=alt.Y("count:Q", title="Volume"),
                                color=alt.Color("sentiment:N", legend=None),
                                tooltip=["sentiment:N", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucune sentiment disponible.")

                with col_b2:
                    st.subheader("Urgence vs sévérité")
                    if not scatter_df.empty:
                        st.altair_chart(
                            alt.Chart(scatter_df)
                            .mark_circle(size=70, opacity=0.6)
                            .encode(
                                x=alt.X("llm_urgency_0_3:Q", title="Urgence (0–3)"),
                                y=alt.Y("llm_severity_0_3:Q", title="Sévérité (0–3)"),
                                color=alt.Color("sentiment_label:N", title="Sentiment"),
                                tooltip=[
                                    "tweet_id:N",
                                    "author:N",
                                    "llm_urgency_0_3:Q",
                                    "llm_severity_0_3:Q",
                                ],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Pas de scores disponibles pour le scatter.")

            # ---------- Onglet 2 : Opérations ----------
            with tabs[2]:
                col_c1, col_c2 = st.columns(2)

                with col_c1:
                    st.subheader("Statuts des tickets")
                    if not status_df.empty:
                        st.altair_chart(
                            alt.Chart(status_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("count:Q", title="Volume"),
                                y=alt.Y("status:N", sort="-x", title="Statut"),
                                tooltip=["status:N", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucun statut recensé.")

                with col_c2:
                    st.subheader("Chaleur jour × heure")
                    if not heatmap.empty:
                        st.altair_chart(
                            alt.Chart(heatmap)
                            .mark_rect()
                            .encode(
                                x=alt.X("hour:O", title="Heure"),
                                y=alt.Y(
                                    "Jour:N",
                                    sort=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
                                ),
                                color=alt.Color("count:Q", title="Volume"),
                                tooltip=["Jour:N", "hour:O", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Chaleur indisponible (dates manquantes).")

                st.subheader("Courbe horaire (toutes équipes)")
                if not hourly.empty:
                    st.altair_chart(
                        alt.Chart(hourly)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("hour:O", title="Heure"),
                            y=alt.Y("count:Q", title="Volume"),
                            tooltip=["hour:O", "count:Q"],
                        )
                        .properties(height=240),
                        use_container_width=True,
                    )
                else:
                    st.caption("Horaires indisponibles.")

            # ---------- Onglet 3 : Équipe & SLA ----------
            with tabs[3]:
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    st.subheader("Volumes par équipe/intent")
                    if not team_volume.empty and team_col:
                        st.altair_chart(
                            alt.Chart(team_volume)
                            .mark_bar()
                            .encode(
                                x=alt.X("count:Q", title="Volume"),
                                y=alt.Y(f"{team_col}:N", sort="-x", title="Équipe / Intent"),
                                tooltip=[f"{team_col}:N", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucune équipe renseignée.")

                with col_d2:
                    st.subheader("Urgence & sévérité moyennes")
                    if not team_scores.empty and team_col:
                        st.altair_chart(
                            alt.Chart(team_scores)
                            .transform_fold(
                                ["llm_urgency_0_3", "llm_severity_0_3"],
                                as_=["indicateur", "valeur"],
                            )
                            .mark_bar()
                            .encode(
                                x=alt.X("valeur:Q", title="Score moyen"),
                                y=alt.Y(
                                    f"{team_col}:N", sort="-x", title="Équipe / Intent"
                                ),
                                color=alt.Color("indicateur:N", title="Indicateur"),
                                tooltip=[
                                    f"{team_col}:N",
                                    "indicateur:N",
                                    "valeur:Q",
                                ],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Scores indisponibles.")

                st.subheader("Heures critiques (% négatif)")
                if not hourly_sentiment.empty:
                    st.altair_chart(
                        alt.Chart(hourly_sentiment)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("hour:O", title="Heure"),
                            y=alt.Y("neg_rate_pct:Q", title="% négatif"),
                            tooltip=[
                                "hour:O",
                                alt.Tooltip("neg_rate_pct:Q", title="% négatif", format=".1f"),
                            ],
                        )
                        .properties(height=280),
                        use_container_width=True,
                    )
                else:
                    st.caption("Aucune granularité horaire disponible.")

            # ---------- Onglet 4 : Détails ----------
            with tabs[4]:
                st.subheader("Timeline des sentiments")
                if not tone_timeline.empty:
                    st.altair_chart(
                        alt.Chart(tone_timeline)
                        .mark_area()
                        .encode(
                            x="date:T",
                            y=alt.Y("count:Q", stack="normalize", title="Part (%)"),
                            color=alt.Color("sentiment:N", title="Sentiment"),
                            tooltip=["date:T", "sentiment:N", "count:Q"],
                        )
                        .properties(height=280),
                        use_container_width=True,
                    )
                else:
                    st.caption("Timeline indisponible.")

                st.subheader("Tweets à surveiller (trié par urgence)")
                if not urgent_table.empty:
                    ut = urgent_table.rename(
                        columns={
                            "theme_primary": "theme",
                            "llm_urgency_0_3": "urgence",
                            "llm_severity_0_3": "sévérité",
                        }
                    )
                    show_white_table(ut, height=320, key="mgr_urgent_table_details")
                else:
                    st.caption("Aucun tweet urgent repéré.")

            # ---------- Onglet 5 : Insights avancés ----------
            with tabs[5]:
                col_e1, col_e2 = st.columns(2)

                with col_e1:
                    st.subheader("Répartition des urgences")
                    if not urgency_split.empty:
                        st.altair_chart(
                            alt.Chart(urgency_split)
                            .mark_arc(innerRadius=60)
                            .encode(
                                theta="count:Q",
                                color=alt.Color("niveau:N", title="Niveau"),
                                tooltip=["niveau:N", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucune mesure d’urgence disponible.")

                with col_e2:
                    st.subheader("Thèmes en progression (7 derniers jours)")
                    if not theme_trends.empty:
                        st.altair_chart(
                            alt.Chart(theme_trends)
                            .mark_bar()
                            .encode(
                                x=alt.X("delta:Q", title="Variation"),
                                y=alt.Y("theme:N", sort="-x", title="Thème"),
                                color=alt.condition(
                                    alt.datum.delta >= 0,
                                    alt.value("#ff7f0e"),
                                    alt.value("#1f77b4"),
                                ),
                                tooltip=[
                                    alt.Tooltip("theme:N", title="Thème"),
                                    alt.Tooltip("recent:Q", title="7 derniers jours"),
                                    alt.Tooltip("previous:Q", title="7 jours avant"),
                                    alt.Tooltip("delta:Q", title="Variation"),
                                ],
                            )
                            .properties(height=320),
                            use_container_width=True,
                        )
                        show_white_table(
                            theme_trends.rename(
                                columns={
                                    "recent": "7 derniers jours",
                                    "previous": "7 jours avant",
                                    "delta": "Variation",
                                }
                            ),
                            height=260,
                            key="mgr_theme_trends_table"
                        )
                    else:
                        st.caption("Pas de tendance significative détectée.")

            # ---------- Onglet 6 : Tous les tweets ----------
            with tabs[6]:
                st.subheader("Tous les tweets (filtrés)")

                all_cols = [
                    "tweet_id",
                    "created_at_dt",
                    "author",
                    "sentiment_label",
                    "theme_primary",
                    "llm_urgency_0_3",
                    "llm_severity_0_3",
                    "status",
                    "text_raw",
                ]
                cols_presentes = [c for c in all_cols if c in flt_graph.columns]

                show_white_table(
                    flt_graph[cols_presentes].sort_values("created_at_dt", ascending=False),
                    height=520,
                    key="mgr_all_tweets_table"
                )
                st.caption(
                    f"{len(flt_graph):,} lignes affichées (correspond à l’export CSV ci-dessus)."
                )

    elif active_tab == "Alertes":
        st.subheader("Tweets à surveiller (focus)")
        if not urgent_table.empty:
            ut = urgent_table.rename(
                columns={
                    "theme_primary": "theme",
                    "llm_urgency_0_3": "urgence",
                    "llm_severity_0_3": "sévérité",
                }
            )
            show_white_table(ut, height=360, key="mgr_urgent_table_alerts")
        else:
            st.caption("Aucun tweet urgent à signaler.")

        st.subheader("Thèmes en progression (7 derniers jours)")
        if not theme_trends.empty:
            st.altair_chart(
                alt.Chart(theme_trends)
                .mark_bar()
                .encode(
                    x=alt.X("delta:Q", title="Variation"),
                    y=alt.Y("theme:N", sort="-x", title="Thème"),
                    color=alt.condition(
                        alt.datum.delta >= 0,
                        alt.value("#ff7f0e"),
                        alt.value("#1f77b4"),
                    ),
                    tooltip=[
                        alt.Tooltip("theme:N", title="Thème"),
                        alt.Tooltip("recent:Q", title="7 derniers jours"),
                        alt.Tooltip("previous:Q", title="7 jours avant"),
                        alt.Tooltip("delta:Q", title="Variation"),
                    ],
                )
                .properties(height=320),
                use_container_width=True,
            )
        else:
            st.caption("Pas de variation significative détectée.")

        st.subheader("Timeline des sentiments")
        if not tone_timeline.empty:
            st.altair_chart(
                alt.Chart(tone_timeline)
                .mark_area()
                .encode(
                    x="date:T",
                    y=alt.Y("count:Q", stack="normalize", title="Part (%)"),
                    color=alt.Color("sentiment:N", title="Sentiment"),
                    tooltip=["date:T", "sentiment:N", "count:Q"],
                )
                .properties(height=280),
                use_container_width=True,
            )
        else:
            st.caption("Timeline indisponible.")

    elif active_tab == "Équipe":
        st.subheader("Volumes par équipe / intent")
        if not team_volume.empty and team_col:
            st.altair_chart(
                alt.Chart(team_volume)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Volume"),
                    y=alt.Y(f"{team_col}:N", sort="-x", title="Équipe / Intent"),
                    tooltip=[f"{team_col}:N", "count:Q"],
                )
                .properties(height=300),
                use_container_width=True,
            )
        else:
            st.caption("Aucune équipe renseignée.")

        st.subheader("Scores moyens (Urgence / Sévérité)")
        if not team_scores.empty and team_col:
            st.altair_chart(
                alt.Chart(team_scores)
                .transform_fold(["llm_urgency_0_3", "llm_severity_0_3"],
                                as_=["indicateur", "valeur"])
                .mark_bar()
                .encode(
                    x=alt.X("valeur:Q", title="Score moyen"),
                    y=alt.Y(f"{team_col}:N", sort="-x", title="Équipe / Intent"),
                    color=alt.Color("indicateur:N", title="Indicateur"),
                    tooltip=[f"{team_col}:N", "indicateur:N", "valeur:Q"],
                )
                .properties(height=300),
                use_container_width=True,
            )
        else:
            st.caption("Scores indisponibles.")

        st.subheader("Heures critiques (% négatif)")
        if not hourly_sentiment.empty:
            st.altair_chart(
                alt.Chart(hourly_sentiment)
                .mark_line(point=True)
                .encode(
                    x=alt.X("hour:O", title="Heure"),
                    y=alt.Y("neg_rate_pct:Q", title="% négatif"),
                    tooltip=[
                        "hour:O",
                        alt.Tooltip("neg_rate_pct:Q", title="% négatif", format=".1f"),
                    ],
                )
                .properties(height=260),
                use_container_width=True,
            )
        else:
            st.caption("Granularité horaire indisponible.")

    else:
        st.subheader("Paramètres & diagnostic")
        st.markdown(f"- Source courant : **{used_path or 'non déterminé'}**")
        st.markdown(f"- Dataset key (load_df) : **{dataset_key}**")
        st.markdown("Historique de chargement :")
        for note in debug_notes:
            st.code(note, language="text")
        st.markdown(
            "Pour aligner encore plus avec l'écran Agent SAV, assure-toi que "
            "`manager_dataset_key` pointe vers le même dataset dans la configuration."
        )

# ------------------------------------------------------------------
# fin
# ------------------------------------------------------------------
