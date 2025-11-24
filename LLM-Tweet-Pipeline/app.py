# app.py — Interface Streamlit (prétraitement + LLM Mistral)
# Fix Windows paths: utilise chemins absolus des scripts + guillemets + cwd=BASE_DIR

import os
import shlex
import tempfile
from pathlib import Path
import subprocess
import platform
import sys
import shutil  # <-- ajouté

import pandas as pd
import streamlit as st

# 🎨 Configuration de la page (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Pipeline SAV Free Mobile",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "🤖 Application d'analyse SAV automatique avec LLM & RAG - Mistral & Ollama"
    }
)

# 🎨 CSS personnalisé pour un design professionnel
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styling - Mode clair professionnel */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to bottom, #ffffff 0%, #f8fafc 100%);
        color: #1e293b;
    }
    
    /* Changer la couleur de fond de l'app entière */
    .stApp {
        background: #ffffff;
    }
    
    /* Header gradient - Bleu professionnel clair */
    .gradient-header {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #38bdf8 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Section cards - Fond blanc */
    .section-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.12);
        border-color: #93c5fd;
    }
    
    /* Success box - Vert clair */
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #22c55e;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.1);
        color: #166534;
    }
    
    .success-box strong {
        color: #15803d;
    }
    
    /* Warning box - Orange clair */
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #f59e0b;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);
        color: #92400e;
    }
    
    .warning-box strong {
        color: #b45309;
    }
    
    /* Info box - Bleu clair */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
        color: #1e40af;
    }
    
    .info-box strong {
        color: #1d4ed8;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #22c55e;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }
    
    /* Buttons styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Primary button */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Secondary button */
    .stButton>button[kind="secondary"] {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #475569;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: #e2e8f0;
    }
    
    /* Sidebar styling - Mode clair */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #1e40af;
    }
    
    [data-testid="stSidebar"] label {
        color: #475569;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.05rem;
        background-color: #f8fafc;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f1f5f9;
        border-color: #93c5fd;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: #f8fafc;
        color: #64748b;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
    }
    
    /* Text inputs */
    .stTextInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
        color: #1e293b;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Text area */
    .stTextArea>div>div>textarea {
        border-radius: 6px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
        color: #1e293b;
    }
    
    /* Select boxes */
    .stSelectbox>div>div {
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #cbd5e1;
    }
    
    /* Number input */
    .stNumberInput>div>div>input {
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #cbd5e1;
    }
    
    /* Radio buttons */
    .stRadio>div {
        color: #1e293b;
    }
    
    /* Checkbox */
    .stCheckbox>label {
        color: #1e293b;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a;
    }
    
    /* Divider */
    hr {
        border-color: #e2e8f0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background-color: #eff6ff;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #ffffff;
        color: #1e293b;
    }
    
    /* Status container */
    [data-testid="stStatusWidget"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        border-radius: 8px;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.3);
    }
    
    /* Spinner */
    .stSpinner>div {
        border-top-color: #3b82f6;
    }
    
    /* Markdown paragraphs */
    .stMarkdown p {
        color: #475569;
    }
    
    /* Captions */
    .stCaption {
        color: #64748b;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Ajouts: dossier fixe + helpers pour clients_only ---
from datetime import datetime

# Configuration des chemins
DEFAULT_CLEAN_DIR = Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\clean_client")
DEFAULT_INPUT_CSV = r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\clean_client\free tweet export.csv"
KB_PATH_DEFAULT = r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\LLM-Tweet-Pipeline\kb_replies_rich_expanded.csv"

def get_clean_dir() -> Path:
    p = Path(st.session_state.get("clean_dir", DEFAULT_CLEAN_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_preprocess_output_dir() -> Path:
    """Retourne le dossier de sortie pour le prétraitement."""
    p = get_clean_dir() / "Prétraitement_LLM"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_mistral_output_dir() -> Path:
    """Retourne le dossier de sortie pour Mistral API."""
    p = get_clean_dir() / "LLM_Mistral"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_ollama_output_dir() -> Path:
    """Retourne le dossier de sortie pour Ollama."""
    p = get_clean_dir() / "LLM_Ollama"
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_temp_dir() -> Path:
    """Retourne le dossier temporaire pour les fichiers filtrés."""
    p = get_clean_dir() / "temp"
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_clients_only(folder: Path):
    """Retourne tous les CSV *clients_only* triés du plus récent au plus ancien, dans folder."""
    try:
        return sorted(
            folder.glob("*clients_only*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    except Exception:
        return []

def pick_latest_clients_only(folder: Path):
    """Retourne le chemin (str) du *clients_only* le plus récent dans folder, sinon None."""
    cands = list_clients_only(folder)
    return str(cands[0].resolve()) if cands else None

def _kb_path_autoload() -> str | None:
    p = Path(KB_PATH_DEFAULT)
    return str(p.resolve()) if p.exists() else None


# --- Remplacer les anciennes fonctions de chargement par des versions dynamiques et cacheables ---
@st.cache_data(show_spinner=False)
def load_subset_for_dates(csv_path: str, date_from_str: str | None, date_to_str: str | None, max_rows: int = 5000):
    """Charge un sous-ensemble depuis un CSV 'clean' en choisissant dynamiquement les colonnes.
    Normalise la sortie en colonnes: ['id','created_at','full_text']."""
    if not csv_path:
        return pd.DataFrame(columns=["id","created_at","full_text"])
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=["id","created_at","full_text"])
    # lecture robuste
    try:
        df = pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=["id","created_at","full_text"])
    # détecte colonnes candidates
    id_col = None
    for c in ("tweet_id","id","id_str","status_id"):
        if c in df.columns:
            id_col = c; break
    text_col = None
    for c in ("text_for_llm","text_for_model","text_clean","text_raw","full_text","text","content"):
        if c in df.columns:
            text_col = c; break
    date_col = None
    for c in ("created_at","_dt","date","timestamp","time"):
        if c in df.columns:
            date_col = c; break
    # fallback
    if id_col is None:
        df["id"] = df.index.astype(str)
        id_col = "id"
    else:
        df[id_col] = df[id_col].astype(str)
    if text_col is None:
        df["full_text"] = ""
        text_col = "full_text"
    if date_col is None:
        df["created_at"] = ""
        date_col = "created_at"
    # normaliser created_at en datetime naive
    df["created_at_dt"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    mask = pd.Series(True, index=df.index)
    if date_from_str:
        try:
            lower = pd.to_datetime(date_from_str)
            mask &= (df["created_at_dt"] >= lower)
        except Exception:
            pass
    if date_to_str:
        try:
            upper = pd.to_datetime(date_to_str) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask &= (df["created_at_dt"] <= upper)
        except Exception:
            pass
    sub = df.loc[mask, [id_col, "created_at_dt", text_col]].copy()
    sub.rename(columns={id_col: "id", "created_at_dt": "created_at", text_col: "full_text"}, inplace=True)
    # format created_at as datetime or iso strings
    sub["created_at"] = pd.to_datetime(sub["created_at"], errors="coerce")
    # sort and limit
    sub = sub.sort_values(["created_at","id"], ascending=[True, True]).head(max_rows)
    # ensure string id and created_at string for UI label
    sub["id"] = sub["id"].astype(str)
    return sub.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def count_tweets_in_range(csv_path: str, date_from_str: str | None, date_to_str: str | None) -> int:
    """Compte les lignes dans l'intervalle pour le CSV clean (utilisé pour info)."""
    df = load_subset_for_dates(csv_path, date_from_str, date_to_str, max_rows=10**9)
    return int(len(df))


# ---------- AJOUT : get_csv_date_range (robuste, cacheé ----------
@st.cache_data(show_spinner=False)
def get_csv_date_range(csv_path: str):
    """Retourne (date_min, date_max) en se basant sur la colonne date disponible."""
    if not csv_path:
        return (None, None)
    p = Path(csv_path)
    if not p.exists():
        return (None, None)

    # lecture tolérante à l'encodage
    try:
        df = pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
        except Exception:
            return (None, None)

    # cherche une colonne date plausible et renvoie min/max (naive)
    for col in ("created_at", "_dt", "date", "timestamp", "time"):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
                if dt.notna().any():
                    return (dt.min().date(), dt.max().date())
            except Exception:
                continue

    return (None, None)
# ---------- FIN AJOUT ----------

st.set_page_config(page_title="Pipeline NLP — Nettoyage + LLM (Mistral)", layout="wide")
st.title("🧹➡️🧠 Pipeline NLP : Prétraitement puis LLM (Mistral)")
st.caption("Chargez un CSV, lancez le script de nettoyage, puis le script LLM.")

# État global
if "paths" not in st.session_state:
    st.session_state.paths = {
        "uploaded_csv": None,
        "clean_csv": None,
        "llm_csv": None,
        "workdir": None,
    }

# Répertoire base = dossier où se trouvent app.py et les scripts
BASE_DIR = Path(__file__).resolve().parent

# -------------------- Utils --------------------

def save_uploaded_file(uploaded_file, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".csv"
    safe_name = Path(uploaded_file.name).stem
    dest = workdir / f"{safe_name}{suffix}"
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


# ---- Helper global: afficher joliment une commande (list ou str)
def fmt_cmd_for_display(cmd):
    if isinstance(cmd, list):
        def q(p):
            p = str(p)
            return f'"{p}"' if (" " in p or "||" in p) else p
        return " ".join(q(x) for x in cmd)
    return str(cmd)

import platform
import subprocess

def run_command(command, workdir: Path | None = None):
    try:
        if isinstance(command, list):
            res = subprocess.run(
                command,
                shell=False,
                cwd=str(workdir) if workdir else None,
                capture_output=True,
                text=True,
            )
        else:
            if platform.system() == "Windows":
                res = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(workdir) if workdir else None,
                    capture_output=True,
                    text=True,
                )
            else:
                import shlex
                args = shlex.split(command)
                res = subprocess.run(
                    args,
                    shell=False,
                    cwd=str(workdir) if workdir else None,
                    capture_output=True,
                    text=True,
                )
        return res.returncode, res.stdout, res.stderr
    except FileNotFoundError as e:
        return 127, "", f"Commande introuvable: {e}"
    except Exception as e:
        return 1, "", f"Erreur d'exécution: {e}"


def preview_csv(path: Path, n: int = 50):
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=';')
        except Exception as e:
            st.error(f"Impossible de lire le CSV: {e}")
            return
    st.dataframe(df.head(n))


def build_command_from_template(template: str, **kwargs) -> str:
    # Les chemins sont quotés dans les templates; on fait juste .format
    return template.format(**kwargs)

# ==================== SIDEBAR : CONFIGURATION ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="margin: 0;">⚙️ Configuration</h2>
        <p style="color: gray; font-size: 0.9rem;">Paramètres avancés</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    if st.session_state.paths["workdir"] is None:
        st.session_state.paths["workdir"] = Path(tempfile.mkdtemp(prefix="streamlit_nlp_"))
    workdir = Path(st.session_state.paths["workdir"]) 
    st.write(f"Dossier de travail : `{workdir}`")
    st.write(f"Dossier scripts : `{BASE_DIR}`")

    st.subheader("Scripts & templates de commande")

    # Chemins par défaut vers les scripts (absolus)
    default_preprocess_script = str((BASE_DIR / "process_tweets_pipeline.py").resolve())
    default_llm_script = str((BASE_DIR / "llm_batch_multitask_pool_mistral.py").resolve())
    default_full_pipeline_script = str((BASE_DIR / "llm_full_ollama_pipeline.py").resolve())

    preprocess_script_path = st.text_input("Chemin script prétraitement", value=default_preprocess_script)
    llm_script_path = st.text_input("Chemin script LLM (Mistral API)", value=default_llm_script)
    full_pipeline_script_path = st.text_input("Chemin script pipeline complet (BERT+Ollama)", value=default_full_pipeline_script)

    # <-- MODIF : nouveau template PREPROCESS avec output-dir -->
    preprocess_tpl = st.text_input(
        "Commande prétraitement",
        value=(
            "python \"{pre_script}\" --input \"{input}\" --output \"{output}\" --output-dir \"{output_dir}\" {extra}"
        ),
        help="Placeholders: {pre_script}, {input}, {output}, {output_dir}, {extra}",
    )

    # <-- AJOUT : dossier configurable pour les exports nettoyés -->
    clean_dir_str = st.text_input("📁 Dossier sortie 'clean_client'", value=str(DEFAULT_CLEAN_DIR))
    # -- définir le dossier proprement et le créer si besoin
    clean_dir = Path(clean_dir_str).expanduser()
    clean_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["clean_dir"] = str(clean_dir)
    st.write(f"Sorties nettoyées → `{clean_dir}`")

    # ⚠️ Nouveau template LLM: --input/--output et --concurrency
    llm_tpl = st.text_input(
        "Commande LLM",
        value=(
            "python \"{llm_script}\" --input \"{input}\" --output \"{output}\" --concurrency {concurrency} {extra}"
        ),
        help="Placeholders: {llm_script}, {input}, {output}, {concurrency}, {extra}",
    )

    st.markdown("### 🔐 Clé API Mistral")
    
    # 🔐 Chargement sécurisé de la clé API (priorité: secrets > env > saisie manuelle)
    mistral_api_key = None
    key_source = "Non configurée"
    
    # 1. Essayer depuis st.secrets (fichier .streamlit/secrets.toml)
    try:
        if "MISTRAL_API_KEY" in st.secrets:
            mistral_api_key = st.secrets["MISTRAL_API_KEY"]
            if mistral_api_key and mistral_api_key != "votre_clé_mistral_ici":
                key_source = "secrets.toml"
                os.environ["MISTRAL_API_KEY"] = mistral_api_key
            else:
                mistral_api_key = None
    except (FileNotFoundError, KeyError):
        pass
    
    # 2. Fallback vers variable d'environnement système
    if not mistral_api_key:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if mistral_api_key:
            key_source = "Variable système"
    
    # 3. Affichage du statut de la clé (sans afficher la clé elle-même)
    if mistral_api_key:
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ Clé API chargée</strong><br>
            <small>Source: <b>{key_source}</b></small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Clé non configurée</strong><br>
            <small>Configurez dans .streamlit/secrets.toml ou saisissez ci-dessous</small>
        </div>
        """, unsafe_allow_html=True)
        mistral_api_key = st.text_input(
            "Saisir la clé API", 
            type="password",
            help="Sera utilisée uniquement pour cette session"
        )
        if mistral_api_key:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
            key_source = "Saisie manuelle"
    
    st.divider()
    
    with st.expander("🔧 Paramètres avancés", expanded=False):
        st.markdown("**Chemins des scripts**")
        preprocess_script_path = st.text_input("Script prétraitement", value=default_preprocess_script, key="adv_preprocess")
        llm_script_path = st.text_input("Script LLM Mistral", value=default_llm_script, key="adv_llm")
        full_pipeline_script_path = st.text_input("Script Ollama", value=default_full_pipeline_script, key="adv_ollama")
        
        st.markdown("**Variables d'environnement**")
        custom_env_kv = st.text_area("Variables (clé=valeur)", height=100)
        if st.button("✅ Appliquer les variables"):
            if mistral_api_key:
                os.environ["MISTRAL_API_KEY"] = mistral_api_key
        if custom_env_kv.strip():
            for line in custom_env_kv.splitlines():
                if not line.strip():
                    continue
                if "=" not in line:
                    st.warning(f"Ligne ignorée (pas de '='): {line}")
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        st.success("Variables d'environnement appliquées.")

    st.subheader("KB réponses (Ollama RAG)")
    kb_path_inp = st.text_input(
        "Chemin KB (CSV)",
        value=_kb_path_autoload() or str(KB_PATH_DEFAULT),
        help="kb_replies_rich_expanded.csv par défaut."
    )
    st.session_state["kb_path"] = kb_path_inp

# ==================== HEADER PRINCIPAL ====================
st.markdown("""
<div class="gradient-header">
    <h1 style="color: white; margin: 0; text-align: center;">
        📡 Pipeline d'Analyse SAV Free Mobile
    </h1>
    <p style="color: white; text-align: center; margin-top: 0.5rem; opacity: 0.9;">
        🤖 Analyse automatique des tweets clients avec LLM & RAG
    </p>
</div>
""", unsafe_allow_html=True)

# 📊 Metrics en haut de page
if "csv_uploaded" in st.session_state and st.session_state.get("csv_uploaded"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Fichier source", "Chargé", delta="✓")
    with col2:
        preprocess_done = st.session_state.get("clients_only_ok", False)
        st.metric("🧼 Prétraitement", "Terminé" if preprocess_done else "En attente", delta="✓" if preprocess_done else None)
    with col3:
        llm_done = "llm_csv" in st.session_state.paths and st.session_state.paths["llm_csv"]
        st.metric("🤖 Analyse LLM", "Terminée" if llm_done else "En attente", delta="✓" if llm_done else None)
    with col4:
        if llm_done:
            try:
                df_result = pd.read_csv(st.session_state.paths["llm_csv"])
                st.metric("📊 Tweets analysés", len(df_result))
            except:
                st.metric("📊 Tweets analysés", "N/A")
        else:
            st.metric("📊 Tweets analysés", "0")
    st.divider()

# -------------------- Upload --------------------
st.header("📁 1) Importer un CSV brut (tweets)")
st.markdown(f"""
<div class="info-box">
    <strong>📂 Fichier par défaut</strong><br>
    <small><code>{Path(DEFAULT_INPUT_CSV).name}</code></small>
</div>
""", unsafe_allow_html=True)

# Chargement automatique du fichier par défaut au démarrage
if "csv_uploaded" not in st.session_state and Path(DEFAULT_INPUT_CSV).exists():
    st.session_state.paths["uploaded_csv"] = DEFAULT_INPUT_CSV
    st.session_state["csv_uploaded"] = True
    dmin, dmax = get_csv_date_range(DEFAULT_INPUT_CSV)
    st.session_state["csv_date_min"] = dmin
    st.session_state["csv_date_max"] = dmax
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Fichier chargé automatiquement</strong><br>
        <small><code>{Path(DEFAULT_INPUT_CSV).name}</code></small>
    </div>
    """, unsafe_allow_html=True)

uploaded = st.file_uploader("Ou uploader un nouveau fichier CSV", type=["csv"]) 
if uploaded:
    in_csv_path = save_uploaded_file(uploaded, workdir)
    st.session_state.paths["uploaded_csv"] = str(in_csv_path)
    st.session_state["csv_uploaded"] = True
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Fichier importé</strong><br>
        <small><code>{Path(in_csv_path).name}</code></small>
    </div>
    """, unsafe_allow_html=True)
    # Récupérer min/max dates du CSV pour piloter les date_input
    dmin, dmax = get_csv_date_range(str(in_csv_path))
    st.session_state["csv_date_min"] = dmin
    st.session_state["csv_date_max"] = dmax
    with st.expander("Aperçu du CSV importé (50 premières lignes)"):
        preview_csv(in_csv_path)
else:
    in_csv_path = st.session_state.paths.get("uploaded_csv")
    dmin = st.session_state.get("csv_date_min")
    dmax = st.session_state.get("csv_date_max")

# Bouton pour recharger le fichier par défaut
if st.button("🔄 Recharger le fichier par défaut", use_container_width=True):
    if Path(DEFAULT_INPUT_CSV).exists():
        st.session_state.paths["uploaded_csv"] = DEFAULT_INPUT_CSV
        st.session_state["csv_uploaded"] = True
        dmin, dmax = get_csv_date_range(DEFAULT_INPUT_CSV)
        st.session_state["csv_date_min"] = dmin
        st.session_state["csv_date_max"] = dmax
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ Fichier rechargé</strong><br>
            <small><code>{Path(DEFAULT_INPUT_CSV).name}</code></small>
        </div>
        """, unsafe_allow_html=True)
        st.rerun()
    else:
        st.error(f"❌ Fichier introuvable : {DEFAULT_INPUT_CSV}")
    if in_csv_path and (dmin is None or dmax is None):
        dmin, dmax = get_csv_date_range(str(in_csv_path))
        st.session_state["csv_date_min"] = dmin
        st.session_state["csv_date_max"] = dmax

# -------------------- Prétraitement --------------------
st.header("🧼 2) Prétraitement & nettoyage")
st.markdown("""
<div class="info-box">
    <strong>💾 Sortie automatique</strong><br>
    <small><code>clean_client/Prétraitement_LLM/tweets_clients_only.csv</code></small>
</div>
""", unsafe_allow_html=True)
col_p1, col_p2 = st.columns([1, 1])
with col_p1:
    default_clean_name = ""
    if in_csv_path:
        base = Path(in_csv_path).with_suffix("")
        # Nom clair : tweets_nettoyes.csv
        default_clean_name = str(Path(in_csv_path).parent / "Prétraitement_LLM" / "tweets_nettoyes.csv")
    out_clean = st.text_input("Nom du CSV de sortie (nettoyé)", value=default_clean_name)
with col_p2:
    extra_args_pre = st.text_input("Arguments supplémentaires (prétraitement)", value="")

# Note: les filtres ont été déplacés vers la section "3) Traitement LLM" (voir plus bas)
# --- FIN SECTION PRÉTRAITEMENT (filtres retirés) ---

pre_btn = st.button("🚿 Lancer le prétraitement", use_container_width=True, type="primary")
if pre_btn:
    if not in_csv_path:
        st.error("Veuillez d'abord importer un CSV.")
    elif not out_clean:
        st.error("Veuillez donner un nom de fichier de sortie pour le CSV nettoyé.")
    else:
        # Construire commande avec sys.executable et --output-dir = Prétraitement_LLM
        python_exec = sys.executable
        preprocess_dir = get_preprocess_output_dir()
        out_dir = str(preprocess_dir.resolve())
        # Recalculer le nom de sortie dans le sous-dossier
        out_clean_name = Path(out_clean).name
        out_clean = str(preprocess_dir / out_clean_name)
        extra_safe = extra_args_pre or ""
        try:
            tpl = preprocess_tpl or 'python "{pre_script}" --input "{input}" --output "{output}" --output-dir "{output_dir}" {extra}'
            cmd_tmp = tpl.format(pre_script=preprocess_script_path, input=in_csv_path, output=out_clean, output_dir=out_dir, extra=extra_safe)
            if cmd_tmp.strip().startswith("python "):
                cmd = cmd_tmp.replace("python", python_exec, 1)
            else:
                cmd = cmd_tmp
        except Exception:
            cmd = f'{python_exec} "{preprocess_script_path}" --input "{in_csv_path}" --output "{out_clean}" --output-dir "{out_dir}" {extra_safe}'

        with st.status("Exécution du prétraitement…", expanded=True) as status:
            st.code(cmd, language="bash")
            code, out, err = run_command(cmd, workdir=BASE_DIR)
            if out:
                st.text_area("stdout", value=out, height=200)
            if err:
                st.text_area("stderr", value=err, height=200)
            if code == 0 and Path(out_clean).exists():
                status.update(label="Prétraitement terminé ✅", state="complete")
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Fichier nettoyé avec succès</strong><br>
                    <small><code>{out_clean}</code></small>
                </div>
                """, unsafe_allow_html=True)

                # --- Détection workdir-only (Prétraitement_LLM) : pick latest *clients_only* ---
                preprocess_dir = get_preprocess_output_dir()
                best = pick_latest_clients_only(preprocess_dir)
                st.session_state.paths["clean_csv_clients"] = best
                st.session_state.paths["clean_csv"] = best  # impose clients_only comme source LLM
                st.session_state["clients_only_ok"] = bool(best)
                st.session_state.paths["clean_csv_saved_from_pre"] = str(Path(out_clean).resolve())
                if best:
                    st.session_state["llm_source_locked"] = best
                    st.success(f"Fichier clients-only détecté et verrouillé pour le LLM : {st.session_state['llm_source_locked']}")
                    with st.expander("Aperçu du CSV nettoyé (50 premières lignes)"):
                        preview_csv(Path(st.session_state['llm_source_locked']))
                    with open(st.session_state['llm_source_locked'], "rb") as f:
                        st.download_button("⬇️ Télécharger le CSV nettoyé", data=f, file_name=Path(st.session_state['llm_source_locked']).name)
                else:
                    st.info("Aucun fichier '*clients_only*.csv' trouvé dans le sous-dossier Prétraitement_LLM.")
            else:
                status.update(label="Échec du prétraitement ❌", state="error")

# -------------------- Re-verrouillage avant la section LLM (workdir-only) --------------------
locked = st.session_state.get("llm_source_locked")
if locked and Path(locked).exists():
    current = locked
else:
    preprocess_dir = get_preprocess_output_dir()
    current = pick_latest_clients_only(preprocess_dir)
    if current:
        st.session_state["llm_source_locked"] = current

st.session_state.paths["clean_csv_clients"] = current
st.session_state.paths["clean_csv"] = current
st.session_state["clients_only_ok"] = bool(current)

# -------------------- LLM --------------------
st.header("🤖 3) Traitement LLM")
st.markdown("""
<div class="info-box">
    <strong>💾 Flux de sortie</strong><br>
    <small>
        📂 <strong>Source</strong> : <code>Prétraitement_LLM/tweets_clients_only.csv</code><br>
        🌐 <strong>Mistral API</strong> → <code>LLM_Mistral/resultats_analyse_mistral.csv</code><br>
        💻 <strong>Ollama Local</strong> → <code>LLM_Ollama/resultats_analyse_ollama.csv</code>
    </small>
</div>
""", unsafe_allow_html=True)

col_engine1, col_engine2 = st.columns(2)
with col_engine1:
    engine = st.radio(
        "Moteur LLM",
        ["Mistral (API)", "Pipeline complet (BERT+Ollama)"],
        index=0,
    )

# ---------- Source LLM : choisir un fichier traité dans Prétraitement_LLM ----------
st.markdown("### 📄 Source des données")

# Lister directement les fichiers du dossier Prétraitement_LLM
preprocess_folder = get_preprocess_output_dir()
pattern = "*.csv"  # Tous les CSV du dossier Prétraitement_LLM

def list_by_pattern(dirpath: Path, patt: str):
    try:
        return sorted(dirpath.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []

cands = list_by_pattern(preprocess_folder, pattern)

# Rescan avec clé unique
if st.button("🔄 Rescanner le dossier", key="btn_rescan_llm"):
    cands = list_by_pattern(preprocess_folder, pattern)

if not cands:
    current = None
    st.code("Source LLM (active) : <aucun fichier correspondant>")
else:
    # Pré-sélectionner le fichier déjà “locké” si présent
    default_idx = 0
    locked_path = st.session_state.get("llm_source_locked")
    if locked_path:
        try:
            default_idx = next(i for i, p in enumerate(cands) if str(p.resolve()) == str(Path(locked_path).resolve()))
        except StopIteration:
            default_idx = 0

    labels = [f"{p.name} — {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}" for p in cands]
    idx = st.selectbox(
        "Choisir le fichier à utiliser",
        list(range(len(cands))),
        index=default_idx,
        format_func=lambda i: labels[i],
        key="llm_select_file"
    )
    current = str(cands[idx].resolve())

    # Utiliser directement le fichier sans créer d'alias (évite doublon)
    st.session_state["llm_source_locked"] = current

    st.markdown(f"""
    <div class="success-box">
        <strong>📝 Fichier actif</strong><br>
        <small><code>{Path(current).name}</code></small>
    </div>
    """, unsafe_allow_html=True)

# Si la source change -> reset des caches/états des filtres
if current != st.session_state.get("_filter_source"):
    st.session_state["_filter_source"] = current
    try: load_subset_for_dates.clear()
    except: pass
    try: count_tweets_in_range.clear()
    except: pass
    st.session_state["_ids_df"] = None
    st.session_state["_ids_total"] = None
    if current:
        dmin, dmax = get_csv_date_range(current)
        st.session_state["csv_date_min"] = dmin
        st.session_state["csv_date_max"] = dmax

# -------------------- FILTRES LLM (optionnels) : utiliser current (active clean file) --------------------
# Garder l'expander ouvert après chargement des tweets
if "filters_expanded" not in st.session_state:
    st.session_state["filters_expanded"] = False

with st.expander("Filtres LLM (optionnels)", expanded=st.session_state["filters_expanded"]):
    csv_date_min = st.session_state.get("csv_date_min")
    csv_date_max = st.session_state.get("csv_date_max")
    use_date = st.checkbox("Activer filtre par date", value=False)
    date_from = None
    date_to = None
    if use_date:
        cdf, cdt = st.columns(2)
        with cdf:
            date_from = st.date_input("Date min (YYYY-MM-DD)", value=None, min_value=csv_date_min, max_value=csv_date_max)
        with cdt:
            default_to = csv_date_max
            date_to = st.date_input("Date max (YYYY-MM-DD)", value=default_to, min_value=csv_date_min, max_value=csv_date_max)

    # Charger les tweets de la période (depuis le CSV clean actif)
    selected_ids = []
    if st.button("🚀 Charger les tweets de la période (max 5000)", use_container_width=True):
        st.session_state["filters_expanded"] = True  # Garder l'expander ouvert
        if not current:
            st.warning("⚠️ Aucune source LLM active. Exécutez le prétraitement pour générer un fichier clients_only.")
        else:
            with st.spinner("🔄 Chargement des tweets en cours..."):
                sub_df = load_subset_for_dates(current,
                                               date_from.strftime("%Y-%m-%d") if date_from else None,
                                               date_to.strftime("%Y-%m-%d") if date_to else None,
                                               max_rows=5000)
                total_cnt = count_tweets_in_range(current,
                                                  date_from.strftime("%Y-%m-%d") if date_from else None,
                                                  date_to.strftime("%Y-%m-%d") if date_to else None)
                st.session_state["_ids_df"] = sub_df
                st.session_state["_ids_total"] = total_cnt
            st.rerun()  # Recharger pour afficher immédiatement les résultats

    sub_df = st.session_state.get("_ids_df")
    total_cnt = st.session_state.get("_ids_total")
    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
        st.info(f"📦 {total_cnt if total_cnt is not None else len(sub_df)} tweets trouvés. Affichés : {len(sub_df)} (max 5000).")
        def mk_label(row):
            t = str(row.get("full_text", ""))[:120].replace("\n", " ")
            d = row.get("created_at")
            d = d.strftime("%Y-%m-%d %H:%M") if pd.notnull(d) else "?"
            return f"{row['id']}  —  {d}  —  {t}"
        labels = {row["id"]: mk_label(row) for _, row in sub_df.iterrows()}
        options = sub_df["id"].astype(str).tolist()
        selected_ids = st.multiselect("Sélectionner un ou plusieurs tweets", options=options, format_func=lambda x: labels.get(x, x), max_selections=500)
        st.caption(f"{len(selected_ids)} tweet(s) sélectionné(s).")
    else:
        st.caption("⚠️ Cliquez sur 'Charger les tweets de la période' pour remplir la liste (depuis le fichier clean actif).")

    max_for_limit = int(st.session_state.get("_ids_total") or 0)
    if max_for_limit > 0:
        limit_n = st.number_input("Limiter à N lignes", min_value=0, max_value=max_for_limit, value=0, step=1)
    else:
        limit_n = st.number_input("Limiter à N lignes", min_value=0, value=0, step=1)

# Concurrence uniquement pour Mistral API
if engine == "Mistral (API)":
    concurrency = st.number_input("Concurrence (threads)", min_value=1, max_value=128, value=4, step=1)
else:
    concurrency = None

# Valeurs par défaut optimisées pour le pipeline BERT + Ollama
pipeline_model_name = os.getenv("OLLAMA_MODEL", "mistral:7b")
pipeline_rag_top_k = 3
pipeline_timeout = 60
pipeline_limit = 0

if engine == "Pipeline complet (BERT+Ollama)":
    col_pl1, col_pl2 = st.columns(2)
    with col_pl1:
        pipeline_model_name = st.text_input(
            "Modèle Ollama (pipeline)",
            value=os.getenv("OLLAMA_MODEL", "mistral:7b"),
            key="pipeline_model_name_input",
        )
    with col_pl2:
        pipeline_rag_top_k = st.number_input(
            "Nombre d'extraits KB (pipeline)",
            min_value=1,
            max_value=10,
            value=3,  # 3 extraits KB → réponses plus riches et mieux contextualisées
            step=1,
            key="pipeline_rag_top_k_input",
        )

    col_pl3, col_pl4 = st.columns(2)
    with col_pl3:
        pipeline_timeout = st.number_input(
            "Timeout Ollama (s)",
            min_value=10,
            max_value=600,
            value=60,
            step=5,
            key="pipeline_timeout_input",
        )
    with col_pl4:
        pipeline_limit = st.number_input(
            "Limiter à N lignes (pipeline)",
            min_value=0,
            value=0,
            step=1,
            key="pipeline_limit_input",
        )

    st.caption("Ce mode exécute automatiquement BERT+Ollama en une seule commande.")

default_extra_ollama = (
    "--fast-ollama "
    "--ollama-workers 2 "
    "--ollama-num-predict 512 "
    "--ollama-max-retries 1 "
    "--temperature 0.3 "
    "--top_p 0.9 "
    "--repeat_penalty 1.05"
)

extra_args_llm = st.text_input(
    "Arguments supplémentaires (LLM)",
    value=default_extra_ollama if engine == "Pipeline complet (BERT+Ollama)" else "",
    help=(
        "Mistral API: ex. --rpm 0 --max-chars 700. "
        "Pipeline complet: options avancées (par défaut: fast-ollama, 2 workers, num_predict=512, "
        "temperature=0.3, top_p=0.9, repeat_penalty=1.05)."
    ),
)

# --- Sortie ---
sav_only = True
st.caption("La sortie générée contient automatiquement les colonnes SAV attendues.")

# Disable LLM if clients_only missing
clients_ok = bool(st.session_state.get("clients_only_ok", False))
if not clients_ok:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ LLM désactivé</strong><br>
        <small>Aucun fichier 'tweets_cleaned_clients_only.csv' détecté. Exécutez le prétraitement.</small>
    </div>
    """, unsafe_allow_html=True)
try:
    llm_btn = st.button("🧠 Lancer le traitement LLM", disabled=not clients_ok, use_container_width=True, type="primary")
except TypeError:
    llm_btn = st.button("🧠 Lancer le traitement LLM")

if llm_btn:
    if not clients_ok:
        st.error("Impossible : aucun fichier clients-only disponible pour le LLM. Générer 'tweets_cleaned_clients_only.csv' d'abord.")
    else:
        # Use only the clients_only locked/source
        source_clean = st.session_state.paths.get("clean_csv_clients")
        if not source_clean or not Path(source_clean).exists():
            st.error("Fichier clients-only introuvable dans le workdir. Vérifiez le prétraitement.")
        else:
            # Charger cleaned CSV (protection fallback encodings)
            try:
                df_clean = pd.read_csv(source_clean, low_memory=False)
            except Exception:
                df_clean = pd.read_csv(source_clean, low_memory=False, encoding='utf-8', on_bad_lines='skip')

            # Appliquer filtres LLM (date, ids, limit) — tentative tweet_id puis id
            if use_date and date_from:
                lower = pd.to_datetime(date_from)
                df_clean["created_at_dt"] = pd.to_datetime(df_clean.get("created_at", ""), errors="coerce")
                df_clean = df_clean[df_clean["created_at_dt"] >= lower]
            if use_date and date_to:
                upper = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                df_clean["created_at_dt"] = pd.to_datetime(df_clean.get("created_at", ""), errors="coerce")
                df_clean = df_clean[df_clean["created_at_dt"] <= upper]

            if selected_ids:
                if "tweet_id" in df_clean.columns:
                    df_clean = df_clean[df_clean["tweet_id"].astype(str).isin(selected_ids)]
                elif "id" in df_clean.columns:
                    df_clean = df_clean[df_clean["id"].astype(str).isin(selected_ids)]

            if limit_n and int(limit_n) > 0:
                df_clean = df_clean.head(int(limit_n))

            # 🚨 Garde-fou : si aucun tweet après filtres, on n'appelle pas le LLM
            if df_clean.empty:
                st.warning("Aucun tweet à traiter après filtres — le LLM n'est pas lancé.")
                st.stop()

            # Write filtered temp CSV in temp folder
            src_path = Path(source_clean)
            temp_dir = get_temp_dir()
            # Nom clair : tweets_filtres_pour_llm.csv
            temp_fp = temp_dir / "tweets_filtres_pour_llm.csv"
            df_clean.to_csv(temp_fp, index=False, encoding="utf-8")
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Fichier temporaire créé</strong><br>
                <small><code>{temp_fp.name}</code> - {len(df_clean)} tweets filtrés</small>
            </div>
            """, unsafe_allow_html=True)

            # compute automatic llm_output (no user input)
            base_noext = str(src_path.with_suffix(""))
            llm_output = None

            # Build command with sys.executable (already used ailleurs)
            python_exec = sys.executable
            if engine == "Mistral (API)":
                mistral_dir = get_mistral_output_dir()
                temp_dir = get_temp_dir()
                input_path = str(temp_fp)
                # Noms clairs pour Mistral
                rag_output = str(temp_dir / "tweets_avec_contexte_rag.csv")
                llm_output = str(mistral_dir / "resultats_analyse_mistral.csv")
                kb_path = r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\LLM-Tweet-Pipeline\kb_replies_rich_expanded.csv"
                model_name = "distilbert-base-multilingual-cased"

                # Vérifier que la clé API est disponible
                current_key = os.getenv("MISTRAL_API_KEY")
                if not current_key:
                    st.error("⚠️ Merci de configurer la clé API Mistral (secrets.toml ou saisie manuelle).")
                    st.stop()
                
                # S'assurer que la clé est bien définie dans l'environnement
                os.environ["MISTRAL_API_KEY"] = current_key

                cmd_rag = (
                    f'{python_exec} "{os.path.join(BASE_DIR, "add_rag_context.py")}" '
                    f'--input "{input_path}" '
                    f'--output "{rag_output}" '
                    f'--kb "{kb_path}" '
                    f'--model "{model_name}" '
                    f'--top-k 1'
                )

                cmd_llm = (
                    f'{python_exec} "{os.path.join(BASE_DIR, "llm_batch_multitask_pool_mistral.py")}" '
                    f'--input "{rag_output}" '
                    f'--output "{llm_output}" '
                    f'--timeout 120 '
                    f'--concurrency 1 '
                    f'--max-chars 900'
                )

                with st.status("Pipeline Mistral (RAG + API)…", expanded=True) as status:
                    st.write("🧩 Exécution du RAG avant Mistral…")
                    st.code(cmd_rag, language="bash")
                    code_rag, out_rag, err_rag = run_command(cmd_rag, workdir=BASE_DIR)
                    if out_rag:
                        st.text_area("stdout RAG", value=out_rag, height=160)
                    if err_rag:
                        st.text_area("stderr RAG", value=err_rag, height=160)
                    if code_rag != 0 or not Path(rag_output).exists():
                        status.update(label="Échec durant l'étape RAG ❌", state="error")
                        st.stop()

                    st.write("🧠 Traitement LLM via Mistral API…")
                    st.code(cmd_llm, language="bash")
                    code_llm, out_llm, err_llm = run_command(cmd_llm, workdir=BASE_DIR)
                    if out_llm:
                        st.text_area("stdout LLM", value=out_llm, height=200)
                    if err_llm:
                        st.text_area("stderr LLM", value=err_llm, height=200)
                    if code_llm == 0 and Path(llm_output).exists():
                        st.session_state.paths["llm_csv"] = str(Path(llm_output).resolve())
                        status.update(label="Traitement complet terminé ✅", state="complete")
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>✅ Analyse terminée avec succès</strong><br>
                            <small><code>{Path(llm_output).name}</code></small>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("Aperçu du CSV LLM (50 premières lignes)"):
                            preview_csv(Path(llm_output))
                        with open(llm_output, "rb") as f:
                            st.download_button("⬇️ Télécharger le CSV LLM", data=f, file_name=Path(llm_output).name, type="primary")
                    else:
                        status.update(label="Échec du traitement LLM ❌", state="error")
                    st.stop()
            elif engine == "Pipeline complet (BERT+Ollama)":
                # Pour ce mode, exécuter la commande fixe demandée par l'utilisateur
                effective_kb = (st.session_state.get("kb_path") or (_kb_path_autoload() or "")).strip()
                if not effective_kb:
                    st.error("Veuillez définir le chemin de la KB dans la barre latérale.")
                    st.stop()
                if not Path(effective_kb).exists():
                    st.error(f"Fichier KB introuvable: {effective_kb}")
                    st.stop()

                # Mode debug (affiche les logs complets)
                debug_mode = st.checkbox("Mode debug (affiche tous les logs)", value=False)

                # Chemins d'entrée/sortie demandés
                ollama_dir = get_ollama_output_dir()
                llm_input = str(temp_fp)
                # Nom clair pour Ollama
                llm_output_path = ollama_dir / "resultats_analyse_ollama.csv"
                llm_output = str(llm_output_path.resolve())

                # Commande exacte à exécuter (identique à la ligne de commande fournie)
                cmd = [
                    python_exec,
                    str(Path(full_pipeline_script_path).resolve()),
                    "--input", llm_input,
                    "--output", llm_output,
                    "--kb", str(effective_kb),
                    "--model", "mistral",
                    "--rag-top-k", "1",
                    "--timeout", "120",
                    "--ollama-num-predict", "512",
                    "--ollama-workers", "1",
                    "--ollama-max-retries", "1",
                ]

                # Afficher la commande et exécuter via run_command
                with st.status("Exécution du LLM…", expanded=True) as status:
                    st.code(fmt_cmd_for_display(cmd), language="bash")
                    # Execute
                    code, out, err = run_command(cmd, workdir=BASE_DIR)

                    # Debug mode: afficher logs complets
                    if debug_mode:
                        if out:
                            st.text_area("stdout", value=out, height=300)
                        if err:
                            st.text_area("stderr", value=err, height=300)
                    else:
                        # Mode non-debug : afficher uniquement un résumé et l'état final
                        if out:
                            st.text_area("stdout (extrait)", value=out[:4000], height=200)
                        if err:
                            st.text_area("stderr (extrait)", value=err[:4000], height=200)

                    if code == 0 and Path(llm_output).exists():
                        st.session_state.paths["llm_csv"] = str(Path(llm_output).resolve())
                        status.update(label="Traitement LLM terminé ✅", state="complete")
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>✅ Analyse terminée avec succès</strong><br>
                            <small><code>{Path(llm_output).name}</code> - Analyse par Ollama terminée</small>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("Aperçu du CSV LLM (50 premières lignes)"):
                            preview_csv(Path(llm_output))
                        with open(llm_output, "rb") as f:
                            st.download_button("⬇️ Télécharger le CSV LLM", data=f, file_name=Path(llm_output).name, type="primary")
                    else:
                        status.update(label="Échec du traitement LLM ❌", state="error")

# La page se termine désormais après l'étape 3 (Mistral ou pipeline complet).
st.divider()
st.caption("Templates de commande quotés + exécution depuis le dossier des scripts pour éviter les erreurs de chemins sous Windows.")
