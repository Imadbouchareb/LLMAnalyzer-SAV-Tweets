# pages/3_Agent_SAV.py
# Refonte complète de l’écran Agent pour l’aligner sur Manager/Analyste.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from itertools import combinations
import unicodedata

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from lib.data import load_df, persist_ticket_updates, apply_edits
from lib.state import get_cfg
from lib.ui import inject_style, set_container_wide, inject_sticky_css, hide_sidebar

HOME_PAGE = "pages/0_Accueil.py"   # Page d’accueil “SAV Tweets — Maquettes”

st.set_page_config(
    page_title="File Agent SAV",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS fallback pour masquer la sidebar de façon sûre
st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], section[data-testid="stSidebar"] { display:none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

hide_sidebar()
inject_style()
set_container_wide()
inject_sticky_css()
cfg = get_cfg()

# Configuration Altair pour fond blanc sur tous les graphiques
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

# Sûr : vider la sidebar si disponible (évite erreurs sur certaines versions/env)
try:
    st.sidebar.empty()
except Exception:
    pass

DOW_LABELS = {
    0: "Lun",
    1: "Mar",
    2: "Mer",
    3: "Jeu",
    4: "Ven",
    5: "Sam",
    6: "Dim",
}
DOW_ORDER = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


# -----------------------------------------------------------------------------
# Configuration pour graphiques Altair avec fond clair
# -----------------------------------------------------------------------------
def configure_chart_light(chart):
    """Configure un graphique Altair avec fond blanc et couleurs claires"""
    return chart.configure(
        background='white'
    ).configure_view(
        stroke=None,
        strokeWidth=0
    ).configure_axis(
        labelColor="#1E293B",
        titleColor="#1E40AF",
        gridColor="#E2E8F0",
        domainColor="#CBD5E1",
        tickColor="#CBD5E1",
        labelFontSize=11,
        titleFontSize=12,
        titleFontWeight=600
    ).configure_legend(
        labelColor="#1E293B",
        titleColor="#1E40AF",
        labelFontSize=11,
        titleFontSize=12
    ).configure_title(
        color="#1E40AF",
        fontSize=14,
        fontWeight=700
    )


# -----------------------------------------------------------------------------
# Helpers de normalisation
# -----------------------------------------------------------------------------
def _parse_dt(series: pd.Series) -> pd.Series:
    """
    Essaie d'interpréter une colonne date/heure -> pandas.Timestamp sans timezone.
    """
    # Première tentative: format standard
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().sum() == 0:
        # 2e tentative style "27/10/2025 14:33"
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # Enlever le timezone si présent
    if pd.api.types.is_datetime64tz_dtype(s):
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            try:
                s = s.dt.tz_convert(None)
            except Exception:
                pass
    return s


def _ensure_list(raw) -> list[str]:
    """
    Garantit que chaque ligne a une liste de thèmes propre.
    Exemples d'inputs possibles :
      - ['facturation','reseau']
      - "['facturation', 'reseau']"
      - "reseau"
      - NaN
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass
    if isinstance(raw, str) and raw.strip():
        # essaie d'évaluer liste python/JSON
        import ast, json
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        # sinon => un seul thème
        return [raw.strip()]
    return []


def _clean_theme_token(x: str) -> str:
    return (
        str(x)
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("{", "")
        .replace("}", "")
        .replace("'", "")
        .replace('"', "")
        .strip()
    )


def _flat_themes(series: pd.Series) -> list[str]:
    tokens: list[str] = []
    for raw in series:
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                cleaned = _clean_theme_token(item)
                if cleaned:
                    tokens.append(cleaned)
    return tokens

# Nouveau helper demandé : extrait tous les thèmes présents dans le DF
def _all_themes_from_df(df: pd.DataFrame) -> list[str]:
    values = set()
    
    # Priorité 1 : utiliser primary_label si disponible
    if "primary_label" in df.columns:
        for val in df["primary_label"].dropna():
            if val and str(val).strip():
                values.add(str(val).strip())
    
    # Priorité 2 : utiliser theme_primary si disponible
    if "theme_primary" in df.columns:
        for val in df["theme_primary"].dropna():
            if val and str(val).strip():
                values.add(str(val).strip())
    
    # Priorité 3 : utiliser themes_list
    if "themes_list" in df.columns:
        for L in df["themes_list"]:
            if isinstance(L, (list, tuple, set)):
                for x in L:
                    tok = _clean_theme_token(x)
                    if tok:
                        values.add(tok)
    
    return sorted(values)


def _prepare_agent_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le DataFrame en colonnes prêtes pour l'analyse Agent.
    On rend les noms cohérents même si le CSV a des variations.
    """
    df = df_in.copy()

    # 1. tweet_id
    if "tweet_id" not in df.columns:
        df["tweet_id"] = df.index.astype(str)
    df["tweet_id"] = df["tweet_id"].astype(str)

    # 2. date
    date_col = next(
        (c for c in [
            "created_at_dt", "created_at", "date", "datetime", "timestamp",
            "posted_at", "tweet_created_at", "tweet_date", "createdAt"
        ] if c in df.columns),
        None,
    )
    if date_col:
        # Parser la colonne date
        parsed_dates = _parse_dt(df[date_col])
        # Si la colonne existe déjà sous le nom created_at_dt et qu'on la réassigne, pas de problème
        df["created_at_dt"] = parsed_dates
    else:
        df["created_at_dt"] = pd.Series(pd.NaT, index=df.index)

    # 3. texte du tweet
    text_col = next(
        (c for c in ["text_raw","text_raw", "text", "tweet", "content", "body"] if c in df.columns),
        None,
    )
    df["text_raw"] = df[text_col].astype(str) if text_col else ""

    # 4. thèmes
    theme_col = next(
        (c for c in ["themes_list", "liste_thèmes", "liste_themes", "themes", "topics", "labels"] if c in df.columns),
        None,
    )
    df["themes_list"] = df[theme_col].apply(_ensure_list) if theme_col else [[] for _ in range(len(df))]
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

    # 5. sentiment
    sentiment_col = next(
        (c for c in ["sentiment_label", "sentiment", "llm_sentiment"] if c in df.columns),
        None,
    )
    df["sentiment_label"] = df[sentiment_col].fillna("").astype(str) if sentiment_col else ""

    # 6. statut SAV (on démarre TOUJOURS vide, peu importe les valeurs du CSV)
    df["status"] = ""

    # 7. assignee
    assignee_col = next(
        (c for c in ["assigned_to", "agent", "handler"] if c in df.columns),
        None,
    )
    df["assigned_to"] = df[assignee_col].fillna("").astype(str) if assignee_col else ""

    # 8. urgence / sévérité
    df["llm_urgency_0_3"] = pd.to_numeric(df.get("llm_urgency_0_3", 0.0), errors="coerce").fillna(0.0)
    df["llm_severity_0_3"] = pd.to_numeric(df.get("llm_severity_0_3", 0.0), errors="coerce").fillna(0.0)

    # 9. auteur
    author_col = next(
        (c for c in ["author", "screen_name", "user", "username", "auteur"] if c in df.columns),
        None,
    )
    df["author"] = df[author_col].fillna("").astype(str) if author_col else ""

    # 10. réponse agent
    reply_col = next(
        (c for c in ["agent_response", "response_text", "reply_body"] if c in df.columns),
        None,
    )
    df["agent_response"] = df[reply_col].fillna("").astype(str) if reply_col else ""
    summary_col = next((c for c in ["llm_summary", "resume_llm", "summary_llm"] if c in df.columns), None)
    suggestion_col = next(
        (c for c in ["llm_reply_suggestion", "llm_suggestion", "suggestion_llm"] if c in df.columns),
        None,
    )
    df["llm_summary"] = df[summary_col].fillna("").astype(str) if summary_col else ""
    df["llm_reply_suggestion"] = df[suggestion_col].fillna("").astype(str) if suggestion_col else ""
    if pd.api.types.is_datetime64tz_dtype(df["created_at_dt"]):
        df["created_at_dt"] = df["created_at_dt"].dt.tz_convert(None)
    return df


def _sample_df() -> pd.DataFrame:
    """
    Jeu factice 30 lignes (fallback démo).
    """
    base = pd.Timestamp.now().floor("H")
    rows = []
    tones = ["negatif", "neutre", "positif"]
    themes = [["réseau"], ["facturation"], ["mobile"], ["facturation", "réseau"]]
    statuses = ["Ouvert", "A traiter", "En attente client", "Clos"]
    for i in range(30):
        rows.append(
            {
                "tweet_id": f"SAMPLE-{i+1:03d}",
                "created_at_dt": base - pd.Timedelta(hours=2 * i),
                "sentiment_label": tones[i % len(tones)],
                "themes_list": themes[i % len(themes)],
                "llm_urgency_0_3": float((i + 1) % 4),
                "llm_severity_0_3": float(i % 4),
                "status": statuses[i % len(statuses)],
                "assigned_to": f"agent_{i%4+1}",
                "author": f"client_{i%8+1}",
                "text_raw": "Message client factice pour la file d’attente.",
                "agent_response": "",
                "llm_summary": "Synthèse automatique du ticket.",
                "llm_reply_suggestion": "Réponse suggérée par l’assistant.",
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES (FIX ESSENTIEL)
# -----------------------------------------------------------------------------
def _load_agent_data(dataset_key: str):
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
                    return _prepare_agent_df(data), uploaded_csv_path, notes
        except Exception as exc:
            notes.append(f"last_dataset.txt: erreur -> {exc}")
    
    # PRIORITÉ 2 : Essayer load_df avec dataset_key
    try:
        direct = load_df(dataset_key)
        if direct is not None and len(direct) > 0:
            notes.append(f"load_df('{dataset_key}') : OK ({len(direct)} lignes)")
            return _prepare_agent_df(direct), f"load_df('{dataset_key}')", notes
        notes.append(f"load_df('{dataset_key}') : vide")
    except Exception as exc:
        notes.append(f"load_df('{dataset_key}') : échec -> {exc}")

    # PRIORITÉ 3 : Fallback sur les chemins candidats
    candidates = [
        APP_ROOT / "tweets_scored_llm.csv",
        APP_ROOT / "data" / "tweets_scored_llm.csv",
        Path(r"C:\projetrncp\tweets_scored_llm.csv"),
        Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\tweets_scored_llm.csv"),
        Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
    ]
    for path in candidates:
        try:
            if not path.exists():
                notes.append(f"{path}: introuvable")
                continue
            data = load_df(str(path))
            if data is not None and len(data) > 0:
                notes.append(f"{path}: OK ({len(data)} lignes)")
                return _prepare_agent_df(data), str(path), notes
            notes.append(f"{path}: vide ou illisible")
        except Exception as exc:
            notes.append(f"{path}: erreur -> {exc}")
    return pd.DataFrame(), "", notes


# -----------------------------------------------------------------------------
# CHARGE LES DONNÉES ICI
# -----------------------------------------------------------------------------
dataset_key = cfg.get("agent_dataset_key", cfg.get("manager_dataset_key", "tweets_scored_llm"))
df_real, used_path, debug_notes = _load_agent_data(dataset_key)
df = df_real if not df_real.empty else _sample_df()
data_loaded_ok = not df_real.empty
if "llm_summary" not in df.columns:
    df["llm_summary"] = ""
if "llm_reply_suggestion" not in df.columns:
    df["llm_reply_suggestion"] = ""

# Appliquer les edits persistés (si présents) — permet de restaurer les statuts/réponses traitées
try:
    df = apply_edits(df)
except Exception:
    # Ne pas casser l'UI si apply_edits échoue ; on continue avec df tel quel
    pass

if "agent_df" not in st.session_state:
    st.session_state["agent_df"] = df.copy()
df = st.session_state["agent_df"]

# --- Defaults session (évite le KeyError) -------------------------------
st.session_state.setdefault("agent_active_view", "queue")
st.session_state.setdefault("agent_thr_urg", 2.0)
st.session_state.setdefault("agent_date_preset", "all")
st.session_state.setdefault("agent_selected_ids", [])
# versioning des filtres pour les resets robustes
st.session_state.setdefault("filters_ver", 1)

# Helper de normalisation pour comparer statuts sans accents / casse
def _norm_status(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()

# Fonction de reset safe (versioning)
def _reset_filters():
    # Supprimer d'anciens états (au cas où)
    for k in [
        "agent_tone", "agent_status", "agent_urgency",
        "agent_theme", "agent_search",
        "agent_period", "agent_time_dates", "agent_time_dates_input",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    # Forcer de NOUVELLES clés de widgets (versioning)
    st.session_state["filters_ver"] = int(st.session_state.get("filters_ver", 1)) + 1
    # Vider la sélection de ticket
    st.session_state["agent_selected_ids"] = []
    st.rerun()

def _update_status(ids: list[str], new_status: str, also_set_response: str | None = None):
    if not ids:
        st.warning("Sélectionne d'abord au moins un ticket.", icon="⚠️")
        return
    df_work = st.session_state["agent_df"]
    mask = df_work["tweet_id"].isin(ids)
    if also_set_response is not None and "agent_response" in df_work.columns:
        df_work.loc[mask, "agent_response"] = also_set_response
    df_work.loc[mask, "status"] = new_status
    st.session_state["agent_df"] = df_work
    # Persister les modifications pour que Analyste/Manager les voient
    try:
        persist_ticket_updates(
            tweet_ids=ids,
            status=new_status,
            assigned_to="",  # adapter si réaffectation connue / passer assigned_to si dispo
            agent_response=(also_set_response or ""),
        )
    except Exception as exc:
        # Ne pas casser l'UI : afficher l'erreur pour debugging (fichier data/sav_edits.csv)
        st.error("Erreur lors de la sauvegarde des modifications (sav_edits.csv). Vérifie permissions/chemin.")
        try:
            st.exception(exc)
        except Exception:
            # fallback si st.exception indisponible dans l'environnement
            st.write(repr(exc))
    # Message utilisateur (toast/feedback)
    if hasattr(st, "toast"):
        st.toast(f"{len(ids)} ticket(s) → statut « {new_status} »", icon="✅")
    else:
        st.success(f"{len(ids)} ticket(s) → statut « {new_status} »")
    # Synchroniser automatiquement le multiselect 'Statut' sur la nouvelle valeur (version courante)
    cur_ver = int(st.session_state.get("filters_ver", 1))
    try:
        st.session_state[f"agent_status_{cur_ver}"] = [new_status]
    except Exception:
        # en cas d'impossibilité d'écrire la clé, ignorer proprement
        pass
    st.rerun()

st.title("File d’attente Agent SAV")

# Bouton retour à l'accueil → envoie vers la page d’accueil “Maquettes”
if st.button("🏠 Retour à l’accueil", key="agent_back_home"):
    st.switch_page(HOME_PAGE)

# Bouton Réinitialiser filtres (au-dessus des widgets, utilise on_click)
st.button("🔄 Réinitialiser filtres", use_container_width=True, on_click=_reset_filters, key="agent_reset_filters_btn")

st.markdown("### 🔍 Filtres Agent")

# versioning keys pour éviter StreamlitAPIException lors du reset
ver = int(st.session_state.get("filters_ver", 1))

if df["created_at_dt"].notna().any():
    min_d_raw = df["created_at_dt"].min().date()
    max_d_raw = df["created_at_dt"].max().date()
    extended_min = (pd.Timestamp(min_d_raw) - pd.Timedelta(days=540)).date()
    extended_max = (pd.Timestamp(max_d_raw) + pd.Timedelta(days=180)).date()
    date_from, date_to = st.date_input(
        "Période",
        value=(min_d_raw, max_d_raw),
        min_value=extended_min,
        max_value=extended_max,
        key=f"agent_period_{ver}",
    )
else:
    date_from = date_to = None
    st.caption("Dates indisponibles dans le fichier.")

# Options du filtre Statut (fixe, aligne avec boutons d'action)
status_options = [
    "Ouvert",                  # ≙ statut NON VIDE (spécial)
    "En attente client",
    "Réaffecté",
    "Répondu",
    "Suggestion acceptée",
    "Clos",
]

tone_options = sorted(df["sentiment_label"].replace("", "Non précisé").unique())
# suppression du filtre "Assigné à" côté UI

# options thèmes
theme_options = _all_themes_from_df(df)

# nouvelle rangée 5 colonnes : Sentiment, Statut, Urgence, Thème, Recherche
c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.2, 2])
with c1:
    tone_filter = st.multiselect("Sentiment", options=tone_options, key=f"agent_tone_{ver}")
with c2:
    status_filter = st.multiselect("Statut", options=status_options, key=f"agent_status_{ver}")
with c3:
    urgency_min = st.slider("Urgence min.", 0.0, 3.0, 0.0, 0.1, key=f"agent_urgency_{ver}")
with c4:
    theme_filter = st.multiselect("Thème", options=theme_options, key=f"agent_theme_{ver}")
with c5:
    search_text = st.text_input("Recherche (ID / texte / auteur)", "", key=f"agent_search_{ver}").strip()

info_col1, info_col2 = st.columns([1, 4])
with info_col1:
    st.caption("📂 Source données")
with info_col2:
    if data_loaded_ok:
        st.success(f"✅ {len(df_real):,} lignes chargées depuis :\n{used_path}")
    else:
        st.error("Aucune donnée réelle trouvée – mode démo 30 tickets.")
with st.expander("Diagnostic import"):
    for note in debug_notes:
        st.text(note)

# applique filtres (utiliser les variables locales, pas st.session_state)
flt = df.copy()
if date_from and date_to:
    flt = flt[
        (flt["created_at_dt"].dt.date >= pd.to_datetime(date_from).date())
        & (flt["created_at_dt"].dt.date <= pd.to_datetime(date_to).date())
    ]
if tone_filter:
    flt = flt[flt["sentiment_label"].replace("", "Non précisé").isin(tone_filter)]
# Filtrage Statut : comparer normalisé
if status_filter:
    # "Ouvert" est un cas spécial = tout ticket dont 'status' est NON VIDE
    if "Ouvert" in status_filter:
        flt = flt[flt["status"].astype(str).str.strip() != ""]
    else:
        flt = flt[flt["status"].isin(status_filter)]
if urgency_min > 0:
    flt = flt[flt["llm_urgency_0_3"] >= urgency_min]

# Appliquer le filtre thème
if theme_filter:
    wanted = set(theme_filter)
    
    # Créer un masque pour chaque ligne
    def row_matches_theme(row):
        # 1. Vérifier primary_label
        if "primary_label" in row.index and pd.notna(row["primary_label"]):
            if str(row["primary_label"]).strip() in wanted:
                return True
        
        # 2. Vérifier theme_primary
        if "theme_primary" in row.index and pd.notna(row["theme_primary"]):
            if str(row["theme_primary"]).strip() in wanted:
                return True
        
        # 3. Vérifier themes_list
        if "themes_list" in row.index:
            L = row["themes_list"]
            if isinstance(L, (list, tuple, set)):
                theme_tokens = set(_clean_theme_token(x) for x in L)
                if theme_tokens & wanted:
                    return True
        
        return False
    
    flt = flt[flt.apply(row_matches_theme, axis=1)]

if search_text:
    lower = search_text.lower()
    flt = flt[
        flt["tweet_id"].str.lower().str.contains(lower, na=False)
        | flt["text_raw"].str.lower().str.contains(lower, na=False)
        | flt["author"].str.lower().str.contains(lower, na=False)
    ]

# si rien après filtre
if flt.empty:
    st.info("Aucun ticket ne correspond aux filtres.")
    st.stop()

open_mask = flt["status"].astype(str).str.strip() != ""
urgent_mask = (flt["llm_urgency_0_3"] >= 2) & (flt["sentiment_label"].str.lower() == "négatif")
today_mask = flt["created_at_dt"].dt.date == pd.Timestamp.today().date()

# KPI — recalculés sur la vue filtrée flt
# Ouverts / À traiter = tickets dont status n'est pas vide (status posé = considéré "ouvert / à traiter")
ouverts_a_traiter = int(open_mask.sum())

# métriques (4 colonnes) — plus de "Traités"
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickets affichés", f"{len(flt):,}")
m2.metric("Ouverts / À traiter", f"{ouverts_a_traiter:,}")
m3.metric("Urgents (≥2)", f"{urgent_mask.sum():,}")
m4.metric("Nouveaux aujourd’hui", f"{today_mask.sum():,}")

flat_theme_tokens = _flat_themes(flt["themes_list"])
theme_counts_df = (
    pd.Series(flat_theme_tokens)
    .value_counts()
    .rename_axis("theme")
    .reset_index(name="count")
    .head(50)
    if flat_theme_tokens
    else pd.DataFrame(columns=["theme", "count"])
)

timeline_source = flt.copy()
timeline_all = (
    timeline_source.assign(date=timeline_source["created_at_dt"].dt.date)
    .groupby("date", dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("date")
    if timeline_source["created_at_dt"].notna().any()
    else pd.DataFrame(columns=["date", "count"])
)
hourly_all = (
    timeline_source.assign(hour=timeline_source["created_at_dt"].dt.hour)
    .groupby("hour")
    .size()
    .reset_index(name="count")
    .sort_values("hour")
    if timeline_source["created_at_dt"].notna().any()
    else pd.DataFrame(columns=["hour", "count"])
)
heatmap_all = (
    timeline_source.assign(dow=timeline_source["created_at_dt"].dt.dayofweek, hour=timeline_source["created_at_dt"].dt.hour)
    .groupby(["dow", "hour"])
    .size()
    .reset_index(name="count")
    if timeline_source["created_at_dt"].notna().any()
    else pd.DataFrame(columns=["dow", "hour", "count"])
)
if not heatmap_all.empty:
    heatmap_all["Jour"] = heatmap_all["dow"].map(DOW_LABELS)
tone_counts = (
    flt["sentiment_label"]
    .replace("", "Non précisé")
    .str.capitalize()
    .value_counts()
    .rename_axis("sentiment")
    .reset_index(name="count")
)

# Extraire le thème principal
if "theme_primary" in flt.columns:
    theme_col = flt["theme_primary"].fillna("").astype(str).str.strip().replace("", "Non précisé")
elif "themes_list" in flt.columns:
    theme_col = flt["themes_list"].apply(
        lambda L: str(L[0]) if isinstance(L, list) and len(L) > 0 else "Non précisé"
    )
else:
    theme_col = pd.Series(["Non précisé"] * len(flt), index=flt.index)

sentiment_theme_all = (
    flt.assign(
        theme=theme_col,
        sentiment=flt["sentiment_label"].replace("", "Non précisé").str.capitalize(),
    )
    .groupby(["theme", "sentiment"])
    .size()
    .reset_index(name="count")
)
combo_rows_all = []
for raw_list in flt["themes_list"]:
    cleaned = sorted(set(_clean_theme_token(t) for t in raw_list if str(t).strip()))
    if len(cleaned) >= 2:
        combo_rows_all.extend(combinations(cleaned, 2))
co_occ_df = (
    pd.Series(combo_rows_all)
    .value_counts()
    .reset_index()
    .rename(columns={"index": "binôme", 0: "count"})
    .assign(binôme=lambda d: d["binôme"].apply(lambda pair: f"{pair[0]} / {pair[1]}"))
    .head(30)
    if combo_rows_all
    else pd.DataFrame(columns=["binôme", "count"])
)
author_counts = (
    flt["author"].replace("", "Non précisé").value_counts().rename_axis("author").reset_index(name="count")
)
created_series = flt["created_at_dt"]
if pd.api.types.is_datetime64tz_dtype(created_series):
    created_series = created_series.dt.tz_convert(None)
now_utc = pd.Timestamp.utcnow().tz_localize(None)
age_hours_series = (now_utc - created_series).dt.total_seconds() / 3600
workload_df = (
    flt.assign(assigned_to=flt["assigned_to"].replace("", "Non assigné"))
    .groupby("assigned_to")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
age_stats_df = (
    flt.assign(
        assigned_to=flt["assigned_to"].replace("", "Non assigné"),
        age_hours=age_hours_series,
    )
    .groupby("assigned_to")["age_hours"]
    .agg(mean="mean", median="median")
    .reset_index()
)
status_counts = flt["status"].value_counts().reset_index().rename(columns={"index": "Statut", "status": "count"})

left, main = st.columns([1, 3.6], gap="medium")
with left:
    st.markdown('<div class="ma-card"><h3>Navigation</h3></div>', unsafe_allow_html=True)
    nav_items = [
        ("File d’attente", "queue"),
        ("Urgences", "urgences"),
        ("Thèmes", "themes"),
        ("Temps & sentiment", "temps"),
        ("Affectations", "affect"),
        ("Exports", "exports"),
        ("Logs", "logs"),
    ]
    for label, view_value in nav_items:
        if st.button(
            label,
            use_container_width=True,
            disabled=(st.session_state.get("agent_active_view", "queue") == view_value),
            key=f"agent_nav_{view_value}",
        ):
            st.session_state["agent_active_view"] = view_value
            st.rerun()

with main:
    st.markdown('<div class="ma-card"><h3>Insights</h3></div>', unsafe_allow_html=True)
    active_view = st.session_state.get("agent_active_view", "queue")

    if active_view == "queue":
        queue_df = flt.copy()
        if queue_df.empty:
            st.info("Aucun ticket dans la file d’attente.")
        else:
            # Construire la vue de la table priorisée
            queue_view = queue_df[
                [
                    "tweet_id",
                    "created_at_dt",
                    "author",
                    "text_raw",
                    "sentiment_label",
                    "themes_list",
                    "llm_urgency_0_3",
                    "status",
                    "assigned_to",
                ]
            ].copy()
            # Supprimer les colonnes dupliquées si elles existent (ex: text_raw en double)
            queue_view = queue_view.loc[:, ~queue_view.columns.duplicated()]
            queue_view["thèmes"] = queue_view["themes_list"].apply(lambda L: ", ".join(L[:3]))
            queue_view = queue_view.drop(columns=["themes_list"])
            queue_view = queue_view.rename(
                columns={
                    "tweet_id": "ID",
                    "created_at_dt": "Date",
                    "author": "Auteur",
                    "text_raw": "Message",
                    "sentiment_label": "Sentiment",
                    "llm_urgency_0_3": "Urgence",
                    "status": "Statut",
                    "assigned_to": "Assigné",
                }
            ).sort_values("Date", ascending=False)

            # Mettre 'Statut' en 2e colonne et éviter de scroller pour y accéder
            col_order = ["ID", "Statut", "Date", "Auteur", "Message", "Sentiment", "Urgence", "Assigné"]
            queue_view = queue_view[[c for c in col_order if c in queue_view.columns]]

            # Remonter la sélection en haut si elle existe en session
            _sel = st.session_state.get("agent_selected_ids", [])
            if _sel:
                queue_view = (
                    queue_view
                    .assign(__sel=queue_view["ID"].isin(_sel))
                    .sort_values(["__sel", "Date"], ascending=[False, False])
                    .drop(columns="__sel")
                )

            st.subheader("File d’attente priorisée")
            # Affichage AgGrid (avec getRowId si possible) + fallback
            if AGGRID_OK:
                gb = GridOptionsBuilder.from_dataframe(queue_view)
                gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, sortable=True, filter=True)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
                gb.configure_selection("single", use_checkbox=False)

                # 'Statut' épinglé à gauche
                try:
                    gb.configure_column("Statut", pinned="left", width=140)
                except Exception:
                    pass

                # Style ligne sélectionnée + getRowId
                row_style = JsCode("""
                function(params) {
                  if (params.node && params.node.isSelected()) {
                    return { 'backgroundColor': 'rgba(255,255,255,0.07)' };
                  }
                  return {};
                }
                """)
                try:
                    gb.configure_grid_options(getRowId=JsCode("function(p){return p.data.ID;}"), rowStyle=row_style)
                    allow_unsafe = True
                except Exception:
                    # fallback si configure_grid_options ne supporte pas ces options
                    try:
                        gb.configure_grid_options(getRowId=JsCode("function(params){ return params.data.ID; }"))
                    except Exception:
                        pass
                    allow_unsafe = False

                grid_resp = AgGrid(
                    queue_view,
                    gridOptions=gb.build(),
                    height=420,
                    fit_columns_on_grid_load=True,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=allow_unsafe,
                    theme='light',
                    key="agent_queue_grid",
                ) or {}
            else:
                grid_resp = {}
                st.dataframe(queue_view, use_container_width=True, height=420)

            with st.expander("Recherche rapide par ID"):
                select_options = ["(Choisir)"] + queue_view["ID"].tolist()
                selected_dropdown = st.selectbox("Sélectionne un ID de tweet", select_options, key="agent_id_select")

            # LOGIQUE DE SÉLECTION "STICKY" (persistante)
            prev_selected = st.session_state.get("agent_selected_ids", [])
            selected_ids = prev_selected[:]  # base = valeur précédente

            # 1) sélection AgGrid (si présente)
            if AGGRID_OK:
                sel_rows = grid_resp.get("selected_rows") if isinstance(grid_resp, dict) else None
                ag_selected = [r.get("ID") for r in (sel_rows or []) if r.get("ID")]
                prev = st.session_state.get("agent_selected_ids", [])
                if ag_selected and ag_selected != prev:
                    st.session_state["agent_selected_ids"] = ag_selected
                    # forcer rerun pour remonter la ligne sélectionnée
                    st.rerun()
                elif ag_selected:
                    selected_ids = ag_selected

            # 2) sélection via la liste déroulante (prioritaire si choisie)
            if selected_dropdown and selected_dropdown != "(Choisir)":
                st.session_state["agent_selected_ids"] = [selected_dropdown]
                st.rerun()

            # 3) si la sélection courante n’est plus visible dans la vue filtrée, on l’efface
            if selected_ids and not queue_view["ID"].isin(selected_ids).any():
                selected_ids = []

            # 4) persistance
            st.session_state["agent_selected_ids"] = selected_ids

            # Actions rapides (désactivées si aucune sélection)
            st.subheader("Actions rapides")
            act_cols = st.columns(3)
            _selected_ids = st.session_state.get("agent_selected_ids", [])
            no_sel = (len(_selected_ids) == 0)

            if act_cols[0].button("↩️ Réaffecter", use_container_width=True, key="agent_action_reassign", disabled=no_sel):
                _update_status(_selected_ids, "Réaffecté")

            if act_cols[1].button("✅ Clore", use_container_width=True, key="agent_action_close", disabled=no_sel):
                _update_status(_selected_ids, "Clos")

            if act_cols[2].button("⏳ Mettre en attente client", use_container_width=True, key="agent_action_pending", disabled=no_sel):
                _update_status(_selected_ids, "En attente client")

            if _selected_ids:
                st.caption(f"🎯 Ticket sélectionné : **{_selected_ids[0]}**")

            # suite : résumé LLM & suggestion (inchangé mais utilise st.session_state["agent_selected_ids"])
            if st.session_state.get("agent_selected_ids"):
                sel_df = queue_df[queue_df["tweet_id"].isin(st.session_state["agent_selected_ids"])][
                    ["tweet_id", "author", "llm_summary", "llm_reply_suggestion", "text_raw"]
                ]
                for row in sel_df.itertuples(index=False):
                    with st.expander(f"Ticket {row.tweet_id} — {row.author}", expanded=True):
                        st.markdown(f"**Résumé LLM :**\n\n{row.llm_summary or '_Non fourni_'}")
                        st.markdown(f"**Suggestion LLM :**\n\n{row.llm_reply_suggestion or '_Non fournie_'}")
                        st.markdown("**Message d’origine :**")
                        st.write(row.text_raw)
                        response_key = f"agent_reply_{row.tweet_id}"
                        if response_key not in st.session_state:
                            st.session_state[response_key] = ""
                        action_cols = st.columns(2)
                        if action_cols[1].button("✅ Accepter la suggestion LLM", key=f"accept_btn_{row.tweet_id}", disabled=False):
                            suggested = row.llm_reply_suggestion or ""
                            st.session_state[response_key] = suggested
                            _update_status([row.tweet_id], "Suggestion acceptée", also_set_response=suggested)
                        reply_text = st.text_area("Réponse agent", key=response_key, placeholder="Rédige ta réponse ici…")
                        if action_cols[0].button("💬 Répondre", key=f"reply_btn_{row.tweet_id}", disabled=False):
                            _update_status([row.tweet_id], "Répondu", also_set_response=reply_text)
            else:
                st.info("Sélectionne un ticket pour afficher le résumé et la suggestion LLM.")

    elif active_view == "urgences":
        # Seuil minimum pour les urgences = 2.0 ET sentiment négatif uniquement
        urgent_threshold = 2.0
        urgent_df_raw = flt.loc[
            (flt["llm_urgency_0_3"] >= urgent_threshold) & 
            (flt["sentiment_label"].str.lower() == "négatif")
        ].copy()
        urgent_count_raw = len(urgent_df_raw)
        total_count = len(flt)
        # Ratio utilisé ensuite pour afficher % urgents
        ratio_pct = (100 * urgent_count_raw / total_count) if total_count else 0.0

        st.session_state.setdefault("agent_urg_year", "Toutes")
        years_all = (
            sorted(
                flt["created_at_dt"]
                .dropna()
                .dt.year
                .astype(int)
                .unique()
                .tolist()
            )
            if "created_at_dt" in flt.columns and flt["created_at_dt"].notna().any()
            else []
        )
        base_year_options = [str(y) for y in years_all]
        current_year_choice = str(st.session_state.get("agent_urg_year", "Toutes"))
        if current_year_choice != "Toutes" and current_year_choice not in base_year_options:
            if current_year_choice.isdigit():
                base_year_options.append(current_year_choice)
                base_year_options = sorted({*base_year_options}, key=int)
            else:
                base_year_options.append(current_year_choice)
        year_options = ["Toutes"] + base_year_options
        st.session_state["agent_urg_year"] = current_year_choice

        selected_year = st.selectbox(
            "Filtrer par année",
            options=year_options,
            key="agent_urg_year",
        )

        if selected_year != "Toutes":
            try:
                selected_year_int = int(selected_year)
            except ValueError:
                urgent_df = urgent_df_raw.iloc[0:0].copy()
            else:
                urgent_df = urgent_df_raw[
                    urgent_df_raw["created_at_dt"].dt.year == selected_year_int
                ].copy()
        else:
            urgent_df = urgent_df_raw.copy()

        if urgent_df.empty:
            st.info(
                f"Aucune urgence détectée avec le seuil actuel ≥ {urgent_threshold:.1f}"
                + (f", année {selected_year}." if selected_year != "Toutes" else ".")
            )
        else:
            urgent_theme_tokens = _flat_themes(urgent_df["themes_list"])
            urgent_theme_counts = (
                pd.Series(urgent_theme_tokens)
                .value_counts()
                .rename_axis("theme")
                .reset_index(name="count")
                if urgent_theme_tokens
                else pd.DataFrame(columns=["theme", "count"])
            )
            top_urgent_theme = (
                urgent_theme_counts.iloc[0]["theme"]
                if not urgent_theme_counts.empty
                else "N/A"
            )

            col_u1, col_u2, col_u3, col_u4 = st.columns(4)
            col_u1.metric("Tickets urgents", f"{len(urgent_df):,}")
            col_u2.metric(
                "Sévérité moyenne",
                f"{urgent_df['llm_severity_0_3'].mean():.2f}",
            )
            col_u3.metric("Thème dominant", top_urgent_theme)
            col_u4.metric(
                "% urgents",
                f"{(100 * len(urgent_df) / len(flt)):.1f} %" if len(flt) else "0.0 %",
            )

            st.caption(
                f"Analyse des urgences (seuil ≥ {urgent_threshold:.1f}"
                + (f", année {selected_year}" if selected_year != "Toutes" else "")
                + ")."
            )

            urgent_view = urgent_df[
                [
                    "tweet_id",
                    "created_at_dt",
                    "author",
                    "text_raw",
                    "sentiment_label",
                    "llm_urgency_0_3",
                    "status",
                    "assigned_to",
                    "themes_list",
                ]
            ].copy()
            # Supprimer les colonnes dupliquées
            urgent_view = urgent_view.loc[:, ~urgent_view.columns.duplicated()]
            urgent_view["Thèmes"] = urgent_view["themes_list"].apply(
                lambda L: ", ".join(L[:3])
            )
            urgent_view = urgent_view.drop(columns=["themes_list"]).rename(
                columns={
                    "tweet_id": "ID",
                    "created_at_dt": "Date",
                    "author": "Auteur",
                    "text_raw": "Message",
                    "sentiment_label": "Sentiment",
                    "llm_urgency_0_3": "Urgence",
                    "status": "Statut",
                    "assigned_to": "Assigné",
                }
            ).sort_values(["Urgence", "Date"], ascending=[False, False])

            st.subheader("Tickets urgents")
            if AGGRID_OK:
                gb_u = GridOptionsBuilder.from_dataframe(urgent_view)
                gb_u.configure_default_column(
                    resizable=True,
                    wrapText=True,
                    autoHeight=True,
                    sortable=True,
                    filter=True,
                )
                gb_u.configure_pagination(
                    paginationAutoPageSize=False,
                    paginationPageSize=20,
                )
                AgGrid(
                    urgent_view,
                    gridOptions=gb_u.build(),
                    height=360,
                    fit_columns_on_grid_load=True,
                    theme='light',
                    key="agent_urgent_grid",
                )
            else:
                st.dataframe(urgent_view, use_container_width=True, height=360)

            st.download_button(
                "⬇️ Exporter urgences (CSV)",
                data=urgent_df.to_csv(index=False).encode("utf-8"),
                file_name="tickets_urgents.csv",
                use_container_width=True,
            )

            urgent_authors = (
                urgent_df["author"]
                .replace("", "Non précisé")
                .value_counts()
                .rename_axis("author")
                .reset_index(name="count")
                .head(15)
            )

            st.subheader("Auteurs les plus urgents")
            if not urgent_authors.empty:
                chart = alt.Chart(urgent_authors).mark_bar().encode(
                    x=alt.X("count:Q", title="Tickets urgents"),
                    y=alt.Y("author:N", sort="-x", title="Auteur"),
                    tooltip=["author:N", "count:Q"],
                ).properties(width='container', height=280)
                st.altair_chart(configure_chart_light(chart), use_container_width=True)
            else:
                st.caption("Aucun auteur urgent identifié.")

            urgent_sentiment = (
                urgent_df["sentiment_label"]
                .replace("", "Non précisé")
                .str.capitalize()
                .value_counts()
                .rename_axis("sentiment")
                .reset_index(name="count")
            )
            st.subheader("Répartition sentiment (tickets urgents)")
            if not urgent_sentiment.empty:
                chart = alt.Chart(urgent_sentiment).mark_bar().encode(
                    x=alt.X("sentiment:N", title="Sentiment"),
                    y=alt.Y("count:Q", title="Volume"),
                    color=alt.Color("sentiment:N", legend=None),
                    tooltip=["sentiment:N", "count:Q"],
                ).properties(width='container', height=260)
                st.altair_chart(configure_chart_light(chart), use_container_width=True)
                st.altair_chart(configure_chart_light(chart), use_container_width=True)
            else:
                st.caption("Aucune sentiment urgente disponible.")

            urgent_scatter = urgent_df[
                [
                    "llm_urgency_0_3",
                    "llm_severity_0_3",
                    "sentiment_label",
                    "tweet_id",
                    "author",
                ]
            ].copy()
            urgent_scatter["sentiment_label"] = urgent_scatter["sentiment_label"].replace(
                "", "Non précisé"
            )

            st.subheader("Urgence vs sévérité (tickets urgents)")
            if not urgent_scatter.empty:
                st.altair_chart(
                    alt.Chart(urgent_scatter)
                    .mark_circle(size=70, opacity=0.65)
                    .encode(
                        x=alt.X("llm_urgency_0_3:Q", title="Urgence"),
                        y=alt.Y("llm_severity_0_3:Q", title="Sévérité"),
                        color=alt.Color("sentiment_label:N", title="Sentiment"),
                        tooltip=[
                            "tweet_id:N",
                            "author:N",
                            alt.Tooltip("llm_urgency_0_3:Q", title="Urgence"),
                            alt.Tooltip("llm_severity_0_3:Q", title="Sévérité"),
                        ],
                    )
                    .properties(width='container', height=280),
                    use_container_width=True,
                )
            else:
                st.caption("Pas de combinaison urgence/sévérité disponible.")

    elif active_view == "themes":
        st.subheader("Volumes par thème")
        if not theme_counts_df.empty:
            chart = alt.Chart(theme_counts_df.head(25)).mark_bar().encode(
                x=alt.X("count:Q", title="Volume"),
                y=alt.Y("theme:N", sort="-x", title="Thème"),
                tooltip=["theme:N", "count:Q"],
            ).properties(width='container', height=320).interactive()
            st.altair_chart(configure_chart_light(chart), use_container_width=True)
        else:
            st.caption("Aucun thème détecté.")

        st.subheader("Co-occurrences de thèmes")
        if not co_occ_df.empty:
            st.dataframe(co_occ_df, use_container_width=True, height=260)
        else:
            st.caption("Pas de co-occurrence significative détectée.")

        st.subheader("Sentiment par thème principal")
        if not sentiment_theme_all.empty:
            chart = alt.Chart(sentiment_theme_all).mark_bar().encode(
                x=alt.X("count:Q", title="Volume"),
                y=alt.Y("theme:N", sort="-x", title="Thème"),
                color=alt.Color("sentiment:N", title="Sentiment"),
                tooltip=["theme:N", "sentiment:N", "count:Q"],
            ).properties(width='container', height=320)
            st.altair_chart(configure_chart_light(chart), use_container_width=True)
        else:
            st.caption("Répartition sentiment indisponible.")

    elif active_view == "temps":
        time_df = flt.copy()
        if time_df["created_at_dt"].notna().any():
            min_t = time_df["created_at_dt"].min().date()
            max_t = time_df["created_at_dt"].max().date()

            def _sanitize_range(raw_val, hard_min, hard_max):
                if isinstance(raw_val, (tuple, list)):
                    if len(raw_val) >= 2:
                        start_raw, end_raw = raw_val[0], raw_val[1]
                    elif len(raw_val) == 1:
                        start_raw = end_raw = raw_val[0]
                    else:
                        start_raw = end_raw = None
                else:
                    start_raw = end_raw = raw_val
                start_dt = pd.to_datetime(start_raw, errors="coerce")
                end_dt = pd.to_datetime(end_raw, errors="coerce")
                if pd.isna(start_dt):
                    start_dt = pd.to_datetime(hard_min)
                if pd.isna(end_dt):
                    end_dt = pd.to_datetime(hard_max)
                start_d = start_dt.date()
                end_d = end_dt.date()
                if start_d > end_d:
                    start_d, end_d = end_d, start_d
                start_d = max(start_d, hard_min)
                end_d = min(end_d, hard_max)
                return (start_d, end_d)

            if "agent_time_dates" not in st.session_state or not st.session_state["agent_time_dates"]:
                st.session_state["agent_time_dates"] = (min_t, max_t)
            default_range = _sanitize_range(st.session_state["agent_time_dates"], min_t, max_t)
            st.session_state["agent_time_dates"] = default_range

            picked_range = st.date_input(
                " ",
                value=default_range,
                min_value=min_t,
                max_value=max_t,
                key="agent_time_dates_input",
            )
            sanitized_range = _sanitize_range(picked_range, min_t, max_t)
            if sanitized_range != tuple(st.session_state["agent_time_dates"]):
                st.session_state["agent_time_dates"] = sanitized_range

            start_t, end_t = st.session_state["agent_time_dates"]
            time_df = time_df[
                time_df["created_at_dt"].dt.date.between(start_t, end_t)
            ]
        else:
            time_df = time_df.iloc[0:0]

        time_df = time_df[time_df["created_at_dt"].notna()]

        if time_df.empty:
            st.info("Aucune donnée pour la période sélectionnée.")
        else:
            tmp = time_df.copy()
            tmp["DateOnly"] = tmp["created_at_dt"].dt.date
            tmp["Heure"] = tmp["created_at_dt"].dt.hour
            tmp["dow_idx"] = tmp["created_at_dt"].dt.dayofweek
            tmp["Jour"] = tmp["dow_idx"].map(DOW_LABELS)
            tmp["Sentiment"] = (
                tmp["sentiment_label"]
                .replace("", "Non précisé")
                .str.capitalize()
            )

            def _join_ids(series, limit=40):
                ids = list(map(str, series))
                if len(ids) > limit:
                    return ", ".join(ids[:limit]) + " …"
                return ", ".join(ids)

            timeline_daily = (
                tmp.groupby("DateOnly", dropna=False)
                .agg(count=("tweet_id", "size"), tickets=("tweet_id", _join_ids))
                .reset_index()
                .sort_values("DateOnly")
            )

            try:
                brush = alt.selection_interval(encodings=["x"], empty="all")
            except TypeError:
                brush = alt.selection_interval(encodings=["x"])

            timeline_chart = (
                alt.Chart(timeline_daily)
                .mark_line(point=True)
                .encode(
                    x=alt.X("DateOnly:T", title="Date"),
                    y=alt.Y("count:Q", title="Tickets"),
                    tooltip=[
                        alt.Tooltip("DateOnly:T", title="Date"),
                        alt.Tooltip("count:Q", title="Volume"),
                        alt.Tooltip("tickets:N", title="Tweets (max 40)"),
                    ],
                )
            )
            try:
                timeline_chart = timeline_chart.add_params(brush)
            except Exception:
                timeline_chart = timeline_chart.add_selection(brush)
            timeline_chart = timeline_chart.properties(width='container', height=260, title="Timeline quotidienne")

            hourly_chart = (
                alt.Chart(tmp)
                .transform_filter(brush)
                .mark_bar()
                .encode(
                    x=alt.X("Heure:O", title="Heure"),
                    y=alt.Y("count():Q", title="Volume"),
                    tooltip=[
                        alt.Tooltip("Heure:O", title="Heure"),
                        alt.Tooltip("count():Q", title="Volume"),
                    ],
                )
                .properties(width='container', height=220, title="Répartition horaire")
            )

            heatmap_chart = (
                alt.Chart(tmp)
                .transform_filter(brush)
                .mark_rect()
                .encode(
                    x=alt.X("Heure:O", title="Heure"),
                    y=alt.Y("Jour:N", sort=DOW_ORDER, title="Jour"),
                    color=alt.Color("count():Q", title="Volume"),
                    tooltip=[
                        alt.Tooltip("Jour:N", title="Jour"),
                        alt.Tooltip("Heure:O", title="Heure"),
                        alt.Tooltip("count():Q", title="Volume"),
                    ],
                )
                .properties(width='container', height=240, title="Chaleur jour × heure")
            )

            tone_chart = (
                alt.Chart(tmp)
                .transform_filter(brush)
                .mark_bar()
                .encode(
                    x=alt.X("Sentiment:N", title="Sentiment"),
                    y=alt.Y("count():Q", title="Volume"),
                    color=alt.Color("Sentiment:N", legend=None, title="Sentiment"),
                    tooltip=[
                        alt.Tooltip("Sentiment:N", title="Sentiment"),
                        alt.Tooltip("count():Q", title="Volume"),
                    ],
                )
                .properties(width='container', height=240, title="Histogramme sentiment")
            )

            combined_chart = alt.vconcat(
                timeline_chart,
                hourly_chart,
                heatmap_chart,
                tone_chart,
                spacing=28,
            ).resolve_scale(color="independent")

            st.altair_chart(combined_chart, use_container_width=True)

    elif active_view == "affect":
        st.subheader("Charge par agent")
        if not workload_df.empty:
            st.altair_chart(
                alt.Chart(workload_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Tickets"),
                    y=alt.Y("assigned_to:N", sort="-x", title="Agent"),
                    tooltip=["assigned_to:N", "count:Q"],
                )
                .properties(width='container', height=280),
                use_container_width=True,
            )
        else:
            st.caption("Aucun ticket assigné.")

        st.subheader("Âge moyen et médian des tickets")
        if not age_stats_df.empty:
            st.altair_chart(
                alt.Chart(age_stats_df)
                .transform_fold(["mean", "median"], as_=["indicateur", "valeur"])
                .mark_bar()
                .encode(
                    x=alt.X("valeur:Q", title="Heures"),
                    y=alt.Y("assigned_to:N", sort="-x", title="Agent"),
                    color=alt.Color("indicateur:N", title="Indicateur"),
                    tooltip=["assigned_to:N", "indicateur:N", alt.Tooltip("valeur:Q", format=".1f")],
                )
                .properties(width='container', height=300),
                use_container_width=True,
            )
        else:
            st.caption("Pas assez de données pour calculer l’âge des tickets.")

    elif active_view == "exports":
        st.subheader("Exports & données brutes")

        # Tickets "ouverts / à traiter" = statut NON vide (aligné avec tes filtres)
        open_mask = flt["status"].astype(str).str.strip() != ""
        ouverts_df = flt.loc[open_mask].copy()

        st.caption(f"{len(ouverts_df):,} ticket(s) **ouverts / à traiter** dans la vue actuelle.")

        if ouverts_df.empty:
            st.info("Aucun ticket ouvert / à traiter à exporter avec les filtres actuels.")
        else:
            st.download_button(
                "⬇️ Exporter les tickets OUVERTS / À TRAITER (CSV)",
                data=ouverts_df.to_csv(index=False).encode("utf-8"),
                file_name="tickets_ouverts_a_traiter.csv",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Exporter les tickets OUVERTS / À TRAITER (JSON)",
                data=ouverts_df.to_json(orient="records", force_ascii=False).encode("utf-8"),
                file_name="tickets_ouverts_a_traiter.json",
                use_container_width=True,
            )

        if st.checkbox("Afficher les données brutes", key="agent_show_raw"):
            st.dataframe(ouverts_df if not ouverts_df.empty else flt, use_container_width=True, height=400)

    elif active_view == "logs":
        st.subheader("Journaux & diagnostic")
        st.markdown(f"- Source courante : **{used_path or 'non déterminé'}**")
        st.markdown(f"- Dataset key Agent : **{dataset_key}**")
        st.markdown(f"- Dataset key Manager (fallback) : **{cfg.get('manager_dataset_key', 'tweets_scored_llm')}**")
        st.markdown(f"- Lignes chargées : **{len(df_real) if data_loaded_ok else len(df)}**")
        st.markdown("Historique de chargement :")
        for note in debug_notes:
            st.code(note, language="text")

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
