# pages/1_Analyste.py
# Écran Analyste — SAV Tweets (Streamlit + Altair + AgGrid si dispo)

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from itertools import combinations

# ⛔️ ancien import (remplacé)
# from st_aggrid import AgGrid, GridOptionsBuilder

# ✅ import sécurisé pour éviter de planter si st_aggrid absent
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from lib.data import (
    load_df,
    normalize_status_empty,
    filter_by_status_like_agent,
    count_open_like_agent,
    STATUS_OPTIONS,
    apply_edits,
    upsert_edits,
    load_edits,
)
from lib.state import get_cfg
from lib.ui import inject_style, set_container_wide, inject_sticky_css, hide_sidebar


# --------------------------- Config Streamlit & UI ---------------------------

st.set_page_config(
    page_title="SAV Tweets — Analyste",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_sidebar()
inject_style()
set_container_wide()
inject_sticky_css()

# --- THEME ALTAIR ---
def custom_light_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {
                "stroke": "transparent"
            },
            "title": {
                "color": "#1E293B",
                "fontSize": 18,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 20
            },
            "axis": {
                "domainColor": "#CBD5E1",
                "gridColor": "#E2E8F0",
                "labelColor": "#64748B",
                "titleColor": "#475569",
                "tickColor": "#CBD5E1",
                "labelFontSize": 11,
                "titleFontSize": 12
            },
            "legend": {
                "labelColor": "#64748B",
                "titleColor": "#475569",
                "strokeColor": "transparent",
                "fillColor": "rgba(255,255,255,0.5)",
                "padding": 10,
                "cornerRadius": 8,
                "symbolType": "circle"
            },
            "range": {
                "category": [
                    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", 
                    "#8B5CF6", "#EC4899", "#6366F1", "#14B8A6"
                ]
            }
        }
    }

alt.themes.register("custom_light_theme", custom_light_theme)
alt.themes.enable("custom_light_theme")
alt.data_transformers.disable_max_rows()

def _parse_dt(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if s.notna().sum() == 0:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
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
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        if pd.isna(raw):
            return []
    except Exception:
        pass
    if isinstance(raw, str) and raw.strip():
        import ast, json
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [raw]
    return []


def _prepare_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # tweet_id
    if "tweet_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "tweet_id"})
        else:
            df["tweet_id"] = df.index.astype(str)
    df["tweet_id"] = df["tweet_id"].astype(str)

    # texte brut (ajouté pour cohérence avec Agent/Manager)
    text_col = next(
        (c for c in ["text_raw", "text_display", "text", "tweet", "content", "body"] if c in df.columns),
        None,
    )
    df["text_raw"] = df[text_col].astype(str) if text_col else ""

    # dates
    date_col = next(
        (
            c for c in [
                "created_at_dt", "created_at", "date", "datetime", "timestamp", "time",
                "posted_at", "tweet_created_at", "tweet_date", "createdAt",
                "created_at_utc", "date_utc", "date_time",
            ] if c in df.columns
        ),
        None,
    )
    df["created_at_dt"] = _parse_dt(df[date_col]) if date_col else pd.Series(pd.NaT, index=df.index)

    # thèmes
    theme_src = next(
        (
            c for c in [
                "themes_list", "liste_thèmes", "liste_themes", "themes", "topics", "labels",
            ] if c in df.columns
        ),
        None,
    )
    if theme_src:
        df["themes_list"] = df[theme_src].apply(_ensure_list)
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

    # sentiment
    sent_col = next(
        (c for c in ["sentiment_label", "sentiment", "llm_sentiment"] if c in df.columns),
        None,
    )
    df["sentiment_label"] = df[sent_col].fillna("").astype(str) if sent_col else ""

    # statut
    status_col = next(
        (c for c in ["status", "statut", "state"] if c in df.columns),
        None,
    )
    df["status"] = df[status_col].astype(str).fillna("Ouvert") if status_col else "Ouvert"

    # prio / urgence / sévérité
    df["prio_score"] = pd.to_numeric(df.get("prio_score", 0.0), errors="coerce").fillna(0.0)

    # - si llm_urgency_0_3 absent => fallback = prio_score * 3 borné [0,3]
    df["llm_urgency_0_3"] = pd.to_numeric(df.get("llm_urgency_0_3", np.nan), errors="coerce")
    if df["llm_urgency_0_3"].isna().all():
        df["llm_urgency_0_3"] = (df["prio_score"] * 3).clip(0, 3)
    else:
        df["llm_urgency_0_3"] = df["llm_urgency_0_3"].fillna(0.0)

    df["llm_severity_0_3"] = pd.to_numeric(df.get("llm_severity_0_3", 0.0), errors="coerce").fillna(0.0)

    # résumé / auteur
    summary_col = next((c for c in ["summary_1l", "resume", "summary"] if c in df.columns), None)
    df["summary_1l"] = df[summary_col].fillna("").astype(str) if summary_col else ""

    author_col = next((c for c in ["author", "screen_name", "user", "username", "auteur"] if c in df.columns), None)
    df["author"] = df[author_col].fillna("").astype(str) if author_col else ""

    return df


def _sample_df() -> pd.DataFrame:
    base = pd.Timestamp.today().normalize()
    rows = []
    tones = ["positif", "neutre", "negatif"]
    themes = [["facturation"], ["réseau"], ["application"], ["facturation", "réseau"]]
    for i in range(30):
        rows.append(
            {
                "tweet_id": f"SAMPLE-{i+1:03d}",
                "created_at_dt": base - pd.Timedelta(hours=6 * i),
                "sentiment_label": tones[i % len(tones)],
                "themes_list": themes[i % len(themes)],
                "theme_primary": themes[i % len(themes)][0],
                "prio_score": round(0.4 + 0.02 * i, 2),
                "status": "Ouvert" if i % 3 else "Clôturé",
                "summary_1l": "Résumé factice pour tester l’écran Analyste.",
                "author": f"user_{i%6:02d}",
            }
        )
    return pd.DataFrame(rows)


def _load_with_debug(dataset_key: str):
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
                    return _prepare_df(data), uploaded_csv_path, notes
        except Exception as exc:
            notes.append(f"last_dataset.txt: erreur -> {exc}")
    
    # PRIORITÉ 2 : Essayer load_df avec dataset_key
    try:
        direct = load_df(dataset_key)
        if direct is not None and len(direct) > 0:
            notes.append(f"load_df('{dataset_key}') : OK ({len(direct)} lignes)")
            return _prepare_df(direct), f"load_df('{dataset_key}')", notes
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
            notes.append(f"{path}: lecture via load_df()")
            data = load_df(str(path))
            if data is not None and len(data) > 0:
                notes.append(f"{path}: OK ({len(data)} lignes)")
                return _prepare_df(data), str(path), notes
            notes.append(f"{path}: vide ou illisible")
        except Exception as exc:
            notes.append(f"{path}: erreur -> {exc}")

    return pd.DataFrame(), "", notes


# ------------------------------ Chargement données ---------------------------

cfg = get_cfg()
dataset_key = cfg.get("analyst_dataset_key", cfg.get("manager_dataset_key", "tweets_scored_llm"))
df_real, used_path, debug_notes = _load_with_debug(dataset_key)
df = df_real if not df_real.empty else _sample_df()
data_loaded_ok = not df_real.empty

# -> Normaliser le statut : on repart toujours d'une colonne vide (même si CSV avait une valeur)
df = normalize_status_empty(df)

# Appliquer les edits persistés (si présents)
df = apply_edits(df)


# ---------------------------------- En-tête ----------------------------------

top_left, top_right = st.columns([0.18, 0.82])
with top_left:
    if st.button("⬅️ Retour à l'accueil", use_container_width=True, key="analyst_back_main"):
        st.switch_page("pages/0_Accueil.py")

with top_right:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); 
            border-radius: 12px; 
            padding: 1rem 1.5rem;
            border: 1px solid #BFDBFE; 
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);
            color: #1E40AF;
            margin-bottom: 1rem;">
          <span style="opacity:0.8; font-weight: 500;">Vous êtes sur l'écran :</span>
          <strong style="color:#1E3A8A; font-size: 1.1rem;">&nbsp;Analyste</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### 🔍 Filtres Analyste")

# Versioning des filtres (aligné Manager/Agent)
st.session_state.setdefault("filters_ver", 1)
ver = int(st.session_state.get("filters_ver", 1))


def _reset_analyst_filters():
    import re
    # clés explicitement connues (version courante)
    keys_to_remove = [
        "a_senti", "a_theme", "a_year_simple",
        f"analyst_status_{ver}",
        "a_dates", "a_thr", "a_search",
        f"a_urg_toggle_{ver}", f"a_urg_thr_{ver}", f"a_urg_btn_{ver}"
    ]
    for k in keys_to_remove:
        if k in st.session_state:
            del st.session_state[k]

    # purge défensive : supprimer d'anciennes versions de a_urg_toggle_* et a_urg_thr_*
    for k in list(st.session_state.keys()):
        if re.match(r"^a_(urg_toggle|urg_thr)_\d+$", k):
            del st.session_state[k]

    # bump version et rerun
    st.session_state["filters_ver"] = int(st.session_state.get("filters_ver", 1)) + 1
    st.rerun()


# ---- Diagnostics / reload edits ----
# edits_df = load_edits()
# st.caption(f"Modifications persistées détectées : {len(edits_df):,}")
# st.button("🔄 Recharger modifications persistées", use_container_width=True,
#           on_click=lambda: st.experimental_rerun(), key="a_reload_edits")

st.button("🔄 Réinitialiser filtres", use_container_width=True, on_click=_reset_analyst_filters, key="a_reset_btn")


# ------------------------------ Prépa options filtres ------------------------

if df["created_at_dt"].notna().any():
    min_d_raw = df["created_at_dt"].min().date()
    max_d_raw = df["created_at_dt"].max().date()
    extended_min = (pd.Timestamp(min_d_raw) - pd.Timedelta(days=540)).date()
    extended_max = (pd.Timestamp(max_d_raw) + pd.Timedelta(days=180)).date()
else:
    min_d_raw = max_d_raw = None
    extended_min = extended_max = None

tone_opts = ["positif", "neutre", "negatif"]

theme_values = (
    df["theme_primary"].fillna("").astype(str).str.strip()
    if "theme_primary" in df.columns else pd.Series([], dtype="object")
)
theme_opts = sorted([t for t in theme_values.unique().tolist() if t])

if df["created_at_dt"].notna().any():
    years_all = (
        df["created_at_dt"]
        .dropna()
        .dt.year
        .astype(int)
        .unique()
        .tolist()
    )
    years_all.sort()
else:
    years_all = []

# Défaut : toujours "(Toutes)" (ne pas présélectionner la dernière année)
default_year = "(Toutes)"
st.session_state.setdefault("a_year_simple", default_year)
current_year_choice = str(st.session_state["a_year_simple"])

year_numbers = set(years_all)
if current_year_choice != "(Toutes)":
    try:
        year_numbers.add(int(current_year_choice))
    except ValueError:
        pass

year_options = ["(Toutes)"] + [str(y) for y in sorted(year_numbers)]
if current_year_choice not in year_options:
    year_options.append(current_year_choice)

# ---- Rangée 6 colonnes : Seuil / Sentiment / Thème / Dates / Année / Statut ----

col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns([1.2, 1, 1.3, 1.4, 1.1, 1.2])

with col_f1:
    thr = st.slider("Seuil de priorité", 0.0, 1.0, 0.70, 0.01, key=f"a_thr_{ver}")

with col_f2:
    senti = st.multiselect("Sentiment", ["positif", "neutre", "negatif"], [], key=f"a_senti_{ver}")

with col_f3:
    themsel = st.multiselect("Thème principal", theme_opts, [], key=f"a_theme_{ver}")

with col_f4:
    if min_d_raw and max_d_raw:
        date_range = st.date_input(
            "Fenêtre temporelle",
            value=(min_d_raw, max_d_raw),
            min_value=extended_min,
            max_value=extended_max,
            key=f"a_dates_{ver}",
        )
    else:
        date_range = ()
        st.caption("Fenêtre temporelle indisponible.")

with col_f5:
    # index piloté par la valeur actuelle de session_state (robuste aux valeurs non présentes)
    try:
        current_year_choice = str(st.session_state.get("a_year_simple", "(Toutes)"))
        idx_year = year_options.index(current_year_choice)
    except Exception:
        idx_year = 0
    year_sel = st.selectbox("Année (graphiques)", options=year_options, index=idx_year, key=f"a_year_simple_{ver}")
    # resynchroniser l'API interne : conserver st.session_state["a_year_simple"]
    st.session_state["a_year_simple"] = year_sel

with col_f6:
    # Statut versionné (clé incluse dans _reset_analyst_filters)
    status_filter = st.multiselect("Statut", options=STATUS_OPTIONS, key=f"analyst_status_{ver}")

    search_col = st.columns([1, 3])[1]
    search = search_col.text_input("Recherche texte (ID, résumé, thème, message)", "", key=f"a_search_{ver}").strip()
    selected_year = st.session_state.get("a_year_simple", "(Toutes)")# ----------------- Rangée urgences (toggle / seuil / bouton) -----------------

u1, u2, u3 = st.columns([1.0, 1.4, 0.9])

with u1:
    # Streamlit n'a pas st.toggle : on utilise checkbox pour le mode ON/OFF
    urg_on = st.checkbox("Mode urgences", value=False, key=f"a_urg_toggle_{ver}")

with u2:
    # le slider est désactivé si le toggle est OFF
    urg_disabled = not st.session_state.get(f"a_urg_toggle_{ver}", False)
    urg_thr = st.slider("Seuil d’urgence", 0.0, 3.0, 2.0, 0.1, key=f"a_urg_thr_{ver}", disabled=urg_disabled)

# ---------------------------- État chargement données ------------------------

status_col1, status_col2 = st.columns([1, 4])
with status_col1:
    st.caption("📂 Source données")
with status_col2:
    if data_loaded_ok:
        st.markdown(
            f"""
            <div style="
                background-color: #d1fae5;
                color: #065f46;
                padding: 0.5rem;
                border-radius: 0.375rem;
                border: 1px solid #a7f3d0;
                font-size: 0.875rem;
                display: flex;
                align-items: center;
            ">
                <span style="margin-right: 0.5rem; font-size: 1.25rem;">✅</span>
                <span>{len(df_real):,} lignes chargées depuis : <strong>{used_path}</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Mode démo : pas de données réelles trouvées", icon="⚠️")

with st.expander("🔎 Diagnostic import"):
    for note in debug_notes:
        st.text(note)

st.title("Tableau de bord Analyste")

# ----------------------------- Application des filtres -----------------------

ver = int(st.session_state.get("filters_ver", 1))

q = df.copy()
if not q.empty:
    # Inclure TOUJOURS les tickets ayant déjà un statut (issus de sav_edits.csv),
    # même si leur prio_score est inférieur au seuil de priorité.
    has_status = q["status"].astype(str).str.strip() != ""
    by_prio = q["prio_score"] >= thr
    q = q[by_prio | has_status]

    if senti:
        prefixes = tuple(s[:3].lower() for s in senti)
        q = q[q["sentiment_label"].str.lower().str.startswith(prefixes)]

    if themsel:
        q = q[q["theme_primary"].isin(themsel)]

    if (
        df["created_at_dt"].notna().any()
        and isinstance(date_range, tuple)
        and len(date_range) == 2
    ):
        start, end = date_range
        if start and end:
            q = q[
                q["created_at_dt"].dt.date.between(
                    pd.to_datetime(start).date(),
                    pd.to_datetime(end).date(),
                )
            ]

    if selected_year != "(Toutes)":
        try:
            year_int = int(selected_year)
        except ValueError:
            q = q.iloc[0:0]
        else:
            q = q[q["created_at_dt"].dt.year == year_int]

    if search:
        lower = search.lower()
        q = q[
            q["tweet_id"].str.lower().str.contains(lower, na=False)
            | q["summary_1l"].str.lower().str.contains(lower, na=False)
            | q["themes_list"].apply(lambda L: any(lower in t.lower() for t in L))
            | q["text_raw"].str.lower().str.contains(lower, na=False)
        ]

    # Filtre Statut via helper centralisé
    q = filter_by_status_like_agent(q, status_filter)

    # Filtre URGENCES : si activé
    if st.session_state.get(f"a_urg_toggle_{ver}", False):
        try:
            urg_threshold = float(st.session_state.get(f"a_urg_thr_{ver}", 2.0))
            q = q[
                pd.to_numeric(q.get("llm_urgency_0_3", 0.0), errors="coerce")
                .fillna(0.0) >= urg_threshold
            ]
        except Exception:
            pass

# Data pour graphiques (filtrée sur année si choisie)
q_graph = q.copy()
selected_year = st.session_state.get("a_year_simple", "(Toutes)")
if not q_graph.empty and selected_year != "(Toutes)":
    try:
        selected_year_int = int(selected_year)
    except ValueError:
        q_graph = q_graph.iloc[0:0]
    else:
        q_graph = q_graph[q_graph["created_at_dt"].dt.year == selected_year_int]


# ------------------------------- Navigation & vues ---------------------------

if "analyst_active_view" not in st.session_state:
    st.session_state["analyst_active_view"] = "exploration"

active_view = st.session_state["analyst_active_view"]

# Garde-fou : empêcher l'accès aux vues retirées depuis le menu
if active_view in {"saved_queries", "logs", "automations"}:
    st.session_state["analyst_active_view"] = "exploration"
    active_view = "exploration"

try:
    left, main = st.columns([1, 3.6], gap="medium")
except TypeError:
    left, main = st.columns([1, 3.6], vertical_alignment="top")

with left:
    st.markdown('<div class="ma-card"><h3>Navigation</h3></div>', unsafe_allow_html=True)
    nav_items = [
        ("Explorer", "exploration"),
        ("Thèmes", "themes"),
        ("Temps & sentiment", "temporal"),
        ("Comparaisons", "comparaisons"),
        ("Réseau thématique", "reseau"),
        ("Alertes auto", "alertes"),
        ("Exports", "exports"),
    ]
    for label, value in nav_items:
        if st.button(label, use_container_width=True, disabled=(active_view == value), key=f"a_nav_{value}"):
            st.session_state["analyst_active_view"] = value
            st.rerun()

with main:
    st.markdown('<div class="ma-card"><h3>Insights</h3></div>', unsafe_allow_html=True)

    if q.empty:
        st.info("Aucune donnée (ou CSV non trouvé).")
    else:
        # Comptages & métriques
        if "themes_list" in q.columns:
            theme_counts = (
                q["theme_primary"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", "Non précisé")
                .value_counts()
                .rename_axis("theme")
                .reset_index(name="count")
            )
        else:
            theme_counts = pd.DataFrame(columns=["theme", "count"])

        top_theme = theme_counts.iloc[0]["theme"] if not theme_counts.empty else "N/A"

        author_counts = (
            q["author"]
            .replace("", "Non précisé")
            .value_counts()
            .rename_axis("author")
            .reset_index(name="count")
        )
        top_author = author_counts.iloc[0]["author"] if not author_counts.empty else "N/A"

        tone_counts = (
            q["sentiment_label"]
            .replace("", "Non précisé")
            .str.capitalize()
            .value_counts()
            .rename_axis("sentiment")
            .reset_index(name="count")
        )

        metrics = st.columns(5)
        metrics[0].metric("Tweets sélectionnés", f"{len(q):,}")
        metrics[1].metric("Score prio médian", f"{q['prio_score'].median():.2f}")
        neg_ratio = (q["sentiment_label"].fillna("").str.lower().str.startswith(("neg", "nég"))).mean()
        metrics[2].metric("% sentiment négative", f"{100 * neg_ratio:.1f} %")
        metrics[3].metric("Thèmes uniques", f"{q['theme_primary'].nunique():,}")
        metrics[4].metric("Ouverts / À traiter", f"{count_open_like_agent(q):,}")

        extra_metrics = st.columns(2)
        extra_metrics[0].metric("Auteur principal", top_author)
        extra_metrics[1].metric("Top thème", top_theme)

        # Découpes temporelles
        if not q_graph.empty and q_graph["created_at_dt"].notna().any():
            timeline = (
                q_graph.assign(date=q_graph["created_at_dt"].dt.date)
                .groupby("date", dropna=False)
                .agg(count=("tweet_id", "size"), prio=("prio_score", "median"))
                .reset_index()
                .sort_values("date")
            )
            hourly = (
                q_graph.assign(hour=q_graph["created_at_dt"].dt.hour)
                .groupby("hour")
                .size()
                .reset_index(name="count")
                .sort_values("hour")
            )
            heatmap = (
                q_graph.assign(dow=q_graph["created_at_dt"].dt.dayofweek, hour=q_graph["created_at_dt"].dt.hour)
                .groupby(["dow", "hour"])
                .size()
                .reset_index(name="count")
            )
        else:
            timeline = pd.DataFrame(columns=["date", "count", "prio"])
            hourly = pd.DataFrame(columns=["hour", "count"])
            heatmap = pd.DataFrame(columns=["dow", "hour", "count"])

        if not heatmap.empty:
            dow_labels = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}
            heatmap["Jour"] = heatmap["dow"].map(dow_labels)

        # Co-occurrences de thèmes
        def _clean_theme_token(x: str) -> str:
            """ Nettoie un libellé de thème qui ressemble à "['Réseau/Internet']" """
            return (
                str(x)
                .replace("[", "").replace("]", "")
                .replace("(", "").replace(")", "")
                .replace("{", "").replace("}", "")
                .replace("'", "").replace('"', "")
                .strip()
            )

        combo_rows = []
        for raw_list in q["themes_list"]:
            cleaned = [_clean_theme_token(t) for t in raw_list if str(t).strip()]
            cleaned = [t for t in cleaned if t]
            cleaned = sorted(set(cleaned))
            if len(cleaned) >= 2:
                combo_rows.extend(combinations(cleaned, 2))

        if combo_rows:
            co_df = (
                pd.Series(combo_rows)
                .value_counts()
                .reset_index()
                .rename(columns={"index": "binôme", 0: "count"})
            )
            co_df["binôme"] = co_df["binôme"].apply(lambda pair: f"{pair[0]} / {pair[1]}")
            co_df = co_df.head(20)
        else:
            co_df = pd.DataFrame(columns=["binôme", "count"])

        sentiment_theme = (
            q.assign(
                theme=q["theme_primary"].fillna("").astype(str).str.strip().replace("", "Non précisé")
            )
            .groupby(["theme", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        rolling = (
            timeline.assign(ma7=timeline["count"].rolling(7, min_periods=1).mean())
            if not timeline.empty else pd.DataFrame()
        )

        status_cmp = (
            q.groupby("status")["prio_score"]
            .agg(["mean", "median", "count"])
            .reset_index()
            .rename(columns={"mean": "moyenne", "median": "médiane"})
        )

        # ----------------------------------- VUES -----------------------------------

        if active_view == "exploration":
            st.subheader("Résultats détaillés")
            view = q[
                [
                    "tweet_id", "summary_1l", "sentiment_label", "theme_primary",
                    "themes_list", "prio_score", "status", "created_at_dt", "author",
                ]
            ].copy()
            view["themes_all"] = view["themes_list"].apply(lambda L: ", ".join(L[:3]))
            view = view.drop(columns=["themes_list"])

            # Tenter de positionner "status" en 2e colonne
            desired = ["tweet_id", "status", "created_at_dt", "author", "summary_1l", "sentiment_label", "prio_score"]
            existing_cols = [c for c in desired if c in view.columns]
            view = view[existing_cols + [c for c in view.columns if c not in existing_cols]]

            # Affichage AgGrid si dispo, sinon dataframe
            if AGGRID_OK:
                gb = GridOptionsBuilder.from_dataframe(view)
                gb.configure_default_column(resizable=True, filter=True, sortable=True, wrapText=True, autoHeight=True)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
                gb.configure_selection("single", use_checkbox=True)
                
                # Configuration du style de grille pour correspondre au design
                gb.configure_grid_options(
                    rowHeight=50,
                    headerHeight=45
                )

                # édition autorisée uniquement sur la colonne 'status' (si présente)
                try:
                    gb.configure_column("status", editable=True)
                except Exception:
                    pass

                grid_resp = AgGrid(
                    view,
                    gridOptions=gb.build(),
                    height=500,
                    fit_columns_on_grid_load=True,
                    theme="light",
                    key="analyst_grid",
                ) or {}

                # Traiter les edits (status/assigned_to/agent_response) si AgGrid a renvoyé des data
                try:
                    resp_data = grid_resp.get("data") if isinstance(grid_resp, dict) else None
                    if resp_data:
                        df_new = pd.DataFrame(resp_data)
                        if "tweet_id" in df_new.columns:
                            cols_to_check = [c for c in ("status", "assigned_to", "agent_response") if c in df_new.columns]
                            if cols_to_check:
                                old_map = (
                                    view.set_index("tweet_id")[cols_to_check].astype(str).fillna("")
                                    .to_dict(orient="index")
                                )
                                rows_to_upsert = []
                                for r in df_new.to_dict("records"):
                                    tid = str(r.get("tweet_id", "")).strip()
                                    if not tid:
                                        continue
                                    old = old_map.get(tid, {c: "" for c in cols_to_check})
                                    changed = {}
                                    for c in cols_to_check:
                                        newv = str(r.get(c, "") or "").strip()
                                        oldv = str(old.get(c, "") or "").strip()
                                        if newv != oldv:
                                            changed[c] = newv
                                    if changed:
                                        row = {"tweet_id": tid}
                                        row.update({c: changed.get(c, "") for c in ("status", "assigned_to", "agent_response")})
                                        rows_to_upsert.append(row)
                                if rows_to_upsert:
                                    try:
                                        upsert_edits(rows_to_upsert)
                                        st.success(f"{len(rows_to_upsert)} modification(s) sauvegardée(s).")
                                    except Exception:
                                        st.error("Échec sauvegarde edits (sav_edits.csv). Consulte les logs.")
                except Exception:
                    # ne pas casser l'affichage si le post-processing échoue
                    pass
            else:
                st.dataframe(view, use_container_width=True, height=360)

            st.subheader("Priorité vs temporalité")
            st.altair_chart(
                alt.Chart(q)
                .mark_circle(size=70, opacity=0.6)
                .encode(
                    x=alt.X("created_at_dt:T", title="Date"),
                    y=alt.Y("prio_score:Q", title="Score prio"),
                    color=alt.Color("sentiment_label:N", title="Sentiment"),
                    tooltip=["tweet_id:N", "theme_primary:N", "prio_score:Q", "sentiment_label:N", "status:N"],
                )
                .properties(height=280)
                .interactive(),
                use_container_width=True,
            )

        elif active_view == "themes":
            st.subheader("Volumes par thème (normalisés)")
            if not theme_counts.empty:
                st.altair_chart(
                    alt.Chart(theme_counts.head(25))
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Volume"),
                        y=alt.Y("theme:N", sort="-x", title="Thème"),
                        tooltip=["theme:N", "count:Q"],
                    )
                    .properties(height=320)
                    .interactive(),
                    use_container_width=True,
                )
            else:
                st.caption("Aucun thème détecté.")

            st.subheader("Top co-occurrences")
            if not co_df.empty:
                st.dataframe(co_df, use_container_width=True, height=240)
            else:
                st.caption("Pas de binômes de thèmes saillants.")

            st.subheader("Répartition sentiment × thème principal")
            if not sentiment_theme.empty:
                st.altair_chart(
                    alt.Chart(sentiment_theme)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Volume"),
                        y=alt.Y("theme:N", sort="-x", title="Thème"),
                        color=alt.Color("sentiment_label:N", title="Sentiment"),
                        tooltip=["theme:N", "sentiment_label:N", "count:Q"],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )
            else:
                st.caption("Répartition sentiment indisponible.")

        elif active_view == "temporal":
            st.subheader("Timeline & priorité")
            if not timeline.empty:
                st.altair_chart(
                    alt.Chart(timeline)
                    .transform_fold(["count", "prio"], as_=["indicateur", "valeur"])
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("valeur:Q", title="Valeur"),
                        color=alt.Color("indicateur:N", title="Indicateur"),
                        tooltip=["date:T", "indicateur:N", alt.Tooltip("valeur:Q", format=".2f")],
                    )
                    .properties(height=260),
                    use_container_width=True,
                )
            else:
                st.caption("Pas de timeline exploitable.")

            st.subheader("Répartition horaire")
            if not hourly.empty:
                st.altair_chart(
                    alt.Chart(hourly)
                    .mark_bar()
                    .encode(
                        x=alt.X("hour:O", title="Heure"),
                        y=alt.Y("count:Q", title="Tweets"),
                        tooltip=["hour:O", "count:Q"],
                    )
                    .properties(height=220),
                    use_container_width=True,
                )
            else:
                st.caption("Aucune granularité horaire disponible.")

            st.subheader("Chaleur jour × heure")
            if not heatmap.empty:
                st.altair_chart(
                    alt.Chart(heatmap)
                    .mark_rect()
                    .encode(
                        x=alt.X("hour:O", title="Heure"),
                        y=alt.Y("Jour:N", sort=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]),
                        color=alt.Color("count:Q", title="Volume"),
                        tooltip=["Jour:N", "hour:O", "count:Q"],
                    )
                    .properties(height=240),
                    use_container_width=True,
                )
            else:
                st.caption("Chaleur indisponible.")

            st.subheader("Volume lissé (MA7)")
            if not rolling.empty:
                st.altair_chart(
                    alt.Chart(rolling)
                    .mark_line(point=True, color="#ff7f0e")
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("ma7:Q", title="Comptes lissés (MA7)"),
                        tooltip=["date:T", alt.Tooltip("ma7:Q", format=".1f")],
                    )
                    .properties(height=220),
                    use_container_width=True,
                )
            else:
                st.caption("Pas assez de points pour calculer la moyenne glissante.")

        elif active_view == "comparaisons":
            st.subheader("Charge par auteur")
            if not author_counts.empty:
                st.altair_chart(
                    alt.Chart(author_counts.head(20))
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Volume"),
                        y=alt.Y("author:N", sort="-x", title="Auteur"),
                        tooltip=["author:N", "count:Q"],
                    )
                    .properties(height=300),
                    use_container_width=True,
                )
            else:
                st.caption("Aucun auteur renseigné.")

            st.subheader("Scores moyens par statut")
            if not status_cmp.empty:
                st.altair_chart(
                    alt.Chart(status_cmp)
                    .transform_fold(["moyenne", "médiane"], as_=["indicateur", "valeur"])
                    .mark_bar()
                    .encode(
                        x=alt.X("valeur:Q", title="Score prio"),
                        y=alt.Y("status:N", sort="-x", title="Statut"),
                        color=alt.Color("indicateur:N", title="Indicateur"),
                        tooltip=["status:N", "indicateur:N", alt.Tooltip("valeur:Q", format=".2f")],
                    )
                    .properties(height=300),
                    use_container_width=True,
                )
                if st.checkbox("Afficher les statistiques détaillées", False, key="a_show_stats"):
                    st.dataframe(status_cmp, use_container_width=True)
            else:
                st.caption("Aucun statut comparatif disponible.")

        elif active_view == "reseau":
            st.subheader("Réseau thématique")
            if not co_df.empty:
                st.altair_chart(
                    alt.Chart(co_df)
                    .mark_circle()
                    .encode(
                        x=alt.X("count:Q", title="Intensité"),
                        y=alt.Y("binôme:N", sort="-x", title="Association"),
                        size=alt.Size("count:Q", legend=None),
                        color=alt.Color("count:Q", legend=None, scale=alt.Scale(scheme="purpleorange")),
                        tooltip=["binôme:N", "count:Q"],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )
            else:
                st.info("💡 Pas de co-occurrence thématique détectée (tweets avec un seul thème)")
            
            # Graphiques alternatifs : corrélations thème-sentiment et urgence par thème
            st.subheader("Analyse thématique approfondie")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sentiment par thème**")
                if not sentiment_theme.empty and len(sentiment_theme) > 0:
                    # Afficher le graphique directement sans capitaliser
                    st.altair_chart(
                        alt.Chart(sentiment_theme)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Nombre de tweets"),
                            y=alt.Y("theme:N", sort="-x", title="Thème"),
                            color=alt.Color("sentiment_label:N", title="Sentiment"),
                            tooltip=["theme:N", "sentiment_label:N", "count:Q"]
                        )
                        .properties(height=300),
                        use_container_width=True
                    )
                else:
                    st.caption("Aucune donnée sentiment-thème disponible.")
            
            with col2:
                st.markdown("**Urgence moyenne par thème**")
                if not q.empty:
                    urgency_by_theme = q.groupby('theme_primary')['llm_urgency_0_3'].mean().reset_index()
                    urgency_by_theme.columns = ['theme', 'urgence_moyenne']
                    urgency_by_theme = urgency_by_theme.sort_values('urgence_moyenne', ascending=False).head(10)
                    
                    st.altair_chart(
                        alt.Chart(urgency_by_theme)
                        .mark_bar(color='#e67e22')
                        .encode(
                            x=alt.X("urgence_moyenne:Q", title="Urgence moyenne", scale=alt.Scale(domain=[0, 3])),
                            y=alt.Y("theme:N", sort="-x", title="Thème"),
                            tooltip=[
                                alt.Tooltip("theme:N", title="Thème"),
                                alt.Tooltip("urgence_moyenne:Q", title="Urgence", format=".2f")
                            ]
                        )
                        .properties(height=300),
                        use_container_width=True
                    )
                else:
                    st.caption("Aucune donnée d'urgence disponible.")

        elif active_view == "exports":
            st.subheader("Exports & données brutes")
            open_view = q[q["status"].astype(str).str.strip() != ""].copy()
            st.caption(f"{len(open_view):,} ticket(s) ouverts / à traiter dans la vue actuelle.")

            if open_view.empty:
                st.info("Aucun ticket ouvert / à traiter à exporter avec les filtres actuels.")
            else:
                st.download_button(
                    "⬇️ Exporter OUVERTS / À TRAITER (CSV)",
                    data=open_view.to_csv(index=False).encode("utf-8"),
                    file_name="tweets_analyste_ouverts.csv",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Exporter OUVERTS / À TRAITER (JSON)",
                    data=open_view.to_json(orient="records", force_ascii=False).encode("utf-8"),
                    file_name="tweets_analyste_ouverts.json",
                    use_container_width=True,
                )

                if st.checkbox("Afficher les données brutes", False, key="a_show_raw"):
                    st.dataframe(open_view if not open_view.empty else q, use_container_width=True, height=400)

        elif active_view == "saved_queries":
            st.subheader("Requêtes sauvegardées")
            st.caption("Ici on affichera les filtres ou requêtes enregistrées par l’analyste.")

        elif active_view == "logs":
            st.subheader("Journaux et chargement")
            st.markdown(f"- Source courante : **{used_path or 'non déterminé'}**")
            st.markdown(f"- Dataset key : **{dataset_key}**")
            st.markdown(f"- Lignes chargées : **{len(df_real) if data_loaded_ok else len(df)}**")
            st.markdown("Diagnostic import :")
            for note in debug_notes:
                st.code(note, language="text")

        elif active_view == "alertes":
            st.subheader("Alertes automatiques")
            st.caption("Suivi des thèmes en hausse, urgences et volume négatif.")

            if not theme_counts.empty:
                st.altair_chart(
                    alt.Chart(theme_counts.head(10))
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Volume"),
                        y=alt.Y("theme:N", sort="-x", title="Thème"),
                        tooltip=["theme:N", "count:Q"],
                    )
                    .properties(height=260),
                    use_container_width=True,
                )

            if not timeline.empty:
                st.altair_chart(
                    alt.Chart(timeline)
                    .mark_line(point=True, color="#f97316")
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("count:Q", title="Tweets / jour"),
                        tooltip=["date:T", "count:Q"],
                    )
                    .properties(height=240),
                    use_container_width=True,
                )

            if not co_df.empty:
                st.dataframe(co_df, use_container_width=True, height=220)

        elif active_view == "automations":
            st.subheader("Automations")
            st.caption(
                "Ici l’analyste pourra définir des règles pour notifier les équipes "
                "(ex : >20 tweets 'Réseau/Internet' en 30 min)."
            )

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
