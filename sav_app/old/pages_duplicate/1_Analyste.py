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

cfg = get_cfg()


# ----------------------------- Helpers & Parsing -----------------------------

def _parse_dt(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.notna().sum() == 0:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
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
    try:
        direct = load_df(dataset_key)
        if direct is not None and len(direct) > 0:
            notes.append(f"load_df('{dataset_key}') : OK ({len(direct)} lignes)")
            return _prepare_df(direct), f"load_df('{dataset_key}')", notes
        notes.append(f"load_df('{dataset_key}') : vide")
    except Exception as exc:
        notes.append(f"load_df('{dataset_key}') : échec -> {exc}")

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
            background-color:#0f1e33; border-radius:4px; padding:0.6rem 0.8rem;
            border:1px solid rgba(255,255,255,.2); font-size:0.9rem; color:#cdd9ff;">
          <span style="opacity:0.8;">Vous êtes sur l'écran :</span>
          <strong style="color:#fff;">&nbsp;Analyste</strong>
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
search = search_col.text_input("Recherche texte (ID, résumé, thème)", "", key=f"a_search_{ver}").strip()
selected_year = st.session_state.get("a_year_simple", "(Toutes)")

# ----------------- Rangée urgences (toggle / seuil / bouton) -----------------

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
        st.success(f"{len(df_real):,} lignes chargées depuis : {used_path}", icon="✅")
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
        neg_ratio = q["sentiment_label"].str.lower().str.startswith("neg").mean()
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

                # édition autorisée uniquement sur la colonne 'status' (si présente)
                try:
                    gb.configure_column("status", editable=True)
                except Exception:
                    pass

                grid_resp = AgGrid(
                    view,
                    gridOptions=gb.build(),
                    height=360,
                    fit_columns_on_grid_load=True,
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
                st.caption("Pas de co-occurrence significative détectée.")

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
