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

# ------------------------------------------------------------------------------
# Rendre importable lib/* même si on lance DIRECT cette page
# (Ex: python -m streamlit run pages/2_Manager.py)
# ------------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]  # dossier racine (ex: sav_app)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from lib.data import load_df              # <- on réutilise EXACTEMENT le loader Agent SAV
from lib.state import get_cfg
from lib.ui import inject_style, set_container_wide, inject_sticky_css  # thème / CSS dark maison

# ------------------------------------------------------------------------------
# Config Streamlit (large + thème custom)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tableau de bord Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        /* cache complet de la sidebar */
        [data-testid="stSidebar"] > div:first-child {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
inject_style()
set_container_wide()
inject_sticky_css()
cfg = get_cfg()

# ------------------------------------------------------------------------------
# Utils parsing date / nettoyage colonnes
# ------------------------------------------------------------------------------

def _parse_dt(series: pd.Series) -> pd.Series:
    """
    Rend les dates propres (timezone, formats variées).
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.notna().sum() == 0:
        # deuxième tentative style '27/10/2025 14:33'
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # drop timezone => naive
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
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
        "text_display",
        "tweet",
        "text",
        "content",
        "body",
    ]
    txt_src = next((c for c in text_candidates if c in df.columns), None)
    if txt_src is not None:
        df["text_display"] = df[txt_src].astype(str)
    else:
        df["text_display"] = ""

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
                "text_display": "Tweet d’exemple généré pour la démo.",
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
        Path(r"C:\Users\IMAD\Desktop\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
        APP_ROOT / "tweets_scored_llm.csv",
        APP_ROOT / "data" / "tweets_scored_llm.csv",
        Path.home() / "OneDrive" / "Bureau" / "IA Free Mobile" / "tweets_scored_llm.csv",
        Path.home() / "OneDrive" / "Bureau" / "IA Free Mobile" / "IA Free Mobile" / "tweets_scored_llm.csv",
    ]

    notes: list[str] = []
    try:
        direct_df = load_df(dataset_key)
        if direct_df is not None and len(direct_df) > 0:
            notes.append(f"load_df('{dataset_key}') : OK ({len(direct_df)} lignes)")
            return _prepare_manager_df(direct_df), f"load_df('{dataset_key}')", notes
        notes.append(f"load_df('{dataset_key}') : vide")
    except Exception as e:
        notes.append(f"load_df('{dataset_key}') : échec -> {e}")

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
            background-color:#0f1e33;
            border-radius:4px;
            padding:0.6rem 0.8rem;
            border:1px solid rgba(255,255,255,.2);
            font-size:0.9rem;
            color:#cdd9ff;
        ">
            <span style="opacity:0.8;">Vous êtes sur l'écran :</span>
            <strong style="color:#fff;">&nbsp;Manager</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
status_opts = ["(Tous)"] + sorted(df["status"].dropna().unique().tolist())

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
default_year_index = len(year_opts) - 1 if year_opts else 0  # dernière année dispo

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
            key="m_period_simple",
        )
    else:
        date_from = date_to = None
        st.caption("Dates indisponibles dans le fichier.")

with col_f2:
    tone_sel = st.selectbox("Sentiment", tone_opts, key="m_tone_simple")

with col_f3:
    status_sel = st.selectbox("Statut", status_opts, key="m_status_simple")

with col_f4:
    theme_sel = st.selectbox("Thème", theme_opts, key="m_theme_simple")

with col_f5:
    year_sel = st.selectbox(
        "Année (graphiques)",
        year_opts,
        index=default_year_index,
        key="m_year_simple",
    )

# Infos source + diagnostic
status_col1, status_col2 = st.columns([1, 4])
with status_col1:
    st.caption("📂 Source données")
with status_col2:
    if data_loaded_ok:
        st.success(f"{len(df):,} lignes chargées depuis : {used_path}", icon="✅")
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

if status_sel != "(Tous)":
    flt = flt[flt["status"] == status_sel]

if theme_sel != "(Tous)":
    flt = flt[flt["theme_primary"] == theme_sel]

flt_graph = flt.copy()
if "year_sel" in locals() and year_sel != "(Toutes)":
    flt_graph = flt_graph[flt_graph["created_at_dt"].dt.year == int(year_sel)]

# ------------------------------------------------------------------------------
# Construction des agrégats pour les graphes et KPIs
# ------------------------------------------------------------------------------
if flt.empty:
    st.info("Aucune donnée pour la sélection courante (filtres trop restrictifs ?).")
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
        flt["sentiment_label"]
        .str.capitalize()
        .replace("", "Non précisé")
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    theme_series = (
        flt["theme_primary"]
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
        flt["author"]
        .replace("", "Non précisé")
        .value_counts()
        .rename_axis("author")
        .reset_index(name="count")
        .head(10)
    )

    status_df = (
        flt["status"]
        .replace("", "Non précisé")
        .str.capitalize()
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )

    heatmap = (
        flt.assign(hour=flt["created_at_dt"].dt.hour,
                   dow=flt["created_at_dt"].dt.dayofweek)
        .groupby(["dow", "hour"])
        .size()
        .reset_index(name="count")
        if flt["created_at_dt"].notna().any()
        else pd.DataFrame(columns=["dow", "hour", "count"])
    )
    if not heatmap.empty:
        dow_labels = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Jeu", 4: "Ven", 5: "Sam", 6: "Dim"}
        heatmap["Jour"] = heatmap["dow"].map(dow_labels)

    scatter_df = flt[
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
        flt.assign(hour=flt["created_at_dt"].dt.hour)
        .groupby("hour")
        .size()
        .reset_index(name="count")
        .sort_values("hour")
        if flt["created_at_dt"].notna().any()
        else pd.DataFrame(columns=["hour", "count"])
    )

    team_col = next((c for c in ("routing_team", "intent_primary") if c in flt.columns), None)
    team_volume = (
        flt.groupby(team_col)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if team_col
        else pd.DataFrame()
    )
    team_scores = (
        flt.groupby(team_col)[["llm_urgency_0_3", "llm_severity_0_3"]]
        .mean()
        .reset_index()
        if team_col
        else pd.DataFrame()
    )

    urgent_table = flt.sort_values(
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
            "text_display",
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

    if flt["created_at_dt"].notna().any():
        valid_dates = flt["created_at_dt"].dropna()
    else:
        valid_dates = pd.Series([], dtype="datetime64[ns]")

    theme_trends = pd.DataFrame(columns=["theme", "recent", "previous", "delta"])
    if not valid_dates.empty:
        max_ts = valid_dates.max()
        recent_start = max_ts - pd.Timedelta(days=6)
        past_start = recent_start - pd.Timedelta(days=7)

        recent_counts = _theme_counts_slice(flt[flt["created_at_dt"] >= recent_start])
        past_counts = _theme_counts_slice(
            flt[
                (flt["created_at_dt"] < recent_start)
                & (flt["created_at_dt"] >= past_start)
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
    if not flt.empty:
        urgency_split = (
            pd.Series(
                np.where(
                    flt["llm_urgency_0_3"] >= 2,
                    "Urgent ≥ 2",
                    "Urgent < 2",
                )
            )
            .value_counts()
            .rename_axis("niveau")
            .reset_index(name="count")
        )

    # Heures critiques : % négatif par heure
    if flt["created_at_dt"].notna().any():
        hourly_sentiment = (
            flt.assign(
                hour=flt["created_at_dt"].dt.hour,
                neg=flt["sentiment_label"]
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
        col1.metric("Tweets", f"{len(flt):,}")
        col2.metric("Pourcentage urgent", f"{100 * (flt['llm_urgency_0_3'] >= 2).mean():.1f} %")
        col3.metric("Pourcentage négatif", f"{100 * flt['sentiment_label'].str.lower().str.startswith('neg').mean():.1f} %")
        col4.metric("Auteurs uniques", f"{flt['author'].nunique():,}")
        col5.metric("Urgence moyenne", f"{flt['llm_urgency_0_3'].mean():.2f}")

        st.download_button(
            "⬇️ Exporter les tweets filtrés (CSV)",
            data=flt.to_csv(index=False).encode("utf-8"),
            file_name="tweets_filtrés_manager.csv",
            mime="text/csv",
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
                # ligne 1 : volume quotidien / volume cumulatif
                col_a1, col_a2 = st.columns(2)
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
                                tooltip=[
                                    "date:T",
                                    alt.Tooltip("cumul:Q", title="Tweets cumulés"),
                                ],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Pas assez de points pour cumuler.")

                # ligne 2 : top thèmes / top auteurs
                col_a3, col_a4 = st.columns(2)

                with col_a3:
                    st.subheader("Top thèmes")
                    if not theme_df.empty:
                        st.altair_chart(
                            alt.Chart(theme_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("count:Q", title="Volume"),
                                y=alt.Y("theme:N", sort="-x", title="Thème"),
                                tooltip=["theme:N", "count:Q"],
                            )
                            .properties(height=280),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Aucun thème recensé.")

                with col_a4:
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
                                alt.Tooltip(
                                    "neg_rate_pct:Q",
                                    title="% négatif",
                                    format=".1f",
                                ),
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
                    st.dataframe(ut, use_container_width=True, height=320)
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
                        st.dataframe(
                            theme_trends.rename(
                                columns={
                                    "recent": "7 derniers jours",
                                    "previous": "7 jours avant",
                                    "delta": "Variation",
                                }
                            ),
                            use_container_width=True,
                            height=260,
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
                    "text_display",
                ]
                cols_presentes = [c for c in all_cols if c in flt.columns]

                st.dataframe(
                    flt[cols_presentes].sort_values("created_at_dt", ascending=False),
                    use_container_width=True,
                    height=520,
                )
                st.caption(
                    f"{len(flt):,} lignes affichées (correspond à l’export CSV ci-dessus)."
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
            st.dataframe(ut, use_container_width=True, height=360)
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
