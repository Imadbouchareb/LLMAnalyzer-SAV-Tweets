# pages/3_Agent_SAV.py
# Refonte complète de l’écran Agent pour l’aligner sur Manager/Analyste.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from itertools import combinations

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from lib.data import load_df
from lib.state import get_cfg
from lib.ui import inject_style, set_container_wide, inject_sticky_css

st.set_page_config(
    page_title="File Agent SAV",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
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
# Helpers de normalisation
# -----------------------------------------------------------------------------
def _parse_dt(series: pd.Series) -> pd.Series:
    """
    Essaie d'interpréter une colonne date/heure -> pandas.Timestamp sans timezone foireuse.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.notna().sum() == 0:
        # 2e tentative style "27/10/2025 14:33"
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # on enlève le tzinfo
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        try:
            s = s.dt.tz_localize(None)
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
    df["created_at_dt"] = _parse_dt(df[date_col]) if date_col else pd.Series(pd.NaT, index=df.index)

    # 3. texte du tweet
    text_col = next(
        (c for c in ["text_raw","text_display", "text", "tweet", "content", "body"] if c in df.columns),
        None,
    )
    df["text_display"] = df[text_col].astype(str) if text_col else ""

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

    # 6. statut SAV
    status_col = next(
        (c for c in ["status", "statut", "state"] if c in df.columns),
        None,
    )
    df["status"] = df[status_col].astype(str).fillna("Ouvert") if status_col else "Ouvert"

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
                "text_display": "Message client factice pour la file d’attente.",
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
    try:
        direct = load_df(dataset_key)
        if direct is not None and len(direct) > 0:
            notes.append(f"load_df('{dataset_key}') : OK ({len(direct)} lignes)")
            return _prepare_agent_df(direct), f"load_df('{dataset_key}')", notes
        notes.append(f"load_df('{dataset_key}') : vide")
    except Exception as exc:
        notes.append(f"load_df('{dataset_key}') : échec -> {exc}")

    candidates = [
        Path(r"C:\Users\IMAD\Desktop\IA Free Mobile\IA Free Mobile\tweets_scored_llm.csv"),
        APP_ROOT / "tweets_scored_llm.csv",
        APP_ROOT / "data" / "tweets_scored_llm.csv",
        Path(r"C:\projetrncp\tweets_scored_llm.csv"),
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

if "agent_active_view" not in st.session_state:
    st.session_state["agent_active_view"] = "queue"
if "agent_thr_urg" not in st.session_state:
    st.session_state["agent_thr_urg"] = 2.0
if "agent_date_preset" not in st.session_state:
    st.session_state["agent_date_preset"] = "all"

with st.sidebar:
    st.title("Navigation")
    st.info("Vous êtes sur l'écran : **Agent SAV**")
    if st.button("⬅️ Revenir à l'accueil", use_container_width=True):
        st.switch_page("pages/0_Accueil.py")
    st.divider()
    if data_loaded_ok:
        st.success(f"✅ {len(df_real):,} lignes chargées depuis :\n{used_path}")
    else:
        st.error("Aucune donnée réelle trouvée – mode démo 30 tickets.")
    with st.expander("Diagnostic import"):
        for note in debug_notes:
            st.text(note)
    st.header("Filtres Agent")
    tone_filter = st.multiselect(
        "Sentiment",
        options=sorted(df["sentiment_label"].replace("", "Non précisé").unique()),
        key="agent_tone",
    )
    status_options = sorted(df["status"].unique())
    status_defaults = status_options
    status_filter = st.multiselect(
        "Statut",
        options=status_options,
        default=status_defaults,
        key="agent_status",
    )
    urgency_min = st.slider("Urgence minimale", 0.0, 3.0, 0.0, 0.1, key="agent_urgency")
    assignee_filter = st.multiselect(
        "Assigné à", options=sorted(set(df["assigned_to"]) - {""}), key="agent_assignee"
    )
    search_text = st.text_input("Recherche (ID / texte / auteur)", "", key="agent_search").strip()

    if st.button("🔄 Réinitialiser filtres", use_container_width=True, key="agent_reset_filters"):
        st.session_state["agent_tone"] = []
        st.session_state["agent_status"] = status_options
        st.session_state["agent_urgency"] = 0.0
        st.session_state["agent_assignee"] = []
        st.session_state["agent_search"] = ""
        st.session_state.pop("agent_time_dates", None)
        st.session_state.pop("agent_time_dates_input", None)
        st.rerun()

back_col, _ = st.columns([1, 5])
with back_col:
    if st.button("⬅️ Retour à l'accueil", use_container_width=True, key="agent_back_main"):
        st.switch_page("pages/0_Accueil.py")

st.title("File d’attente Agent SAV")

# applique filtres
flt = df.copy()

if tone_filter:
    flt = flt[flt["sentiment_label"].replace("", "Non précisé").isin(tone_filter)]
if status_filter:
    flt = flt[flt["status"].isin(status_filter)]
if urgency_min > 0:
    flt = flt[flt["llm_urgency_0_3"] >= urgency_min]
if assignee_filter:
    flt = flt[flt["assigned_to"].isin(assignee_filter)]
if search_text:
    lower = search_text.lower()
    flt = flt[
        flt["tweet_id"].str.lower().str.contains(lower, na=False)
        | flt["text_display"].str.lower().str.contains(lower, na=False)
        | flt["author"].str.lower().str.contains(lower, na=False)
    ]

# si rien après filtre
if flt.empty:
    st.info("Aucun ticket ne correspond aux filtres.")
    st.stop()

open_mask = flt["status"].str.lower().isin({"ouvert", "a traiter"})
urgent_mask = flt["llm_urgency_0_3"] >= 2
today_mask = flt["created_at_dt"].dt.date == pd.Timestamp.today().date()

# métriques (4 colonnes)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickets affichés", f"{len(df):,}")
m2.metric("Ouverts / À traiter", f"{open_mask.sum():,}")
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
sentiment_theme_all = (
    flt.assign(
        theme=flt["theme_primary"].fillna("").astype(str).str.strip().replace("", "Non précisé"),
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

left, main = st.columns([1, 3.6], vertical_alignment="top")
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
            disabled=(st.session_state["agent_active_view"] == view_value),
            key=f"agent_nav_{view_value}",
        ):
            st.session_state["agent_active_view"] = view_value
            st.rerun()

with main:
    st.markdown('<div class="ma-card"><h3>Insights</h3></div>', unsafe_allow_html=True)
    active_view = st.session_state["agent_active_view"]

    if active_view == "queue":
        queue_df = flt.copy()
        if queue_df.empty:
            st.info("Aucun ticket dans la file d’attente.")
        else:
            st.subheader("Actions rapides")
            act_cols = st.columns(3)
            if act_cols[0].button("↩️ Réaffecter", use_container_width=True, key="agent_action_reassign"):
                st.toast("Ticket marqué pour réaffectation.", icon="🔄")
            if act_cols[1].button("✅ Clore", use_container_width=True, key="agent_action_close"):
                st.toast("Ticket marqué comme clos (simulation).", icon="✅")
            if act_cols[2].button("⏳ Mettre en attente client", use_container_width=True, key="agent_action_pending"):
                st.toast("Ticket mis en attente client (simulation).", icon="⏳")

            queue_view = queue_df[
                [
                    "tweet_id",
                    "created_at_dt",
                    "author",
                    "text_display",
                    "sentiment_label",
                    "themes_list",
                    "llm_urgency_0_3",
                    "status",
                    "assigned_to",
                ]
            ].copy()
            queue_view["thèmes"] = queue_view["themes_list"].apply(lambda L: ", ".join(L[:3]))
            queue_view = queue_view.drop(columns=["themes_list"])
            queue_view = queue_view.rename(
                columns={
                    "tweet_id": "ID",
                    "created_at_dt": "Date",
                    "author": "Auteur",
                    "text_display": "Message",
                    "sentiment_label": "Sentiment",
                    "llm_urgency_0_3": "Urgence",
                    "status": "Statut",
                    "assigned_to": "Assigné",
                }
            ).sort_values("Date", ascending=False)

            st.subheader("File d’attente priorisée")
            if AGGRID_OK:
                gb = GridOptionsBuilder.from_dataframe(queue_view)
                gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True, sortable=True, filter=True)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
                gb.configure_selection("single", use_checkbox=False)
                gb.configure_grid_options(rowSelection="single", suppressRowClickSelection=False)
                grid_resp = AgGrid(
                    queue_view,
                    gridOptions=gb.build(),
                    height=420,
                    fit_columns_on_grid_load=True,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    key="agent_queue_grid",
                ) or {}
            else:
                grid_resp = {}
                st.dataframe(queue_view, use_container_width=True, height=420)

            with st.expander("Recherche rapide par ID"):
                select_options = ["(Choisir)"] + queue_view["ID"].tolist()
                selected_dropdown = st.selectbox("Sélectionne un ID de tweet", select_options, key="agent_id_select")

            selected_ids = []
            if AGGRID_OK:
                selected_rows = grid_resp.get("selected_rows", []) if isinstance(grid_resp, dict) else []
                selected_ids = [row.get("ID") for row in selected_rows if row.get("ID")]
            elif not queue_view.empty:
                selected_ids = [queue_view.iloc[0]["ID"]]

            if selected_dropdown and selected_dropdown != "(Choisir)":
                selected_ids = [selected_dropdown]

            if selected_ids:
                st.subheader("Résumé LLM & suggestion")
                sel_df = queue_df[queue_df["tweet_id"].isin(selected_ids)][
                    ["tweet_id", "author", "llm_summary", "llm_reply_suggestion", "text_display"]
                ]
                for row in sel_df.itertuples(index=False):
                    with st.expander(f"Ticket {row.tweet_id} — {row.author}", expanded=True):
                        st.markdown(f"**Résumé LLM :**\n\n{row.llm_summary or '_Non fourni_'}")
                        st.markdown(f"**Suggestion LLM :**\n\n{row.llm_reply_suggestion or '_Non fournie_'}")
                        st.markdown("**Message d’origine :**")
                        st.write(row.text_display)
                        response_key = f"agent_reply_{row.tweet_id}"
                        if response_key not in st.session_state:
                            st.session_state[response_key] = ""
                        action_cols = st.columns(2)
                        if action_cols[1].button("✅ Accepter la suggestion LLM", key=f"accept_btn_{row.tweet_id}"):
                            st.session_state[response_key] = row.llm_reply_suggestion or ""
                            st.toast("Suggestion copiée dans la zone de réponse.", icon="✅")
                        st.text_area("Réponse agent", key=response_key, placeholder="Rédige ta réponse ici…")
                        if action_cols[0].button("💬 Répondre", key=f"reply_btn_{row.tweet_id}"):
                            st.success("Réponse enregistrée localement (simulation).")
            else:
                st.info("Sélectionne un ticket pour afficher le résumé et la suggestion LLM.")

    elif active_view == "urgences":
        slider_col, kpi_col = st.columns([4, 1.4])
        with slider_col:
            st.slider(
                "Seuil d’urgence (≥)",
                min_value=0.0,
                max_value=3.0,
                step=0.1,
                key="agent_thr_urg",
                help="Ajuste le seuil : KPIs, table et graphes se mettent à jour instantanément.",
            )

        current_thr = float(st.session_state["agent_thr_urg"])
        urgent_df_raw = flt.loc[flt["llm_urgency_0_3"] >= current_thr].copy()
        urgent_count_raw = len(urgent_df_raw)
        total_count = len(flt)
        prev_thr = st.session_state.get("agent_prev_thr", current_thr)
        prev_count = st.session_state.get("agent_prev_urgent_count", urgent_count_raw)
        delta = urgent_count_raw - prev_count

        with kpi_col:
            st.metric(
                label=f"Urgents restants (≥ {current_thr:.1f})",
                value=f"{urgent_count_raw:,}",
                delta=f"{delta:+,} vs {prev_thr:.1f}",
            )

        if prev_thr != current_thr:
            st.session_state["agent_prev_thr"] = current_thr
            st.session_state["agent_prev_urgent_count"] = urgent_count_raw

        ratio_pct = (100 * urgent_count_raw / total_count) if total_count else 0.0
        st.caption(
            f"**{urgent_count_raw:,}** / {total_count:,} tickets "
            f"(**{ratio_pct:.1f}%**) marqués urgents pour un seuil ≥ **{current_thr:.1f}**."
        )

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
                f"Aucune urgence détectée avec le seuil actuel ≥ {current_thr:.1f}"
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
                f"Analyse des urgences (seuil ≥ {current_thr:.1f}"
                + (f", année {selected_year}" if selected_year != "Toutes" else "")
                + ")."
            )

            urgent_view = urgent_df[
                [
                    "tweet_id",
                    "created_at_dt",
                    "author",
                    "text_display",
                    "sentiment_label",
                    "llm_urgency_0_3",
                    "status",
                    "assigned_to",
                    "themes_list",
                ]
            ].copy()
            urgent_view["Thèmes"] = urgent_view["themes_list"].apply(
                lambda L: ", ".join(L[:3])
            )
            urgent_view = urgent_view.drop(columns=["themes_list"]).rename(
                columns={
                    "tweet_id": "ID",
                    "created_at_dt": "Date",
                    "author": "Auteur",
                    "text_display": "Message",
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
                st.altair_chart(
                    alt.Chart(urgent_authors)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Tickets urgents"),
                        y=alt.Y("author:N", sort="-x", title="Auteur"),
                        tooltip=["author:N", "count:Q"],
                    )
                    .properties(height=280),
                    use_container_width=True,
                )
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
                st.altair_chart(
                    alt.Chart(urgent_sentiment)
                    .mark_bar()
                    .encode(
                        x=alt.X("sentiment:N", title="Sentiment"),
                        y=alt.Y("count:Q", title="Volume"),
                        color=alt.Color("sentiment:N", legend=None),
                        tooltip=["sentiment:N", "count:Q"],
                    )
                    .properties(height=260),
                    use_container_width=True,
                )
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
                    .properties(height=280),
                    use_container_width=True,
                )
            else:
                st.caption("Pas de combinaison urgence/sévérité disponible.")

    elif active_view == "themes":
        st.subheader("Volumes par thème")
        if not theme_counts_df.empty:
            st.altair_chart(
                alt.Chart(theme_counts_df.head(25))
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

        st.subheader("Co-occurrences de thèmes")
        if not co_occ_df.empty:
            st.dataframe(co_occ_df, use_container_width=True, height=260)
        else:
            st.caption("Pas de co-occurrence significative détectée.")

        st.subheader("Sentiment par thème principal")
        if not sentiment_theme_all.empty:
            st.altair_chart(
                alt.Chart(sentiment_theme_all)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Volume"),
                    y=alt.Y("theme:N", sort="-x", title="Thème"),
                    color=alt.Color("sentiment:N", title="Sentiment"),
                    tooltip=["theme:N", "sentiment:N", "count:Q"],
                )
                .properties(height=320),
                use_container_width=True,
            )
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
            timeline_chart = timeline_chart.properties(height=260, title="Timeline quotidienne")

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
                .properties(height=220, title="Répartition horaire")
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
                .properties(height=240, title="Chaleur jour × heure")
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
                .properties(height=240, title="Histogramme sentiment")
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
                .properties(height=280),
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
                .properties(height=300),
                use_container_width=True,
            )
        else:
            st.caption("Pas assez de données pour calculer l’âge des tickets.")

    elif active_view == "exports":
        st.subheader("Exports & données brutes")
        st.download_button(
            "⬇️ Exporter les tickets filtrés (CSV)",
            data=flt.to_csv(index=False).encode("utf-8"),
            file_name="file_agent_sav_filtrée.csv",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Exporter les tickets filtrés (JSON)",
            data=flt.to_json(orient="records", force_ascii=False).encode("utf-8"),
            file_name="file_agent_sav_filtrée.json",
            use_container_width=True,
        )
        if st.checkbox("Afficher les données brutes", key="agent_show_raw"):
            st.dataframe(flt, use_container_width=True, height=400)

    elif active_view == "logs":
        st.subheader("Journaux & diagnostic")
        st.markdown(f"- Source courante : **{used_path or 'non déterminé'}**")
        st.markdown(f"- Dataset key Agent : **{dataset_key}**")
        st.markdown(f"- Dataset key Manager (fallback) : **{cfg.get('manager_dataset_key', 'tweets_scored_llm')}**")
        st.markdown(f"- Lignes chargées : **{len(df_real) if data_loaded_ok else len(df)}**")
        st.markdown("Historique de chargement :")
        for note in debug_notes:
            st.code(note, language="text")
