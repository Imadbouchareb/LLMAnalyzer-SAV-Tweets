import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import time

from .state import get_cfg

# --- Helpers statut partagés (alignés sur l'écran Agent SAV) -----------------
STATUS_OPTIONS = [
    "Ouvert",              # spécial = status NON VIDE
    "En attente client",
    "Réaffecté",
    "Répondu",
    "Suggestion acceptée",
    "Clos",
]

def normalize_status_empty(df: pd.DataFrame) -> pd.DataFrame:
    """On ignore le statut du CSV et on repart toujours d'une colonne vide."""
    out = df.copy()
    out["status"] = ""
    return out

def filter_by_status_like_agent(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Même logique que l'écran Agent:
    - 'Ouvert' -> status != ''
    - sinon -> status ∈ selected
    - vide -> pas de filtre
    """
    if not selected:
        return df
    if "Ouvert" in selected:
        return df[df["status"].astype(str).str.strip() != ""]
    return df[df["status"].isin(selected)]

def count_open_like_agent(df: pd.DataFrame) -> int:
    """Compteur 'Ouverts / À traiter' = status NON vide."""
    return int((df["status"].astype(str).str.strip() != "").sum())

@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Gestion de la colonne date - ne pas écraser si created_at_dt existe déjà
    if "created_at_dt" not in df.columns:
        ca = df.get("created_at")
        if ca is not None:
            try:
                df["created_at_dt"] = pd.to_datetime(ca, errors="coerce", utc=True).dt.tz_convert(None)
            except Exception:
                df["created_at_dt"] = pd.to_datetime(ca, errors="coerce")
        else:
            df["created_at_dt"] = pd.NaT
    else:
        # Si created_at_dt existe déjà, la parser correctement
        try:
            df["created_at_dt"] = pd.to_datetime(df["created_at_dt"], errors="coerce")
        except Exception:
            pass

    tcol = _trycol(df, ["text_masked", "text_clean", "text_raw", "text_for_llm"]) or "text_raw"
    if tcol != "text_display":
        df = df.rename(columns={tcol: "text_display"})

    if "intent_primary" not in df and "llm_intent" in df:
        df["intent_primary"] = df["llm_intent"]
    if "sentiment_label" not in df and "llm_sentiment" in df:
        df["sentiment_label"] = df["llm_sentiment"]

    for c in ["llm_urgency_0_3", "llm_severity_0_3"]:
        if c not in df.columns:
            df[c] = np.nan

    if "summary_1l" not in df and "llm_summary" in df:
        df["summary_1l"] = df["llm_summary"]
    if "suggested_reply" not in df and "llm_reply_suggestion" in df:
        df["suggested_reply"] = df["llm_reply_suggestion"]

    # Gestion des thèmes - ne pas écraser si themes_list existe déjà
    if "themes_list" not in df.columns:
        if "themes" in df.columns:
            df["themes_list"] = df["themes"].apply(_aslist)
        else:
            df["themes_list"] = [[] for _ in range(len(df))]
    else:
        # Si themes_list existe déjà, la parser correctement (peut être une chaîne JSON)
        df["themes_list"] = df["themes_list"].apply(_aslist)

    df["prio_score"] = df.apply(_prio_score, axis=1)

    if "status" not in df:
        df["status"] = ""

    # Gestion de l'auteur - ne pas écraser si author existe déjà
    if "author" not in df.columns:
        df["author"] = df.get("screen_name", df.get("user", ""))

    return df

def _trycol(df: pd.DataFrame, cols):
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if col in df.columns:
            return col
    return None

def _aslist(x):
    try:
        parsed = json.loads(x) if isinstance(x, str) else x
        if isinstance(parsed, dict):
            return list(parsed.keys())
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []

def _prio_score(row) -> float:
    try:
        urg = float(row.get("llm_urgency_0_3") or 0) / 3
    except Exception:
        urg = 0.0
    try:
        sev = float(row.get("llm_severity_0_3") or 0) / 3
    except Exception:
        sev = 0.0
    sent = str(row.get("sentiment_label", "")).lower()
    neg = 1.0 if sent.startswith("neg") else 0.5 if sent.startswith("neu") else 0.0
    return round(0.45 * urg + 0.40 * sev + 0.15 * neg, 2)

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EDITS_PATH = DATA_DIR / "sav_edits.csv"
EDIT_COLS = ["tweet_id", "status", "assigned_to", "agent_response", "updated_at"]

def _empty_edits_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EDIT_COLS).astype({
        "tweet_id": "string",
        "status": "string",
        "assigned_to": "string",
        "agent_response": "string",
        "updated_at": "string",
    })

def load_edits() -> pd.DataFrame:
    if EDITS_PATH.exists():
        try:
            df = pd.read_csv(EDITS_PATH, dtype="string")
            for c in EDIT_COLS:
                if c not in df.columns:
                    df[c] = ""
            return df[EDIT_COLS].fillna("")
        except Exception:
            return _empty_edits_df()
    return _empty_edits_df()

def upsert_edits(rows: list[dict]) -> None:
    """rows: [{tweet_id, status?, assigned_to?, agent_response?}] — upsert par tweet_id."""
    base = load_edits()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    up = []
    for r in rows:
        rid = str(r.get("tweet_id", "")).strip()
        if not rid:
            continue
        up.append({
            "tweet_id": rid,
            "status": str(r.get("status", "") or "").strip(),
            "assigned_to": str(r.get("assigned_to", "") or "").strip(),
            "agent_response": str(r.get("agent_response", "") or "").strip(),
            "updated_at": now_iso,
        })
    if not up:
        return
    up_df = pd.DataFrame(up, columns=EDIT_COLS).astype("string").fillna("")
    # Upsert sur tweet_id (les champs vides n'écrasent pas des valeurs existantes)
    merged = base.set_index("tweet_id").combine_first(
        _empty_edits_df().set_index("tweet_id")
    )
    for row in up_df.itertuples(index=False):
        tid = row.tweet_id
        if tid not in merged.index:
            merged.loc[tid] = [row.status, row.assigned_to, row.agent_response, row.updated_at]
        else:
            cur = merged.loc[tid]
            merged.loc[tid, "status"] = row.status or cur.get("status", "")
            merged.loc[tid, "assigned_to"] = row.assigned_to or cur.get("assigned_to", "")
            merged.loc[tid, "agent_response"] = row.agent_response or cur.get("agent_response", "")
            merged.loc[tid, "updated_at"] = row.updated_at
    merged = merged.reset_index()[EDIT_COLS].fillna("")
    merged.to_csv(EDITS_PATH, index=False, encoding="utf-8")

def apply_edits(df: pd.DataFrame) -> pd.DataFrame:
    """Écrase df.status / assigned_to / agent_response avec les éditions persistées si présentes."""
    edits = load_edits()
    if edits.empty or df.empty or "tweet_id" not in df.columns:
        return df
    left = df.copy()
    m = left.merge(edits, on="tweet_id", how="left", suffixes=("", "_edit"))
    for col in ("status", "assigned_to", "agent_response"):
        edit_col = f"{col}_edit"
        if edit_col in m.columns:
            m[col] = m.apply(
                lambda r: r[edit_col] if isinstance(r[edit_col], str) and r[edit_col].strip() else r[col],
                axis=1
            )
    drop_cols = [c for c in m.columns if c.endswith("_edit") or c == "updated_at"]
    return m.drop(columns=drop_cols, errors="ignore")

def persist_ticket_updates(tweet_ids: list[str], status: str = "", assigned_to: str = "", agent_response: str = "") -> None:
    rows = []
    for tid in tweet_ids:
        rows.append({
            "tweet_id": str(tid),
            "status": status,
            "assigned_to": assigned_to,
            "agent_response": agent_response,
        })
    upsert_edits(rows)
