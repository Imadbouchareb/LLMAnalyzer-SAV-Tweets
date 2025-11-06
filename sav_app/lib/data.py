import json
import numpy as np
import pandas as pd
import streamlit as st

from .state import get_cfg

@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    ca = df.get("created_at")
    if ca is not None:
        try:
            df["created_at_dt"] = pd.to_datetime(ca, errors="coerce", utc=True).dt.tz_convert(None)
        except Exception:
            df["created_at_dt"] = pd.to_datetime(ca, errors="coerce")
    else:
        df["created_at_dt"] = pd.NaT

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

    if "themes" in df.columns:
        df["themes_list"] = df["themes"].apply(_aslist)
    else:
        df["themes_list"] = [[] for _ in range(len(df))]

    df["prio_score"] = df.apply(_prio_score, axis=1)

    if "status" not in df:
        df["status"] = "Ouvert"

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
