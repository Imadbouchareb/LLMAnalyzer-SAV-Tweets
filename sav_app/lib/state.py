import copy
from pathlib import Path
import streamlit as st

CURRENT_KEY = "agent_current_id"

DEFAULT_CFG = {
    "data": {"prio_threshold": 0.70},
    "ui": {"page_size": 15, "scroll_where": "top", "pin_selected_first": True},
    "paths": {
        "csv_path": r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\sav_app\tweets_scored_llm.csv"
    },
    "llm": {"temperature": 0.2, "max_tokens": 512},
}

def get_cfg():
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = copy.deepcopy(DEFAULT_CFG)
    return st.session_state["cfg"]

def cfg_vars():
    cfg = get_cfg()
    return (
        int(cfg["ui"]["page_size"]),
        cfg["ui"]["scroll_where"],
        bool(cfg["ui"]["pin_selected_first"]),
        float(cfg["data"].get("prio_threshold", 0.0)),
    )
