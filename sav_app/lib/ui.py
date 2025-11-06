import streamlit as st

LIGHT_CSS = """
<style>
:root {
    --card-bg: #11182c;
    --card-border: rgba(255,255,255,0.05);
    --card-shadow: 0 10px 24px rgba(8,13,26,0.45);
    --muted: #94a3b8;
    --chip-bg: #16213b;
    --chip-border: rgba(255,255,255,0.05);
}
body, .stApp { background:#0b1120; color:#e2e8f0; }
h1,h2,h3 { color:#f8fafc; }
.ma-card { background:var(--card-bg); border:1px solid var(--card-border);
           box-shadow:var(--card-shadow); border-radius:16px; padding:14px 16px; margin-bottom:12px; }
.ma-pill { float:right; background:#1b2542; border:1px solid rgba(255,255,255,0.08);
           border-radius:999px; padding:6px 10px; color:#cbd5f5; font-weight:600; }
.ma-filterbar { background:var(--card-bg); border:1px solid var(--card-border); border-radius:14px;
                padding:10px 14px; box-shadow:var(--card-shadow); font-weight:600; color:#f8fafc; }
.ma-filterbar a { color:#cbd5f5; margin-right:16px; text-decoration:underline; }
input, textarea, select, .stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background:#0f172a !important; color:#e2e8f0 !important;
    border:1px solid rgba(255,255,255,0.1) !important; box-shadow:none !important;
}
input::placeholder, textarea::placeholder { color:rgba(226,232,240,0.7) !important; }
[data-testid="stDataFrame"] { color:#f1f5f9; }
</style>
"""

def inject_style():
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

def set_container_wide(max_width_px: int = 1680, top_padding: str = ".6rem"):
    st.markdown(
        f"<style>.block-container{{max-width:{max_width_px}px; padding-top:{top_padding};}}</style>",
        unsafe_allow_html=True,
    )

_STICKY = """
<style>
  .ma-sticky{ position: sticky; top: 10px; }
</style>
"""

def inject_sticky_css():
    st.markdown(_STICKY, unsafe_allow_html=True)

def render_readonly_text(container, label: str, value: str, key: str, height: int = 220):
    if label:
        container.subheader(label)
    container.text_area("", value=value or "", height=height, disabled=True, key=key)
