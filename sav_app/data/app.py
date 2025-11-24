import streamlit as st

st.set_page_config(page_title="SAV Tweets — Bienvenue", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        background: linear-gradient(145deg, #1e1e3f, #121221);
        border-radius: 18px;
        padding: 4rem 3rem;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <h1 style="font-size: 3.5rem; font-weight: 700; color: #FFFFFF; margin-bottom: 1.5rem;">
            Bienvenue sur SAV Tweets
        </h1>
        <p style="font-size: 1.25rem; color: #E0E0E0; max-width: 700px; margin: 0 auto 2.5rem auto; line-height: 1.6;">
            Votre plateforme d'analyse et de gestion du support client propulsée par l'IA.<br>
            Analysez les sentiments, pilotez vos KPI et accélérez les réponses clients.
        </p>
    </div>cd "/c/Users/hallo/OneDrive/Bureau/IA Free Mobile/sav_app"
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

cols = st.columns([2, 3, 2])
with cols[1]:
    if st.button("Accéder à l’application", use_container_width=True, type="primary"):
        st.switch_page("pages/0_Accueil.py")

st.markdown("---")
st.caption("© 2024 SAV Tweets — Propulsé par Streamlit & IA.")
