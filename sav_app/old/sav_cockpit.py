import streamlit as st
from PIL import Image

# --- 1. CHARGEMENT DE L'IMAGE ---
# On charge l'image pour l'utiliser comme favicon (icône d'onglet) et logo
try:
    logo = Image.open("logo_sav.png")
except FileNotFoundError:
    st.error("Erreur : L'image 'logo_sav.png' est introuvable. Vérifie qu'elle est bien dans le même dossier que le script.")
    logo = None # On continue sans logo pour ne pas planter l'app

# --- 2. CONFIGURATION DE LA PAGE ---
# Doit être la première commande Streamlit
st.set_page_config(
    page_title="SAV Free - Pilotage",
    page_icon=logo, # Met le logo dans l'onglet du navigateur
    layout="wide"   # Mode large pour bien voir les dashboards
)

# --- 3. BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    if logo:
        st.image(logo, width=180) # Affiche le logo en haut à gauche
    
    st.markdown("---") # Ligne de séparation
    
    # Menu de navigation entre tes deux fonctionnalités
    choix_page = st.radio(
        "Navigation", 
        ["📊 Dashboards", "🐦 Répondre aux Tweets"]
    )
    
    st.markdown("---")
    st.info("Support Technique Free")

# --- 4. AFFICHAGE DU CONTENU PRINCIPAL ---

if choix_page == "📊 Dashboards":
    st.title("📊 Vue d'ensemble - KPIs")
    # --- ICI TU METS TON CODE DE DASHBOARD ---
    st.write("Graphiques et stats ici...")
    
    # Exemple de structure colonnes pour dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric("Tweets en attente", "12", "-2")
    col2.metric("Temps de réponse", "4min", "OK")
    col3.metric("Satisfaction", "4.8/5", "+0.1")

elif choix_page == "🐦 Répondre aux Tweets":
    st.title("🐦 Gestion des réponses")
    # --- ICI TU METS TON CODE POUR RÉPONDRE AUX TWEETS ---
    st.write("Interface de réponse ici...")
    
    st.text_area("Tweet du client", "J'ai plus de connexion depuis ce matin...")
    st.text_area("Votre réponse", "Bonjour, navré pour la gêne...")
    if st.button("Envoyer la réponse"):
        st.success("Réponse envoyée !")