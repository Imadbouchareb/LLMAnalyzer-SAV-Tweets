import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector import create_vector_store_from_pdf,create_vector_store_from_jsonl
import re
from dotenv import load_dotenv
import json
import os

load_dotenv()  # Load environment variables from .env file


st.set_page_config(
    page_title="Assistant Free Mobile",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface lisible et moderne
st.markdown("""
<style>
    /* Style général - Fond doux beige/crème */
    .stApp {
        background: linear-gradient(135deg, #faf8f5 0%, #f0ede8 100%);
    }
    
    /* Zone de chat - Messages sur fond légèrement teinté */
    .stChatMessage {
        background-color: #fefdfb;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e3db;
    }
    
    /* Messages utilisateur - fond bleu très doux */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #f0f5ff;
        border-left: 4px solid #3b82f6;
    }
    
    /* Messages assistant - fond crème légèrement vert */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f6faf7;
        border-left: 4px solid #10b981;
    }
    
    /* Texte des messages - Noir pour lisibilité */
    [data-testid="stChatMessageContent"] {
        color: #1e293b;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Input de chat */
    .stChatInputContainer {
        border-radius: 25px;
        background-color: #fefdfb;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
        border: 2px solid #3b82f6;
    }
    
    /* Sidebar - Fond moderne avec dégradé Free Mobile */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 4px 0 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Tous les textes de la sidebar en blanc */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .st-emotion-cache-1n543e5,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Bouton dans la sidebar */
    [data-testid="stSidebar"] .stButton button {
        background: #ffffff;
        color: #2563eb;
        font-weight: 700;
        border: 2px solid transparent;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: #f0f9ff;
        border: 2px solid #ffffff;
        color: #1e40af;
    }
    
    /* Radio buttons dans la sidebar - style amélioré */
    [data-testid="stSidebar"] .stRadio > div {
        background-color: transparent;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.12);
        padding: 12px 18px;
        border-radius: 12px;
        margin: 6px 0;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        border: 2px solid transparent;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.25);
        border: 2px solid rgba(255, 255, 255, 0.4);
        transform: translateX(5px);
    }
    
    /* Radio button sélectionné */
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background-color: rgba(255, 255, 255, 0.3);
        border: 2px solid #ffffff;
        font-weight: 600;
    }
    
    /* Headers dans la sidebar */
    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-size: 1.4rem;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Subheader dans la sidebar */
    [data-testid="stSidebar"] .st-emotion-cache-1n543e5 h3 {
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Caption dans la sidebar */
    [data-testid="stSidebar"] .st-emotion-cache-1wmy9hl {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem;
    }
    
    /* Séparateur dans la sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 15px 0;
    }
    
    /* Boutons principaux - style cohérent */
    .stButton button {
        border-radius: 20px;
        background: #3b82f6;
        color: white;
        font-weight: 600;
        border: none;
        padding: 10px 25px;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Titre - Bleu foncé sur fond clair */
    h1 {
        color: #1e3a8a;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    
    /* Sous-titre - Gris foncé lisible */
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

st.title("📱 Assistant Free Mobile")
st.markdown('<p class="subtitle">Votre assistant virtuel intelligent pour toutes vos questions</p>', unsafe_allow_html=True)

def clean_deepseek_response(response):
    """
    Remove <think> tags and their content from DeepSeek R1 responses
    """
    # Remove everything between <think> and </think> tags (including the tags)
    cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove any remaining isolated <think> or </think> tags
    cleaned_response = re.sub(r'</?think>', '', cleaned_response)
    
    # Clean up extra whitespace that might be left
    cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

# Streaming filter to remove <think> blocks while streaming
def stream_without_think(chunks_iterable):
    in_think = False
    buffer = ""
    for chunk in chunks_iterable:
        text = chunk if isinstance(chunk, str) else str(chunk)
        buffer += text
        output = []
        idx = 0
        while True:
            if in_think:
                close_pos = buffer.find("</think>", idx)
                if close_pos == -1:
                    # Still inside think; drop buffer content and wait for more
                    buffer = ""
                    idx = 0
                    break
                else:
                    idx = close_pos + len("</think>")
                    in_think = False
                    # Continue scanning remaining buffer
                    continue
            else:
                open_pos = buffer.find("<think>", idx)
                if open_pos == -1:
                    # No open tag; emit remainder
                    output.append(buffer[idx:])
                    buffer = ""
                    idx = 0
                    break
                else:
                    # Emit text before think, then enter think mode
                    output.append(buffer[idx:open_pos])
                    idx = open_pos + len("<think>")
                    in_think = True
                    # Loop to look for closing tag
                    continue
        if output:
            yield "".join(output)

# Initialize the LLM with temperature control
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # Updated to reflect your model choice
        temperature=0.2,  # Lower temperature for more focused responses
    )

llm = get_llm()

@st.cache_resource
def get_retriever(path: str = "free_mobile_rag_qas_full.jsonl", collection_name: str = "free_mobile"):
    # create_vector_store_from_jsonl doit persister la DB et ne pas la supprimer à chaque appel
    return create_vector_store_from_jsonl(path, collection_name=collection_name)

# Create the prompt template with improved context handling
template = """

Vous êtes un assistant Free Mobile professionnel. Répondez à la question du client en vous basant sur les informations fournies.

Conversation précédente ( si cette derniere est essentielle pour la compréhension de la question ) :
{chat_history}

Contexte de la base de connaissances:
{context}

Question du client: {question}

Instructions:
- Répondez uniquement en français
- Utilisez un ton professionnel et bienveillant
- Basez votre réponse sur les informations fournies
- Soyez concis mais complet
- Ne répétez pas la question
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# Gestion multi-conversations dans la session
if "conversations" not in st.session_state:
    st.session_state.conversations = {"Conversation 1": []}
    st.session_state.current_conversation = "Conversation 1"

# Ensure a per-conversation retriever mapping exists
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}

# Sidebar : historique et gestion des conversations
with st.sidebar:
    st.header("💬 Mes Conversations")
    st.markdown("---")
    
    # Bouton pour nouvelle conversation (en haut)
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        new_name = f"Conversation {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[new_name] = []
        # Do not carry over retriever to the new conversation
        st.session_state.retrievers[new_name] = None
        st.session_state.current_conversation = new_name
        st.rerun()
    
    st.markdown("---")
    
    # Liste des conversations existantes
    st.subheader("📋 Historique")
    conversation_names = list(st.session_state.conversations.keys())
    selected_conversation = st.radio(
        "Sélectionnez une conversation :",
        conversation_names,
        index=conversation_names.index(st.session_state.current_conversation),
        label_visibility="collapsed"
    )
    st.session_state.current_conversation = selected_conversation
    
    # Informations sur la conversation actuelle
    st.markdown("---")
    st.caption(f"💬 {len(st.session_state.conversations[st.session_state.current_conversation])} messages")

# Zone principale avec container
with st.container():
    # Synchronisation de l'historique courant
    st.session_state.messages = st.session_state.conversations[st.session_state.current_conversation]

    # Affichage de l'historique courant
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

# React to user input
if prompt_text := st.chat_input("💬 Posez votre question sur Free Mobile..."):
    # Crée / récupère le retriever en cache (ne pas recréer à chaque message)
    if st.session_state.retrievers.get(st.session_state.current_conversation) is None:
        st.session_state.retrievers[st.session_state.current_conversation] = get_retriever()

    retriever = st.session_state.retrievers[st.session_state.current_conversation]

    # Affichage du message utilisateur
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.conversations[st.session_state.current_conversation] = st.session_state.messages

    # Récupération des docs via retriever (fallback robuste)
    docs = []
    if retriever:
        try:
            docs = retriever.get_relevant_documents(prompt_text)
        except Exception:
            raw = getattr(retriever, "raw", retriever)
            if hasattr(raw, "similarity_search"):
                docs = raw.similarity_search(prompt_text, k=None)  # pas de limite
            elif hasattr(raw, "retrieve"):
                docs = raw.retrieve(prompt_text)
            else:
                docs = []

    # Construire le contexte avec tous les documents
    if docs:
        ctx_parts = []
        for i, d in enumerate(docs):  # plus de slice [:5]
            src = d.metadata.get("source", "") if getattr(d, "metadata", None) else ""
            line = d.metadata.get("line", "") if getattr(d, "metadata", None) else ""
            header = f"Document {i+1}"
            if src or line:
                header += f" (source: {src}{', line: '+str(line) if line else ''})"
            ctx_parts.append(f"{header}:\n{d.page_content.strip()}")
        context = "\n\n---\n\n".join(ctx_parts)
    else:
        context = "Aucune information pertinente trouvée dans la base de connaissances."

    # Format chat history for context
    chat_history = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in st.session_state.messages[:-1]
    ])

    # Stream response — context est injecté dans la template via la clé "context"
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Réflexion en cours..."):
            streamed_text = st.write_stream(
                chain.stream({
                    "context": context,
                    "chat_history": chat_history,
                    "question": prompt_text
                })
            )

    # Add assistant response to chat history (store the streamed, cleaned version)
    final_response = clean_deepseek_response(streamed_text)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.session_state.conversations[st.session_state.current_conversation] = st.session_state.messages
