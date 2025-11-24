# 🤖 Free Mobile Chatbot - Assistant Client Intelligent

Application de chatbot conversationnel utilisant le RAG (Retrieval-Augmented Generation) pour répondre aux questions des clients Free Mobile.

## 📋 Description

Cette application Streamlit combine la recherche sémantique vectorielle avec un modèle de langage puissant (LLM) pour fournir des réponses précises et contextuelles basées sur une base de connaissances Free Mobile.

### ✨ Fonctionnalités principales

- 💬 **Conversations multiples** : Créez et gérez plusieurs conversations simultanément
- 🔍 **Recherche sémantique** : Récupération intelligente des documents pertinents via ChromaDB
- 🧠 **RAG (Retrieval-Augmented Generation)** : Réponses générées à partir de la base de connaissances
- 📚 **Base de connaissances** : Questions/réponses Free Mobile vectorisées
- 🌊 **Streaming en temps réel** : Affichage progressif des réponses
- 💾 **Historique persistant** : Conservation de toutes vos conversations

## 🏗️ Architecture technique

```
┌─────────────────┐
│  Utilisateur    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   Streamlit Interface   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Question utilisateur   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ChromaDB Vector Store          │
│  (Recherche sémantique k=100)   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Contexte + Historique          │
│  + Question                     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ChatGroq LLM                   │
│  (llama-3.3-70b-versatile)      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Réponse streamée et nettoyée   │
└─────────────────────────────────┘
```

### 🔧 Stack technique

- **Interface** : Streamlit
- **LLM** : ChatGroq (llama-3.3-70b-versatile)
- **Embeddings** : Ollama (mxbai-embed-large)
- **Base vectorielle** : ChromaDB
- **Framework** : LangChain

## 📦 Installation

### Prérequis

- Python 3.8+
- Ollama installé et en cours d'exécution
- Clé API Groq

### Étapes d'installation

1. **Cloner le projet**
```bash
cd "c:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app"
```

2. **Créer un environnement virtuel**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Installer les dépendances**
```powershell
pip install -r requirements.txt
```

4. **Installer et lancer Ollama**
   - Téléchargez Ollama depuis [ollama.ai](https://ollama.ai)
   - Téléchargez le modèle d'embeddings :
```powershell
ollama pull mxbai-embed-large
```

5. **Configurer les variables d'environnement**

Créez un fichier `.env` à la racine du projet :
```env
GROQ_API_KEY=votre_clé_api_groq_ici
```

Pour obtenir une clé API Groq :
- Rendez-vous sur [console.groq.com](https://console.groq.com)
- Créez un compte gratuit
- Générez une clé API

## 🚀 Lancement de l'application

### Méthode 1 : Raccourci (Recommandé) ⭐
Double-cliquez sur le raccourci **"Assistant Free Mobile"** sur votre Bureau

### Méthode 2 : Fichier batch
Double-cliquez sur `Lancer_Application.bat` dans le dossier du projet

### Méthode 3 : Ligne de commande
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

L'application sera accessible sur : `http://localhost:8501` (ou 8502, 8503)

**Note** : Un raccourci de bureau a été créé lors de l'installation pour un lancement rapide en un clic !

## 📁 Structure du projet

```
chatboot_app/
│
├── app.py                          # Application principale Streamlit
├── vector.py                       # Gestion de la base vectorielle
│
├── free_mobile_rag_qas_full.jsonl  # Base de connaissances Q&A (39 Q&A)
│
├── database/                       # Bases ChromaDB persistées
│   └── free_mobile/
│       ├── chroma.sqlite3
│       └── be27cf18-.../
│
├── old/                            # Fichiers archivés
│   ├── app_c21_light.py
│   ├── dashboard_sav.py
│   ├── streamlit_sav_app.py
│   ├── ui_utils.py
│   └── ...
│
├── requirements.txt                # Dépendances Python
├── .env                            # Variables d'environnement (sécurisé)
├── .gitignore                      # Fichiers à ignorer par Git
├── README.md                       # Ce fichier
├── COMMANDES.md                    # Guide des commandes utiles
│
├── Lancer_Application.bat          # Fichier de lancement rapide
└── venv/                           # Environnement virtuel Python
```

**Note** : Un raccourci "Assistant Free Mobile" est également présent sur votre Bureau.

## 💡 Utilisation

### Démarrer une conversation

1. Lancez l'application
2. Tapez votre question dans la zone de chat
3. L'assistant recherche dans la base de connaissances et génère une réponse

### Gérer plusieurs conversations

- **Créer une nouvelle conversation** : Cliquez sur "➕ Nouvelle conversation" dans la sidebar
- **Basculer entre conversations** : Sélectionnez la conversation souhaitée dans la liste

### Exemples de questions

- "Comment activer la 5G sur mon forfait ?"
- "Quel est le délai de livraison de ma carte SIM ?"
- "Comment résilier mon abonnement ?"
- "Quels sont les frais de roaming à l'étranger ?"

## 🔧 Configuration

### Modifier le modèle LLM

Dans `app.py`, ligne 78-81 :
```python
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",  # Changez ici
        temperature=0.2,
    )
```

### Modifier le nombre de documents récupérés

Dans `app.py`, ligne 175 ou `vector.py`, ligne 103 :
```python
docs = retriever.get_relevant_documents(prompt_text)  # k=100 par défaut
```

### Personnaliser le prompt

Dans `app.py`, lignes 89-107, modifiez le template selon vos besoins.

## 📊 Base de connaissances

### Format JSONL

Chaque ligne du fichier `free_mobile_rag_qas_full.jsonl` contient :
```json
{"question": "Comment...", "answer": "Pour..."}
```

### Ajouter des documents

1. Ajoutez vos Q&A au fichier JSONL
2. Supprimez le dossier `./database/free_mobile/`
3. Relancez l'application (la base sera recréée automatiquement)

### Support PDF

Le fichier `vector.py` inclut également `create_vector_store_from_pdf()` pour ingérer des PDFs (non utilisé actuellement dans `app.py`).

## 🐛 Dépannage

### Problèmes courants résolus

L'application a été testée et les problèmes suivants ont été corrigés :
- ✅ Incompatibilités de versions (numpy, rpds-py, grpcio, pyarrow)
- ✅ Modules manquants (langchain-groq, pandas, zstandard)
- ✅ Erreurs d'import ChromaDB

### Erreur Ollama
```
Si vous obtenez une erreur de connexion à Ollama :
1. Vérifiez qu'Ollama est lancé : ollama list
2. Vérifiez que le modèle est installé : ollama pull mxbai-embed-large
3. Relancez : ollama serve
```

### Erreur Groq API
```
Si vous obtenez une erreur d'authentification :
1. Vérifiez votre fichier .env
2. Assurez-vous que GROQ_API_KEY est défini
3. Vérifiez que votre clé est valide sur console.groq.com
```

### Base vectorielle corrompue
```powershell
# Supprimez et recréez la base
Remove-Item -Recurse -Force .\database\free_mobile\
# Relancez l'application
```

### Commandes de réparation rapide
```powershell
# Réinstaller les packages problématiques
.\venv\Scripts\python.exe -m pip install --force-reinstall numpy==1.26.4
.\venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall rpds-py grpcio protobuf
.\venv\Scripts\python.exe -m pip install --force-reinstall "pyarrow<22,>=7.0"
```

**💡 Astuce** : Consultez le fichier `COMMANDES.md` pour toutes les commandes utiles !

## 🎨 Interface utilisateur

L'application dispose d'une interface moderne et lisible avec :
- **Fond doux** : Couleurs beige/crème apaisantes pour les yeux
- **Messages différenciés** : Bleu doux pour l'utilisateur, vert clair pour l'assistant
- **Sidebar élégante** : Fond bleu avec texte blanc, conversations bien organisées
- **Boutons stylisés** : Design moderne avec effets de survol
- **Responsive** : S'adapte à toutes les tailles d'écran

## 🔐 Sécurité

- ✅ `.gitignore` configuré pour protéger `.env`
- ✅ Clé API Groq sécurisée dans `.env`
- ⚠️ Ne commitez **jamais** votre fichier `.env`
- ⚠️ Ne partagez jamais vos clés API

## 📚 Fichiers de référence

- **README.md** : Documentation complète (ce fichier)
- **COMMANDES.md** : Liste de toutes les commandes PowerShell utiles
- **Lancer_Application.bat** : Script de lancement rapide

## 🚀 Démarrage rapide

1. Double-cliquez sur le raccourci Bureau **"Assistant Free Mobile"**
2. Attendez l'ouverture du navigateur
3. Commencez à poser vos questions !

## 📝 Licence

Projet interne Free Mobile

## 👥 Contribution

Pour contribuer :
1. Forkez le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Commitez vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Pushez vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

## 📞 Support

Pour toute question ou problème, contactez l'équipe technique.

---

**Développé avec ❤️ pour Free Mobile**
