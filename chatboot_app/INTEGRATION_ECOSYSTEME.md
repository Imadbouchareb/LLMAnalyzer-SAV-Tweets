# 🔗 Intégration avec l'Écosystème Free Mobile SAV

## 📋 Vue d'ensemble

Ce document explique comment **l'Assistant Free Mobile RAG** (application actuelle) s'intègre dans l'écosystème complet du Service Après-Vente Free Mobile, composé de **3 applications complémentaires** :

```
┌─────────────────────────────────────────────────────────────────┐
│                  ÉCOSYSTÈME SAV FREE MOBILE                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│  APPLICATION  │    │  APPLICATION   │    │ APPLICATION  │
│   ACTUELLE    │    │     BLOC2      │    │   SAV_APP    │
│               │    │                │    │              │
│  Assistant    │    │   Pipeline     │    │  Plateforme  │
│     RAG       │    │  Traitement    │    │    Gestion   │
│  Chatbot KB   │    │   LLM Tweets   │    │  Cockpit SAV │
└───────────────┘    └────────────────┘    └──────────────┘
```

---

## 🎯 Rôle de chaque application

### 1️⃣ **Application Actuelle : Assistant Free Mobile RAG**
**Localisation** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app`

#### 🔍 Fonction principale
- **Chatbot de connaissances** pour le support interne
- Répond aux questions des agents SAV sur les procédures Free Mobile
- Base de connaissances : 39 Q&A sur les services Free Mobile

#### ⚙️ Technologies
- **LangChain** : orchestration RAG
- **ChatGroq** (Llama 3.3 70B) : génération de réponses *(Phase POC/Test)*
- **ChromaDB** : stockage vectoriel des connaissances
- **Ollama** (mxbai-embed-large) : embeddings
- **Streamlit** : interface web

> **📌 Note Stratégique** : L'application utilise actuellement **Groq API en phase de test** pour valider l'architecture RAG. En production, une migration vers **Mistral API** (français, hébergé en Europe) est prévue pour garantir la souveraineté des données et la conformité RGPD.

#### 💡 Cas d'usage
- Agent SAV cherche une réponse rapide sur la portabilité
- Formation des nouveaux agents
- Consultation rapide des procédures sans chercher dans les documents

---

### 2️⃣ **Bloc2 : Pipeline de Traitement LLM des Tweets**
**Localisation** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\bloc2`

#### 🔍 Fonction principale
- **Traitement automatique des tweets clients** de Free Mobile
- Analyse et classification intelligente via LLM (Mistral AI + Ollama)
- Enrichissement contextuel via RAG avec la base de connaissances

#### ⚙️ Technologies
- **Mistral API** (Cloud) : `mistral-small-latest` pour production
- **Ollama** (Local) : `mistral:7b` pour développement
- **RAG enrichissement** : avec base KB similaire à l'Assistant actuel
- **Streamlit** : interface de pipeline
- **Power BI** : visualisation avancée des résultats

#### 📊 Capacités d'analyse
Pour chaque tweet client, le système extrait :
- **Thème** : facturation, réseau, portabilité, offres, technique, SAV
- **Sentiment** : positif, négatif, neutre (avec score de confiance)
- **Urgence** : échelle de 0 à 3
- **Gravité** : échelle de 0 à 3
- **Réponse suggérée** : générée automatiquement par le LLM avec contexte RAG

#### 💾 Fichiers de sortie
- `tweets_processed_mistral.csv` : résultats avec analyse LLM
- `tweets_with_rag.csv` : tweets enrichis avec contexte RAG
- Exports pour intégration dans SAV_APP

---

### 3️⃣ **SAV_APP : Plateforme de Gestion du Support Client**
**Localisation** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\sav_app`

#### 🔍 Fonction principale
- **Cockpit complet pour le SAV** : gestion des tickets, KPI, dashboards
- Interface multi-rôles (Analyste, Manager, Agent)
- Traitement temps réel des demandes clients

#### ⚙️ Modules
**📈 Interface Analyste**
- Visualisation des données de support
- Analyse des sentiments (graphiques interactifs)
- Filtres avancés (statut, période, thème, agent)
- Réseau thématique et tendances temporelles

**📊 Tableau de bord Manager**
- KPI du service client en temps réel
- Performances par agent/équipe
- Statistiques d'urgence et satisfaction
- Identification des heures/thèmes critiques

**🎧 File Agent SAV**
- Traitement optimisé des tickets
- Gestion du statut (Ouvert, En cours, Résolu, Fermé)
- Priorisation automatique des urgents
- **Suggestions de réponse IA** (enrichies par RAG)
- Édition temps réel via AgGrid

**📤 Upload de Données**
- Import CSV depuis BLOC2
- Intégration automatique des analyses LLM
- Suivi du dernier dataset chargé

---

## 🔄 Flux de Travail Intégré

### Scénario complet : Du Tweet à la Réponse Client

```
┌────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1 : Collecte et Traitement (BLOC2)                          │
└────────────────────────────────────────────────────────────────────┘
  Client tweete : "@free_mobile Mon réseau 4G ne marche plus depuis ce matin !"
                      │
                      ▼
  ┌─────────────────────────────────────┐
  │ BLOC2 : Pipeline LLM                │
  │                                     │
  │ 1. Classification automatique       │
  │    - Thème: "réseau"                │
  │    - Sentiment: "négatif" (-0.8)    │
  │    - Urgence: 2/3                   │
  │    - Gravité: 2/3                   │
  │                                     │
  │ 2. Enrichissement RAG               │
  │    → Recherche dans KB :            │
  │      "problème réseau 4G"           │
  │    → Contexte trouvé :              │
  │      "Vérifier APN, redémarrer..."  │
  │                                     │
  │ 3. Génération réponse IA            │
  │    "Bonjour, nous comprenons...     │
  │     Pouvez-vous vérifier les        │
  │     paramètres APN ?..."            │
  └─────────────────────────────────────┘
                      │
                      ▼ (Export CSV)
┌────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2 : Gestion et Traitement (SAV_APP)                         │
└────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ SAV_APP : Upload de données         │
  │ Import : tweets_processed_mistral.csv│
  └─────────────────────────────────────┘
                      │
                      ▼
  ┌─────────────────────────────────────┐
  │ Manager : Tableau de bord           │
  │ - Visualise pic de tickets "réseau" │
  │ - Alerte : +50% urgence niveau 2    │
  │ - Assigne priorité aux agents       │
  └─────────────────────────────────────┘
                      │
                      ▼
  ┌─────────────────────────────────────┐
  │ Agent SAV : File de tickets         │
  │                                     │
  │ 1. Voit le ticket priorisé          │
  │ 2. Lit la suggestion IA :           │
  │    "Bonjour, nous comprenons..."    │
  │ 3. Peut modifier/valider            │
  │ 4. Change statut : "En cours"       │
  │ 5. Répond au client                 │
  │ 6. Clôture : "Résolu"               │
  └─────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────────────────┐
│ ÉTAPE 3 : Support Agent (ASSISTANT RAG - App actuelle)            │
└────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ Assistant RAG : Consultation KB     │
  │                                     │
  │ Agent demande :                     │
  │ "Comment configurer les APN Free ?" │
  │                                     │
  │ Assistant répond :                  │
  │ "Les paramètres APN Free Mobile :   │
  │  - APN : free                       │
  │  - MCC : 208                        │
  │  - MNC : 15                         │
  │  Procédure : Réglages > Réseau..."  │
  └─────────────────────────────────────┘
```

---

## 🔗 Points d'Intégration Possibles

### 🎯 Intégration 1 : Base de Connaissances Partagée

**Objectif** : Unifier la base de connaissances entre les 3 applications

#### Architecture proposée
```
┌─────────────────────────────────────┐
│  Base de Connaissances Centralisée  │
│         (ChromaDB Partagée)         │
└─────────────────────────────────────┘
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
 ┌────────┐ ┌────┐ ┌────────┐
 │Assistant│ │BLOC2│ │SAV_APP│
 │   RAG  │ │ RAG│ │  RAG  │
 └────────┘ └────┘ └────────┘
```

#### Implémentation
1. **Créer un dossier partagé** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\shared_kb\`
2. **Centraliser ChromaDB** : 
   - Migrer `database/free_mobile/` vers `shared_kb/chroma_db/`
   - Pointer les 3 apps vers ce même emplacement
3. **Synchroniser les mises à jour** :
   - Toute modification KB se propage aux 3 apps
   - Versioning avec git pour traçabilité

#### Code à modifier dans `app.py` (App actuelle)
```python
# Avant
CHROMA_PATH = "database/free_mobile"

# Après
CHROMA_PATH = r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\shared_kb\chroma_db"
```

#### Code à modifier dans `bloc2/add_rag_context.py`
```python
# Avant
CHROMA_PATH = "./database/kb_free_mobile"

# Après
CHROMA_PATH = r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\shared_kb\chroma_db"
```

---

### 🎯 Intégration 2 : Pipeline Automatisé Bout-en-Bout

**Objectif** : Automatiser le flux Tweet → Traitement → Cockpit SAV

#### Architecture proposée
```
 Twitter API
      │
      ▼
┌──────────────┐
│   BLOC2      │  1. Récupère tweets
│              │  2. Analyse LLM
│  Pipeline    │  3. Enrichit RAG
│  Automatique │  4. Exporte CSV
└──────────────┘
      │
      ▼ (API ou File Watcher)
┌──────────────┐
│  SAV_APP     │  5. Import auto
│              │  6. Priorise tickets
│  Cockpit     │  7. Notifie agents
└──────────────┘
      │
      ▼ (Webhook ou Interface)
┌──────────────┐
│ Assistant RAG│  8. Support contextuel
│              │     pour réponses
└──────────────┘
```

#### Implémentation technique

**1. BLOC2 : Ajout d'export automatique**
```python
# Dans bloc2/process_tweets_pipeline.py
def export_to_sav_app(df_processed):
    """Exporte automatiquement vers SAV_APP"""
    output_path = r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\sav_app\data\incoming_tweets.csv"
    df_processed.to_csv(output_path, index=False)
    
    # Notification webhook (optionnel)
    requests.post(
        "http://localhost:8501/api/import",
        json={"file": "incoming_tweets.csv"}
    )
```

**2. SAV_APP : File watcher automatique**
```python
# Dans sav_app/lib/auto_import.py
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TweetImportHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith("incoming_tweets.csv"):
            import_tweets_to_dashboard(event.src_path)

observer = Observer()
observer.schedule(TweetImportHandler(), path="./data")
observer.start()
```

**3. Assistant RAG : API de consultation**
```python
# Dans app.py (App actuelle)
from fastapi import FastAPI
import uvicorn

# Ajouter endpoint API
api = FastAPI()

@api.post("/query_kb")
def query_knowledge_base(question: str):
    """Permet aux autres apps d'interroger la KB"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    return {"context": [doc.page_content for doc in docs]}

# Lancer FastAPI en parallèle de Streamlit
if __name__ == "__main__":
    uvicorn.run(api, host="localhost", port=8000)
```

---

### 🎯 Intégration 3 : Interface Unifiée (Future Version)

**Objectif** : Créer une application "Maître" regroupant les 3 fonctionnalités

#### Architecture proposée
```
┌──────────────────────────────────────────────────┐
│         FREE MOBILE SAV - PLATEFORME UNIFIÉE     │
├──────────────────────────────────────────────────┤
│  Navigation:  [Chatbot KB] [Pipeline] [Cockpit] │
└──────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Page 1      │  │  Page 2      │  │  Page 3      │
│  Chatbot RAG │  │  Pipeline    │  │  Cockpit SAV │
│  (app.py)    │  │  (bloc2)     │  │  (sav_app)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

#### Implémentation Streamlit

**Créer `unified_app.py` dans un nouveau dossier**
```python
import streamlit as st

st.set_page_config(
    page_title="Free Mobile SAV - Plateforme Complète",
    page_icon="📡",
    layout="wide"
)

# Navigation principale
page = st.sidebar.radio(
    "Navigation",
    ["🤖 Assistant Chatbot", "⚙️ Pipeline Tweets", "📊 Cockpit SAV"]
)

if page == "🤖 Assistant Chatbot":
    # Importer le code de l'app actuelle
    exec(open("../chatboot_app/app.py").read())
    
elif page == "⚙️ Pipeline Tweets":
    # Importer le code de bloc2
    exec(open("../bloc2/app.py").read())
    
elif page == "📊 Cockpit SAV":
    # Importer le code de sav_app
    exec(open("../sav_app/app.py").read())
```

---

## 📊 Tableau Comparatif des Applications

| Critère | **Assistant RAG** | **BLOC2** | **SAV_APP** |
|---------|------------------|----------|-------------|
| **Fonction** | Chatbot KB interne | Pipeline traitement tweets | Cockpit gestion SAV |
| **Utilisateurs** | Agents SAV | Responsables traitement données | Agents + Managers + Analystes |
| **Données entrée** | Questions textuelles | Tweets clients CSV | CSV traités par BLOC2 |
| **IA utilisée** | ChatGroq (Llama 3.3) | Mistral API + Ollama | Analyses de BLOC2 |
| **Base de données** | ChromaDB (39 Q&A) | ChromaDB KB + Cache LLM | CSV + base locale |
| **Interface** | Chat conversationnel | Pipeline multi-étapes | Dashboard multi-vues |
| **Temps réel** | ✅ Oui (streaming) | ❌ Non (batch) | ✅ Oui (édition live) |
| **Exportation** | ❌ Non | ✅ Oui (CSV) | ✅ Oui (CSV + Excel) |

---

## 🛠️ Plan d'Action pour l'Intégration

### Phase 1️⃣ : Base de Connaissances Partagée (Court terme - 1 semaine)
- [ ] Créer dossier `shared_kb/`
- [ ] Migrer ChromaDB vers emplacement partagé
- [ ] Modifier chemins dans les 3 apps
- [ ] Tester synchronisation
- [ ] Documenter procédure de mise à jour KB

### Phase 2️⃣ : Pipeline Automatisé (Moyen terme - 2 semaines)
- [ ] Implémenter export auto dans BLOC2
- [ ] Créer file watcher dans SAV_APP
- [ ] Ajouter API REST à l'Assistant RAG
- [ ] Tester flux bout-en-bout
- [ ] Configurer webhooks/notifications

### Phase 3️⃣ : Interface Unifiée (Long terme - 1 mois)
- [ ] Créer structure `unified_sav_platform/`
- [ ] Développer navigation multi-pages
- [ ] Intégrer les 3 apps existantes
- [ ] Harmoniser le design (thème commun)
- [ ] Tester performance et stabilité
- [ ] Déployer version production

---

## 📝 Checklist de Compatibilité

Avant d'intégrer les applications, vérifier :

### Environnements Python
- [ ] Versions Python compatibles (toutes en 3.10+ ?)
- [ ] Dépendances communes documentées
- [ ] Virtual environments séparés ou fusionnés ?

### Clés API et Configuration
- [ ] `.env` standardisé entre les 3 apps
- [ ] Clés API (GROQ, MISTRAL) partagées ou séparées ?
- [ ] Variables d'environnement documentées

### Chemins et Fichiers
- [ ] Chemins absolus vs relatifs harmonisés
- [ ] Structure de dossiers cohérente
- [ ] Nommage des fichiers standardisé

### Modèles et Données
- [ ] Ollama `mxbai-embed-large` installé partout
- [ ] Formats CSV compatibles
- [ ] Schémas de données documentés

---

## 🎓 Ressources Techniques

### Documentation des Applications
- **Assistant RAG** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app\README.md`
- **BLOC2** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\bloc2\README.md`
- **SAV_APP** : `C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\sav_app\README.md`

### Commandes Utiles
- **Assistant RAG** : `COMMANDES.md`
- **BLOC2** : `GUIDE_COMMANDES.md`
- **SAV_APP** : `COMMANDES.md`

### Guides Spécifiques
- **BLOC2** : `AMELIORATIONS_INTERFACE.md`, `ORGANISATION_DOSSIERS.md`, `SECRETS_CONFIG.md`
- **SAV_APP** : `DOCUMENTATION.md`, `GUIDE_SOUTENANCE.md`, `INTEGRATION_TWITTER.md`

---

## 🔐 Stratégie Multi-LLM et Souveraineté

### 📌 Pourquoi deux APIs différentes ?

#### **Groq API** (Assistant RAG actuel)
✅ **Spécialisé pour la conversation temps réel**
- **Llama 3.3 70B** via infrastructure Groq LPU (Language Processing Units)
- **Ultra-rapide** : latence < 1 seconde pour une UX fluide
- **Streaming natif** : affichage progressif des réponses
- **Idéal pour** : chatbot interactif avec agents SAV

#### **Mistral API** (BLOC2 - Pipeline tweets)
✅ **Spécialisé pour l'analyse structurée française**
- **Mistral Small** optimisé pour classification et extraction
- **Excellence sur le français** : modèle français, corpus français
- **Output JSON structuré** : parfait pour traitement batch
- **Idéal pour** : analyse automatique de sentiment/urgence

### 🎯 Justification Technique

| Critère | **Groq (Llama 3.3)** | **Mistral** |
|---------|---------------------|-------------|
| **Usage** | Chatbot conversationnel | Analyse batch structurée |
| **Latence** | < 1s (LPU) | ~15s par tweet |
| **Force** | Rapidité, streaming | Précision français, JSON |
| **Cas d'usage** | Réponses agents temps réel | Classification automatique |

### 🇫🇷 Approche Souveraineté

> **🔄 Phase actuelle : POC/Test avec Groq**
> 
> L'application utilise **Groq API en phase de validation** pour :
> - ✅ Rapidité de prototypage et test de concept
> - ✅ Compatibilité LangChain pour migration facile
> - ✅ Performance optimale pour démonstration

> **🚀 Phase production : Migration Mistral prévue**
>
> Pour le déploiement chez Free Mobile :
> - 🇫🇷 **Mistral API** : société française, hébergement Europe
> - 🔒 **Conformité RGPD** : protection des données européennes
> - 🏢 **Souveraineté numérique** : données restent en Europe
> - 🔄 **Alternative Ollama** : traitement 100% local si requis

### 💬 Phrase de Soutenance

*"Cette application est un **prototype de validation technique**. Nous utilisons Groq pour sa rapidité de développement et ses performances lors des tests. Pour le déploiement production chez Free Mobile, nous prévoyons une **migration vers Mistral API** (souveraineté française) avec possibilité de basculer sur **Ollama local** si exigences RGPD strictes."*

---

## 🚀 Conclusion

L'**écosystème Free Mobile SAV** est composé de 3 applications complémentaires :

1. **Assistant RAG** (App actuelle) : Chatbot de connaissances pour support interne
2. **BLOC2** : Pipeline LLM pour traitement automatique des tweets
3. **SAV_APP** : Cockpit de gestion complète du service client

### Synergies à exploiter
- ✅ **Base de connaissances commune** : éviter la redondance
- ✅ **Flux automatisé** : Tweet → Analyse → Cockpit → Réponse
- ✅ **Interface unifiée** : une seule application pour 3 fonctions
- ✅ **Standardisation** : harmoniser code, dépendances, design
- ✅ **Stratégie LLM évolutive** : POC → Production souveraine

### Prochaine étape recommandée
**Commencer par la Phase 1** (Base de connaissances partagée) car c'est :
- Le plus simple à implémenter
- Le plus utile immédiatement
- La fondation pour les autres intégrations

---

**📅 Dernière mise à jour** : 22 novembre 2025  
**👤 Auteur** : Assistant GitHub Copilot  
**📧 Contact** : Documentation générée pour le projet Free Mobile SAV
