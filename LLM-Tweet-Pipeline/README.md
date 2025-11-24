# 🚀 Pipeline d'Analyse SAV Free Mobile - LLM & RAG

> Application professionnelle de traitement automatique des tweets clients via LLM (Mistral AI & Ollama) avec enrichissement RAG pour le support client.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table des matières
- [Présentation](#-présentation)
- [Installation](#-installation)
- [Démarrage rapide](#-démarrage-rapide)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Scripts disponibles](#-scripts-disponibles)
- [Technologies](#-technologies)

---

## 🎯 Présentation

Cette application analyse automatiquement les messages clients de Free Mobile pour :
- 📊 **Classifier** les demandes (thème, sentiment, urgence, gravité)
- 🤖 **Générer** des réponses personnalisées via LLM
- 🔍 **Enrichir** le contexte avec RAG (base de connaissances)
- 📈 **Exporter** les résultats au format CSV standardisé

### ✨ Fonctionnalités principales
- Interface Streamlit moderne et intuitive
- Support dual mode : **Mistral API** (cloud) ou **Ollama** (local)
- Prétraitement automatique des tweets (nettoyage, déduplication)
- Enrichissement sémantique avec embeddings BERT
- Classification multi-tâches avec LLM
- Cache intelligent pour optimiser les performances

### 📊 Cas d'usage
- Détection automatique des problèmes urgents
- Routing intelligent vers les équipes SAV
- Génération de réponses pré-rédigées
- Analyse de sentiment et KPIs clients

---

## 💾 Installation

### Prérequis
- **Python 3.10+**
- **Compte Mistral AI** avec clé API ([console.mistral.ai](https://console.mistral.ai))
- **Git** (optionnel)

### Étapes d'installation

**1. Cloner le projet**
```powershell
git clone https://github.com/Imadbouchareb/LLMAnalyzer-SAV-Tweets.git
cd LLMAnalyzer-SAV-Tweets/LLM-Tweet-Pipeline
```

**2. Créer un environnement virtuel**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**3. Installer les dépendances**
```powershell
pip install -r requirements.txt
```

**4. Configurer la clé API**

Créer le fichier `.streamlit/secrets.toml` :
```toml
MISTRAL_API_KEY = "votre_clé_mistral_ici"
```

**5. (Optionnel) Installer Ollama pour mode local**
```powershell
# Télécharger depuis https://ollama.ai
# Puis installer un modèle
ollama pull mistral
```

---

## 🚀 Démarrage rapide

### Méthode 1 : Lanceur automatique (Windows)

Double-cliquez sur **`lancer_app.bat`**

✅ Active automatiquement l'environnement virtuel  
✅ Lance l'application Streamlit  
✅ Ouvre votre navigateur sur l'interface

### Méthode 2 : Ligne de commande

```powershell
# Activer l'environnement
.venv\Scripts\activate

# Lancer l'application
streamlit run app.py
```

L'interface s'ouvre automatiquement sur `http://localhost:8501`

---

## 🏗️ Architecture

```
📦 Pipeline de traitement
│
├─ 📤 1. Import & Filtrage
│   └─ process_tweets_pipeline.py
│      • Extraction tweets clients
│      • Suppression réponses Free
│      • Détection langue
│
├─ 🔍 2. Enrichissement RAG
│   └─ add_rag_context.py
│      • Recherche sémantique
│      • Embeddings BERT
│      • Injection contexte
│
├─ 🤖 3. Analyse LLM (2 options)
│   ├─ Option A: Mistral API
│   │   └─ llm_batch_multitask_pool_mistral.py
│   │      • Cloud, rapide
│   │      • Nécessite clé API
│   │
│   └─ Option B: Ollama Local
│       └─ llm_full_ollama_pipeline.py
│          • Local, gratuit
│          • Sans clé API
│
└─ 📊 4. Export CSV
    └─ Format standardisé 16 colonnes
```

---

## 🔐 Configuration

### Configuration de la clé API Mistral

**Méthode 1 : Streamlit Secrets (RECOMMANDÉ)**

Créer `.streamlit/secrets.toml` :
```toml
MISTRAL_API_KEY = "votre_clé_mistral_ici"
```

**Méthode 2 : Variable d'environnement**
```powershell
$env:MISTRAL_API_KEY = "votre_clé_mistral_ici"
```

**Méthode 3 : Fichier .env**
```bash
MISTRAL_API_KEY=votre_clé_mistral_ici
```

### Sécurité

⚠️ **Important** : Ne jamais committer les fichiers contenant des clés API

Fichiers protégés dans `.gitignore` :
- `.streamlit/secrets.toml`
- `.env`
- `*.sqlite` (caches)
- `*.pt` (embeddings)

---

## 📦 Scripts disponibles

| Script | Description | Usage principal |
|--------|-------------|-----------------|
| `app.py` | Interface Streamlit | Interface utilisateur complète |
| `process_tweets_pipeline.py` | Prétraitement | Extraction et nettoyage des tweets |
| `add_rag_context.py` | Enrichissement RAG | Injection de contexte sémantique |
| `llm_batch_multitask_pool_mistral.py` | Pipeline Mistral | Classification et génération (cloud) |
| `llm_full_ollama_pipeline.py` | Pipeline Ollama | Alternative locale sans API |
| `lancer_app.bat` | Lanceur Windows | Démarrage automatique |

### 📂 Structure des fichiers

```
LLM-Tweet-Pipeline/
├── 📄 app.py                              # Interface Streamlit
├── 📄 process_tweets_pipeline.py          # Prétraitement
├── 📄 add_rag_context.py                  # RAG
├── 📄 llm_batch_multitask_pool_mistral.py # Pipeline Mistral
├── 📄 llm_full_ollama_pipeline.py         # Pipeline Ollama
├── 📄 lancer_app.bat                      # Lanceur Windows
├── 📄 requirements.txt                    # Dépendances
├── 📊 kb_replies_rich_expanded.csv        # Base de connaissances
├── 🔧 .streamlit/secrets.toml             # Configuration API
├── 💾 llm_cache.sqlite                    # Cache requêtes
└── 📁 old/                                # Fichiers archivés
```

---

## 🛠️ Technologies

### Backend
- **Python 3.10+** - Langage principal
- **Pandas** - Manipulation de données
- **LangChain** - Orchestration LLM
- **Mistral AI** - Modèle de langage (API)
- **Sentence-Transformers** - Embeddings sémantiques
- **PyTorch** - Calculs tensoriels

### Frontend
- **Streamlit** - Interface web interactive

### Infrastructure
- **SQLite** - Cache des requêtes LLM
- **Git** - Versioning
- **python-dotenv** - Gestion variables d'environnement

---

## 📊 Format de sortie

Le pipeline génère un CSV avec **16 colonnes standardisées** :

| Colonne | Type | Description |
|---------|------|-------------|
| `tweet_id` | str | Identifiant unique |
| `created_at_dt` | datetime | Date de publication |
| `text_display` | str | Texte du tweet |
| `rag_context` | str | Contexte RAG injecté |
| `themes_list` | json | Liste des thèmes détectés |
| `primary_label` | str | Thème principal |
| `sentiment_label` | str | Sentiment (positif/négatif/neutre) |
| `llm_urgency_0_3` | int | Urgence (0=faible, 3=critique) |
| `llm_severity_0_3` | int | Gravité (0=mineure, 3=majeure) |
| `status` | str | État (open/closed) |
| `summary_1l` | str | Résumé en une ligne |
| `author` | str | Auteur du tweet |
| `assigned_to` | str | Équipe responsable |
| `llm_summary` | str | Résumé détaillé |
| `llm_reply_suggestion` | str | Réponse suggérée |
| `routing_team` | str | Équipe de routage |

---

## 🐛 Dépannage

### Problème : "Clé API non configurée"
**Solution :** Vérifier que `.streamlit/secrets.toml` existe et contient `MISTRAL_API_KEY`

### Problème : "ModuleNotFoundError"
**Solution :** Réinstaller les dépendances
```powershell
pip install -r requirements.txt
```

### Problème : Ollama ne répond pas
**Solution :** 
1. Vérifier qu'Ollama est installé : `ollama --version`
2. Vérifier qu'un modèle est téléchargé : `ollama list`
3. Lancer le service : `ollama serve`

### Problème : Fichier CSV invalide
**Solution :** Vérifier que le CSV contient au moins les colonnes `id`, `created_at`, `full_text`

---

## 📞 Support

Pour toute question ou problème :
- 📖 Consultez cette documentation
- 🐛 Ouvrez une issue sur [GitHub](https://github.com/Imadbouchareb/LLMAnalyzer-SAV-Tweets/issues)
- 📧 Contactez l'équipe de développement

---

## 📄 Licence

Ce projet est développé pour le traitement et l'analyse automatique des demandes SAV Free Mobile.

---

<div align="center">

**🤖 Pipeline d'Analyse SAV Free Mobile - LLM & RAG**

*Application de traitement automatique avec Mistral AI & Ollama*

**Version 2.0** • Novembre 2025

[Documentation](README.md) • [Issues](https://github.com/Imadbouchareb/LLMAnalyzer-SAV-Tweets/issues) • [GitHub](https://github.com/Imadbouchareb/LLMAnalyzer-SAV-Tweets)
