# 🎧 SAV Tweets - Support Client IA

Application de gestion et d'analyse des tweets du service après-vente, propulsée par l'intelligence artificielle.

## 📋 Table des matières

- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Lancement de l'application](#lancement-de-lapplication)
- [Structure du projet](#structure-du-projet)
- [Sécurité](#sécurité)
- [Contribution](#contribution)
- [Support](#support)

## 🎯 Présentation

**SAV Tweets** est une plateforme complète de gestion des interactions client sur Twitter, intégrant :
- Analyse automatique des sentiments et priorités
- Tableau de bord Manager pour la supervision
- Interface Analyste pour l'analyse approfondie
- File d'attente Agent SAV pour le traitement des tickets

## ✨ Fonctionnalités

### 👤 Page Analyste
- Analyse détaillée des tweets avec visualisations (Altair)
- Filtres avancés (sentiment, urgence, thème, agent)
- Export des données (CSV, JSON)
- Statistiques et tendances

### 👔 Page Manager
- Vue d'ensemble des KPI (tickets ouverts, temps de réponse, satisfaction)
- Suivi de la charge de travail par agent
- Graphiques interactifs et tableaux de bord
- Export et rapports personnalisés

### 🎧 Page Agent SAV
- File d'attente intelligente avec priorisation automatique
- Actions rapides (Clore, Réaffecter, Répondre)
- Historique des modifications persisté
- Interface optimisée pour le traitement rapide

## 🔧 Prérequis

- **Python** : Version 3.8 ou supérieure
- **Système d'exploitation** : Windows, macOS ou Linux
- **Mémoire RAM** : 4 Go minimum (8 Go recommandé)
- **Espace disque** : 500 Mo minimum

## 📦 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-organisation/sav_app.git
cd sav_app
```

### 2. Créer un environnement virtuel

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration (Optionnel)

Créez un fichier `.env` à la racine si vous avez des configurations spécifiques :

```env
# Exemple de configuration (NE PAS COMMITER CE FICHIER)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

## 🚀 Lancement de l'application

### Méthode 1 : Scripts de lancement (Recommandé sur Windows)

**Windows - Fichier Batch :**
```bash
# Double-cliquer sur :
lancer_sav_app.bat
```

**Windows - PowerShell :**
```bash
# Clic droit > "Exécuter avec PowerShell" sur :
lancer_sav_app.ps1
```

### Méthode 2 : Ligne de commande

```bash
# Activer l'environnement virtuel (si pas déjà fait)
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # macOS/Linux

# Lancer l'application
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse : **http://localhost:8501**

### Arrêt de l'application

- Fermez la fenêtre du terminal/PowerShell
- Ou appuyez sur `Ctrl+C` dans le terminal

## 📁 Structure du projet

```
sav_app/
├── app.py                      # Landing page principale
├── requirements.txt            # Dépendances Python
├── .gitignore                  # Fichiers exclus du versioning
├── lancer_sav_app.bat         # Lanceur Windows (Batch)
├── lancer_sav_app.ps1         # Lanceur Windows (PowerShell)
├── README.md                   # Documentation (ce fichier)
│
├── pages/                      # Pages Streamlit multi-pages
│   ├── 0_Accueil.py           # Menu principal
│   ├── 1_Analyste.py          # Interface Analyste
│   ├── 2_Manager.py           # Tableau de bord Manager
│   └── 3_Agent_SAV.py         # File d'attente Agent
│
├── lib/                        # Bibliothèques partagées
│   ├── __init__.py            # Initialisation du module
│   ├── data.py                # Gestion des données (load, persist)
│   ├── state.py               # Gestion de l'état session
│   ├── ui.py                  # Helpers UI et CSS
│   └── aggrid_utils.py        # Utilitaires AgGrid
│
├── data/                       # Données persistées
│   ├── sav_edits.csv          # Historique des modifications
│   └── last_dataset.txt       # Dernier dataset chargé
│
└── uploads/                    # Fichiers uploadés (temporaires)
```

## 🔒 Sécurité

### Bonnes pratiques

#### ✅ À FAIRE

- ✅ Toujours utiliser un environnement virtuel (`venv/`)
- ✅ Garder les dépendances à jour : `pip install --upgrade -r requirements.txt`
- ✅ Ne jamais commiter de données sensibles (clés API, mots de passe)
- ✅ Vérifier le fichier `.gitignore` avant chaque commit
- ✅ Utiliser des variables d'environnement pour les configurations sensibles

#### ❌ À NE PAS FAIRE

- ❌ Ne JAMAIS commiter les fichiers suivants :
  - `data/uploads/*` (fichiers temporaires uploadés)
  - `.env` ou `.env.*` (configurations locales)
  - Fichiers contenant des tokens/clés API
  - `__pycache__/`, `*.pyc` (fichiers Python compilés)
  - `venv/` ou `.venv/` (environnement virtuel)

### Fichiers protégés par `.gitignore`

Le fichier `.gitignore` exclut automatiquement :
- Environnements virtuels (`venv/`, `.venv/`)
- Fichiers de cache Python (`__pycache__/`, `*.pyc`)
- Données sensibles (`.env`, `secrets.*`, `credentials.*`)
- Fichiers temporaires (`uploads/*.csv`, `*.tmp`, `*.log`)
- Fichiers de configuration IDE (`.vscode/`, `.idea/`)

### Gestion des données

- **Données persistées** : Seul `data/sav_edits.csv` est versionné (historique des modifications)
- **Uploads temporaires** : Les fichiers dans `uploads/` sont automatiquement exclus
- **Datasets volumineux** : À stocker en dehors du dépôt Git ou utiliser Git LFS

### Recommandations supplémentaires

1. **Contrôle d'accès** : Limitez l'accès au dépôt aux membres de l'équipe uniquement
2. **Revue de code** : Effectuez des pull requests pour toute modification importante
3. **Mots de passe** : Utilisez un gestionnaire de secrets (GitHub Secrets, Azure Key Vault, etc.)
4. **Logs** : Ne loggez jamais d'informations sensibles (tokens, emails, etc.)
5. **HTTPS** : Utilisez toujours HTTPS pour les communications réseau

## 🛠️ Technologies utilisées

- **Streamlit** : Framework web Python pour applications de données
- **Pandas** : Manipulation et analyse de données
- **Altair** : Visualisations interactives déclaratives
- **AgGrid** : Tableaux de données avancés
- **Python 3.8+** : Langage principal

## 👥 Contribution

### Équipe de développement

- **Asma** : Page Analyste, Accueil, Landing page
- **Walid** : Page Manager, Bibliothèques partagées (`lib/`)
- **Imad** : Page Agent SAV, Gestion des données

### Workflow Git

```bash
# 1. Créer une branche pour votre fonctionnalité
git checkout -b feat/ma-nouvelle-fonctionnalite

# 2. Faire vos modifications et commits
git add <fichiers-modifiés>
git commit -m "Feat: Description claire de la fonctionnalité"

# 3. Pusher votre branche
git push origin feat/ma-nouvelle-fonctionnalite

# 4. Créer une Pull Request sur GitHub
```

### Conventions de commit

- `Feat:` Nouvelle fonctionnalité
- `Fix:` Correction de bug
- `Refactor:` Refactorisation du code
- `Docs:` Mise à jour de la documentation
- `Style:` Changements de style (CSS, formatting)
- `Test:` Ajout ou modification de tests

## 📞 Support

### Problèmes courants

**L'application ne démarre pas :**
```bash
# Vérifier que Python est installé
python --version

# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

**Erreur "Module not found" :**
```bash
# Vérifier que l'environnement virtuel est activé
# Réinstaller les dépendances
pip install -r requirements.txt
```

**Port déjà utilisé :**
```bash
# Lancer sur un port différent
streamlit run app.py --server.port 8502
```

### Contact

Pour toute question ou problème, veuillez :
1. Ouvrir une issue sur GitHub
2. Contacter l'équipe de développement
3. Consulter la documentation Streamlit : https://docs.streamlit.io

## 📄 Licence

Ce projet est destiné à un usage interne pour Free Mobile.

---

**Développé avec ❤️ par l'équipe SAV IA**
