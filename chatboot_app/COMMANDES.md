# 📋 Commandes Utiles - Assistant Free Mobile

## 🚀 Lancement de l'application

### Démarrer l'application
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

### Démarrer avec le navigateur par défaut
```powershell
cd "c:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app"
.\venv\Scripts\streamlit.exe run app.py
```

### Arrêter l'application
Appuyez sur `Ctrl + C` dans le terminal

---

## 🔧 Gestion de l'environnement virtuel

### Activer l'environnement virtuel
```powershell
.\venv\Scripts\Activate.ps1
```

### Désactiver l'environnement virtuel
```powershell
deactivate
```

### Vérifier la version de Python
```powershell
.\venv\Scripts\python.exe --version
```

---

## 📦 Gestion des dépendances

### Installer toutes les dépendances
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Installer un package spécifique
```powershell
.\venv\Scripts\python.exe -m pip install nom_du_package
```

### Mettre à jour pip
```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

### Lister les packages installés
```powershell
.\venv\Scripts\python.exe -m pip list
```

### Générer un nouveau requirements.txt
```powershell
.\venv\Scripts\python.exe -m pip freeze > requirements.txt
```

---

## 🤖 Ollama (Embeddings)

### Vérifier les modèles installés
```powershell
ollama list
```

### Télécharger le modèle d'embeddings
```powershell
ollama pull mxbai-embed-large
```

### Lancer Ollama
```powershell
ollama serve
```

### Tester Ollama
```powershell
ollama run mxbai-embed-large
```

---

## 🔍 Débuggage

### Tester l'import des modules Python
```powershell
.\venv\Scripts\python.exe -c "import streamlit; print('Streamlit OK')"
.\venv\Scripts\python.exe -c "from langchain_groq import ChatGroq; print('LangChain OK')"
.\venv\Scripts\python.exe -c "import chromadb; print('ChromaDB OK')"
```

### Vérifier la configuration Python
```powershell
.\venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

### Exécuter l'application en mode debug
```powershell
.\venv\Scripts\streamlit.exe run app.py --logger.level=debug
```

---

## 🗄️ Gestion de la base ChromaDB

### Supprimer la base de données (reset)
```powershell
Remove-Item -Recurse -Force .\database\free_mobile\
```

### Vérifier la taille de la base
```powershell
Get-ChildItem .\database\free_mobile\ -Recurse | Measure-Object -Property Length -Sum
```

---

## 🔐 Configuration

### Vérifier le fichier .env
```powershell
Get-Content .env
```

### Éditer le fichier .env
```powershell
notepad .env
```

---

## 📊 Informations système

### Vérifier l'espace disque
```powershell
Get-PSDrive C
```

### Voir les processus Python en cours
```powershell
Get-Process python
```

### Tuer un processus Python bloqué
```powershell
Stop-Process -Name python -Force
```

---

## 🌐 Accès à l'application

### URLs par défaut
- **Local** : http://localhost:8501
- **Local (alternatif)** : http://localhost:8502
- **Local (alternatif 2)** : http://localhost:8503
- **Réseau** : http://192.168.1.179:8501

### Ouvrir dans le navigateur
```powershell
Start-Process "http://localhost:8501"
```

---

## 🛠️ Maintenance

### Nettoyer le cache pip
```powershell
.\venv\Scripts\python.exe -m pip cache purge
```

### Nettoyer les fichiers __pycache__
```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

### Recréer l'environnement virtuel (si corrompu)
```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📝 Git (si initialisé)

### Initialiser un dépôt Git
```powershell
git init
git add .gitignore
git commit -m "Initial commit"
```

### Vérifier le statut
```powershell
git status
```

### Commit des changements
```powershell
git add .
git commit -m "Description des changements"
```

---

## 🔥 Commandes de secours

### Réinstaller les packages critiques
```powershell
.\venv\Scripts\python.exe -m pip install --force-reinstall numpy==1.26.4
.\venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall rpds-py
.\venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall grpcio protobuf
.\venv\Scripts\python.exe -m pip install --force-reinstall "pyarrow<22,>=7.0"
```

### Forcer la réinstallation de LangChain
```powershell
.\venv\Scripts\python.exe -m pip uninstall -y langchain langchain-core langchain-ollama langchain-groq langchain-chroma
.\venv\Scripts\python.exe -m pip install langchain langchain-ollama langchain-groq langchain-chroma
```

---

## 💡 Astuces

### Lancer rapidement (commande complète)
```powershell
cd "c:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app" ; .\venv\Scripts\streamlit.exe run app.py
```

### Ouvrir VS Code dans le projet
```powershell
code .
```

### Ouvrir l'explorateur Windows
```powershell
explorer .
```

---

**📌 Conseil** : Gardez ce fichier ouvert dans un onglet pour accéder rapidement aux commandes !
