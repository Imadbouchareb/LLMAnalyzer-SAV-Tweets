# 🔐 Checklist de Sécurité - AVANT de pusher sur GitHub

## ⚠️ À LIRE ABSOLUMENT par TOUS

Avant de faire **TOUT commit ou push**, vérifiez cette checklist !

---

## 🛡️ Règles de Sécurité Critiques

### ❌ NE JAMAIS PUSHER :

1. **Fichier .env** (contient les clés API réelles)
   - ✅ Utilisez `.env.example` à la place
   - ✅ Le `.gitignore` doit exclure `.env`

2. **Clés API ou secrets**
   - ❌ `GROQ_API_KEY=gsk_xxxxxxxxxxxxx`
   - ✅ `GROQ_API_KEY=your_groq_api_key_here`

3. **Base de données avec données réelles**
   - ❌ `database/` (peut contenir des données clients)
   - ✅ Exclu dans `.gitignore`

4. **Fichiers volumineux**
   - ❌ `old/*.csv` (tweets avec données clients)
   - ✅ Exclu dans `.gitignore`

5. **Informations personnelles**
   - ❌ Emails, numéros de téléphone, adresses
   - ❌ Logs contenant des données sensibles

---

## ✅ Checklist AVANT chaque commit

### Pour WALID :

```powershell
# 1. Vérifier que .env n'est PAS ajouté
git status
# Si tu vois .env en vert → DANGER !

# 2. Vérifier .env.example
Get-Content .env.example
# Doit contenir : GROQ_API_KEY=your_groq_api_key_here
# NE DOIT PAS contenir : gsk_xxxxx

# 3. Vérifier .gitignore
Select-String -Path .gitignore -Pattern "^\.env$"
# Doit retourner : .env
```

**Fichiers à pusher (WALID)** :
- ✅ `.env.example` (template seulement)
- ✅ `.gitignore`
- ✅ `requirements.txt`
- ✅ `Lancer_Application.bat`
- ✅ `README.md`
- ✅ `COMMANDES.md`
- ❌ `.env` (JAMAIS !)
- ❌ `database/` (exclu automatiquement)
- ❌ `__pycache__/` (exclu automatiquement)

---

### Pour ASMA :

```powershell
# 1. Vérifier que database/ n'est PAS ajouté
git status
# database/ ne doit PAS apparaître en vert

# 2. Vérifier le contenu de app.py
Select-String -Path app.py -Pattern "gsk_|api.*key.*=.*['\"]gsk"
# NE DOIT RIEN retourner (pas de clés hardcodées)

# 3. Vérifier vector.py
Select-String -Path vector.py -Pattern "password|secret|key.*=.*['\"]"
# NE DOIT PAS contenir de credentials
```

**Fichiers à pusher (ASMA)** :
- ✅ `app.py` (sans clés hardcodées)
- ✅ `vector.py` (sans credentials)
- ✅ `free_mobile_rag_qas_full.jsonl` (données publiques OK)
- ❌ `database/` (exclu automatiquement)
- ❌ `.env` (JAMAIS !)
- ❌ `__pycache__/` (exclu automatiquement)

---

### Pour IMAD :

```powershell
# 1. Vérifier le dossier old/
Get-ChildItem old\ -Recurse | Select-Object Name, Length | Where-Object {$_.Length -gt 10MB}
# Si fichiers > 10MB → les exclure

# 2. Vérifier qu'aucun CSV sensible n'est inclus
git status
Select-String -Path .gitignore -Pattern "\.csv$"

# 3. Vérifier INTEGRATION_ECOSYSTEME.md
Select-String -Path INTEGRATION_ECOSYSTEME.md -Pattern "gsk_|password|secret.*=|api.*key.*=.*gsk"
# NE DOIT PAS contenir de secrets
```

**Fichiers à pusher (IMAD)** :
- ✅ `INTEGRATION_ECOSYSTEME.md` (documentation uniquement)
- ✅ `old/` (fichiers Python legacy OK)
- ⚠️ `old/images/` (si < 5MB)
- ❌ `old/database/` (exclu automatiquement)
- ❌ `old/*.csv` (exclu automatiquement si gros fichiers)
- ❌ `.env` (JAMAIS !)

---

## 🔍 Commande de Vérification Globale

**À exécuter AVANT tout push** :

```powershell
cd "c:\Users\hallo\OneDrive\Bureau\IA Free Mobile\chatboot_app"

# Script de vérification automatique
Write-Host "🔐 AUDIT DE SÉCURITÉ" -ForegroundColor Cyan
Write-Host ""

# 1. Vérifier .env n'est pas staged
$envStaged = git diff --cached --name-only | Select-String "^\.env$"
if ($envStaged) {
    Write-Host "❌ DANGER : .env est staged ! Exécute : git reset HEAD .env" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ .env n'est pas staged" -ForegroundColor Green
}

# 2. Vérifier .gitignore contient .env
if (Select-String -Path .gitignore -Pattern "^\.env$" -Quiet) {
    Write-Host "✅ .gitignore exclut .env" -ForegroundColor Green
} else {
    Write-Host "❌ DANGER : .gitignore n'exclut pas .env !" -ForegroundColor Red
    exit 1
}

# 3. Rechercher des clés API dans les fichiers staged
$stagedFiles = git diff --cached --name-only
foreach ($file in $stagedFiles) {
    if ($file -match "\.(py|md|txt|bat)$") {
        $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
        if ($content -match "gsk_[a-zA-Z0-9]{32,}") {
            Write-Host "❌ DANGER : Clé API trouvée dans $file !" -ForegroundColor Red
            exit 1
        }
    }
}
Write-Host "✅ Aucune clé API détectée dans les fichiers staged" -ForegroundColor Green

# 4. Vérifier la taille des fichiers
$largeFiles = git diff --cached --name-only | ForEach-Object { 
    if (Test-Path $_) {
        $size = (Get-Item $_).Length / 1MB
        if ($size -gt 50) { $_ }
    }
}
if ($largeFiles) {
    Write-Host "⚠️  ATTENTION : Fichiers volumineux détectés (>50MB)" -ForegroundColor Yellow
    $largeFiles
} else {
    Write-Host "✅ Aucun fichier trop volumineux" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Audit de sécurité réussi ! Vous pouvez pusher." -ForegroundColor Cyan
```

---

## 🚨 En cas d'erreur

### Si vous avez committé .env par erreur :

```powershell
# AVANT de pusher sur GitHub
git reset HEAD .env
git commit --amend --no-edit

# SI DÉJÀ PUSHÉ (URGENT !)
# 1. Révoquer immédiatement la clé API sur groq.com
# 2. Générer une nouvelle clé
# 3. Supprimer l'historique Git :
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all
git push origin --force --all
```

### Si vous avez committé une clé API hardcodée :

```powershell
# Trouver le commit contenant la clé
git log -S "gsk_" --source --all

# Révoquer la clé immédiatement sur groq.com
# Puis supprimer le commit :
git rebase -i <commit_hash>~1
# Marquer le commit comme 'drop' ou 'edit'
```

---

## 📊 Récapitulatif des exclusions (.gitignore)

```
✅ EXCLU (ne sera JAMAIS pushé) :
├── .env                          # Clés API réelles
├── database/                     # Base de données ChromaDB
├── __pycache__/                  # Cache Python
├── venv/                         # Environnement virtuel
├── *.sqlite3                     # Fichiers base de données
├── old/database/                 # Ancienne base de données
├── old/*.csv                     # CSV potentiellement gros
└── .streamlit/secrets.toml       # Secrets Streamlit

✅ INCLUS (sera pushé) :
├── .env.example                  # Template de configuration
├── .gitignore                    # Configuration Git
├── requirements.txt              # Dépendances
├── app.py                        # Code application
├── vector.py                     # Code vector store
├── free_mobile_rag_qas_full.jsonl # Base de connaissances
├── README.md                     # Documentation
├── COMMANDES.md                  # Référence commandes
├── INTEGRATION_ECOSYSTEME.md     # Guide intégration
└── old/*.py                      # Scripts legacy (OK si petits)
```

---

## 🎯 Points de Contrôle Finaux

Avant de pusher, **chaque personne** doit vérifier :

### ✅ WALID
- [ ] `.env.example` ne contient QUE des templates
- [ ] `.gitignore` exclut bien `.env`
- [ ] Aucune clé API dans `README.md` ou `COMMANDES.md`
- [ ] `Lancer_Application.bat` ne contient pas de secrets

### ✅ ASMA
- [ ] `app.py` utilise `load_dotenv()` et non des clés hardcodées
- [ ] `vector.py` n'a pas de credentials
- [ ] `database/` n'apparaît PAS dans `git status`
- [ ] `free_mobile_rag_qas_full.jsonl` contient des données publiques

### ✅ IMAD
- [ ] `INTEGRATION_ECOSYSTEME.md` n'a pas de clés API
- [ ] `old/` ne contient pas de fichiers > 50MB
- [ ] Aucun fichier CSV sensible inclus
- [ ] Pull Request vérifiée avant merge

---

## 🏆 Bonnes Pratiques

1. **Toujours utiliser `git status` avant `git add`**
2. **Toujours utiliser `git diff` avant `git commit`**
3. **Exécuter l'audit de sécurité avant `git push`**
4. **Ne JAMAIS committer sous pression**
5. **En cas de doute, demander à l'équipe**

---

## 📞 Contact en cas d'urgence

Si vous avez pushé un secret par erreur :
1. **STOP** : Ne pas paniquer
2. **Révoquer** : Changer immédiatement la clé API
3. **Prévenir** : Alerter Imad (responsable du repo)
4. **Nettoyer** : Utiliser `git filter-branch` ou contacter GitHub Support

---

**📅 Date** : 23 novembre 2025  
**🔒 Criticité** : HAUTE - À lire par TOUS avant TOUT push  
**✅ Status** : Checklist de sécurité complète
