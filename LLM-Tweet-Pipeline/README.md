# 📦 Dossier Old - Fichiers archivés

Ce dossier contient les anciens fichiers et scripts qui ne sont plus utilisés par l'application principale.

## 🗂️ Contenu

### Scripts Python obsolètes
- `llm_batch_local_bertsim.py` - Ancienne version avec BERTScore local
- `llm_batch_mistral_api.py` - Version simplifiée remplacée par `llm_batch_multitask_pool_mistral.py`
- `llm_rag_ollama.py` - Version RAG standalone (intégrée dans pipeline principal)
- `test_ollama_json.py` - Tests unitaires Ollama

### Fichiers de cache obsolètes
- `llm_cache_bertsim.sqlite` - Cache BERTScore
- `llm_cache_ollama.sqlite` - Ancien cache Ollama
- `ma_base_cache.sqlite` - Cache de test

### Données de test
- `free tweet export.csv` - Données brutes initiales
- `free tweet export - Copie.csv` - Copie de sauvegarde
- `free tweet export - Copie_clean_llm.csv` - Anciens résultats nettoyés
- `tweets_scored_llm.csv` - Anciens résultats d'analyse

### Fichiers temporaires
- `tmp_test.txt` - Fichier de test temporaire
- `Executable.txt` - Anciennes instructions d'exécution

## ⚠️ Important

Ces fichiers sont conservés pour référence historique mais ne sont **plus utilisés** par l'application.

Pour l'application actuelle, référez-vous au [README principal](../README.md).

## 🗑️ Suppression

Ces fichiers peuvent être supprimés en toute sécurité si vous avez besoin de libérer de l'espace disque :

```powershell
Remove-Item -Path "C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\bloc2\old" -Recurse -Force
```
