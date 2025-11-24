# ==============================================
# 📘 add_rag_context.py
# Auteur : Walid Hallouche
# Objectif : Ajouter un champ "rag_context" à un CSV
# à partir de similarités BERT avec la KB
# ==============================================

import os

# 🔒 Forcer Transformers à ignorer complètement TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import argparse

# ------------------------------------------------------------------
# 🧠 Fonction utilitaire : charger ou encoder la KB
# ------------------------------------------------------------------
def load_or_encode_kb(kb_csv_path, model_name, cache_path):
    print("[INFO] Modèle pour RAG :", model_name)
    if os.path.exists(cache_path):
        print("[INFO] Cache KB trouvé :", cache_path)
        obj = torch.load(cache_path)

        # Cas 1 : cache dict (format Ollama)
        if isinstance(obj, dict) and "embeddings" in obj:
            kb_emb = obj["embeddings"]
            kb_df = pd.read_csv(kb_csv_path)
            print("[INFO] Cache KB (dict) chargé, embeddings shape:", kb_emb.shape)

        # Cas 2 : ancien format tuple
        else:
            kb_df, kb_emb = obj
            print("[INFO] Cache KB (tuple) chargé, embeddings shape:", kb_emb.shape)

    else:
        print("[INFO] Pas de cache KB, encodage complet…")
        kb_df = pd.read_csv(kb_csv_path)
        model = SentenceTransformer(model_name)
        kb_emb = model.encode(
            kb_df["reply_text"].astype(str).tolist(),
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        # On sauvegarde dans le format dict pour compatibilité avec Ollama
        torch.save(
            {
                "kb_signature": f"kb={os.path.basename(kb_csv_path)}|model={model_name}|n={len(kb_df)}",
                "bert_model": model_name,
                "embeddings": kb_emb,
            },
            cache_path,
        )
        print("[INFO] Cache KB sauvegardé dans", cache_path)

    return kb_df, kb_emb

# ------------------------------------------------------------------
# ⚙️ Construction du contexte RAG (Top-K similarités)
# ------------------------------------------------------------------
def build_rag_context_for_tweets(input_df, kb_df, kb_emb, model, top_k=1, max_len=300):
    text_candidates = [
        "text_display",
        "text_clean",
        "text_raw",
        "text",
        "full_text",
    ]

    text_col = None
    for c in text_candidates:
        if c in input_df.columns:
            text_col = c
            break

    if text_col is None:
        raise ValueError(
            f"Aucune colonne texte trouvée dans le fichier input. Colonnes dispo : {list(input_df.columns)}"
        )

    print(f"[INFO] Colonne utilisée pour les textes (similarités KB): {text_col}")

    input_emb = model.encode(
        input_df[text_col].astype(str).tolist(),
        convert_to_tensor=True,
    )

    print(f"[INFO] Calcul des similarités (Top-{top_k})…")
    results = []
    for i in tqdm(range(len(input_df))):
        scores = util.pytorch_cos_sim(input_emb[i], kb_emb)[0]
        top_indices = torch.topk(scores, k=top_k).indices.cpu().numpy()

        kb_text_candidates = ["reply", "body", "opener", "cta", "reply_text"]
        kb_text_col = next((c for c in kb_text_candidates if c in kb_df.columns), None)
        if kb_text_col is None:
            raise KeyError(f"Aucune colonne texte trouvée dans la KB. Colonnes KB : {list(kb_df.columns)}")

        contexts = [str(kb_df.iloc[idx][kb_text_col])[:max_len] for idx in top_indices]
        results.append(" ".join(contexts))

    input_df["rag_context"] = results

    base_candidates = ["text_display", "text_clean", "text_raw", "text", "full_text"]
    base_col = next((c for c in base_candidates if c in input_df.columns), None)
    if base_col is None:
        base_col = "text_display"
        input_df[base_col] = ""

    input_df["text_for_llm"] = (
        input_df[base_col].fillna("").astype(str)
        + "\n\n[Contexte KB]\n"
        + input_df["rag_context"].fillna("").astype(str)
    )

    input_df["text_display"] = input_df[base_col].fillna("").astype(str)

    if "created_at_dt" not in input_df.columns:
        date_col = None
        for c in ("created_at", "date", "timestamp", "time"):
            if c in input_df.columns:
                date_col = c
                break
        if date_col:
            input_df["created_at_dt"] = pd.to_datetime(input_df[date_col], errors="coerce")

    return input_df

# ------------------------------------------------------------------
# 🧩 Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fichier CSV à enrichir")
    parser.add_argument("--output", required=True, help="Fichier de sortie avec colonne rag_context")
    parser.add_argument("--kb", required=True, help="Fichier CSV KB (knowledge base)")
    parser.add_argument("--model", default="distilbert-base-multilingual-cased", help="Modèle SentenceTransformer")
    parser.add_argument("--top-k", type=int, default=1, help="Nombre d'extraits à inclure")
    parser.add_argument("--cache", default=None, help="Chemin du cache KB (torch)")
    args = parser.parse_args()

    cache_path = args.cache or args.kb.replace(".csv", f"__{args.model}_emb.pt")

    kb_df, kb_emb = load_or_encode_kb(args.kb, args.model, cache_path)
    model = SentenceTransformer(args.model)
    input_df = pd.read_csv(args.input)

    enriched_df = build_rag_context_for_tweets(input_df, kb_df, kb_emb, model, top_k=args.top_k)
    enriched_df.to_csv(args.output, index=False)
    print(f"[OK] Fichier enrichi écrit dans : {args.output}", flush=True)
