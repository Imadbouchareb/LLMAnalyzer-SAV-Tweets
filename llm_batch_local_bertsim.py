#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_batch_local_bertsim.py
Version BERT-sim rapide (CPU) — même sorties que la version Mistral.
Ce fichier est une copie fonctionnelle du script "llm_batch_local_bert_bertsim.py"
mais nommé exactement comme l'app l'attend (llm_batch_local_bertsim.py).
"""
import os
# Désactiver TF / logs bruyants
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse, json, time, sqlite3, hashlib, sys
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import joblib

# INTENT_LABELS et mapping minimal (doivent correspondre à la version app)
INTENT_LABELS = [
    "Facturation","Reseau/Internet","Mobile/SIM/Portabilite","Box/TV","Commande/Livraison",
    "Resiliation","Support/SAV/Reclamation","Commercial/Offre","Compte/Acces",
    "Annonce/Marketing","Insatisfaction/Colère","Incident/Actualité","Securite/Fraude",
]

URGENCY_TERMS_CRIT = {"urgent","urgence","immédiat","immediat","bloqué","bloquee","sos"}
URGENCY_TERMS_HIGH = {"impossible","aucun reseau","pas de reseau","panne","hs","hors service","coupure"}
NEG_WORDS = {"honte","scandale","arnaque","inadmissible","nul","pourri","déçu","catastrophe","colère","marre","lamentable"}

LLM_COLS = [
    "intent_text","intent_confidence","sentiment_text","sentiment_confidence",
    "urgency_0_3","severity_0_3","summary","reply_suggestion","needs_handoff"
]

DEFAULT_OUTPUT = dict(
    intent_text="Service client",
    intent_confidence=0.0,
    sentiment_text="neutre",
    sentiment_confidence=0.0,
    urgency_0_3=1,
    severity_0_3=1,
    summary="",
    reply_suggestion="",
    needs_handoff=False,
)

def set_threads(n: int):
    try:
        torch.set_num_threads(max(1, int(n)))
        os.environ.setdefault("OMP_NUM_THREADS", str(int(n)))
        os.environ.setdefault("MKL_NUM_THREADS", str(int(n)))
    except Exception:
        pass

def pick_text_col(df: pd.DataFrame) -> str:
    if "text_for_model" in df.columns: return "text_for_model"
    if "text_for_llm" in df.columns: return "text_for_llm"
    if "text_clean" in df.columns: return "text_clean"
    if "full_text" in df.columns: return "full_text"
    raise ValueError("Aucune colonne texte trouvée.")

def sha1_of_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()

def mean_pooling_last_hidden(model, tokenizer, texts: List[str], device, bs=64, max_length=128):
    all_emb = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=att, return_dict=True)
        token_embeddings = out.last_hidden_state
        mask = att.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * mask).sum(1)
        sum_mask = mask.sum(1).clamp(min=1e-9)
        emb = (sum_embeddings / sum_mask).cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        all_emb.append(emb)
    return np.vstack(all_emb) if all_emb else np.zeros((0, model.config.hidden_size))

def softmax_conf_from_sims(sims: np.ndarray, temp: float = 0.1):
    ex = np.exp(sims / max(temp, 1e-6))
    probs = ex / ex.sum(axis=1, keepdims=True)
    idxs = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    return idxs, confs

def score_urgency(text: str):
    t = str(text or "").lower()
    u = 0
    if any(k in t for k in URGENCY_TERMS_HIGH): u = 2
    if any(k in t for k in URGENCY_TERMS_CRIT): u = 3
    sev = u
    if any(k in t for k in NEG_WORDS): sev = max(sev, 2)
    if text.count("!") >= 2: sev = max(sev, 2)
    return min(u,3), min(sev,3)

def reply_template(intent: str, sentiment: str, urgency: int) -> str:
    base = "Merci pour votre message."
    if intent in {"Securite/Fraude","Incident/Actualité"}:
        base = "Alerte reçue. Merci de votre signalement."
    if sentiment == "négatif" or urgency >= 2:
        base += " Nous sommes désolés pour la gêne occasionnée."
    add = " Pouvez-vous nous écrire en DM avec votre identifiant client et un numéro de contact ?"
    if intent in {"Facturation","Compte/Acces"}:
        add = " Pour vérifier votre dossier, envoyez-nous en DM votre identifiant client."
    if intent in {"Reseau/Internet","Box/TV","Mobile/SIM/Portabilite"}:
        add = " Donnez-nous en DM votre code postal et une description rapide de l’incident."
    return (base + add).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache", default="llm_cache_bertsim.sqlite")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hf-batch", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--sentiment-model", default="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    ap.add_argument("--intent-bert", default="distilbert-base-multilingual-cased")
    ap.add_argument("--intent-max-length", type=int, default=128)
    ap.add_argument("--sim-temperature", type=float, default=0.1)
    ap.add_argument("--distilled-dir", default="models/local_llm")
    ap.add_argument("--prefer-distilled", action="store_true")
    ap.add_argument("--eval-against", default=None)
    args = ap.parse_args()

    set_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    df_in = pd.read_csv(args.input, low_memory=False)
    orig_cols = list(df_in.columns)

    text_col = pick_text_col(df_in)
    df = df_in.copy()
    df["text_for_model"] = df[text_col].astype("string").fillna("").astype(str).apply(lambda s: s[:700])

    rows_idx = list(df.index)
    texts = df["text_for_model"].tolist()
    hashes = [sha1_of_text(t) for t in texts]

    cache = sqlite3.connect(args.cache, check_same_thread=False)
    cur = cache.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS cache (h TEXT PRIMARY KEY, json TEXT, ts INTEGER)")
    cache.commit()

    # get cached values
    q = ",".join("?" for _ in hashes) if hashes else ""
    cached_map = {}
    if hashes:
        cur.execute(f"SELECT h,json FROM cache WHERE h IN ({q})", hashes)
        for h,j in cur.fetchall():
            cached_map[h] = j

    to_call_idx = [i for i,h in enumerate(hashes) if h not in cached_map]
    to_call_texts = [texts[i] for i in to_call_idx]

    # try load distilled pack if requested
    distilled_pack = None
    if args.prefer_distilled:
        try:
            meta_fp = os.path.join(args.distilled_dir, "meta.json")
            if os.path.exists(meta_fp):
                with open(meta_fp, "r", encoding="utf-8") as f: meta = json.load(f)
                # attempt to load joblibs
                pack = {}
                pack["intent_clf"] = joblib.load(os.path.join(args.distilled_dir, "intent_clf.joblib"))
                pack["sentiment_clf"] = joblib.load(os.path.join(args.distilled_dir, "sentiment_clf.joblib"))
                pack["urgency_clf"] = joblib.load(os.path.join(args.distilled_dir, "urgency_clf.joblib"))
                pack["severity_clf"] = joblib.load(os.path.join(args.distilled_dir, "severity_clf.joblib"))
                try:
                    from sentence_transformers import SentenceTransformer
                    pack["encoder"] = SentenceTransformer(meta.get("encoder"))
                except Exception:
                    pack["encoder"] = None
                pack["intent_classes"] = meta.get("intent_classes")
                pack["sentiment_classes"] = meta.get("sentiment_classes")
                distilled_pack = pack
        except Exception:
            distilled_pack = None

    use_distilled = args.prefer_distilled and distilled_pack is not None

    # sentiment pipeline
    sentiment_pipe = pipeline("sentiment-analysis", model=args.sentiment_model, device=0 if torch.cuda.is_available() else -1, framework="pt")
    summarizer = None
    if not args.no_summarize:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1, framework="pt")

    # intent encoder
    tok = AutoTokenizer.from_pretrained(args.intent_bert, use_fast=True)
    mdl = AutoModel.from_pretrained(args.intent_bert)
    mdl.to(device)
    mdl.eval()
    label_embeds = mean_pooling_last_hidden(mdl, tok, INTENT_LABELS, device, bs=args.hf_batch, max_length=args.intent_max_length)

    pre_emb = None
    if not use_distilled and len(to_call_texts) > 0:
        pre_emb = mean_pooling_last_hidden(mdl, tok, to_call_texts, device, bs=args.hf_batch, max_length=args.intent_max_length)

    results = {}
    to_store_items = []
    pbar = tqdm(total=len(to_call_texts), desc="BERT-sim", unit="item")
    for offset in range(0, len(to_call_texts), max(1, args.batch)):
        chunk = to_call_texts[offset: offset + args.batch]
        if use_distilled:
            # use distilled classifiers via encoder (sentence-transformers) if available
            enc = distilled_pack.get("encoder")
            if enc is None:
                # fallback to default small pipeline behavior
                X = mean_pooling_last_hidden(mdl, tok, chunk, device, bs=args.hf_batch, max_length=args.intent_max_length)
                sims = X @ label_embeds.T
                idxs, confs = softmax_conf_from_sims(sims, temp=args.sim_temperature)
                intents = [INTENT_LABELS[i] for i in idxs]
                sent_batch = sentiment_pipe(chunk, truncation=True, batch_size=max(1, args.hf_batch))
                summ_batch = summarizer(chunk, max_length=60, min_length=15, truncation=True, batch_size=max(1, args.hf_batch)) if summarizer else None
                outs = []
                for j, txt in enumerate(chunk):
                    s_label_raw = sent_batch[j].get("label","")
                    s_label = {"positive":"positif","neutral":"neutre","negative":"négatif"}.get(s_label_raw.lower(), "neutre")
                    s_conf = float(sent_batch[j].get("score", 0.0))
                    intent = intents[j]
                    i_conf = float(confs[j])
                    urg, sev = score_urgency(txt)
                    summary_text = (summ_batch[j]["summary_text"].strip() if (summ_batch and isinstance(summ_batch[j], dict) and "summary_text" in summ_batch[j]) else txt[:180])
                    reply = reply_template(intent, s_label, urg)
                    out = dict(
                        intent_text=intent,
                        intent_confidence=i_conf,
                        sentiment_text=s_label,
                        sentiment_confidence=s_conf,
                        urgency_0_3=int(urg),
                        severity_0_3=int(sev),
                        summary=summary_text,
                        reply_suggestion=reply,
                        needs_handoff=bool(urg>=2 or sev>=2 or intent in {"Securite/Fraude","Reseau/Internet","Support/SAV/Reclamation","Resiliation"})
                    )
                    outs.append(out)
            else:
                # encode with sentence-transformers + predict via joblib
                X = enc.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
                p_int = distilled_pack["intent_clf"].predict_proba(X)
                idxs = p_int.argmax(axis=1); confs = p_int.max(axis=1)
                p_s = distilled_pack["sentiment_clf"].predict_proba(X); idxs_s = p_s.argmax(axis=1); confs_s = p_s.max(axis=1)
                urg_pred = distilled_pack["urgency_clf"].predict(X)
                sev_pred = distilled_pack["severity_clf"].predict(X)
                outs = []
                for j, txt in enumerate(chunk):
                    intent = distilled_pack["intent_classes"][int(idxs[j])]
                    s_label = distilled_pack["sentiment_classes"][int(idxs_s[j])]
                    out = dict(
                        intent_text=intent,
                        intent_confidence=float(confs[j]),
                        sentiment_text=s_label,
                        sentiment_confidence=float(confs_s[j]),
                        urgency_0_3=int(urg_pred[j]),
                        severity_0_3=int(sev_pred[j]),
                        summary=txt[:180],
                        reply_suggestion=reply_template(intent, s_label, int(urg_pred[j])),
                        needs_handoff=bool(int(urg_pred[j])>=2 or int(sev_pred[j])>=2 or intent in {"Securite/Fraude","Reseau/Internet","Support/SAV/Reclamation","Resiliation"})
                    )
                    outs.append(out)
        else:
            if pre_emb is not None:
                X = pre_emb[offset: offset + len(chunk)]
            else:
                X = mean_pooling_last_hidden(mdl, tok, chunk, device, bs=args.hf_batch, max_length=args.intent_max_length)
            sims = X @ label_embeds.T
            idxs, confs = softmax_conf_from_sims(sims, temp=args.sim_temperature)
            intents = [INTENT_LABELS[i] for i in idxs]
            sent_batch = sentiment_pipe(chunk, truncation=True, batch_size=max(1, args.hf_batch))
            summ_batch = summarizer(chunk, max_length=60, min_length=15, truncation=True, batch_size=max(1, args.hf_batch)) if summarizer else None
            outs = []
            for j, txt in enumerate(chunk):
                s_label_raw = sent_batch[j].get("label","")
                s_label = {"positive":"positif","neutral":"neutre","negative":"négatif"}.get(s_label_raw.lower(), "neutre")
                s_conf = float(sent_batch[j].get("score", 0.0))
                intent = intents[j]
                i_conf = float(confs[j])
                urg, sev = score_urgency(txt)
                summary_text = (summ_batch[j]["summary_text"].strip() if (summ_batch and isinstance(summ_batch[j], dict) and "summary_text" in summ_batch[j]) else txt[:180])
                reply = reply_template(intent, s_label, urg)
                out = dict(
                    intent_text=intent,
                    intent_confidence=i_conf,
                    sentiment_text=s_label,
                    sentiment_confidence=s_conf,
                    urgency_0_3=int(urg),
                    severity_0_3=int(sev),
                    summary=summary_text,
                    reply_suggestion=reply,
                    needs_handoff=bool(urg>=2 or sev>=2 or intent in {"Securite/Fraude","Reseau/Internet","Support/SAV/Reclamation","Resiliation"})
                )
                outs.append(out)
        # write outs to cache list
        for j, out in enumerate(outs):
            global_idx = to_call_idx[offset + j] if (offset + j) < len(to_call_idx) else None
            if global_idx is not None:
                results[global_idx] = out
            h = sha1_of_text(chunk[j])
            cur.execute("INSERT OR REPLACE INTO cache(h,json,ts) VALUES(?,?,?)", (h, json.dumps(out, ensure_ascii=False), int(time.time())))
        cache.commit()
        pbar.update(len(outs))
    pbar.close()

    # build out rows aligning with original df order
    out_rows = []
    for i,h in enumerate(hashes):
        if h in cached_map:
            out_rows.append(json.loads(cached_map[h]))
        else:
            out_rows.append(results.get(i, DEFAULT_OUTPUT.copy()))

    out_df = pd.DataFrame(out_rows)
    for c in LLM_COLS:
        if c not in out_df.columns:
            if c == "intent_text": out_df[c] = "Service client"
            elif c == "intent_confidence": out_df[c] = 0.0
            elif c == "sentiment_text": out_df[c] = "neutre"
            elif c == "sentiment_confidence": out_df[c] = 0.0
            elif c in {"urgency_0_3","severity_0_3"}: out_df[c] = 1
            elif c == "summary": out_df[c] = ""
            elif c == "reply_suggestion": out_df[c] = ""
            elif c == "needs_handoff": out_df[c] = False

    merged = pd.concat([df_in.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)
    final_cols = list(df_in.columns) + [c for c in LLM_COLS if c not in df_in.columns]
    final = merged.loc[:, final_cols]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    final.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[OK] Écrit: {args.output}")

if __name__ == "__main__":
    main()
