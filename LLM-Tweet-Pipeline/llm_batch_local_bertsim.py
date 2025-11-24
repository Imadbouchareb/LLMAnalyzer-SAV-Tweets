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

import argparse, json, time, sqlite3, hashlib, sys, csv, re
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


# ---------------------------
# KB / RAG helpers (local, silent if absent)
# ---------------------------
def load_kb_rich(path: str) -> List[Dict]:
    """Charge la KB CSV UTF-8. Retourne liste de dicts avec colonnes attendues.
    Silencieux si fichier absent/illisible: retourne [].
    """
    cols = [
        "body","cta","intent","lang","length","opener","reply","source_intent_col",
        "source_sentiment_col","source_theme_col","subintent","tags","tone","urgency_bucket",
    ]
    if not path or not os.path.exists(path):
        return []
    try:
        dfk = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    except Exception:
        return []
    rows = []
    for _, r in dfk.iterrows():
        row = {c: (r[c] if c in dfk.columns else "") for c in cols}
        # ensure strings
        for k in row:
            if pd.isna(row[k]):
                row[k] = ""
            row[k] = str(row[k])
        rows.append(row)
    return rows


def normalize_intent(x: str) -> str:
    if not x:
        return "Generic"
    s = x.strip().lower()
    mapping = {
        "factur": "Facturation",
        "paiement": "Facturation",
        "reseau": "Reseau/Internet",
        "internet": "Reseau/Internet",
        "box": "Box/TV",
        "freebox": "Box/TV",
        "mobile": "Mobile/SIM/Portabilite",
        "sim": "Mobile/SIM/Portabilite",
        "portabil": "Mobile/SIM/Portabilite",
        "sav": "Support/SAV/Reclamation",
        "réclamation": "Support/SAV/Reclamation",
        "reclamation": "Support/SAV/Reclamation",
        "support": "Support/SAV/Reclamation",
        "compte": "Compte/Acces",
        "connexion": "Compte/Acces",

        # nouveaux mappages utiles
        "commande": "Support/SAV/Reclamation",
        "livraison": "Support/SAV/Reclamation",
        "resiliation": "Support/SAV/Reclamation",
        "insatisfaction": "Support/SAV/Reclamation",
        "colère": "Support/SAV/Reclamation",
        "securite": "Support/SAV/Reclamation",
        "fraude": "Support/SAV/Reclamation",
        "commercial": "Generic",
        "offre": "Generic",
        "annonce": "Generic",
        "marketing": "Generic",
        "incident/actualit": "Reseau/Internet",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    for cand in (
        "Facturation","Reseau/Internet","Box/TV","Mobile/SIM/Portabilite",
        "Support/SAV/Reclamation","Compte/Acces"
    ):
        if cand.lower() in s:
            return cand
    return "Generic"


def infer_tone_from(s_label: str, urg: Optional[int]) -> str:
    try:
        if urg is not None and int(urg) >= 2:
            return "neg"
    except Exception:
        pass
    s = (s_label or "").strip().lower()
    if any(k in s for k in ("neg","nég","negative","négatif")):
        return "neg"
    if any(k in s for k in ("pos","positif","positive","positif")):
        return "pos"
    return "neu"


def infer_urgency_bucket(urg: Optional[int]) -> str:
    if urg is None:
        return "unknown"
    try:
        u = int(urg)
    except Exception:
        return "unknown"
    if u <= 0:
        return "low"
    if u == 1:
        return "normal"
    return "high"


def get_subintent_hint(df_row) -> str:
    # prefer explicit theme columns if present
    for col in ("primary_theme","themes_primary","themes","source_theme_col","theme","topic"):
        if col in df_row and df_row.get(col) not in (None, ""):
            return str(df_row.get(col))
    return ""


def select_kb_candidates(kb: List[Dict], intent_norm: str, tone: str, urg_bucket: str, subintent_hint: str) -> List[str]:
    if not kb:
        return []
    intent_norm_l = (intent_norm or "Generic").strip().lower()
    # primary intent match (case-insensitive)
    primary = [r for r in kb if (r.get("intent","") or "").strip().lower() == intent_norm_l]
    if not primary:
        primary = [r for r in kb if (r.get("intent","") or "").strip().lower() == "generic"]
    if not primary:
        primary = kb

    # soft filter: prefer tone/urg matches but keep others
    def score_row(r):
        score = 0
        if r.get("tone") and tone and tone.strip().lower() in r.get("tone","" ).strip().lower():
            score += 2
        if r.get("urgency_bucket") and urg_bucket and urg_bucket.strip().lower() in r.get("urgency_bucket","" ).strip().lower():
            score += 1
        # subintent boost
        if subintent_hint:
            if subintent_hint.lower() in (r.get("subintent","") or "").lower():
                score += 3
            if subintent_hint.lower() in (r.get("tags","") or "").lower():
                score += 2
        return score

    scored = sorted(primary, key=lambda r: score_row(r), reverse=True)
    # collect replies, dedupe preserving order
    out = []
    seen = set()
    for r in scored:
        rep = (r.get("reply") or "").strip()
        if not rep:
            continue
        k = rep.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(rep)
        if len(out) >= 10:
            break
    return out


def pick_deterministic(candidates: List[str], seed_fields: List[str]) -> str:
    if not candidates:
        return ""
    key = "".join([str(s or "") for s in seed_fields])
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    idx = seed % len(candidates)
    return candidates[idx]


# --- helpers variété + perfs ---
_norm_space_re = re.compile(r"\s+")
_norm_punct_re = re.compile(r"\s+([?.!])")

def normalize_reply(s: str) -> str:
    s = (_norm_space_re.sub(" ", (s or "").strip()))
    s = (_norm_punct_re.sub(r"\1", s))
    return s

def build_kb_index(kb_rows: List[Dict]):
    """Indexe la KB par intent (lower) et pré-calcule champs normalisés pour scoring rapide."""
    by_intent = {}
    for r in kb_rows:
        it = (r.get("intent") or "").strip().lower() or "generic"
        r["_tone"] = (r.get("tone") or "").lower()
        r["_urg"]  = (r.get("urgency_bucket") or "").lower()
        r["_sub"]  = (r.get("subintent") or "").lower()
        r["_tags"] = (r.get("tags") or "").lower()
        r["_op"]   = (r.get("opener") or "").strip()
        r["_bd"]   = (r.get("body") or "").strip()
        r["_cta"]  = (r.get("cta") or "").strip()
        r["_rep"]  = normalize_reply(r.get("reply",""))
        by_intent.setdefault(it, []).append(r)
    return by_intent

def select_kb_candidates_indexed(by_intent: Dict[str, List[Dict]], fallback_rows: List[Dict],
                                 intent_norm: str, tone: str, urg_bucket: str, subintent_hint: str) -> List[str]:
    """Retourne jusqu'à 30 candidats (reply + recomposition opener+body+cta), dédupliqués."""
    key = (intent_norm or "Generic").strip().lower()
    primary = by_intent.get(key) or by_intent.get("generic") or fallback_rows
    subhint = (subintent_hint or "").lower()

    def score_row(r):
        s = 0
        if tone and tone in r["_tone"]: s += 2
        if urg_bucket and urg_bucket in r["_urg"]: s += 1
        if subhint:
            if subhint in r["_sub"]:  s += 3
            if subhint in r["_tags"]: s += 2
        return s

    scored = sorted(primary, key=score_row, reverse=True)

    seen, out = set(), []
    for r in scored:
        # 1) reply prêt à l'emploi
        if r["_rep"]:
            k = r["_rep"].casefold()
            if k not in seen:
                seen.add(k); out.append(r["_rep"])
        # 2) recomposition opener + body + cta
        if r["_bd"] and r["_cta"]:
            rep2 = r["_op"] + (" " if r["_op"] else "") + r["_bd"]
            rep2 = normalize_reply(rep2)
            if not rep2.endswith((".", "!", "?")):
                rep2 += "."
            rep2 = normalize_reply(rep2 + " " + r["_cta"])
            k2 = rep2.casefold()
            if k2 not in seen:
                seen.add(k2); out.append(rep2)
        if len(out) >= 30:
            break
    return out

def get_subintent_hint(row) -> str:
    for col in ("primary_theme","themes_primary","themes","theme","topic"):
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val)
    return ""


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

    # Load local KB once (silent if absent) with multiple fallbacks
    kb_candidates = [
        os.path.join(os.getcwd(), "kb_replies_rich.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_replies_rich.csv"),
    ]
    try:
        kb_candidates.append(os.path.join(os.path.dirname(os.path.abspath(args.input)), "kb_replies_rich.csv"))
    except Exception:
        pass

    kb_path = next((p for p in kb_candidates if p and os.path.exists(p)), None)
    KB_RICH = load_kb_rich(kb_path) if kb_path else []
    KB_AVAILABLE = bool(KB_RICH)
    KB_INDEX = build_kb_index(KB_RICH) if KB_AVAILABLE else {}

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
                # integrate KB-based deterministic reply selection if KB available
                try:
                    if KB_AVAILABLE:
                        # df is aligned; global_idx is position in rows
                        row = df.iloc[global_idx]
                        intent_curr = out.get("intent_text", "")
                        s_label = out.get("sentiment_text", "")
                        urg = out.get("urgency_0_3", None)
                        intent_norm = normalize_intent(intent_curr)
                        tone = infer_tone_from(s_label, urg)
                        urg_bucket = infer_urgency_bucket(urg)
                        subintent_hint = get_subintent_hint(row)
                        # use indexed selection for speed/diversity
                        candidates = select_kb_candidates_indexed(KB_INDEX, KB_RICH, intent_norm, tone, urg_bucket, subintent_hint)
                        if candidates:
                            # build seed from available id or text for deterministic selection
                            tweet_id_or_text = ""
                            for cid in ("id_str","tweet_id","status_id","id"):
                                if cid in df.columns:
                                    val = row.get(cid, "")
                                    if pd.notna(val) and str(val).strip():
                                        tweet_id_or_text = str(val).strip()
                                        break
                            if not tweet_id_or_text:
                                tweet_id_or_text = row.get(text_col, "")
                            chosen = pick_deterministic(candidates, [tweet_id_or_text, intent_norm, str(urg or "")])
                            if chosen:
                                out["reply_suggestion"] = chosen
                except Exception:
                    # on any error, keep existing out["reply_suggestion"] (fallback)
                    pass
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
