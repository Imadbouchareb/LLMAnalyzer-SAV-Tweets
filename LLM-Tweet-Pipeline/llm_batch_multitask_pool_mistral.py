#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from tqdm import tqdm

ChatOllama = None
try:  # pragma: no cover
    from langchain_ollama import ChatOllama as _ChatOllama  # type: ignore

    ChatOllama = _ChatOllama
except Exception:  # noqa: BLE001
    try:
        from langchain_community.chat_models import ChatOllama as _ChatOllama  # type: ignore

        ChatOllama = _ChatOllama
    except Exception:  # noqa: BLE001
        ChatOllama = None

OUTPUT_COLUMNS = [
    "tweet_id",
    "created_at_dt",
    "text_raw",
    "text_display",
    "rag_context",
    "themes_list",
    "primary_label",
    "sentiment_label",
    "llm_urgency_0_3",
    "llm_severity_0_3",
    "status",
    "summary_1l",
    "author",
    "assigned_to",
    "llm_summary",
    "llm_reply_suggestion",
    "routing_team",
]

SYSTEM_PROMPT = """
Tu es un assistant expert du service client de l'opérateur télécom Free.
Tu analyses des messages clients sur les réseaux sociaux (tweets, posts, commentaires).
Pour chaque message, tu dois produire un diagnostic structuré pour aider les équipes SAV.

### FORMAT ATTENDU (clés JSON) :
- intent_text : thème principal du message (parmi : "Problème réseau", "Facturation", "Freebox",
  "Portabilité", "Ligne mobile", "Résiliation", "SAV", "Livraison", "Autre").
- sentiment_label : tonalité du message ("positif", "négatif" ou "neutre").
- urgency_0_3 : urgence perçue (0 à 3)
- severity_0_3 : gravité perçue (0 à 3)
- summary_1l : résumé concis du message (20 mots max)
- llm_reply_suggestion : réponse naturelle et personnalisée que Free pourrait publier
- needs_handoff : true/false → intervention humaine nécessaire ou non
- routing_team : équipe concernée (exemples : "SAV Mobile", "Facturation", "Technique", "Freebox", "Portabilité")

### CONSIGNES :
- Si le message exprime un problème, une réclamation ou une colère → sentiment_label = "négatif"
- Si le message contient des remerciements, satisfaction ou reconnaissance → sentiment_label = "positif"
- Sinon → sentiment_label = "neutre"
- Varie le ton dans les résumés et les réponses, évite les phrases répétitives.
- Ne copie jamais le texte original.
- Réponds toujours avec un JSON propre et complet, sans texte avant ni après.
"""

HUMAN_PROMPT = """
Analyse ce message client et fournis les champs demandés.

TEXTE DU CLIENT :
{text}

💡 Consignes :
- Utilise bien le contexte pour déterminer le thème (intent_text) et le sentiment.
- Varie ton style de résumé et de réponse pour éviter les répétitions.
- Réponds sous forme structurée en respectant exactement les champs de TweetLabels.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

THEME_MAPPING_RAW = {
    "SAV": "Service après-vente",
    "Service client": "Service après-vente",
    "Service après vente": "Service après-vente",
    "Support technique": "Service après-vente",
    "Facturation": "Facturation",
    "Paiement": "Facturation",
    "Freebox": "Freebox",
    "Connexion": "Problème réseau",
    "Réseau": "Problème réseau",
    "Signal": "Problème réseau",
    "Panne": "Problème réseau",
    "Internet": "Problème réseau",
    "Resiliation": "Résiliation",
    "Résiliation": "Résiliation",
    "Livraison": "Livraison",
    "Portabilité": "Portabilité",
    "Transfert": "Portabilité",
    "Ligne mobile": "Ligne mobile",
    "Mobile": "Ligne mobile",
    "Autre": "Autre",
}

SENTIMENT_MAPPING_RAW = {
    "colère": "négatif",
    "frustration": "négatif",
    "mécontentement": "négatif",
    "ironie": "négatif",
    "satisfaction": "positif",
    "reconnaissance": "positif",
    "positif": "positif",
    "négatif": "négatif",
    "neutre": "neutre",
}

THEME_MAPPING = {k.lower(): v for k, v in THEME_MAPPING_RAW.items()}
SENTIMENT_MAPPING = {k.lower(): v for k, v in SENTIMENT_MAPPING_RAW.items()}

AUTHOR_CANDIDATES = ["author", "user_name", "username", "screen_name", "user", "name"]
SUMMARY_FALLBACKS = [
    "Résumé indisponible (analyse automatique échouée).",
    "Analyse indisponible, merci de transmettre à un conseiller Free.",
]
REPLY_FALLBACKS = [
    "Merci de transférer ce cas à un agent du service client pour un suivi personnalisé.",
    "Un conseiller doit reprendre ce dossier manuellement. Merci de l'escalader.",
]


class TweetLabels(BaseModel):
    intent_text: str = Field("Autre")
    sentiment_label: str = Field("neutre")
    urgency_0_3: int = Field(1)
    severity_0_3: int = Field(1)
    summary_1l: str = Field("")
    llm_reply_suggestion: str = Field("")
    needs_handoff: bool = Field(False)
    routing_team: str = Field("SAV Mobile")


def normalize_theme(value: str | None) -> str:
    if not value:
        return "Autre"
    key = str(value).strip()
    return THEME_MAPPING.get(key.lower(), key) if key else "Autre"


def normalize_sentiment(value: str | None) -> str:
    if not value:
        return "neutre"
    return SENTIMENT_MAPPING.get(str(value).strip().lower(), "neutre")


def clamp_score(value: Any, default: int = 1) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(3, score))


def clean_text(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def serialize_themes(theme: str) -> str:
    return json.dumps([theme], ensure_ascii=False)


def extract_author(row: pd.Series) -> str:
    for col in AUTHOR_CANDIDATES:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()
    return "inconnu"


def extract_text_display(row: pd.Series) -> str:
    for col in ("text_display", "text_for_llm", "text_clean", "text", "text_for_model", "full_text"):
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return ""


def ensure_text_column(df: pd.DataFrame) -> None:
    if "text_for_model" in df.columns:
        df["text_for_model"] = df["text_for_model"].astype("string").fillna("")
        return
    for candidate in ("text_for_llm", "text_display", "text_clean", "text", "full_text"):
        if candidate in df.columns:
            df["text_for_model"] = df[candidate].astype("string").fillna("")
            return
    df["text_for_model"] = pd.Series([""] * len(df), dtype="string")


def payload_for_row(row: pd.Series, max_chars: int) -> dict:
    txt = clean_text(row.get("text_for_model"), "[Message vide ou tweet supprimé]")
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    context = clean_text(row.get("rag_context"), "")
    full_input = f"TEXTE : {txt}\n\nCONTEXTE FREE (RAG) : {context}"
    return {"text": full_input}


class Cache:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (h TEXT PRIMARY KEY, json TEXT, ts INTEGER)")
        self.conn.commit()

    def get_many(self, hashes: List[str]) -> Dict[str, str]:
        if not hashes:
            return {}
        qmarks = ",".join("?" for _ in hashes)
        cur = self.conn.execute(f"SELECT h,json FROM cache WHERE h IN ({qmarks})", hashes)
        return {h: blob for h, blob in cur.fetchall()}

    def set_many(self, items: List[tuple[str, str, int]]) -> None:
        if not items:
            return
        self.conn.executemany("INSERT OR REPLACE INTO cache(h,json,ts) VALUES(?,?,?)", items)
        self.conn.commit()


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.lock = threading.Lock()
        self.last = 0.0
        self.min_interval = 60.0 / rpm if rpm else 0.0

    def acquire(self) -> None:
        if not self.rpm:
            return
        with self.lock:
            now = time.time()
            wait = self.min_interval - (now - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()


def sha1_of_payload(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def default_llm_output() -> dict:
    return dict(
        intent_text="Service après-vente",
        sentiment_label="neutre",
        urgency_0_3=1,
        severity_0_3=1,
        summary_1l=random.choice(SUMMARY_FALLBACKS),
        llm_reply_suggestion=random.choice(REPLY_FALLBACKS),
        needs_handoff=True,
        routing_team="SAV Mobile",
    )


def build_chain(use_ollama: bool = False):
    if use_ollama:
        if ChatOllama is None:
            raise RuntimeError("ChatOllama indisponible. Installez `langchain-ollama`.")
        model = (os.getenv("OLLAMA_MODEL") or "mistral").strip()
        llm = ChatOllama(model=model, temperature=0.4)
        print(f"[INFO] Chaîne Ollama prête (modèle={model}).")
        return prompt | llm.with_structured_output(TweetLabels)

    key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY manquant.")
    model = (os.getenv("MISTRAL_MODEL") or "mistral-small-latest").strip()
    temp = random.uniform(0.75, 0.95)
    llm = ChatMistralAI(model=model, temperature=temp, api_key=key)
    print(f"[INFO] Chaîne Mistral prête (modèle={model}, temp={temp:.2f}).")
    return prompt | llm.with_structured_output(TweetLabels)


def worker(chain, payload: dict, rl: RateLimiter, retries: int, backoff: float) -> dict:
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            rl.acquire()
            out = chain.invoke(payload)
            if isinstance(out, dict):
                return out
            if isinstance(out, BaseModel):
                return out.model_dump()
            parsed = getattr(out, "parsed", None)
            if isinstance(parsed, dict):
                return parsed
            raise TypeError(f"Type inattendu de sortie LLM: {type(out)}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[LLM ERROR] tentative {attempt + 1}: {type(exc).__name__} -> {exc}")
            time.sleep(backoff * (attempt + 1))

    print(f"[LLM ERROR FATAL] Dernière erreur: {last_err!r}")
    fallback = default_llm_output()
    fallback["_error"] = str(last_err) if last_err else "Unknown error"
    return fallback


def build_result_row(row: pd.Series, llm_output: dict) -> dict:
    merged = default_llm_output()
    merged.update(llm_output or {})
    theme = normalize_theme(merged.get("intent_text"))
    sentiment = normalize_sentiment(merged.get("sentiment_label"))
    urgency = clamp_score(merged.get("urgency_0_3"))
    severity = clamp_score(merged.get("severity_0_3"))
    needs_handoff = bool(merged.get("needs_handoff", False))
    routing_team = clean_text(merged.get("routing_team"), "SAV Mobile")
    summary = clean_text(merged.get("summary_1l"), random.choice(SUMMARY_FALLBACKS))
    reply = clean_text(merged.get("llm_reply_suggestion"), random.choice(REPLY_FALLBACKS))

    tweet_id = row.get("tweet_id") or row.get("id") or row.get("id_str") or str(row.name)
    created_at = row.get("created_at_dt")
    if pd.isna(created_at):
        created_at = ""

    text_display = extract_text_display(row)
    text_raw = clean_text(row.get("text_raw"), "")
    rag_context = clean_text(row.get("rag_context"), "")
    author = extract_author(row)
    assigned_to = clean_text(row.get("assigned_to"), routing_team)

    return {
        "tweet_id": tweet_id,
        "created_at_dt": created_at,
        "text_raw": text_raw,
        "text_display": text_display,
        "rag_context": rag_context,
        "themes_list": serialize_themes(theme),
        "primary_label": theme,
        "sentiment_label": sentiment,
        "llm_urgency_0_3": urgency,
        "llm_severity_0_3": severity,
        "status": "open" if needs_handoff else "closed",
        "summary_1l": summary,
        "author": author,
        "assigned_to": assigned_to,
        "llm_summary": summary,
        "llm_reply_suggestion": reply,
        "routing_team": routing_team,
    }


def finalize_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.reindex(columns=OUTPUT_COLUMNS)
    text_cols = [
        "text_raw",
        "text_display",
        "rag_context",
        "themes_list",
        "summary_1l",
        "author",
        "assigned_to",
        "llm_summary",
        "llm_reply_suggestion",
        "routing_team",
        "status",
        "primary_label",
        "sentiment_label",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(lambda v: str(v).strip())
    for col in ("llm_urgency_0_3", "llm_severity_0_3"):
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna(1)
                .apply(lambda v: clamp_score(v))
                .astype(int)
                .clip(lower=0, upper=3)
            )
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch Mistral pipeline (JSON → SAV dashboard)")
    ap.add_argument("--data", default=os.getenv("DATA_CSV", "free_tweets_cleaned_fallback.csv"))
    ap.add_argument("--out", default=os.getenv("OUT_CSV", "tweets_scored_llm.csv"))
    ap.add_argument("--input", dest="input_alias")
    ap.add_argument("--output", dest="output_alias")
    ap.add_argument("--cache", default="llm_cache.sqlite")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", 4)))
    ap.add_argument("--max-chars", type=int, default=int(os.getenv("MAX_CHARS", 700)))
    ap.add_argument("--retries", type=int, default=int(os.getenv("RETRIES", 4)))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("TIMEOUT", 60)))
    ap.add_argument("--rpm", type=int, default=int(os.getenv("RPM", 0)))
    ap.add_argument("--ollama", action="store_true", help="Active le mode local Ollama")
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if getattr(args, "input_alias", None):
        args.data = args.input_alias
    if getattr(args, "output_alias", None):
        args.out = args.output_alias

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"CSV introuvable: {args.data}")

    df = pd.read_csv(args.data, low_memory=False)
    ensure_text_column(df)

    rows_idx = list(df.index)
    payloads = [payload_for_row(df.loc[i], args.max_chars) for i in rows_idx]
    hashes = [sha1_of_payload(p) for p in payloads]

    cache = Cache(args.cache)
    cached = cache.get_many(hashes)

    chain = build_chain(use_ollama=args.ollama)
    rl = RateLimiter(args.rpm if (args.rpm and not args.ollama) else 0)

    pending = [(pos, h, payloads[pos]) for pos, h in enumerate(hashes) if h not in cached]
    llm_results: Dict[int, dict] = {}
    new_cache_rows: List[tuple[str, str, int]] = []

    if pending:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {
                executor.submit(worker, chain, payload, rl, args.retries, 1.2): (pos, hash_key)
                for pos, hash_key, payload in pending
            }
            pbar = tqdm(total=len(futures), desc="LLM (Mistral)", unit="tweet")
            for fut in cf.as_completed(futures):
                pos, hash_key = futures[fut]
                try:
                    out_dict = fut.result(timeout=args.timeout)
                except Exception:  # noqa: BLE001
                    out_dict = default_llm_output()
                llm_results[pos] = out_dict
                new_cache_rows.append((hash_key, json.dumps(out_dict, ensure_ascii=False), int(time.time())))
                pbar.update(1)
            pbar.close()
        if new_cache_rows:
            cache.set_many(new_cache_rows)
            for hash_key, blob, _ in new_cache_rows:
                cached[hash_key] = blob

    results: List[Dict[str, Any]] = []
    for pos, idx in enumerate(rows_idx):
        row = df.loc[idx]
        hash_key = hashes[pos]
        if hash_key in cached:
            try:
                llm_output = json.loads(cached[hash_key])
            except json.JSONDecodeError:
                llm_output = default_llm_output()
        else:
            llm_output = llm_results.get(pos) or default_llm_output()
        results.append(build_result_row(row, llm_output))

    final_df = finalize_dataframe(results)
    print(f"[INFO] Colonnes standardisées : {len(final_df.columns)} colonnes")
    print(final_df.columns.tolist())
    final_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Écrit : {args.out}")


if __name__ == "__main__":
    main()
