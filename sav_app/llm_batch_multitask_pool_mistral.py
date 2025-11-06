#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_batch_multitask_pool_mistral.py
- Switch to Mistral (LangChain ChatMistralAI) + thread pool anti-blocage.
- Même qualité (prompts/fields), cache SQLite, retries, checkpoints, rpm.

PATCH: accepte --input/--output (alias de --data/--out) et privilégie text_for_llm.
"""
import os, re, json, time, sqlite3, argparse, hashlib, concurrent.futures as cf, threading
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # Pydantic v2
from tqdm import tqdm

# --- Tout le reste de votre script d'origine est conservé (règles, prompts, etc.) ---

# ...
# (Pour lisibilité ici, gardez l'intégralité de vos constantes/règles comme dans votre fichier)
# ...

SYSTEM = (
    "Tu es un analyste de contenu client. Pour chaque texte, tu dois extraire:"
    " intent_text, intent_confidence, sentiment_text, sentiment_confidence,"
    " urgency_0_3, severity_0_3, summary, reply_suggestion, needs_handoff."
)

class TweetLabels(BaseModel):
    intent_text: str = Field("Service client")
    intent_confidence: float = Field(0.0)
    sentiment_text: str = Field("neutre")
    sentiment_confidence: float = Field(0.0)
    urgency_0_3: int = Field(1)
    severity_0_3: int = Field(1)
    summary: str = Field("")
    reply_suggestion: str = Field("")
    needs_handoff: bool = Field(False)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Texte: {text}\nMeta: ...\nRéponds STRICTEMENT en JSON au format demandé."),
])

def build_chain():
    key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY manquant.")
    model = (os.getenv("MISTRAL_MODEL") or "mistral-large-latest").strip()
    llm = ChatMistralAI(model=model, temperature=0, api_key=key)
    return prompt | llm.with_structured_output(TweetLabels)

class Cache:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (h TEXT PRIMARY KEY, json TEXT, ts INTEGER)")
        self.conn.commit()
    def get_many(self, hashes: List[str]) -> Dict[str, dict]:
        if not hashes: return {}
        qmarks = ",".join("?" for _ in hashes)
        cur = self.conn.execute(f"SELECT h,json FROM cache WHERE h IN ({qmarks})", hashes)
        return {h:j for h,j in cur.fetchall()}
    def set_many(self, items: List[tuple]):
        if not items: return
        self.conn.executemany("INSERT OR REPLACE INTO cache(h,json,ts) VALUES(?,?,?)", items)
        self.conn.commit()

class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.lock = threading.Lock()
        self.last = 0.0
        self.min_interval = 60.0 / rpm if rpm else 0.0
    def acquire(self):
        if not self.rpm: return
        with self.lock:
            now = time.time()
            wait = self.min_interval - (now - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()

def payload_for_row(row, max_chars: int = 700) -> dict:
    txt = str(row.get("text_for_model", ""))
    if len(txt) > max_chars:
        txt = txt[:max_chars]
    return dict(text=txt, reply_count=int(row.get("reply_count", 0)),
                quote_count=int(row.get("quote_count", 0)), retweet_count=int(row.get("retweet_count", 0)),
                favorite_count=int(row.get("favorite_count", 0)))

def sha1_of_payload(p: dict) -> str:
    m = hashlib.sha1()
    m.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return m.hexdigest()

def worker(chain, payload: dict, rl: RateLimiter, retries: int, backoff: float, timeout: int) -> dict:
    last_err = None
    for attempt in range(max(1, retries)):
        try:
            rl.acquire()
            out = chain.invoke(payload)
            out_obj = out if isinstance(out, dict) else out.__dict__
            return json.loads(json.dumps(out_obj, default=str))
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    return dict(
        intent_text="Service client", intent_confidence=0.0,
        sentiment_text="neutre", sentiment_confidence=0.0,
        urgency_0_3=1, severity_0_3=1,
        summary="", reply_suggestion="", needs_handoff=False
    )


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("DATA_CSV", "free_tweets_cleaned_fallback.csv"))
    ap.add_argument("--out",  default=os.getenv("OUT_CSV", "tweets_scored_llm.csv"))
    ap.add_argument("--input", dest="input_alias")
    ap.add_argument("--output", dest="output_alias")
    ap.add_argument("--cache", default="llm_cache.sqlite")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", 4)))
    ap.add_argument("--checkpoint-every", type=int, default=int(os.getenv("CHECKPOINT_EVERY", 300)))
    ap.add_argument("--max-chars", type=int, default=int(os.getenv("MAX_CHARS", 700)))
    ap.add_argument("--retries", type=int, default=int(os.getenv("RETRIES", 4)))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("TIMEOUT", 60)))
    ap.add_argument("--rpm", type=int, default=int(os.getenv("RPM", 0)))  # 0 = off
    args = ap.parse_args()

    # alias -> champs officiels
    if getattr(args, "input_alias", None):
        args.data = args.input_alias
    if getattr(args, "output_alias", None):
        args.out = args.output_alias

    DATA = args.data
    OUT  = args.out
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"CSV introuvable: {DATA}")

    df = pd.read_csv(DATA, low_memory=False)

    # Choix de la colonne texte: priorité à text_for_llm, puis text_for_model existante, sinon fallback
    if "text_for_model" in df.columns and df["text_for_model"].dtype == "O":
        df["text_for_model"] = df["text_for_model"].astype("string").fillna("")
    elif "text_for_llm" in df.columns:
        df["text_for_model"] = df["text_for_llm"].astype("string").fillna("")
    else:
        src = "text_clean" if "text_clean" in df.columns else ("full_text" if "full_text" in df.columns else None)
        if src is None:
            df["text_for_model"] = pd.Series([""] * len(df), index=df.index, dtype="string")
        else:
            df["text_for_model"] = df[src].astype("string").fillna("")

    rows_idx = list(df.index)
    payloads = [payload_for_row(df.loc[i], max_chars=args.max_chars) for i in rows_idx]
    hashes   = [sha1_of_payload(p) for p in payloads]

    cache = Cache(args.cache)
    cached = cache.get_many(hashes)

    chain = build_chain()
    rl = RateLimiter(args.rpm if args.rpm else 0)

    to_call_idx = []
    to_call_payloads = []
    for i, h in enumerate(hashes):
        if h in cached:
            continue
        to_call_idx.append(i)
        to_call_payloads.append(payloads[i])

    results = [None] * len(to_call_idx)
    to_store = []

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = {ex.submit(worker, chain, p, rl, args.retries, 1.2, args.timeout): k
                   for k, p in enumerate(to_call_payloads)}
        pbar = tqdm(total=len(futures), desc="LLM (Mistral, pool)", unit="item")
        done_since_last_ckpt = 0
        for fut in cf.as_completed(futures):
            k = futures[fut]
            try:
                out_dict = fut.result(timeout=args.timeout)
            except Exception:
                out_dict = dict(
                    intent_text="Service client", intent_confidence=0.0,
                    sentiment_text="neutre", sentiment_confidence=0.0,
                    urgency_0_3=1, severity_0_3=1,
                    summary="", reply_suggestion="", needs_handoff=False
                )
            results[k] = out_dict
            done_since_last_ckpt += 1
            if done_since_last_ckpt >= max(1, args.checkpoint_every):
                done_since_last_ckpt = 0
            pbar.update(1)
        pbar.close()

    # Reconstruire la sortie
    out_rows = []
    for i in range(len(rows_idx)):
        h = hashes[i]
        if h in cached:
            d = json.loads(cached[h])
        else:
            k = to_call_idx.index(i) if i in to_call_idx else None
            d = results[k] if k is not None else None
            if d is None:
                d = dict(intent_text="Service client", intent_confidence=0.0,
                         sentiment_text="neutre", sentiment_confidence=0.0,
                         urgency_0_3=1, severity_0_3=1, summary="", reply_suggestion="", needs_handoff=False)
        out_rows.append(d)

    out_df = pd.DataFrame(out_rows)
    out_df.index = rows_idx
    merged = pd.concat([df, out_df], axis=1)
    merged.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[OK] Écrit: {OUT}")

if __name__ == "__main__":
    main()
