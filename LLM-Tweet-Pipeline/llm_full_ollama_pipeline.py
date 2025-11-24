#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline complet : RAG BERT + analyse/réponse Ollama, sortie CSV scoring complet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from requests.adapters import HTTPAdapter

DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"
FAST_OLLAMA_MODEL = "mistral:7b"
DEFAULT_OLLAMA_CACHE = "llm_cache_ollama.sqlite"
MAX_CHARS_TEXT = 700
REQUIRED_COLUMNS = [
    "tweet_id",
    "created_at_dt",
    "text_raw",
    "text_display",
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

thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        thread_local.session = session
    return thread_local.session


def default_kb_cache_path(kb_path: str, bert_model: str) -> str:
    base = Path(kb_path).resolve()
    safe_model = bert_model.replace("/", "_")
    cache_name = f"{base.stem}__{safe_model}_emb.pt"
    return str(base.with_name(cache_name))


def compute_file_signature(file_path: str) -> str:
    try:
        stat = os.stat(file_path)
        return f"{stat.st_size}-{int(stat.st_mtime)}"
    except OSError:
        return ""


def load_cached_embeddings(cache_path: str, kb_signature: str, bert_model: str) -> torch.Tensor | None:
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Cache KB illisible ({cache_path}): {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("kb_signature") != kb_signature or payload.get("bert_model") != bert_model:
        return None
    emb = payload.get("embeddings")
    if isinstance(emb, torch.Tensor):
        return emb
    return None


def save_cached_embeddings(
    cache_path: str,
    kb_signature: str,
    bert_model: str,
    embeddings: torch.Tensor,
) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(Path(cache_path).parent, exist_ok=True)
        torch.save(
            {
                "kb_signature": kb_signature,
                "bert_model": bert_model,
                "embeddings": embeddings.cpu(),
            },
            cache_path,
        )
        print(f"[INFO] Cache KB écrit -> {cache_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Impossible d'enregistrer le cache KB ({cache_path}): {exc}")


def resolve_ollama_cache_path(user_path: str | None, output_path: str | None) -> str | None:  # noqa: ARG001
    if user_path is not None:
        user_path = user_path.strip()
        if not user_path:
            return None
        return str(Path(user_path).expanduser().resolve())

    script_dir = Path(__file__).resolve().parent
    return str(script_dir / DEFAULT_OLLAMA_CACHE)


def hash_prompt(model: str, prompt: str) -> str:
    key = f"{model}|{prompt}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


class OllamaCache:
    def __init__(self, cache_path: str | None):
        self.path = cache_path
        self.lock = threading.Lock()
        self.conn: sqlite3.Connection | None = None
        if self.path:
            self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(Path(self.path).parent, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ollama_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ollama_cache_model ON ollama_cache(model)"
            )
            self.conn.commit()

    def get(self, prompt_hash: str, model: str) -> str | None:
        if not self.conn:
            return None
        with self.lock:
            cur = self.conn.execute(
                "SELECT raw_response FROM ollama_cache WHERE prompt_hash = ? AND model = ?",
                (prompt_hash, model),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def set(self, prompt_hash: str, model: str, raw_response: str) -> None:
        if not self.conn:
            return
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO ollama_cache(prompt_hash, model, raw_response) VALUES (?, ?, ?)",
                (prompt_hash, model, raw_response),
            )
            self.conn.commit()

    def close(self) -> None:
        if self.conn:
            with self.lock:
                self.conn.close()
            self.conn = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline complet : BERT RAG + Ollama JSON")
    parser.add_argument("--input", required=True, help="CSV d'entrée (tweets clients_only)")
    parser.add_argument("--output", required=True, help="CSV de sortie final")
    parser.add_argument("--kb", required=True, help="CSV base de connaissances")
    parser.add_argument("--model", default="mistral", help="Nom du modèle Ollama")
    parser.add_argument("--rag-top-k", type=int, default=3, help="Nombre d'extraits KB à injecter")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout HTTP Ollama (s)")
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre de lignes traitées (debug)")
    parser.add_argument("--bert-model", default=DEFAULT_MODEL_NAME, help="Modèle HF pour les embeddings BERT")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch pour l'encodage BERT")
    parser.add_argument(
        "--kb-cache",
        default=None,
        help="Chemin du cache embeddings KB (.pt). Laisser vide pour désactiver (auto si None).",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=512,
        help="Nombre max de tokens générés par Ollama (limite les réponses).",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default="10m",
        help="Durée keep_alive transmise à Ollama (ex: '5m', '1h').",
    )
    parser.add_argument(
        "--ollama-workers",
        type=int,
        default=2,
        help="Nombre de requêtes Ollama en parallèle (>=1).",
    )
    parser.add_argument(
        "--ollama-max-retries",
        type=int,
        default=1,
        help="Nombre de tentatives Ollama (0 = pas de retry).",
    )
    parser.add_argument(
        "--fast-ollama",
        action="store_true",
        help="Active le mode Fast Ollama (llama3.1:8b-instruct-q4_0 optimisé CPU).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Température pour Ollama (0-1).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) pour Ollama (0-1).",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.1,
        help="Pénalisation de répétition pour Ollama.",
    )
    parser.add_argument(
        "--ollama-cache-path",
        default=None,
        help=(
            "Chemin de la base SQLite pour mettre en cache les réponses Ollama. "
            "Laisser vide pour utiliser llm_cache_ollama.sqlite dans le dossier de sortie."
        ),
    )
    return parser.parse_args()


def pick_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "text_for_model",
        "text_for_llm",
        "text_clean",
        "text_display",
        "full_text",
        "text",
        "content",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    df["text_for_model"] = ""
    return "text_for_model"


def ensure_text_for_model(df: pd.DataFrame, base_col: str) -> pd.Series:
    series = df[base_col].astype(str).fillna("")
    truncated = series.apply(lambda txt: txt[:MAX_CHARS_TEXT])
    df["text_for_model"] = truncated
    return df["text_for_model"]


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            pooled = mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu())
    return torch.cat(embeddings, dim=0)


def compute_rag_indices(
    df_tweets: pd.DataFrame,
    kb_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    top_k: int,
    batch_size: int,
    kb_cache_path: str | None = None,
    kb_signature: str | None = None,
    bert_model_name: str | None = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    print("[INFO] Encodage tweets…")
    t0 = time.perf_counter()
    tweet_emb = encode_texts(tokenizer, model, df_tweets["text_for_model"].tolist(), device, batch_size)
    print(f"[INFO] Tweets encodés en {time.perf_counter() - t0:.2f}s")

    kb_emb = None
    kb_texts = [build_kb_text(row) for _, row in kb_df.iterrows()]
    if kb_cache_path and kb_signature and bert_model_name:
        kb_emb = load_cached_embeddings(kb_cache_path, kb_signature, bert_model_name)
        if kb_emb is not None:
            print(f"[INFO] Cache KB trouvé ({kb_cache_path}), saut de l'encodage.")

    if kb_emb is None:
        print("[INFO] Encodage KB…")
        t1 = time.perf_counter()
        kb_emb = encode_texts(tokenizer, model, kb_texts, device, batch_size)
        print(f"[INFO] KB encodée en {time.perf_counter() - t1:.2f}s")
        save_cached_embeddings(kb_cache_path or "", kb_signature or "", bert_model_name or "", kb_emb)

    print("[INFO] Calcul des similarités…")
    sims = tweet_emb @ kb_emb.T
    indices_list: List[List[int]] = []
    scores_list: List[List[float]] = []
    for row in sims:
        top = torch.topk(row, k=min(top_k, kb_emb.shape[0]))
        idxs = top.indices.tolist()
        scs = top.values.tolist()
        indices_list.append(idxs)
        scores_list.append(scs)
    return indices_list, scores_list


def build_kb_text(row: pd.Series) -> str:
    parts = []
    for col in ("intent", "subintent", "tags", "tone", "body", "reply"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts) if parts else str(row.to_dict())


def build_rag_context_from_indices(kb_df: pd.DataFrame, indices: List[int]) -> str:
    lines = []
    for pos, idx in enumerate(indices, start=1):
        if idx < 0 or idx >= len(kb_df):
            continue
        row = kb_df.iloc[idx]
        snippet_parts = []
        for col in ("intent", "subintent", "tags", "tone", "body", "reply"):
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                snippet_parts.append(val.strip())
        if snippet_parts:
            lines.append(f"[KB {pos}] {' '.join(snippet_parts)}")
    if not lines:
        return "[Aucun extrait KB disponible]"
    return "\n".join(lines)


def build_ollama_prompt(row: pd.Series, rag_context: str) -> str:
    raw_text = str(row.get("text_for_model", "") or "")
    raw_text = raw_text.replace('"', "'").replace("\n", " ").replace("\r", " ")
    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    tweet_id = row.get("tweet_id", "n/a")
    return f"""
Tu es un agent du service client Free qui analyse des tweets et propose une réponse adaptée et concise.

TWEET NORMALISÉ (id={tweet_id}) :
"{raw_text}"

CONTEXTE RAG (extraits de base de connaissances si utile) :
{rag_context}

RÈGLES DE SORTIE (DOIVENT TOUTES ÊTRE RESPECTÉES) :
1. Réponds STRICTEMENT avec UN SEUL objet JSON, sans texte avant ni après.
2. Interdiction d’utiliser ``` ou des balises Markdown : commence par {{ et termine par }}.
3. Ferme TOUJOURS correctement l’objet JSON. Si un champ texte n’a rien à dire, mets "" et termine proprement.
4. summary_1l = 1 phrase courte (≤ 140 caractères, ton clair).
5. llm_summary = 1 à 2 phrases (≤ 240 caractères) synthétisant le tweet + contexte.
6. llm_reply_suggestion = réponse style tweet (≤ 280 caractères, 1 à 3 phrases) personnalisée et cohérente.
7. llm_urgency_0_3 et llm_severity_0_3 sont des entiers 0, 1, 2 ou 3.
8. themes_list contient 1 à 5 thèmes maximum, mots/expressions courtes en français.
9. primary_label doit être UNE SEULE catégorie de la liste ci-dessous (n’utilise Autre/Indéterminé qu’en dernier recours).
10. sentiment_label ∈ {"positif", "neutre", "négatif"} et reflète vraiment le tweet.
11. status ∈ {"a_traiter", "clos"} selon l’action requise.
12. Varie la formulation de llm_reply_suggestion et adapte le ton (empathique si client en colère).
13. Ne commence pas toutes les réponses par la même formule : personnalise l’accroche.
14. Utilise les extraits RAG uniquement s’ils renforcent la réponse, sans inventer d’informations.
15. Toujours vérifier que la longueur totale reste courte et que chaque champ texte respecte les limites ci-dessus.

CATEGORIES POSSIBLES POUR "primary_label" :
- Incident/Actualité
- Insatisfaction/Colère
- Mobile/SIM/Portabilite
- Reseau/Internet
- Resiliation
- Securite/Fraude
- Support/SAV/Reclamation
- Annonce/Marketing
- Autre/Indéterminé
- Box/TV
- Commande/Livraison
- Commercial/Offre
- Compte/Acces
- Facturation

FORMAT JSON EXACT A PRODUIRE (EXEMPLE VALIDE) :

{{
    "themes_list": ["facturation", "prelevement"],
    "primary_label": "Facturation",
    "sentiment_label": "neutre",
    "llm_urgency_0_3": 1,
    "llm_severity_0_3": 1,
    "status": "a_traiter",
    "summary_1l": "Résumé très court en une phrase",
    "llm_summary": "Résumé plus détaillé en quelques phrases",
    "llm_reply_suggestion": "Réponse personnalisée à envoyer sur Twitter/X",
    "routing_team": "Equipe SAV",
    "assigned_to": "auto"
}}
"""

def _extract_first_json_block(text: str) -> str | None:
    """
    Extrait le premier bloc JSON équilibré ({...} ou [...]) du texte.
    Ignore les accolades à l'intérieur des chaînes de caractères.
    """
    cleaned = text.strip().replace("\r", "")
    # Nettoyage rapide d'éventuelles fences Markdown
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    if not cleaned:
        return None

    # On cherche soit un objet { }, soit un tableau [ ]
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        return None

    start = min(starts)
    opening = cleaned[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(cleaned[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if in_string:
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                # Premier bloc JSON complet trouvé
                return cleaned[start : i + 1]

def parse_ollama_response(raw_text: str) -> dict:
    """
    Essaie de récupérer un dict à partir du texte brut renvoyé par Ollama.
    Tolère les fences Markdown, du texte avant/après, plusieurs blocs JSON, etc.
    """
    if not isinstance(raw_text, str):
        raise RuntimeError("Réponse Ollama vide ou non textuelle")

    text = raw_text.strip().replace("\r", "")

    # 1) Tentative directe : le modèle a renvoyé un JSON propre
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass

    # 2) On extrait le premier bloc JSON équilibré ({...} ou [...])
    json_blob = _extract_first_json_block(text)
    if not json_blob:
        raise RuntimeError(
            f"Impossible d'extraire le JSON dans la réponse: {text[:200]}..."
        )

    # 3) On retente un json.loads sur ce bloc
    try:
        obj = json.loads(json_blob)
    except json.JSONDecodeError:
        # 4) Dernier recours : ast.literal_eval pour les JSON "presque Python"
        try:
            candidate_obj = ast.literal_eval(json_blob)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"JSON invalide (extrait: {json_blob[:200]}...)"
            ) from exc
        else:
            if isinstance(candidate_obj, dict):
                return candidate_obj
            if isinstance(candidate_obj, list) and candidate_obj and isinstance(candidate_obj[0], dict):
                return candidate_obj[0]
            raise RuntimeError("La réponse ne contient pas d'objet JSON de niveau racine.")
    else:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        raise RuntimeError("La réponse ne contient pas d'objet JSON de niveau racine.")


def parse_ollama_response(raw_text: str) -> dict:
    """
    Essaie de récupérer un dict à partir du texte brut renvoyé par Ollama.
    Tolère les fences Markdown, du texte avant/après, plusieurs blocs JSON, etc.
    """
    if not isinstance(raw_text, str):
        raise RuntimeError("Réponse Ollama vide ou non textuelle")

    text = raw_text.strip().replace("\r", "")

    # 1) Tentative directe : le modèle a renvoyé un JSON propre
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass

    # 2) On extrait le premier bloc JSON équilibré ({...} ou [...])
    json_blob = _extract_first_json_block(text)
    if not json_blob:
        raise RuntimeError(
            f"Impossible d'extraire le JSON dans la réponse: {text[:200]}..."
        )

    # 3) On retente un json.loads sur ce bloc
    try:
        obj = json.loads(json_blob)
    except json.JSONDecodeError:
        # 4) Dernier recours : ast.literal_eval pour les JSON "presque Python"
        try:
            candidate_obj = ast.literal_eval(json_blob)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"JSON invalide (extrait: {json_blob[:200]}...)"
            ) from exc
        else:
            if isinstance(candidate_obj, dict):
                return candidate_obj
            if isinstance(candidate_obj, list) and candidate_obj and isinstance(candidate_obj[0], dict):
                return candidate_obj[0]
            raise RuntimeError("La réponse ne contient pas d'objet JSON de niveau racine.")
    else:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        raise RuntimeError("La réponse ne contient pas d'objet JSON de niveau racine.")


def try_repair_truncated_json(inner_raw: str) -> dict | None:
    """Essaie de nettoyer puis de reconstituer un JSON tronqué renvoyé par le LLM."""
    import json
    import re

    if not isinstance(inner_raw, str):
        return None

    text = inner_raw.strip()
    if not text:
        return None

    # 1) Normalisation des guillemets et suppression des balises/bruit HTML ou Markdown.
    fancy_map = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "„": '"',
            "«": '"',
            "»": '"',
            "’": "'",
            "‘": "'",
            "‚": "'",
        }
    )
    text = text.translate(fancy_map).replace("\r", "")
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # 2) On ne garde que la portion supposée JSON entre la première { et la dernière }.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None
    candidate = text[first_brace : last_brace + 1]

    # 3) Réparations regex : clés en double quotes, valeurs simples, virgules orphelines.
    candidate = re.sub(r"'([A-Za-z0-9_]+)'\s*:", r'"\1":', candidate)

    def _fix_simple_string(match: re.Match[str]) -> str:
        inner = match.group(1)
        inner = inner.replace('"', '\\"')
        return ': "' + inner + '"'

    candidate = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", _fix_simple_string, candidate)
    candidate = re.sub(r",\s*(?=[}\]])", "", candidate)
    candidate = candidate.strip()

    # 4) Tentative de récupération du bloc JSON principal via regex (au cas où du texte traîne encore).
    block_match = re.search(r"\{[\s\S]*\}", candidate)
    if block_match:
        candidate = block_match.group(0)

    # 5) Rééquilibrage final des accolades, puis json.loads.
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if close_braces < open_braces:
        candidate += "}" * (open_braces - close_braces)

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        print(f"[FIX] Échec de réparation regex: {exc}")
        return None

    if isinstance(obj, dict):
        print("[FIX] JSON tronqué réparé via regex.")
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        print("[FIX] JSON tronqué réparé via regex (liste).")
        return obj[0]

    return None


def prediction_template() -> dict:
    """
    Gabarit de base pour une prédiction, sans log.
    Utilisé comme base dans normalize_prediction().
    """
    return {
        "themes_list": ["Autre/Indéterminé"],
        "primary_label": "Autre/Indéterminé",
        "sentiment_label": "neutre",
        "llm_urgency_0_3": 1,
        "llm_severity_0_3": 1,
        "status": "a_traiter",
        "summary_1l": "Résumé non généré automatiquement (fallback).",
        "author": "",
        "assigned_to": "auto",
        "llm_summary": (
            "Analyse détaillée non générée automatiquement par le LLM. "
            "Merci de traiter ce tweet manuellement."
        ),
        "llm_reply_suggestion": (
            "Merci pour votre message. Afin de vous aider, pouvez-vous nous écrire en "
            "message privé avec vos informations client (numéro de ligne / identifiant) ?"
        ),
        "routing_team": "Equipe SAV",
    }


def default_prediction() -> dict:
    """Prédiction de secours utilisée uniquement quand le LLM échoue."""
    print("[WARN] Utilisation de default_prediction() — Ollama a échoué ou réponse invalide")
    return prediction_template()


def build_output_row(row: pd.Series, pred: dict) -> dict:
    tweet_id = row.get("tweet_id") or row.get("id") or row.get("id_str") or str(row.name)
    created_at_dt = row.get("created_at_dt")
    if pd.isna(created_at_dt) and row.get("created_at") is not None:
        created_at_dt = pd.to_datetime(row.get("created_at"), errors="coerce")
    author = row.get("screen_name") or row.get("user") or "inconnu"
    text_display = (
        row.get("text_for_llm")
        or row.get("text_clean")
        or row.get("text")
        or row.get("text_for_model")
        or ""
    )
    text_raw = (
        row.get("text_raw")
        or row.get("full_text")
        or row.get("text")
        or ""
    )
    themes_list = ensure_theme_list(pred.get("themes_list", []))
    primary_label = pred.get("primary_label", "")
    if primary_label == "Autre/Indéterminé":
        hints = list(themes_list)
        if isinstance(text_display, str) and text_display.strip():
            hints.append(text_display)
        better_label = canonical_primary_label("", hints)
        if better_label and better_label != "Autre/Indéterminé":
            primary_label = better_label
            themes_list = [better_label] + [t for t in themes_list if t != better_label]
    if not themes_list and primary_label:
        themes_list = [primary_label]

    return {
        "tweet_id": tweet_id,
        "created_at_dt": created_at_dt,
        "text_raw": text_raw,
        "text_display": text_display,
        "themes_list": safe_themes_value(themes_list),
        "primary_label": primary_label,
        "sentiment_label": pred.get("sentiment_label", "neutre"),
        "llm_urgency_0_3": clamp_llm_score(pred.get("llm_urgency_0_3", 1)),
        "llm_severity_0_3": clamp_llm_score(pred.get("llm_severity_0_3", 1)),
        "status": pred.get("status", "a_traiter"),
        "summary_1l": pred.get("summary_1l", ""),
        "author": author,
        "assigned_to": pred.get("assigned_to", ""),
        "llm_summary": pred.get("llm_summary", ""),
        "llm_reply_suggestion": pred.get("llm_reply_suggestion", ""),
        "routing_team": pred.get("routing_team", ""),
    }


def call_ollama_json(
    prompt: str,
    model: str,
    timeout: int,
    num_predict: int | None = None,
    keep_alive: str | None = None,
    temperature: float = 0.4,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    session: requests.Session | None = None,
) -> Tuple[dict, str]:
    """
    Appelle Ollama et renvoie un dictionnaire Python décodé à partir du JSON.
    Corrige le cas où Ollama encapsule le JSON complet dans le champ "response"
    sous forme de texte.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = base_url.rstrip("/") + "/api/generate"

    effective_num_predict = num_predict if (num_predict and num_predict > 0) else 200
    options = {
        "temperature": max(0.0, temperature if temperature is not None else 0.4),
        "top_p": max(0.0, min(1.0, top_p if top_p is not None else 0.95)),
        "repeat_penalty": max(0.0, repeat_penalty if repeat_penalty is not None else 1.1),
        "num_predict": effective_num_predict,
    }

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if keep_alive:
        payload["keep_alive"] = keep_alive

    sess = session or requests.Session()
    try:
        resp = sess.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERREUR] Appel Ollama échoué: {exc}")
        return default_prediction(), ""

    try:
        data = resp.json()
    except ValueError as exc:  # noqa: B904
        print(f"[ERREUR] Réponse non JSON: {exc}\nTexte brut: {resp.text[:400]}")
        return default_prediction(), resp.text[:400]

    inner_raw = data.get("response", "")
    if not inner_raw:
        print("[WARN] Champ 'response' vide dans la réponse Ollama.")
        return default_prediction(), ""

    try:
        parsed = parse_ollama_response(inner_raw)
        if isinstance(parsed, dict) and "data" in parsed:
            parsed = parsed["data"]
        return parsed, inner_raw
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ERREUR] JSON interne invalide: {exc}\n--- Début du texte ---\n"
            f"{inner_raw[:400]}\n--- Fin ---"
        )
        repaired = try_repair_truncated_json(inner_raw)
        if repaired is not None:
            return repaired, inner_raw
        return default_prediction(), inner_raw


def clamp_llm_score(value: Any, default: int = 1) -> int:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = float(default)
    val = max(0.0, min(3.0, val))
    return int(round(val))


def ensure_theme_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except json.JSONDecodeError:
                pass
        return [s.strip() for s in stripped.split(",") if s.strip()]
    return []


PRIMARY_LABELS = [
    "Incident/Actualité",
    "Insatisfaction/Colère",
    "Mobile/SIM/Portabilite",
    "Reseau/Internet",
    "Resiliation",
    "Securite/Fraude",
    "Support/SAV/Reclamation",
    "Annonce/Marketing",
    "Autre/Indéterminé",
    "Box/TV",
    "Commande/Livraison",
    "Commercial/Offre",
    "Compte/Acces",
    "Facturation",
]

PRIMARY_PATTERNS: list[tuple[str, str]] = [
    ("incident", "Incident/Actualité"),
    ("actualite", "Incident/Actualité"),
    ("actus", "Incident/Actualité"),

    ("insatisfaction", "Insatisfaction/Colère"),
    ("colere", "Insatisfaction/Colère"),
    ("mécontent", "Insatisfaction/Colère"),
    ("mecontent", "Insatisfaction/Colère"),

    ("mobile", "Mobile/SIM/Portabilite"),
    ("sim", "Mobile/SIM/Portabilite"),
    ("portabilite", "Mobile/SIM/Portabilite"),
    ("portabilité", "Mobile/SIM/Portabilite"),

    ("reseauinternet", "Reseau/Internet"),
    ("reseau", "Reseau/Internet"),
    ("réseau", "Reseau/Internet"),
    ("internet", "Reseau/Internet"),

    ("resiliation", "Resiliation"),
    ("résiliation", "Resiliation"),
    ("resili", "Resiliation"),

    ("securite", "Securite/Fraude"),
    ("sécurité", "Securite/Fraude"),
    ("fraude", "Securite/Fraude"),

    ("sav", "Support/SAV/Reclamation"),
    ("reclamation", "Support/SAV/Reclamation"),
    ("réclamation", "Support/SAV/Reclamation"),
    ("support", "Support/SAV/Reclamation"),

    ("annonce", "Annonce/Marketing"),
    ("marketing", "Annonce/Marketing"),
    ("pub", "Annonce/Marketing"),
    ("publicite", "Annonce/Marketing"),

    ("box", "Box/TV"),
    ("tv", "Box/TV"),
    ("decodeur", "Box/TV"),

    ("commande", "Commande/Livraison"),
    ("livraison", "Commande/Livraison"),
    ("colis", "Commande/Livraison"),

    ("commercial", "Commercial/Offre"),
    ("offre", "Commercial/Offre"),
    ("promotion", "Commercial/Offre"),

    ("compte", "Compte/Acces"),
    ("acces", "Compte/Acces"),
    ("identifiant", "Compte/Acces"),
    ("connexion", "Compte/Acces"),

    ("facturation", "Facturation"),
    ("facture", "Facturation"),
    ("paiement", "Facturation"),
    ("prelevement", "Facturation"),
]

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


PRIMARY_LABELS_MAP: dict[str, str] = {}
for lbl in PRIMARY_LABELS:
    norm = strip_accents(lbl).lower().replace(" ", "")
    PRIMARY_LABELS_MAP[norm] = lbl


def canonical_primary_label(label: str, hints: Sequence[str] | None = None) -> str:
    """
    Normalise le label fourni par le LLM vers l'une des valeurs de PRIMARY_LABELS.
    - label : ce que renvoie le LLM ("Facturation", "facture", "Reseau internet", etc.)
    - hints : liste d'indices (themes_list, texte, etc.) pour aider à deviner.
    """
    if label:
        normalized = strip_accents(label).lower().replace(" ", "")
        if normalized in PRIMARY_LABELS_MAP:
            return PRIMARY_LABELS_MAP[normalized]

    if hints:
        joined = " ".join(str(h) for h in hints)
        norm_hint = strip_accents(joined).lower().replace(" ", "")
        for pattern, canon in PRIMARY_PATTERNS:
            if pattern in norm_hint:
                return canon

    if label:
        norm_label = strip_accents(label).lower().replace(" ", "")
        for pattern, canon in PRIMARY_PATTERNS:
            if pattern in norm_label:
                return canon

    return "Autre/Indéterminé"


def guess_label_from_text_and_themes(text: str, themes: list[str] | None) -> str:
    """
    Essaie de deviner un primary_label à partir du texte du tweet et de themes_list,
    en utilisant des règles simples (mots-clés).
    Retourne un label parmi PRIMARY_LABELS ou 'Autre/Indéterminé' si rien ne matche.
    """
    if themes is None:
        themes = []

    corpus = (text or "") + " " + " ".join(map(str, themes))
    corpus_norm = strip_accents(corpus).lower()

    rules: list[tuple[list[str], str]] = [
        (["facture", "facturation", "prelevement", "prélèvement", "paiement"], "Facturation"),
        (["box", "freebox", "player", "tv", "decodeur", "décodeur"], "Box/TV"),
        (["internet", "fibre", "adsl", "wifi", "wi-fi", "connexion"], "Reseau/Internet"),
        (["reseau", "réseau", "4g", "5g", "antenne"], "Reseau/Internet"),
        (["mobile", "sim", "carte sim", "portabilite", "portabilité"], "Mobile/SIM/Portabilite"),
        (["compte", "identifiant", "mdp", "mot de passe", "acces", "accès", "espace client"], "Compte/Acces"),
        (["resiliation", "résiliation", "resilier", "quitter", "résilier"], "Resiliation"),
        (["arnaque", "fraude", "pirate", "piratage", "usurpation"], "Securite/Fraude"),
        (["sav", "reclamation", "réclamation", "support"], "Support/SAV/Reclamation"),
        (["colis", "livraison", "commande"], "Commande/Livraison"),
        (["pub", "publicite", "publicité", "marketing", "promo"], "Annonce/Marketing"),
        (["offre", "abonnement", "forfait"], "Commercial/Offre"),
        (["hors sujet", "politique", "actualite", "actualité"], "Incident/Actualité"),
        (["marre", "ras le bol", "scandale", "honteux"], "Insatisfaction/Colère"),
    ]

    for keywords, label in rules:
        for kw in keywords:
            if kw in corpus_norm:
                return label

    return "Autre/Indéterminé"


def post_fix_prediction_with_rules(pred: dict, text_display: str) -> dict:
    """
    Applique une 2e passe de correction quand primary_label est Autre/Indéterminé
    ou quand le summary_1l est un fallback.
    """
    if not isinstance(pred, dict):
        return pred

    primary = pred.get("primary_label", "")
    themes = pred.get("themes_list", []) or []

    summary = str(pred.get("summary_1l", "")).strip()
    is_fallback_summary = summary == "Résumé non généré automatiquement (fallback)."

    if primary == "Autre/Indéterminé" or is_fallback_summary:
        guessed = guess_label_from_text_and_themes(text_display, themes)
        if guessed != "Autre/Indéterminé":
            pred["primary_label"] = guessed
            if not themes or (len(themes) == 1 and "Autre/Indéterminé" in themes[0]):
                pred["themes_list"] = [guessed]

    return pred


def normalize_prediction(raw: dict | None) -> dict:
    base = prediction_template()
    if isinstance(raw, dict):
        for key, value in raw.items():
            if value is not None:
                base[key] = value
    base["themes_list"] = ensure_theme_list(base.get("themes_list"))
    canonical = canonical_primary_label(base.get("primary_label", ""), base["themes_list"])
    base["primary_label"] = canonical
    normalized_themes: list[str] = []
    if canonical:
        normalized_themes.append(canonical)
    for theme in base["themes_list"]:
        theme_str = str(theme).strip()
        if not theme_str:
            continue
        if theme_str == canonical and canonical in normalized_themes:
            continue
        if theme_str not in normalized_themes:
            normalized_themes.append(theme_str)

    # Si on a déjà au moins un thème spécifique, on enlève "Autre/Indéterminé"
    if len(normalized_themes) > 1 and "Autre/Indéterminé" in normalized_themes:
        normalized_themes = [t for t in normalized_themes if t != "Autre/Indéterminé"]

    if not normalized_themes:
        normalized_themes = [canonical or "Autre/Indéterminé"]
    base["themes_list"] = normalized_themes
    base["sentiment_label"] = (
        str(base.get("sentiment_label", "neutre") or "neutre").strip() or "neutre"
    )
    base["status"] = (
        str(base.get("status", "a_traiter") or "a_traiter").strip() or "a_traiter"
    )
    base["summary_1l"] = str(base.get("summary_1l", "") or "")
    base["llm_summary"] = str(base.get("llm_summary", "") or "")
    base["llm_reply_suggestion"] = str(base.get("llm_reply_suggestion", "") or "")
    base["routing_team"] = str(base.get("routing_team", "") or "")
    base["assigned_to"] = str(base.get("assigned_to", "") or "")
    base["llm_urgency_0_3"] = clamp_llm_score(base.get("llm_urgency_0_3", 1))
    base["llm_severity_0_3"] = clamp_llm_score(base.get("llm_severity_0_3", 1))
    return base


def sanitize_llm_output(pred: dict | None) -> dict:
    """
    Nettoie les sorties de l'LLM Ollama :
    - Corrige les labels ambigus
    - Vide les textes de fallback génériques
    - Tronque les textes trop longs
    """
    if pred is None:
        return prediction_template()
    if not isinstance(pred, dict):
        return prediction_template()

    allowed_labels = set(PRIMARY_LABELS)
    raw_label = str(pred.get("primary_label", "")).strip()
    if "/" in raw_label and raw_label not in allowed_labels:
        raw_label = raw_label.split("/")[0].strip()
        pred["primary_label"] = raw_label

    for key in ["summary_1l", "llm_summary", "llm_reply_suggestion"]:
        # Si la clé n'existe pas dans la prédiction brute, on ne touche à rien :
        # prediction_template() fournira une valeur par défaut plus tard.
        if key not in pred:
            continue

        val = str(pred.get(key, "")).strip()

        # Si le LLM a explicitement renvoyé une chaîne vide, on la garde vide.
        if not val:
            pred[key] = ""
            continue

        val_lower = val.lower()

        # Pour les résumés, on enlève explicitement les textes de fallback
        # générés éventuellement par le modèle, mais on ne touche pas
        # à la suggestion de réponse.
        if key in ("summary_1l", "llm_summary") and "(fallback)" in val_lower:
            pred[key] = ""
            continue

        # On limite la taille pour éviter des JSON énormes
        if len(val) > 600:
            pred[key] = val[:600] + "..."
            continue

        pred[key] = val

    return pred


def infer_label_from_themes(pred: dict) -> dict:
    label = pred.get("primary_label", "")
    if label not in ("", None, "Autre/Indéterminé"):
        return pred

    themes = pred.get("themes_list", []) or []
    text_content = f"{pred.get('summary_1l', '')} {' '.join(map(str, themes))}"
    norm = strip_accents(text_content).lower()

    rules = [
        (["fibre", "panne", "debit"], "Reseau/Internet"),
        (["technicien", "rdv"], "Support/SAV/Reclamation"),
        (["freebox", "delta", "tv"], "Box/TV"),
        (["fibre", "adsl", "wifi", "debit", "internet", "panne", "coupure", "reseau", "hs"], "Reseau/Internet"),
        (["box", "freebox", "player", "delta", "pop", "ultra", "tv", "telecommande"], "Box/TV"),
        (["mobile", "sim", "portabilite", "4g", "5g", "esim", "capte pas"], "Mobile/SIM/Portabilite"),
        (["facture", "prelevement", "hors forfait", "paiement", "remboursement"], "Facturation"),
        (["resiliation", "resilier", "abonnement", "partir"], "Resiliation"),
        (["technicien", "rdv", "rendez-vous", "ticket", "assistance", "1044", "3244"], "Support/SAV/Reclamation"),
        (["commande", "livraison", "ups", "chronopost", "reçu"], "Commande/Livraison"),
        (["offre", "promo", "vente privee"], "Commercial/Offre"),
        (["identifiant", "mot de passe", "compte", "acces", "mail"], "Compte/Acces"),
        (["arnaque", "fraude", "phishing", "piratage"], "Securite/Fraude"),
        (["colere", "honte", "scandale", "inadmissible", "marre", "merde"], "Insatisfaction/Colère"),
    ]

    for keywords, new_label in rules:
        if any(k in norm for k in keywords):
            pred["primary_label"] = new_label
            if themes:
                pred["themes_list"] = [t for t in themes if t != "Autre/Indéterminé"]
            pred.setdefault("themes_list", []).append(new_label)
            break

    return pred


def finalize_prediction(raw_pred: dict | None) -> dict:
    # 1) Nettoyage de la sortie brute de l'LLM (ou du default_prediction)
    pred = sanitize_llm_output(raw_pred)
    # 2) Fusion avec le gabarit (valeurs par défaut)
    pred = normalize_prediction(pred)
    # 3) Deuxième passe de nettoyage pour enlever d'éventuels fallback réintroduits
    pred = sanitize_llm_output(pred)
    # 4) Ajustement automatique du primary_label à partir des thèmes, si besoin
    pred = infer_label_from_themes(pred)
    return pred


def call_ollama_with_retry(
    prompt: str,
    model: str,
    timeout: int,
    num_predict: int | None,
    keep_alive: str | None,
    max_retries: int,
    temperature: float = 0.4,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    cache: OllamaCache | None = None,
    context: str | None = None,
) -> dict:
    prompt_hash = hash_prompt(model, prompt)

    if cache:
        cached_raw = cache.get(prompt_hash, model)
        if cached_raw is not None:
            try:
                cached_pred = parse_ollama_response(cached_raw)
                return finalize_prediction(cached_pred)
            except Exception as exc:  # noqa: BLE001
                label = context or prompt_hash
                print(f"[WARN] Cache Ollama invalide ({label}): {exc}. Recalcul en cours…")

    attempts = max(0, max_retries) + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            prediction, raw_response = call_ollama_json(
                prompt,
                model,
                timeout,
                num_predict=num_predict,
                keep_alive=keep_alive,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                session=_get_session(),
            )
            if cache and raw_response:
                cache.set(prompt_hash, model, raw_response)
            return finalize_prediction(prediction)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                wait_time = min(8.0, 1.5 * attempt)
                label = context or "tweet"
                print(
                    f"[WARN] Ollama tentative {attempt}/{attempts} échouée ({label}): {exc}. "
                    f"Nouveau test dans {wait_time:.1f}s"
                )
                time.sleep(wait_time)
    if last_error:
        label = context or "tweet"
        print(f"[WARN] Ollama échec définitif ({label}) après {attempts} tentatives: {last_error}")
    return finalize_prediction(None)


def safe_themes_value(value) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value
    return "[]"


def main() -> None:
    args = parse_args()

    if args.fast_ollama:
        args.model = FAST_OLLAMA_MODEL
        print(f"[INFO] Mode Fast Ollama actif -> modèle '{args.model}'")

    df = pd.read_csv(args.input, low_memory=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    kb_df = pd.read_csv(args.kb, low_memory=False)

    text_col = pick_text_col(df)
    ensure_text_for_model(df, text_col)

    cache_path = resolve_ollama_cache_path(args.ollama_cache_path, args.output)
    ollama_cache = OllamaCache(cache_path)
    if cache_path:
        print(f"[INFO] Cache Ollama SQLite -> {cache_path}")

    rag_enabled = args.rag_top_k and args.rag_top_k > 0 and not df.empty
    if rag_enabled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        model = AutoModel.from_pretrained(args.bert_model).to(device)
        if args.kb_cache is None:
            kb_cache_path = default_kb_cache_path(args.kb, args.bert_model)
        else:
            kb_cache_path = args.kb_cache.strip() or None
        kb_signature = compute_file_signature(args.kb)

        indices_list, _ = compute_rag_indices(
            df,
            kb_df,
            tokenizer,
            model,
            device,
            args.rag_top_k,
            args.batch_size,
            kb_cache_path=kb_cache_path,
            kb_signature=kb_signature,
            bert_model_name=args.bert_model,
        )
        df["rag_context"] = [build_rag_context_from_indices(kb_df, idxs) for idxs in indices_list]
    else:
        if args.rag_top_k <= 0:
            print("[INFO] RAG désactivé (--rag-top-k=0), saut de l'encodage BERT.")
        df["rag_context"] = ["[RAG désactivé]" for _ in range(len(df))]

    def process_row(idx_row: int) -> tuple[int, dict]:
        row = df.iloc[idx_row]
        rag_context = row.get("rag_context") or "[Aucun extrait KB disponible]"
        prompt = build_ollama_prompt(row, rag_context)
        tweet_id = row.get("tweet_id") or row.get("id") or row.get("id_str") or str(idx_row)
        context_label = f"idx={idx_row} tweet_id={tweet_id}"
        pred = call_ollama_with_retry(
            prompt,
            args.model,
            args.timeout,
            num_predict=args.ollama_num_predict,
            keep_alive=args.ollama_keep_alive or None,
            max_retries=args.ollama_max_retries,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            cache=ollama_cache,
            context=context_label,
        )
        return idx_row, build_output_row(row, pred)

    rows_buffer: List[dict | None] = [None] * len(df)
    max_workers = max(1, args.ollama_workers)
    print(f"[INFO] Appels Ollama… (threads={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, idx): idx for idx in range(len(df))}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Ollama", unit="tweet"):
            idx_hint = futures[fut]
            tweet_row = df.iloc[idx_hint]
            tweet_id = tweet_row.get("tweet_id") or tweet_row.get("id") or tweet_row.get("id_str") or idx_hint
            try:
                idx_row, out_row = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Traitement tweet idx={idx_hint} tweet_id={tweet_id} échoué: {exc}")
                fallback_pred = finalize_prediction(None)
                rows_buffer[idx_hint] = build_output_row(tweet_row, fallback_pred)
                continue
            rows_buffer[idx_row] = out_row

    rows = [row for row in rows_buffer if row is not None]

    if rows:
        df_final = pd.DataFrame(rows)
    else:
        df_final = pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df_final.columns:
            default_value: Any = ""
            if col in {"llm_urgency_0_3", "llm_severity_0_3"}:
                default_value = 1
            df_final[col] = default_value

    df_final = df_final[REQUIRED_COLUMNS]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_final.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[OK] Écrit : {args.output}")

    ollama_cache.close()


if __name__ == "__main__":
    main()
