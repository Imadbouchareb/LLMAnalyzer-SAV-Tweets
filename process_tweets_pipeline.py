# process_tweets_pipeline.py
# ------------------------------------------------------------
# Usage:
#   python process_tweets_pipeline.py free_tweet_export.csv
#
# Sorties:
#   - tweets_cleaned.parquet / .csv        (dédup: Clients uniquement)
#   - tweets_cleaned_clients_only.parquet / .csv   (seulement Clients)
#
# Dépendances conseillées:
#   pip install pandas numpy emoji langdetect ftfy python-dateutil pyarrow
# ------------------------------------------------------------

import sys
import os
import re
import unicodedata
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Libs optionnelles
try:
    import emoji
except Exception:
    emoji = None

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
except Exception:
    detect = None

try:
    import ftfy
except Exception:
    ftfy = None


# ---------------------------
# Helpers généraux
# ---------------------------

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def safe_lower(s: str) -> str:
    return s.lower() if isinstance(s, str) else s

def fix_text(s: str) -> str:
    """Normalise le texte en préservant la structure.
    - Convertit les séquences échappées \\n / \\r\\n en vrais retours
    - Normalise espaces, limite les \n consécutifs
    """
    if not isinstance(s, str):
        return s
    if ftfy is not None:
        try:
            s = ftfy.fix_text(s)
        except Exception:
            pass
    s = s.replace('\\r\\n', '\n').replace('\\n', '\n')  # séquences échappées -> vrais \n
    s = s.replace('\r', '\n')                           # normalise CR
    s = s.replace('\t', ' ')
    s = re.sub(r'[ \u00A0]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)                    # au plus 2 \n d'affilée
    return s.strip()

def detect_lang(text: str) -> str:
    if not detect or not isinstance(text, str) or not text.strip():
        return ''
    try:
        return detect(text)
    except Exception:
        return ''

def try_parse_date(s):
    if not isinstance(s, str):
        return None
    for fmt in (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None


# ---------------------------
# Colonnes candidates
# ---------------------------
TEXT_CANDIDATES = ["full_text", "text", "content", "tweet", "body", "message", "Text", "TEXT"]
USER_CANDIDATES = ["user_screen_name", "screen_name", "username", "user", "author", "name"]
ID_CANDIDATES   = ["id_str", "tweet_id", "id", "status_id"]
DATE_CANDIDATES = ["created_at", "date", "timestamp", "time"]


# ---------------------------
# Regex & constantes (cleaning)
# ---------------------------
URL_RE         = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
HASHTAG_RE     = re.compile(r'#(\w{1,100})', flags=re.UNICODE)
RT_PREFIX_RE   = re.compile(r'^\s*RT\s+@[\w_]+:\s*', flags=re.IGNORECASE)
ELLONG_RE      = re.compile(r'([a-zA-Zàâçéèêëîïôûùüÿñæœ])\1{2,}')
PUNCT_SPACE_RE = re.compile(r'\s+')

BRAND_WHITELIST = {"free", "freebox", "freemobile", "iliad", "fingapp", "outagedetect"}

MENTION_RE = re.compile(r'(^|[^@\w])@([A-Za-z0-9_]{1,15})', flags=re.UNICODE)
def mention_to_token(m):
    prefix, handle = m.group(1), m.group(2)
    low = handle.lower()
    if low in BRAND_WHITELIST:
        return f"{prefix}<BRAND:{low}>"
    return f"{prefix}<USER>"

ABBREV_MAP = {
    "svp":"s'il vous plaît","stp":"s'il te plaît","pls":"s'il te plaît",
    "cdt":"cordialement","ps":"post-scriptum",
    "qd":"quand","tt":"tout","bcp":"beaucoup",
    "qlq":"quelques","qqch":"quelque chose","qqn":"quelqu'un",
    "cad":"c'est-à-dire","aka":"également connu sous le nom de",
    "asap":"au plus vite","imho":"selon moi",
    "tjrs":"toujours","tjr":"toujours","tjs":"toujours",
    "mdr":"mort de rire","ptdr":"explosé de rire","lol":"mort de rire",
    "rip":"repose en paix","gg":"bien joué","wtf":"c'est quoi ce délire",
    "pb":"problème","prb":"problème","ras":"rien à signaler",
    "rdv":"rendez-vous","sav":"service après-vente","faq":"foire aux questions",
    "fw":"firmware","maj":"mise à jour","os":"système d'exploitation",
    "apk":"package android","vpn":"réseau privé virtuel",
    "sms":"message texte","mms":"message multimédia",
    "dl":"téléchargement","ul":"téléversement",
    "dsl":"désolé","deso":"désolé","slt":"salut","bjr":"bonjour","bsr":"bonsoir",
}
ABBREV_REGEX_RULES = [
    (re.compile(r"\b(pck|pcq|psk|parsk|pqk)\b", re.IGNORECASE), "parce que"),
    (re.compile(r"\b(pk|pq)\b", re.IGNORECASE), "pourquoi"),
    (re.compile(r"\bpr\b", re.IGNORECASE), "pour"),
    (re.compile(r"\bpdt\b", re.IGNORECASE), "pendant"),
    (re.compile(r"\btj\b", re.IGNORECASE), "toujours"),
    (re.compile(r"\bjms\b", re.IGNORECASE), "jamais"),
]
_ABBR_KEYS = sorted(map(re.escape, ABBREV_MAP.keys()), key=len, reverse=True)
ABBREV_TOKEN_RE = re.compile(r'\b(' + '|'.join(_ABBR_KEYS) + r')\b', flags=re.IGNORECASE) if _ABBR_KEYS else None


# ---------------------------
# Lexiques (émotions/urgence/intents)
# ---------------------------
def build_word_re(terms):
    esc = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
    pat = r'(?<!\w)(' + r'|'.join(esc) + r')(?!\w)'
    return re.compile(pat, flags=re.IGNORECASE)

EMO_LEX = {
    "anger":{"colere","furieux","enerve","marre","scandale","honte","abus","arnaque","inadmissible","rage","mecontent","nul","pourri"},
    "sadness":{"triste","decu","deception","dommage","desole","malheureux"},
    "joy":{"merci","bravo","top","parfait","genial","super","heureux","content","nickel"},
    "fear":{"peur","inquiet","inquietude","anxieux","dangereux","risque"},
    "surprise":{"surpris","incroyable","hallucinant","etonner","wow","wtf"},
    "trust":{"confiance","fiable","sur","garantie","satisfaction"},
    "disgust":{"degoute","ecoeure","beurk","merde"},
    "anticipation":{"bientot","attente","hate","impatient"},
}
URGENCY_TERMS = {
    "urgent","urgence","immédiat","immediat","vite","rapidement","bloque","bloquee","panne","hs","hors service",
    "impossible","aucun reseau","pas de reseau","internet coupe","ligne coupee","depuis","depuis hier","depuis 2 jours","sos"
}
INTENT_RULES = {
    "Facturation":{"facture","prelevement","paiement","montant","remboursement","surfacturation","forfait","abonnement","echeance"},
    "Reseau/Internet":{"reseau","4g","5g","h+","internet","wifi","fibre","adsl","debit","ping","latence","coupure","panne","antenne"},
    "Mobile/SIM/Portabilite":{"sim","carte sim","esim","portabilite","activation","desimlock","puk","imei"},
    "Box/TV":{"freebox","box","tv","player","chaine","reboot","firmware","redemarrer","oqee", "replay", "enregistrer", "télécommande","l'appli","appli","l'application","application"},
    "Commande/Livraison":{"commande","livraison","colis","relais","retard","suivi","expedition"},
    "Resiliation":{"resiliation","resilier","rompre","arreter","arret"},
    "Support/SAV/Reclamation":{"sav","reclamation","plainte","ticket","dossier","assistance","service client","hotline","chat", "joindre", "contacter", "réponse", "conseiller", "messagerie", "zimbra"},
    "Commercial/Offre":{"offre","promo","promotion","remise","prix","option","serie","eligibilite","eligible"},
    "Compte/Acces":{"compte","identifiant","mot de passe","mdp","connexion","espace client","login"},
    "Annonce/Marketing": {"partenaire", "partenariat", "annonce", "officiel", "communiqué", "gratuit", "jeuconcours", "concours", "gagner", "participer", "tirage au sort", "Ligue 1"},
    "Insatisfaction/Colère": {"pire", "honte", "scandaleux", "incompétent", "rendez fou", "marre", "jamais de reponse", "catastrophique", "lamentable"},
    "Incident/Actualité": {"cyberattaque", "panne nationale", "données personnelles", "fuite de données", "alerte", "données bancaires", "piratage"},
    "Securite/Fraude":{"fraude","phishing","hameconnage","arnaque","piratage","pirate"},
}
EMO_RE     = {k: build_word_re(v) for k, v in EMO_LEX.items()}
URGENCY_RE = build_word_re(URGENCY_TERMS)
INTENT_RE  = {k: build_word_re(v) for k, v in INTENT_RULES.items()}


# ---------------------------
# Nettoyage/normalisation
# ---------------------------

def normalize_for_match(s: str) -> str:
    s = safe_lower(s)
    s = strip_accents(s)
    s = re.sub(r'[^\w#@]+', ' ', s, flags=re.UNICODE).strip()
    s = PUNCT_SPACE_RE.sub(' ', s)
    return s

def demap_structure(s: str) -> str:
    """Convertit nos marqueurs en vrais retours à la ligne pour le LLM."""
    return s.replace(' <PARA> ', '\n\n').replace(' <NL> ', '\n')

def clean_tweet_text(text: str):
    """
    text_clean: texte pour règles/analyses, avec sauts de ligne mappés:
      - '\n\n' -> ' <PARA> '
      - '\n'   -> ' <NL> '
    Mentions brand-aware, URLs -> <URL>, compaction lettres.
    Retourne (text_clean, text_for_llm) où text_for_llm contient de vrais \n.
    """
    if not isinstance(text, str):
        return "", ""

    raw = fix_text(text)

    tmp = RT_PREFIX_RE.sub('', raw)                 # retire "RT @user:"
    tmp = URL_RE.sub('<URL>', tmp)                  # URLs -> <URL>
    tmp = MENTION_RE.sub(mention_to_token, tmp)     # @brand / @user
    tmp = ELLONG_RE.sub(r'\1\1', tmp)               # loooong letters -> 2

    # Abréviations
    if ABBREV_TOKEN_RE:
        tmp = ABBREV_TOKEN_RE.sub(lambda m: ABBREV_MAP.get(m.group(0).lower(), m.group(0)), tmp)
    for rgx, repl in ABBREV_REGEX_RULES:
        tmp = rgx.sub(repl, tmp)

    # Mapping des sauts de ligne en jetons + espaces propres
    tmp = tmp.replace('\n\n', ' <PARA> ').replace('\n', ' <NL> ')
    tmp = re.sub(r'[ ]{2,}', ' ', tmp).strip()

    text_clean   = tmp
    text_for_llm = demap_structure(text_clean)  # vrais \n pour le LLM

    # (Optionnel) alias d'emojis
    if emoji is not None:
        try:
            found = emoji.emoji_list(text_for_llm)
            if found:
                aliases = []
                for item in found[:6]:
                    ch = item.get('emoji')
                    if ch:
                        aliases.append(emoji.demojize(ch))
                aliases = list(dict.fromkeys(aliases))
                if aliases:
                    text_for_llm = f"{text_for_llm} [emoji: {' '.join(aliases)}]"
        except Exception:
            pass

    return text_clean, text_for_llm


# ---------------------------
# Extraction d'entités & features
# ---------------------------

def count_emojis(text: str) -> int:
    if not isinstance(text, str) or not emoji:
        return 0
    try:
        return len(emoji.emoji_list(text))
    except Exception:
        try:
            return sum(1 for ch in text if hasattr(emoji, 'is_emoji') and emoji.is_emoji(ch))
        except Exception:
            return 0

def extract_entities(text: str):
    if not isinstance(text, str):
        return [], 0, 0, 0
    urls = URL_RE.findall(text) or []
    mentions = MENTION_RE.findall(text) or []  # tuples
    hashtags = HASHTAG_RE.findall(text) or []
    n_emojis = count_emojis(text)
    return hashtags, len(urls), len(mentions), n_emojis


# ---------------------------
# Scorings (émotions/urgence/intents)
# ---------------------------

def score_emotions(text_norm: str):
    scores = {emo: len(rgx.findall(text_norm)) for emo, rgx in EMO_RE.items()}
    top = [k for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if v > 0][:3]
    return scores, top

def detect_urgency(text_norm: str):
    hits = URGENCY_RE.findall(text_norm)
    return int(len(hits) > 0), list(dict.fromkeys([h.lower() for h in hits]))

def classify_intents(text_norm: str):
    labels, rule_hits = [], {}
    for label, rgx in INTENT_RE.items():
        hits = rgx.findall(text_norm)
        if hits:
            labels.append(label)
            rule_hits[label] = list(dict.fromkeys([h.lower() for h in hits]))[:5]
    primary = max(labels, key=lambda lab: len(rule_hits.get(lab, []))) if labels else "Autre/Indéterminé"
    return labels, primary, rule_hits

# ---------------------------
# Classification des comptes
# ---------------------------

def classify_account(screen_name: str) -> str:
    if not isinstance(screen_name, str) or not screen_name.strip():
        return "Client"
    s = screen_name.lower()
    if s in ["free"]:
        return "Officiel Free"
    if s in ["freebox"]:
        return "Support Freebox"
    if s in ["free_1337"]:
        return "Free Incidents"
    # --- AJOUT ---
    # On ajoute une nouvelle catégorie pour les comptes liés à l'écosystème Free
    if s in ["groupeiliad", "xavier75", "universfreebox"]:
        return "Compte Associé"
    # --- FIN AJOUT ---
    return "Client"


# ---------------------------
# Lecture CSV robuste
# ---------------------------

def robust_read_csv(path):
    tried = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines='skip', low_memory=False)
        except Exception as e:
            tried.append(f"{enc}: {e}")
    raise RuntimeError("Impossible de lire le CSV. Tentatives:\n" + "\n".join(tried))

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    text_like = [c for c in df.columns if df[c].dtype == object]
    return text_like[0] if text_like else None


# ---------------------------
# MAIN
# ---------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("positional_input", nargs="?", help="Entrée CSV (compat historique)")
    ap.add_argument("--input", help="Chemin du CSV d'entrée (prioritaire sur l'argument positionnel)")
    ap.add_argument("--output", help="Chemin du CSV de sortie principal (full, dédup Clients)")
    ap.add_argument("--output-dir", help="Dossier où écrire les exports standards")
    ap.add_argument("--no-standard-exports", action="store_true",
                    help="N'écrit pas les exports standards (tweets_cleaned*.csv/parquet)")
    # --- AJOUT FILTRES ---
    ap.add_argument("--date-from", help="Début (YYYY-MM-DD ou ISO) inclus")
    ap.add_argument("--date-to", help="Fin (YYYY-MM-DD ou ISO) inclus")
    ap.add_argument("--id-min", type=str, help="tweet_id min (inclus)")
    ap.add_argument("--id-max", type=str, help="tweet_id max (inclus)")
    ap.add_argument("--id-list", type=str, help="Liste d'IDs séparés par virgules")
    ap.add_argument("--limit", type=int, help="Limiter le nombre de lignes après filtres")
    ap.add_argument("--offset", type=int, default=0, help="Décalage après filtres (0 = pas d'offset)")
    ap.add_argument("--pick-last", action="store_true", help="Si --limit: prendre les N dernières lignes après tri")
    # --- FIN AJOUT FILTRES ---
    args = ap.parse_args()

    in_csv = args.input or args.positional_input
    if not in_csv:
        print("Usage: python process_tweets_pipeline.py --input free_tweet_export.csv [--output cleaned.csv]")
        sys.exit(1)
    if not os.path.exists(in_csv):
        print(f"Fichier introuvable: {in_csv}")
        sys.exit(2)

    print(f"[INFO] Lecture du CSV: {in_csv}")
    df = robust_read_csv(in_csv)
    n0 = len(df)

    # Colonnes clés
    text_col = pick_first_existing(df, TEXT_CANDIDATES)
    id_col   = pick_first_existing(df, ID_CANDIDATES)
    date_col = pick_first_existing(df, DATE_CANDIDATES)
    user_col = pick_first_existing(df, USER_CANDIDATES)

    if text_col is None:
        raise ValueError("Aucune colonne texte trouvée. Ajoutez 'text' ou 'full_text'.")

    # Normalisation colonnes
    df = df.copy()
    df.rename(columns={text_col: "text_raw"}, inplace=True)
    if id_col and id_col != "tweet_id":
        df.rename(columns={id_col: "tweet_id"}, inplace=True)
    elif "tweet_id" not in df.columns:
        df["tweet_id"] = np.arange(1, len(df) + 1)

    original_user_col = user_col or "user"
    if user_col and user_col != "user":
        df.rename(columns={user_col: "user"}, inplace=True)
    elif "user" not in df.columns:
        df["user"] = ""
    if "screen_name" not in df.columns:
        df["screen_name"] = df["user"]

    if date_col and date_col != "created_at":
        df.rename(columns={date_col: "created_at"}, inplace=True)
    elif "created_at" not in df.columns:
        df["created_at"] = ""

    # Fix textes & date
    df["text_raw"]   = df["text_raw"].astype(str).map(fix_text)
    # --- AJOUT FILTRES ---
    # Parse -> UTC (tz-aware), puis on retire le fuseau pour obtenir des datetimes NAÏFS
    df["_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)

    # Filtre par date
    if args.date_from:
        df = df[df["_dt"] >= pd.to_datetime(args.date_from, errors="coerce")]
    if args.date_to:
        to_dt = pd.to_datetime(args.date_to, errors="coerce")
        if pd.notnull(to_dt) and to_dt.floor("D") == to_dt:
            to_dt = to_dt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["_dt"] <= to_dt]

    # Filtre par tweet_id
    df["_tweet_id_str"] = df["tweet_id"].astype(str).str.strip()
    def _to_int_or_nan(s):
        try: return int(str(s).strip())
        except Exception: return np.nan
    df["_tweet_id_num"] = df["_tweet_id_str"].apply(_to_int_or_nan)

    if args.id_list:
        idset = {s.strip() for s in args.id_list.split(",") if s.strip()}
        df = df[df["_tweet_id_str"].isin(idset)]
    if args.id_min is not None:
        df = df[df["_tweet_id_num"] >= _to_int_or_nan(args.id_min)]
    if args.id_max is not None:
        df = df[df["_tweet_id_num"] <= _to_int_or_nan(args.id_max)]

    # Tri, offset, limit
    df = df.sort_values(["_dt", "tweet_id"], ascending=[True, True])
    off = max(0, int(args.offset)) if args.offset else 0
    if off: df = df.iloc[off:]
    if args.limit and int(args.limit) > 0:
        N = int(args.limit)
        df = df.tail(N) if args.pick_last else df.head(N)

    # Réécrit created_at final + nettoie colonnes techniques
    df["created_at"] = df["_dt"].apply(lambda dt: dt.isoformat() if pd.notnull(dt) else "")
    for _c in ["_dt", "_tweet_id_str", "_tweet_id_num"]:
        if _c in df.columns: del df[_c]

    # --- FIN AJOUT FILTRES ---

    # Nettoyage + version LLM
    out_clean = df["text_raw"].apply(lambda s: clean_tweet_text(s))
    df["text_clean"]   = out_clean.map(lambda x: x[0])
    df["text_for_llm"] = out_clean.map(lambda x: x[1])

    # Entités & compteurs (sur brut)
    ents = df["text_raw"].apply(extract_entities)
    df["hashtags"]    = ents.map(lambda x: x[0])
    df["n_urls"]      = ents.map(lambda x: x[1])
    df["n_mentions"]  = ents.map(lambda x: x[2])
    df["n_emojis"]    = ents.map(lambda x: x[3])

    # Langue
    if detect is not None:
        df["lang"] = df["text_clean"].apply(detect_lang)
        df["is_french"] = df["lang"].map(lambda x: x.startswith('fr'))
    else:
        df["lang"] = ""
        df["is_french"] = True

    # Normalisation pour matching
    df["_norm"] = df["text_clean"].map(normalize_for_match)

    df["contains_free"] = df["text_raw"].str.contains(
        r'\bfree\b|@OQEEbyFree|@Free_1337|@Freebox|@freemobile|@UniversFreebox|#Free_1337|#Freebox|#freemobile|Freebox|iliad|@Xavier75',
        case=False, na=False, regex=True
    )

    # Émotions / Urgence / Intents
    emo_scores = df["_norm"].apply(score_emotions)
    for emo in EMO_RE.keys():
        df[f"emo_{emo}"] = emo_scores.map(lambda x: x[0].get(emo, 0))
    df["top_emotions"] = emo_scores.map(lambda x: x[1])

    urg = df["_norm"].apply(detect_urgency)
    df["is_urgent"]    = int  # keep signature
    df["is_urgent"]    = df["_norm"].apply(lambda s: detect_urgency(s)[0]).astype(int)
    df["urgency_hits"] = df["_norm"].apply(lambda s: detect_urgency(s)[1])

    intents = df["_norm"].apply(classify_intents)
    df["labels"]        = intents.map(lambda x: x[0])
    df["primary_label"] = intents.map(lambda x: x[1])
    df["rule_hits"]     = intents.map(lambda x: x[2])

    # Comptes
    df["account_type"] = df["user"].apply(classify_account)

    # Dédup Clients
    def day_only(s):
        return s[:10] if isinstance(s, str) and len(s) >= 10 else s
    df["_date_day"] = df["created_at"].map(day_only)

    df_clients     = df[df["account_type"] == "Client"].copy()
    df_non_clients = df[df["account_type"] != "Client"].copy()

    before_clients = len(df_clients)
    df_clients = df_clients.sort_values(["created_at", "tweet_id"]).drop_duplicates(
        subset=["_norm", "user", "_date_day"], keep="first"
    )
    dedup_removed_clients = before_clients - len(df_clients)
    df_clients = df_clients[df_clients["contains_free"] == True]

    df_final = pd.concat([df_clients, df_non_clients], ignore_index=True)
    df_final = df_final.sort_values(["created_at", "tweet_id"]).reset_index(drop=True)

    final_cols = [
        "tweet_id","created_at","user","screen_name","account_type",
        "text_raw","text_clean","text_for_llm",
        "lang","is_french",
        "contains_free",
        "n_urls","n_mentions","n_emojis","hashtags",
        "emo_anger","emo_sadness","emo_joy","emo_fear","emo_surprise","emo_trust","emo_disgust","emo_anticipation",
        "top_emotions","is_urgent","urgency_hits",
        "labels","primary_label","rule_hits"
    ]
    # Nettoyage des colonnes de travail avant export
    for tmpc in ["_dt", "_tweet_id_str", "_tweet_id_num"]:
        if tmpc in df_final.columns:
            df_final.drop(columns=[tmpc], inplace=True)
        if tmpc in df_clients.columns:
            df_clients.drop(columns=[tmpc], inplace=True)

    final_cols = [c for c in final_cols if c in df_final.columns]
    df_out_all     = df_final[final_cols].reset_index(drop=True)
    df_out_clients = df_clients[final_cols].reset_index(drop=True)

    # Écritures
    if not args.no_standard_exports:
        out_dir = args.output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        df_out_all.to_parquet(os.path.join(out_dir, "tweets_cleaned.parquet"), index=False)
        df_out_all.to_csv(os.path.join(out_dir, "tweets_cleaned.csv"), index=False, encoding="utf-8")
        df_out_clients.to_parquet(os.path.join(out_dir, "tweets_cleaned_clients_only.parquet"), index=False)
        df_out_clients.to_csv(os.path.join(out_dir, "tweets_cleaned_clients_only.csv"), index=False, encoding="utf-8")

    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df_out_all.to_csv(out_path, index=False, encoding="utf-8")

    # Stats
    account_counts = df["account_type"].value_counts(dropna=False)
    account_pct    = (account_counts / account_counts.sum() * 100).round(1)
    n_all = len(df_out_all)
    n_cli = len(df_out_clients)
    n_fr  = int(df_out_all["is_french"].sum()) if "is_french" in df_out_all.columns else n_all

    print("\n[SUMMARY]")
    print(f" - Entrées brutes: {n0}")
    print(f" - Après dédup (Clients uniquement): {n_all}  (Clients supprimés: {dedup_removed_clients})")
    print(f" - Dont Clients (export dédié): {n_cli}")
    print(f" - Tweets FR détectés (sur full): {n_fr} ({(100*n_fr/max(n_all,1)):.1f}%)")
    print(" - Répartition comptes:")
    for k, v in account_counts.items():
        pct = account_pct.get(k, 0.0)
        print(f"    * {k}: {v} ({pct}%)")
    print(" - Fichiers écrits:")
    if not args.no_standard_exports:
        print("    * tweets_cleaned.parquet / tweets_cleaned.csv (full, dédup Clients)")
        print("    * tweets_cleaned_clients_only.parquet / tweets_cleaned_clients_only.csv (Clients)")
    if args.output:
        print(f"    * {args.output} (CSV principal)")

if __name__ == "__main__":
    main()
