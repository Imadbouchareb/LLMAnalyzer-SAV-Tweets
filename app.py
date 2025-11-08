# app.py — Interface Streamlit (prétraitement + LLM Mistral)
# Fix Windows paths: utilise chemins absolus des scripts + guillemets + cwd=BASE_DIR

import os
import shlex
import tempfile
from pathlib import Path
import subprocess
import platform
import sys
import shutil  # <-- ajouté

import pandas as pd
import streamlit as st

# --- Ajouts: dossier fixe + helpers pour clients_only ---
from datetime import datetime

DEFAULT_CLEAN_DIR = Path(r"C:\Users\hallo\OneDrive\Bureau\IA Free Mobile\clean_client")

def get_clean_dir() -> Path:
    p = Path(st.session_state.get("clean_dir", DEFAULT_CLEAN_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_clients_only(folder: Path):
    """Retourne tous les CSV *clients_only* triés du plus récent au plus ancien, dans folder."""
    try:
        return sorted(
            folder.glob("*clients_only*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    except Exception:
        return []

def pick_latest_clients_only(folder: Path):
    """Retourne le chemin (str) du *clients_only* le plus récent dans folder, sinon None."""
    cands = list_clients_only(folder)
    return str(cands[0].resolve()) if cands else None

def write_latest_alias(src_path: str | Path):
    """Copie src_path vers clean_dir/latest_clients_only.csv (écrase si existant)."""
    try:
        if not src_path:
            return None
        src = Path(src_path)
        if not src.exists():
            return None
        # Utiliser le dossier configuré dans la session (sécurisé même après reruns)
        target_dir = Path(st.session_state.get("clean_dir", DEFAULT_CLEAN_DIR))
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "latest_clients_only.csv"
        shutil.copy2(str(src), str(target))
        return str(target.resolve())
    except Exception:
        return None


# --- Remplacer les anciennes fonctions de chargement par des versions dynamiques et cacheables ---
@st.cache_data(show_spinner=False)
def load_subset_for_dates(csv_path: str, date_from_str: str | None, date_to_str: str | None, max_rows: int = 5000):
    """Charge un sous-ensemble depuis un CSV 'clean' en choisissant dynamiquement les colonnes.
    Normalise la sortie en colonnes: ['id','created_at','full_text']."""
    if not csv_path:
        return pd.DataFrame(columns=["id","created_at","full_text"])
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=["id","created_at","full_text"])
    # lecture robuste
    try:
        df = pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=["id","created_at","full_text"])
    # détecte colonnes candidates
    id_col = None
    for c in ("tweet_id","id","id_str","status_id"):
        if c in df.columns:
            id_col = c; break
    text_col = None
    for c in ("text_for_llm","text_for_model","text_clean","text_raw","full_text","text","content"):
        if c in df.columns:
            text_col = c; break
    date_col = None
    for c in ("created_at","_dt","date","timestamp","time"):
        if c in df.columns:
            date_col = c; break
    # fallback
    if id_col is None:
        df["id"] = df.index.astype(str)
        id_col = "id"
    else:
        df[id_col] = df[id_col].astype(str)
    if text_col is None:
        df["full_text"] = ""
        text_col = "full_text"
    if date_col is None:
        df["created_at"] = ""
        date_col = "created_at"
    # normaliser created_at en datetime naive
    df["created_at_dt"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    mask = pd.Series(True, index=df.index)
    if date_from_str:
        try:
            lower = pd.to_datetime(date_from_str)
            mask &= (df["created_at_dt"] >= lower)
        except Exception:
            pass
    if date_to_str:
        try:
            upper = pd.to_datetime(date_to_str) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask &= (df["created_at_dt"] <= upper)
        except Exception:
            pass
    sub = df.loc[mask, [id_col, "created_at_dt", text_col]].copy()
    sub.rename(columns={id_col: "id", "created_at_dt": "created_at", text_col: "full_text"}, inplace=True)
    # format created_at as datetime or iso strings
    sub["created_at"] = pd.to_datetime(sub["created_at"], errors="coerce")
    # sort and limit
    sub = sub.sort_values(["created_at","id"], ascending=[True, True]).head(max_rows)
    # ensure string id and created_at string for UI label
    sub["id"] = sub["id"].astype(str)
    return sub.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def count_tweets_in_range(csv_path: str, date_from_str: str | None, date_to_str: str | None) -> int:
    """Compte les lignes dans l'intervalle pour le CSV clean (utilisé pour info)."""
    df = load_subset_for_dates(csv_path, date_from_str, date_to_str, max_rows=10**9)
    return int(len(df))


# ---------- AJOUT : get_csv_date_range (robuste, cacheé ----------
@st.cache_data(show_spinner=False)
def get_csv_date_range(csv_path: str):
    """Retourne (date_min, date_max) en se basant sur la colonne date disponible."""
    if not csv_path:
        return (None, None)
    p = Path(csv_path)
    if not p.exists():
        return (None, None)

    # lecture tolérante à l'encodage
    try:
        df = pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(p, low_memory=False, on_bad_lines="skip")
        except Exception:
            return (None, None)

    # cherche une colonne date plausible et renvoie min/max (naive)
    for col in ("created_at", "_dt", "date", "timestamp", "time"):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
                if dt.notna().any():
                    return (dt.min().date(), dt.max().date())
            except Exception:
                continue

    return (None, None)
# ---------- FIN AJOUT ----------

st.set_page_config(page_title="Pipeline NLP — Nettoyage + LLM (Mistral)", layout="wide")
st.title("🧹➡️🧠 Pipeline NLP : Prétraitement puis LLM (Mistral)")
st.caption("Chargez un CSV, lancez le script de nettoyage, puis le script LLM.")

# État global
if "paths" not in st.session_state:
    st.session_state.paths = {
        "uploaded_csv": None,
        "clean_csv": None,
        "llm_csv": None,
        "workdir": None,
    }

# Répertoire base = dossier où se trouvent app.py et les scripts
BASE_DIR = Path(__file__).resolve().parent

# -------------------- Utils --------------------

def save_uploaded_file(uploaded_file, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".csv"
    safe_name = Path(uploaded_file.name).stem
    dest = workdir / f"{safe_name}{suffix}"
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


import platform
import subprocess

def run_command(command: str, workdir: Path | None = None):
    try:
        if platform.system() == "Windows":
            # Laisser CMD parser la ligne avec ses guillemets
            res = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir) if workdir else None,
                capture_output=True,
                text=True,
            )
        else:
            # Unix: pas besoin de shell=True, découpe sûre
            import shlex
            args = shlex.split(command)
            res = subprocess.run(
                args,
                shell=False,
                cwd=str(workdir) if workdir else None,
                capture_output=True,
                text=True,
            )
        return res.returncode, res.stdout, res.stderr
    except FileNotFoundError as e:
        return 127, "", f"Commande introuvable: {e}"
    except Exception as e:
        return 1, "", f"Erreur d'exécution: {e}"


def preview_csv(path: Path, n: int = 50):
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=';')
        except Exception as e:
            st.error(f"Impossible de lire le CSV: {e}")
            return
    st.dataframe(df.head(n))


def build_command_from_template(template: str, **kwargs) -> str:
    # Les chemins sont quotés dans les templates; on fait juste .format
    return template.format(**kwargs)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Paramètres")

    if st.session_state.paths["workdir"] is None:
        st.session_state.paths["workdir"] = Path(tempfile.mkdtemp(prefix="streamlit_nlp_"))
    workdir = Path(st.session_state.paths["workdir"]) 
    st.write(f"Dossier de travail : `{workdir}`")
    st.write(f"Dossier scripts : `{BASE_DIR}`")

    st.subheader("Scripts & templates de commande")

    # Chemins par défaut vers les scripts (absolus)
    default_preprocess_script = str((BASE_DIR / "process_tweets_pipeline.py").resolve())
    default_llm_script = str((BASE_DIR / "llm_batch_multitask_pool_mistral.py").resolve())
    default_bert_script = str((BASE_DIR / "llm_batch_local_bertsim.py").resolve())  # NEW default

    preprocess_script_path = st.text_input("Chemin script prétraitement", value=default_preprocess_script)
    llm_script_path = st.text_input("Chemin script LLM (Mistral API)", value=default_llm_script)
    bert_script_path = st.text_input("Chemin script LLM local (BERT)", value=default_bert_script)

    # <-- MODIF : nouveau template PREPROCESS avec output-dir -->
    preprocess_tpl = st.text_input(
        "Commande prétraitement",
        value=(
            "python \"{pre_script}\" --input \"{input}\" --output \"{output}\" --output-dir \"{output_dir}\" {extra}"
        ),
        help="Placeholders: {pre_script}, {input}, {output}, {output_dir}, {extra}",
    )

    # <-- AJOUT : dossier configurable pour les exports nettoyés -->
    clean_dir_str = st.text_input("📁 Dossier sortie 'clean_client'", value=str(DEFAULT_CLEAN_DIR))
    # -- définir le dossier proprement et le créer si besoin
    clean_dir = Path(clean_dir_str).expanduser()
    clean_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["clean_dir"] = str(clean_dir)
    st.write(f"Sorties nettoyées → `{clean_dir}`")

    # ⚠️ Nouveau template LLM: --input/--output et --concurrency
    llm_tpl = st.text_input(
        "Commande LLM",
        value=(
            "python \"{llm_script}\" --input \"{input}\" --output \"{output}\" --concurrency {concurrency} {extra}"
        ),
        help="Placeholders: {llm_script}, {input}, {output}, {concurrency}, {extra}",
    )

    st.divider()
    st.subheader("Clés/API & environnement")
    mistral_api_key = st.text_input("MISTRAL_API_KEY (optionnel)", type="password")
    custom_env_kv = st.text_area("Variables d'env (clé=valeur, une par ligne)")
    if st.button("Appliquer les variables d'environnement"):
        if mistral_api_key:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
        if custom_env_kv.strip():
            for line in custom_env_kv.splitlines():
                if not line.strip():
                    continue
                if "=" not in line:
                    st.warning(f"Ligne ignorée (pas de '='): {line}")
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        st.success("Variables d'environnement appliquées.")

# -------------------- Upload --------------------
st.header("1) Importer le CSV")
uploaded = st.file_uploader("Sélectionnez votre fichier CSV", type=["csv"]) 
if uploaded:
    in_csv_path = save_uploaded_file(uploaded, workdir)
    st.session_state.paths["uploaded_csv"] = str(in_csv_path)
    st.success(f"Fichier importé: {in_csv_path}")
    # Récupérer min/max dates du CSV pour piloter les date_input
    dmin, dmax = get_csv_date_range(str(in_csv_path))
    st.session_state["csv_date_min"] = dmin
    st.session_state["csv_date_max"] = dmax
    with st.expander("Aperçu du CSV importé (50 premières lignes)"):
        preview_csv(in_csv_path)
else:
    in_csv_path = st.session_state.paths.get("uploaded_csv")
    dmin = st.session_state.get("csv_date_min")
    dmax = st.session_state.get("csv_date_max")
    if in_csv_path and (dmin is None or dmax is None):
        dmin, dmax = get_csv_date_range(str(in_csv_path))
        st.session_state["csv_date_min"] = dmin
        st.session_state["csv_date_max"] = dmax

# -------------------- Prétraitement --------------------
st.header("2) Prétraitement & nettoyage")
col_p1, col_p2 = st.columns([1, 1])
with col_p1:
    default_clean_name = ""
    if in_csv_path:
        base = Path(in_csv_path).with_suffix("")
        default_clean_name = str(base) + "_clean.csv"
    out_clean = st.text_input("Nom du CSV de sortie (nettoyé)", value=default_clean_name)
with col_p2:
    extra_args_pre = st.text_input("Arguments supplémentaires (prétraitement)", value="")

# Note: les filtres ont été déplacés vers la section "3) Traitement LLM" (voir plus bas)
# --- FIN SECTION PRÉTRAITEMENT (filtres retirés) ---

pre_btn = st.button("🚿 Lancer le prétraitement")
if pre_btn:
    if not in_csv_path:
        st.error("Veuillez d'abord importer un CSV.")
    elif not out_clean:
        st.error("Veuillez donner un nom de fichier de sortie pour le CSV nettoyé.")
    else:
        # Construire commande avec sys.executable et --output-dir = clean_dir (workdir fixe demandé)
        python_exec = sys.executable
        out_dir = str(clean_dir.resolve())               # <-- forcé vers clean_dir
        extra_safe = extra_args_pre or ""
        try:
            tpl = preprocess_tpl or 'python "{pre_script}" --input "{input}" --output "{output}" --output-dir "{output_dir}" {extra}'
            cmd_tmp = tpl.format(pre_script=preprocess_script_path, input=in_csv_path, output=out_clean, output_dir=out_dir, extra=extra_safe)
            if cmd_tmp.strip().startswith("python "):
                cmd = cmd_tmp.replace("python", python_exec, 1)
            else:
                cmd = cmd_tmp
        except Exception:
            cmd = f'{python_exec} "{preprocess_script_path}" --input "{in_csv_path}" --output "{out_clean}" --output-dir "{out_dir}" {extra_safe}'

        with st.status("Exécution du prétraitement…", expanded=True) as status:
            st.code(cmd, language="bash")
            code, out, err = run_command(cmd, workdir=BASE_DIR)
            if out:
                st.text_area("stdout", value=out, height=200)
            if err:
                st.text_area("stderr", value=err, height=200)
            if code == 0 and Path(out_clean).exists():
                status.update(label="Prétraitement terminé ✅", state="complete")
                st.success(f"Fichier nettoyé : {out_clean}")

                # --- Détection workdir-only (clean_dir) : pick latest *clients_only* dans clean_dir ---
                best = pick_latest_clients_only(clean_dir)
                st.session_state.paths["clean_csv_clients"] = best
                st.session_state.paths["clean_csv"] = best  # impose clients_only comme source LLM
                st.session_state["clients_only_ok"] = bool(best)
                st.session_state.paths["clean_csv_saved_from_pre"] = str(Path(out_clean).resolve())
                if best:
                    alias = write_latest_alias(best)
                    st.session_state["llm_source_locked"] = alias or best
                    st.success(f"Fichier clients-only détecté et verrouillé pour le LLM : {st.session_state['llm_source_locked']}")
                    with st.expander("Aperçu du CSV nettoyé (50 premières lignes)"):
                        preview_csv(Path(st.session_state['llm_source_locked']))
                    with open(st.session_state['llm_source_locked'], "rb") as f:
                        st.download_button("⬇️ Télécharger le CSV nettoyé", data=f, file_name=Path(st.session_state['llm_source_locked']).name)
                else:
                    st.info("Aucun fichier '*clients_only*.csv' trouvé dans le dossier 'clean_client'.")
            else:
                status.update(label="Échec du prétraitement ❌", state="error")

# -------------------- Re-verrouillage avant la section LLM (workdir-only) --------------------
locked = st.session_state.get("llm_source_locked")
if locked and Path(locked).exists():
    current = locked
else:
    current = pick_latest_clients_only(clean_dir)
    if current:
        st.session_state["llm_source_locked"] = current

st.session_state.paths["clean_csv_clients"] = current
st.session_state.paths["clean_csv"] = current
st.session_state["clients_only_ok"] = bool(current)

# -------------------- LLM --------------------
st.header("3) Traitement LLM")
engine = st.radio("Moteur LLM", ["Mistral (API)", "BERT local (offline)"], index=0, horizontal=True)

# ---------- Source LLM : choisir un fichier traité dans clean_client ----------
st.subheader("Source LLM")

folder = Path(st.session_state.get("clean_dir", DEFAULT_CLEAN_DIR))
# Choix du type de fichiers à lister dans le dossier
pattern = st.selectbox(
    "Type de fichiers à lister",
    ["*clients_only*.csv", "*clean*.csv", "*.csv"],
    index=0,
    key="llm_pattern"
)

def list_by_pattern(dirpath: Path, patt: str):
    try:
        return sorted(dirpath.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []

cands = list_by_pattern(folder, pattern)

# Rescan avec clé unique
if st.button("🔄 Rescanner le dossier", key="btn_rescan_llm"):
    cands = list_by_pattern(folder, pattern)

if not cands:
    current = None
    st.code("Source LLM (active) : <aucun fichier correspondant>")
else:
    # Pré-sélectionner le fichier déjà “locké” si présent
    default_idx = 0
    locked_path = st.session_state.get("llm_source_locked")
    if locked_path:
        try:
            default_idx = next(i for i, p in enumerate(cands) if str(p.resolve()) == str(Path(locked_path).resolve()))
        except StopIteration:
            default_idx = 0

    labels = [f"{p.name} — {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}" for p in cands]
    idx = st.selectbox(
        "Choisir le fichier à utiliser",
        list(range(len(cands))),
        index=default_idx,
        format_func=lambda i: labels[i],
        key="llm_select_file"
    )
    current = str(cands[idx].resolve())

    # On maintient l’alias latest_clients_only.csv si on est sur ce pattern
    alias_written = write_latest_alias(current) if pattern == "*clients_only*.csv" else None
    st.session_state["llm_source_locked"] = alias_written or current

    st.code(f"Source LLM (active) : {current}")

# Si la source change -> reset des caches/états des filtres
if current != st.session_state.get("_filter_source"):
    st.session_state["_filter_source"] = current
    try: load_subset_for_dates.clear()
    except: pass
    try: count_tweets_in_range.clear()
    except: pass
    st.session_state["_ids_df"] = None
    st.session_state["_ids_total"] = None
    if current:
        dmin, dmax = get_csv_date_range(current)
        st.session_state["csv_date_min"] = dmin
        st.session_state["csv_date_max"] = dmax

# -------------------- FILTRES LLM (optionnels) : utiliser current (active clean file) --------------------
with st.expander("Filtres LLM (optionnels)"):
    csv_date_min = st.session_state.get("csv_date_min")
    csv_date_max = st.session_state.get("csv_date_max")
    use_date = st.checkbox("Activer filtre par date", value=False)
    date_from = None
    date_to = None
    if use_date:
        cdf, cdt = st.columns(2)
        with cdf:
            date_from = st.date_input("Date min (YYYY-MM-DD)", value=None, min_value=csv_date_min, max_value=csv_date_max)
        with cdt:
            default_to = csv_date_max
            date_to = st.date_input("Date max (YYYY-MM-DD)", value=default_to, min_value=csv_date_min, max_value=csv_date_max)

    # Charger les tweets de la période (depuis le CSV clean actif)
    selected_ids = []
    if st.button("Charger les tweets de la période (max 5000)"):
        if not current:
            st.warning("Aucune source LLM active. Exécutez le prétraitement pour générer un fichier clients_only.")
        else:
            sub_df = load_subset_for_dates(current,
                                           date_from.strftime("%Y-%m-%d") if date_from else None,
                                           date_to.strftime("%Y-%m-%d") if date_to else None,
                                           max_rows=5000)
            total_cnt = count_tweets_in_range(current,
                                              date_from.strftime("%Y-%m-%d") if date_from else None,
                                              date_to.strftime("%Y-%m-%d") if date_to else None)
            st.session_state["_ids_df"] = sub_df
            st.session_state["_ids_total"] = total_cnt

    sub_df = st.session_state.get("_ids_df")
    total_cnt = st.session_state.get("_ids_total")
    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
        st.info(f"📦 {total_cnt if total_cnt is not None else len(sub_df)} tweets trouvés. Affichés : {len(sub_df)} (max 5000).")
        def mk_label(row):
            t = str(row.get("full_text", ""))[:120].replace("\n", " ")
            d = row.get("created_at")
            d = d.strftime("%Y-%m-%d %H:%M") if pd.notnull(d) else "?"
            return f"{row['id']}  —  {d}  —  {t}"
        labels = {row["id"]: mk_label(row) for _, row in sub_df.iterrows()}
        options = sub_df["id"].astype(str).tolist()
        selected_ids = st.multiselect("Sélectionner un ou plusieurs tweets", options=options, format_func=lambda x: labels.get(x, x), max_selections=500)
        st.caption(f"{len(selected_ids)} tweet(s) sélectionné(s).")
    else:
        st.caption("⚠️ Cliquez sur 'Charger les tweets de la période' pour remplir la liste (depuis le fichier clean actif).")

    max_for_limit = int(st.session_state.get("_ids_total") or 0)
    if max_for_limit > 0:
        limit_n = st.number_input("Limiter à N lignes", min_value=0, max_value=max_for_limit, value=0, step=1)
    else:
        limit_n = st.number_input("Limiter à N lignes", min_value=0, value=0, step=1)

# Concurrency only for Mistral
if engine.startswith("Mistral"):
    concurrency = st.number_input("Concurrence (threads)", min_value=1, max_value=128, value=4, step=1)
else:
    concurrency = None

extra_args_llm = st.text_input(
    "Arguments supplémentaires (LLM)",
    value="--no-summarize --hf-batch 32 --batch 128 --torch-threads 2 --intent-max-length 128" if engine.endswith("(offline)") else "",
    help="Local (BERT): ex. --no-summarize --hf-batch 32 --batch 128 --torch-threads 2 --intent-max-length 128  |  Mistral: ex. --rpm 0 --max-chars 700"
)

# Disable LLM if clients_only missing
clients_ok = bool(st.session_state.get("clients_only_ok", False))
if not clients_ok:
    st.info("LLM désactivé : aucun fichier 'tweets_cleaned_clients_only.csv' détecté dans le workdir. Exécutez le prétraitement.")
try:
    llm_btn = st.button("🧠 Lancer le traitement LLM", disabled=not clients_ok)
except TypeError:
    llm_btn = st.button("🧠 Lancer le traitement LLM")

if llm_btn:
    if not clients_ok:
        st.error("Impossible : aucun fichier clients-only disponible pour le LLM. Générer 'tweets_cleaned_clients_only.csv' d'abord.")
    else:
        # Use only the clients_only locked/source
        source_clean = st.session_state.paths.get("clean_csv_clients")
        if not source_clean or not Path(source_clean).exists():
            st.error("Fichier clients-only introuvable dans le workdir. Vérifiez le prétraitement.")
        else:
            # Charger cleaned CSV (protection fallback encodings)
            try:
                df_clean = pd.read_csv(source_clean, low_memory=False)
            except Exception:
                df_clean = pd.read_csv(source_clean, low_memory=False, encoding='utf-8', on_bad_lines='skip')

            # Appliquer filtres LLM (date, ids, limit) — tentative tweet_id puis id
            if use_date and date_from:
                lower = pd.to_datetime(date_from)
                df_clean["created_at_dt"] = pd.to_datetime(df_clean.get("created_at", ""), errors="coerce")
                df_clean = df_clean[df_clean["created_at_dt"] >= lower]
            if use_date and date_to:
                upper = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                df_clean["created_at_dt"] = pd.to_datetime(df_clean.get("created_at", ""), errors="coerce")
                df_clean = df_clean[df_clean["created_at_dt"] <= upper]

            if selected_ids:
                if "tweet_id" in df_clean.columns:
                    df_clean = df_clean[df_clean["tweet_id"].astype(str).isin(selected_ids)]
                elif "id" in df_clean.columns:
                    df_clean = df_clean[df_clean["id"].astype(str).isin(selected_ids)]

            if limit_n and int(limit_n) > 0:
                df_clean = df_clean.head(int(limit_n))

            # Write filtered temp CSV in same folder (workdir)
            src_path = Path(source_clean)
            temp_fp = src_path.with_name(f"{src_path.stem}_for_llm_filtered.csv")
            df_clean.to_csv(temp_fp, index=False, encoding="utf-8")
            st.success(f"Fichier temporaire filtré créé: {temp_fp}")

            # compute automatic llm_output (no user input)
            base_noext = str(src_path.with_suffix(""))
            llm_output = f"{base_noext}{'_llm_mistral.csv' if engine.startswith('Mistral') else '_llm_bert.csv'}"

            # Build command with sys.executable (already used ailleurs)
            python_exec = sys.executable
            if engine.startswith("Mistral"):
                cmd = f'{python_exec} "{llm_script_path}" --input "{temp_fp}" --output "{llm_output}" --concurrency {int(concurrency)} {extra_args_llm or ""}'
            else:
                cmd = f'{python_exec} "{bert_script_path}" --input "{temp_fp}" --output "{llm_output}" --no-summarize --hf-batch 32 --batch 128 --torch-threads 2 --cache "llm_cache_bertsim.sqlite" {extra_args_llm or ""}'

            with st.status("Exécution du LLM…", expanded=True) as status:
                st.code(cmd, language="bash")
                code, out, err = run_command(cmd, workdir=BASE_DIR)
                if out:
                    st.text_area("stdout", value=out, height=200)
                if err:
                    st.text_area("stderr", value=err, height=200)
                if code == 0 and Path(llm_output).exists():
                    st.session_state.paths["llm_csv"] = str(Path(llm_output).resolve())
                    status.update(label="Traitement LLM terminé ✅", state="complete")
                    st.success(f"Fichier LLM : {st.session_state.paths['llm_csv']}")
                    with st.expander("Aperçu du CSV LLM (50 premières lignes)"):
                        preview_csv(Path(llm_output))
                    with open(llm_output, "rb") as f:
                        st.download_button("⬇️ Télécharger le CSV LLM", data=f, file_name=Path(llm_output).name)
                else:
                    status.update(label="Échec du traitement LLM ❌", state="error")

st.divider()
st.caption("Templates de commande quotés + exécution depuis le dossier des scripts pour éviter les erreurs de chemins sous Windows.")
