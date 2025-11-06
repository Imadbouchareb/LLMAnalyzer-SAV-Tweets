# app.py — Interface Streamlit (prétraitement + LLM Mistral)
# Fix Windows paths: utilise chemins absolus des scripts + guillemets + cwd=BASE_DIR

import os
import shlex
import tempfile
from pathlib import Path
import subprocess
import platform

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_subset_for_dates(csv_path: str, date_from_str: str | None, date_to_str: str | None, max_rows: int = 5000):
    """Sous-ensemble filtré par date, prêt pour l'assistant d'IDs."""
    try:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")

    if "created_at" not in df.columns:
        return pd.DataFrame(columns=["id", "created_at", "full_text"])

    # => tz-aware -> naive
    dt = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert(None)

    mask = pd.Series(True, index=df.index)
    if date_from_str:
        lower = pd.to_datetime(date_from_str)  # naive
        mask &= (dt >= lower)
    if date_to_str:
        upper = pd.to_datetime(date_to_str) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= (dt <= upper)

    sub = df.loc[mask, ["id", "created_at", "full_text"]].copy()
    sub["id"] = sub["id"].astype(str).str.strip()
    sub["created_at"] = pd.to_datetime(sub["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    return sub.sort_values(["created_at", "id"]).head(max_rows)


@st.cache_data(show_spinner=False)
def count_tweets_in_range(csv_path: str, date_from_str: str | None, date_to_str: str | None) -> int:
    """Nombre total de tweets dans l'intervalle (sans limite)."""
    try:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")

    if "created_at" not in df.columns:
        return 0

    dt = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert(None)

    mask = pd.Series(True, index=df.index)
    if date_from_str:
        lower = pd.to_datetime(date_from_str)
        mask &= (dt >= lower)
    if date_to_str:
        upper = pd.to_datetime(date_to_str) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask &= (dt <= upper)

    return int(mask.sum())


@st.cache_data(show_spinner=False)
def get_csv_date_range(csv_path: str):
    """Retourne (date_min, date_max) basées sur created_at."""
    try:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip", encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False, on_bad_lines="skip")

    if "created_at" not in df.columns:
        return (None, None)

    dt = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_convert(None)
    if dt.notna().any():
        return (dt.min().date(), dt.max().date())
    return (None, None)

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

    preprocess_tpl = st.text_input(
        "Commande prétraitement",
        value=(
            "python \"{pre_script}\" --input \"{input}\" --output \"{output}\" --no-standard-exports {extra}"
        ),
        help="Placeholders: {pre_script}, {input}, {output}, {extra}",
    )

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

    # --- AJOUT FILTRES ---
    with st.expander("Filtres de prétraitement (optionnels)"):
        # === 1) Filtre par date (Date max = dernière date du CSV et dates futures grisées) ===
        use_date = st.checkbox("Activer filtre par date", value=True)
        csv_date_min = st.session_state.get("csv_date_min")
        csv_date_max = st.session_state.get("csv_date_max")

        date_from = None
        date_to = None
        if use_date:
            cdf, cdt = st.columns(2)
            with cdf:
                # Date min : libre (optionnelle), bornée par min/max du CSV si dispo
                date_from = st.date_input(
                    "Date min (YYYY-MM-DD)",
                    value=None,
                    min_value=csv_date_min,
                    max_value=csv_date_max
                )
            with cdt:
                # Date max : par défaut = dernière date du CSV, dates futures grisées via max_value
                default_to = csv_date_max
                date_to = st.date_input(
                    "Date max (YYYY-MM-DD)",
                    value=default_to,
                    min_value=csv_date_min,
                    max_value=csv_date_max
                )

        # === 2) Sélectionner un ou plusieurs tweets (multiselect) ===
        selected_ids = []
        if use_date and in_csv_path:
            if st.button("Charger les tweets de la période (max 5000)"):
                # DataFrame limité à 5000 pour la liste + total exact pour info/limite
                sub_df = load_subset_for_dates(
                    in_csv_path,
                    date_from.strftime("%Y-%m-%d") if date_from else None,
                    date_to.strftime("%Y-%m-%d") if date_to else None,
                    max_rows=5000,
                )
                total_cnt = count_tweets_in_range(
                    in_csv_path,
                    date_from.strftime("%Y-%m-%d") if date_from else None,
                    date_to.strftime("%Y-%m-%d") if date_to else None,
                )
                st.session_state["_ids_df"] = sub_df
                st.session_state["_ids_total"] = total_cnt

            sub_df = st.session_state.get("_ids_df")
            total_cnt = st.session_state.get("_ids_total")

            if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                # Info claire sur le volume
                info_msg = f"📦 {total_cnt if total_cnt is not None else len(sub_df)} tweets trouvés dans l'intervalle. " \
                           f"Affichés : {len(sub_df)} (max 5000)."
                st.info(info_msg)

                # Options lisibles (ID + date + extrait texte), valeur = ID complet
                def mk_label(row):
                    t = str(row.get("full_text", ""))[:80].replace("\n", " ")
                    d = row["created_at"]
                    d = d.strftime("%Y-%m-%d %H:%M") if pd.notnull(d) else "?"
                    return f"{row['id']}  —  {d}  —  {t}"
                labels = {row["id"]: mk_label(row) for _, row in sub_df.iterrows()}
                options = sub_df["id"].astype(str).tolist()

                selected_ids = st.multiselect(
                    "Sélectionner un ou plusieurs tweets",
                    options=options,
                    format_func=lambda x: labels.get(x, x),
                    max_selections=500
                )
                st.caption(f"{len(selected_ids)} tweet(s) sélectionné(s).")
            else:
                st.caption("⚠️ Clique sur « Charger les tweets de la période » pour remplir la liste.")

        # === 3) Limiter à N lignes (0 = pas de limite) ===
        max_for_limit = int(st.session_state.get("_ids_total") or 0)
        if max_for_limit > 0:
            limit_n = st.number_input(
                "Limiter à N lignes",
                min_value=0,
                max_value=max_for_limit,
                value=0,
                step=1,
                help=f"0 = pas de limite. Max disponible dans l'intervalle : {max_for_limit}."
            )
        else:
            limit_n = st.number_input(
                "Limiter à N lignes",
                min_value=0,
                value=0,
                step=1,
                help="0 = pas de limite. (Charge la période pour connaître le maximum disponible.)"
            )

    # --- TOKENS argparse (seulement date + id-list + limit) ---
    tokens = []
    if date_from: tokens += ["--date-from", date_from.strftime("%Y-%m-%d")]
    if date_to:   tokens += ["--date-to",   date_to.strftime("%Y-%m-%d")]
    if selected_ids:
        tokens += ["--id-list", ",".join(selected_ids)]
    if limit_n and int(limit_n) > 0:
        tokens += ["--limit", str(int(limit_n))]

    user_extra = (extra_args_pre or "").strip()
    final_extra_args_pre = " ".join([user_extra] + [shlex.quote(t) for t in tokens]).strip()
    # --- FIN AJOUT FILTRES ---

pre_btn = st.button("🚿 Lancer le prétraitement")
if pre_btn:
    if not in_csv_path:
        st.error("Veuillez d'abord importer un CSV.")
    elif not out_clean:
        st.error("Veuillez donner un nom de fichier de sortie pour le CSV nettoyé.")
    else:
        cmd = build_command_from_template(
            preprocess_tpl,
            pre_script=preprocess_script_path,
            input=in_csv_path,
            output=out_clean,
            extra=final_extra_args_pre,
        )
        with st.status("Exécution du prétraitement…", expanded=True) as status:
            st.code(cmd, language="bash")

            code, out, err = run_command(cmd, workdir=BASE_DIR)
            if out:
                st.text_area("stdout", value=out, height=200)
            if err:
                st.text_area("stderr", value=err, height=200)
            if code == 0 and Path(out_clean).exists():
                st.session_state.paths["clean_csv"] = str(Path(out_clean).resolve())
                status.update(label="Prétraitement terminé ✅", state="complete")
                st.success(f"Fichier nettoyé : {st.session_state.paths['clean_csv']}")
                with st.expander("Aperçu du CSV nettoyé (50 premières lignes)"):
                    preview_csv(Path(out_clean))
                with open(out_clean, "rb") as f:
                    st.download_button("⬇️ Télécharger le CSV nettoyé", data=f, file_name=Path(out_clean).name)
            else:
                status.update(label="Échec du prétraitement ❌", state="error")

# -------------------- LLM --------------------
st.header("3) Traitement LLM")
engine = st.radio("Moteur LLM", ["Mistral (API)", "BERT local (offline)"], index=0, horizontal=True)

llm_input_default = st.session_state.paths.get("clean_csv") or st.session_state.paths.get("uploaded_csv") or ""
col_l1, col_l2 = st.columns([1, 1])
with col_l1:
    llm_input = st.text_input("CSV d'entrée pour le LLM", value=llm_input_default)
    llm_output_default = ""
    if llm_input:
        base = Path(llm_input).with_suffix("")
        suffix = "_llm_mistral.csv" if engine.startswith("Mistral") else "_llm_bert.csv"
        llm_output_default = str(base) + suffix
    llm_output = st.text_input("Nom du CSV de sortie (LLM)", value=llm_output_default)
with col_l2:
    concurrency = st.number_input("Concurrence (threads)", min_value=1, max_value=128, value=4, step=1)
    # Arguments supplémentaires (LLM) - valeur par défaut rapide pour le mode local
    extra_args_llm = st.text_input(
        "Arguments supplémentaires (LLM)",
        value="--no-summarize --hf-batch 32 --batch 128 --torch-threads 2 --intent-max-length 128" if engine.endswith("(offline)") else "",
        help="Local (BERT): ex. --no-summarize --hf-batch 32 --batch 128 --torch-threads 2 --intent-max-length 128  |  Mistral: ex. --rpm 0 --max-chars 700"
    )

llm_btn = st.button("🧠 Lancer le traitement LLM")
if llm_btn:
    if not llm_input:
        st.error("Veuillez indiquer un CSV d'entrée pour le LLM.")
    elif not llm_output:
        st.error("Veuillez indiquer un fichier de sortie pour le LLM.")
    else:
        # When building command for local BERT, ensure quoting on Windows:
        if engine.startswith("Mistral"):
            cmd = build_command_from_template(
                llm_tpl,
                llm_script=llm_script_path,
                input=llm_input,
                output=llm_output,
                concurrency=int(concurrency),
                extra=extra_args_llm,
            )
        else:
            cmd = build_command_from_template(
                "python \"{llm_script}\" --input \"{input}\" --output \"{output}\" {extra}",
                llm_script=bert_script_path,
                input=llm_input,
                output=llm_output,
                extra=extra_args_llm,
            )

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
