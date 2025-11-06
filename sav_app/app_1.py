# app.py — Interface Streamlit (prétraitement + LLM Mistral)
# Fix Windows paths: utilise chemins absolus des scripts + guillemets + cwd=BASE_DIR

import os
import shlex
import tempfile
from pathlib import Path
import subprocess
import platform
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pipeline NLP — Nettoyage + LLM (Mistral)", layout="wide")
st.title("🧹➡🧠 Pipeline NLP : Prétraitement puis LLM (Mistral)")
st.caption("Chargez un CSV, lancez le script de nettoyage, puis le script LLM.")

# État global
if "paths" not in st.session_state:
    st.session_state.paths = {
        "uploaded_csv": None,
        "clean_csv": None,   # sera forcé vers tweets_cleaned_clients_only.csv si présent
        "llm_csv": None,
        "workdir": None,
    }
if "clients_only_ok" not in st.session_state:
    st.session_state.clients_only_ok = False  # True si clients_only existe et est sélectionné

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


def find_first_existing(pathnames: list[Path]) -> Path | None:
    for p in pathnames:
        try:
            if p and p.exists():
                return p.resolve()
        except Exception:
            continue
    return None

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙ Paramètres")

    if st.session_state.paths["workdir"] is None:
        st.session_state.paths["workdir"] = Path(tempfile.mkdtemp(prefix="streamlit_nlp_"))
    workdir = Path(st.session_state.paths["workdir"])
    st.write(f"Dossier de travail : {workdir}")
    st.write(f"Dossier scripts : {BASE_DIR}")

    st.subheader("Scripts & templates de commande")

    # Chemins par défaut vers les scripts (absolus)
    default_preprocess_script = str((BASE_DIR / "process_tweets_pipeline.py").resolve())
    default_llm_script = str((BASE_DIR / "llm_batch_multitask_pool_mistral.py").resolve())

    preprocess_script_path = st.text_input("Chemin script prétraitement", value=default_preprocess_script)
    llm_script_path = st.text_input("Chemin script LLM", value=default_llm_script)

    # ⛔ IMPORTANT: on ne met plus --no-standard-exports
    # Utiliser explicitement l'interpréteur Python courant (sys.executable) pour
    # garantir que les subprocess utilisent le même environnement (.venv)
    preprocess_tpl = st.text_input(
        "Commande prétraitement",
        value=(
            f'"{sys.executable}" "{{pre_script}}" --input "{{input}}" --output "{{output}}" {{extra}}'
        ),
        help="Placeholders: {pre_script}, {input}, {output}, {extra} — Ne pas mettre --no-standard-exports pour générer les exports standards (dont clients_only).",
    )

    # Nouveau template LLM: --input/--output et --concurrency
    llm_tpl = st.text_input(
        "Commande LLM",
        value=(
            f'"{sys.executable}" "{{llm_script}}" --input "{{input}}" --output "{{output}}" --concurrency {{concurrency}} {{extra}}'
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
    with st.expander("Aperçu du CSV importé (50 premières lignes)"):
        preview_csv(in_csv_path)
else:
    in_csv_path = st.session_state.paths.get("uploaded_csv")

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

pre_btn = st.button("🚿 Lancer le prétraitement")
if pre_btn:
    if not in_csv_path:
        st.error("Veuillez d'abord importer un CSV.")
    elif not out_clean:
        st.error("Veuillez donner un nom de fichier de sortie pour le CSV nettoyé.")
    else:
        # Réinitialiser l'état clients_only avant de lancer
        st.session_state.clients_only_ok = False
        st.session_state.paths["clean_csv"] = None

        cmd = build_command_from_template(
            preprocess_tpl,
            pre_script=preprocess_script_path,
            input=in_csv_path,
            output=out_clean,
            extra=extra_args_pre,
        )
        with st.status("Exécution du prétraitement…", expanded=True) as status:
            st.code(cmd, language="bash")

            code, out, err = run_command(cmd, workdir=BASE_DIR)
            if out:
                st.text_area("stdout", value=out, height=200)
            if err:
                st.text_area("stderr", value=err, height=200)

            # Détection robuste des exports
            clients_only_name = "tweets_cleaned_clients_only.csv"
            full_name = "tweets_cleaned.csv"

            # On cherche dans plusieurs emplacements probables
            search_dirs = [
                Path(out_clean).parent if out_clean else None,
                BASE_DIR,
                workdir,
                Path.cwd(),
            ]
            candidates_clients = [d / clients_only_name for d in search_dirs if d]
            candidates_full = [d / full_name for d in search_dirs if d]

            clients_only_path = find_first_existing(candidates_clients)
            full_path = find_first_existing(candidates_full)

            # Succès si le script s'est terminé et qu'au moins un export standard existe
            if code == 0 and (clients_only_path or full_path or Path(out_clean).exists()):
                status.update(label="Prétraitement terminé ✅", state="complete")

                if clients_only_path:
                    # ✅ Utiliser exclusivement le fichier clients_only
                    st.session_state.clients_only_ok = True
                    st.session_state.paths["clean_csv"] = str(clients_only_path)
                    st.success(f"Fichier 'clients_only' détecté : {clients_only_path.name} — il sera utilisé pour le LLM.")
                    with st.expander("Aperçu (clients uniquement) — 50 premières lignes"):
                        preview_csv(clients_only_path)
                    try:
                        with open(clients_only_path, "rb") as f:
                            st.download_button(
                                "⬇ Télécharger le CSV clients_only",
                                data=f,
                                file_name=clients_only_path.name,
                            )
                    except Exception as e:
                        st.warning(f"Impossible d'activer le téléchargement: {e}")

                else:
                    # ⚠ Aucun fichier clients_only — on informe et on n'affiche pas le full ici
                    st.session_state.clients_only_ok = False
                    st.session_state.paths["clean_csv"] = None  # ne pas laisser de fallback
                    st.warning("⚠ Aucun tweet client détecté — le fichier 'tweets_cleaned_clients_only.csv' n’a pas été généré. L’étape LLM est désactivée.")
            else:
                status.update(label="Échec du prétraitement ❌", state="error")

# -------------------- LLM --------------------
st.header("3) Traitement LLM (Mistral)")

# On N'UTILISE QUE le fichier clients_only si présent
llm_input_default = st.session_state.paths.get("clean_csv") or ""

col_l1, col_l2 = st.columns([1, 1])
llm_disabled = not st.session_state.clients_only_ok  # bouton LLM grisé si pas de clients

with col_l1:
    llm_input = st.text_input(
        "CSV d'entrée pour le LLM (clients uniquement)",
        value=llm_input_default,
        disabled=llm_disabled,
        help="Ce champ est automatiquement rempli avec 'tweets_cleaned_clients_only.csv' quand il existe."
    )
    llm_output_default = ""
    if llm_input:
        base = Path(llm_input).with_suffix("")
        llm_output_default = str(base) + "_llm.csv"
    llm_output = st.text_input(
        "Nom du CSV de sortie (LLM)",
        value=llm_output_default,
        disabled=llm_disabled
    )

with col_l2:
    concurrency = st.number_input("Concurrence (threads)", min_value=1, max_value=128, value=4, step=1, disabled=llm_disabled)
    extra_args_llm = st.text_input("Arguments supplémentaires (LLM)", value="", help="Ex: --max-chars 700 --timeout 60", disabled=llm_disabled)

if llm_disabled:
    st.info("ℹ L’étape LLM est désactivée car aucun fichier clients_only n’a été généré au prétraitement.")

llm_btn = st.button("🧠 Lancer le traitement LLM", disabled=llm_disabled)
if llm_btn:
    if not llm_input:
        st.error("Aucun fichier clients_only disponible pour le LLM.")
    elif not llm_output:
        st.error("Veuillez indiquer un fichier de sortie pour le LLM.")
    else:
        cmd = build_command_from_template(
            llm_tpl,
            llm_script=llm_script_path,
            input=llm_input,
            output=llm_output,
            concurrency=int(concurrency),
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
                try:
                    with open(llm_output, "rb") as f:
                        st.download_button("⬇ Télécharger le CSV LLM", data=f, file_name=Path(llm_output).name)
                except Exception as e:
                    st.warning(f"Impossible d'activer le téléchargement du CSV LLM: {e}")
            else:
                status.update(label="Échec du traitement LLM ❌", state="error")

st.divider()
st.caption("Les commandes s’exécutent depuis le dossier des scripts (cwd=BASE_DIR) pour éviter les erreurs de chemins sous Windows. Le LLM n’utilise que le fichier clients_only.")
