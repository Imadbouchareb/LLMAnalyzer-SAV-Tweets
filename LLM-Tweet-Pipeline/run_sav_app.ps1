param(
  [int]$Port = 8501
)

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$app = Join-Path $PSScriptRoot "sav_app\app_1.py"

if (-not (Test-Path $python)) {
  Write-Error "Python du venv introuvable : $python`nCréez le venv avec : python -m venv .venv"
  exit 1
}
if (-not (Test-Path $app)) {
  Write-Error "Fichier d'application introuvable : $app"
  exit 1
}

Write-Host "Démarrage Streamlit avec : $python -m streamlit run $app --server.port $Port" -ForegroundColor Green
& $python -m streamlit run $app --server.port $Port
