# ==============================================================================
# CONFIGURATION - WORD PIPELINE
# ==============================================================================
import os
import sys
import torch

# Set this pod's GIT_TOKEN as an Environment Variable in the RunPod pod config
# (Pod Settings -> Environment Variables), not here - never hardcode a token
# in a notebook cell. Falls back to the credential helper already cached on
# this pod (from earlier manual setup) if the env var isn't set.
GIT_TOKEN = os.environ.get("GIT_TOKEN")
GIT_REPO_URL = "https://github.com/fabsGitHub/machine-translation.git"
GIT_USER_NAME = "fabsGitHub"
GIT_USER_EMAIL = "fabianhensel@live.de"
GIT_BRANCH = "main"

# This pipeline's own dedicated checkout - do not point both notebooks at the
# same path, they'd race on run_studies.log, matrix caches, and results.
REPO_PATH = "/workspace/machine-translation"

TOKEN_TYPE = "word"
STUDY_NAME = "all"
RUN_DATA_EXPLORATION = True  # Task 1 - corpus-level stats, token-type independent.
                             # Only True here; the char notebook skips it as
                             # redundant (identical output, char has much less
                             # time to spare).
AUTO_SHUTDOWN = True  # stop this pod once the full run succeeds

# ==============================================================================
# 1. GIT SYNC & CLONE
# ==============================================================================
os.system(f'git config --global user.name "{GIT_USER_NAME}"')
os.system(f'git config --global user.email "{GIT_USER_EMAIL}"')
os.system("git config --global advice.addIgnoredFile false")
os.system("git config --global credential.helper store")

auth_url = (
    GIT_REPO_URL.replace("https://", f"https://{GIT_USER_NAME}:{GIT_TOKEN}@")
    if GIT_TOKEN else GIT_REPO_URL
)

if os.path.exists(REPO_PATH):
    print(f"📦 Repository found at {REPO_PATH}. Syncing branch '{GIT_BRANCH}'...")
    os.chdir(REPO_PATH)
    if GIT_TOKEN:
        os.system(f"git remote set-url origin {auth_url}")
    os.system("git fetch origin")
    os.system(f"git checkout {GIT_BRANCH} || git checkout -b {GIT_BRANCH} origin/{GIT_BRANCH}")
    os.system(f"git pull origin {GIT_BRANCH} --rebase --autostash")
    os.system(f"git remote set-url origin {GIT_REPO_URL}")
else:
    print(f"📥 Cloning branch '{GIT_BRANCH}' into {REPO_PATH}...")
    os.chdir(os.path.dirname(REPO_PATH))
    if GIT_TOKEN:
        os.system(f"git clone -b {GIT_BRANCH} {auth_url} {os.path.basename(REPO_PATH)}")
    else:
        os.system(f"git clone -b {GIT_BRANCH} {GIT_REPO_URL} {os.path.basename(REPO_PATH)}")
        print("⚠️ No GIT_TOKEN set - if this prompts for a password, use a Personal "
              "Access Token, not your account password.")
    os.chdir(REPO_PATH)
    os.system(f"git remote set-url origin {GIT_REPO_URL}")

print("\n📌 Active Commit Details:")
os.system("git log -1 --oneline")

# ==============================================================================
# 2. DEPENDENCY & HARDWARE VALIDATION
# ==============================================================================
venv_python = os.path.join(REPO_PATH, ".venv", "bin", "python")
venv_pip = os.path.join(REPO_PATH, ".venv", "bin", "pip")

if not os.path.exists(venv_python):
    print("\n📦 No .venv found - creating one (reusing system torch/CUDA install)...")
    os.system(f"python3 -m venv --system-site-packages {os.path.join(REPO_PATH, '.venv')}")

if os.path.exists("requirements.txt"):
    print("\n📦 Verifying / installing requirements into .venv...")
    os.system(f"{venv_pip} install -r requirements.txt --quiet --disable-pip-version-check")

print(f"\n⚡ CUDA Available: {torch.cuda.is_available()} | GPU Count: {torch.cuda.device_count()}")

# ==============================================================================
# 3. TASK 1: DATA EXPLORATION (word notebook only - see RUN_DATA_EXPLORATION)
# ==============================================================================
if RUN_DATA_EXPLORATION:
    print("\n📊 Running Task 1 data exploration (corpus stats + figures)...")
    os.system(f"{venv_python} src/explore_data.py")

# ==============================================================================
# 4. RUN EXPERIMENTS
# ==============================================================================
shutdown_flag = "--auto_shutdown" if AUTO_SHUTDOWN else ""
os.system(f"{venv_python} src/run_studies.py --study {STUDY_NAME} --token_type {TOKEN_TYPE} {shutdown_flag}")

# ==============================================================================
# 5. BACKUP & SYNC RESULTS TO GITHUB
# ==============================================================================
print("\n🔄 Execution complete. Initializing backup sequence...")

os.chdir(REPO_PATH)
if GIT_TOKEN:
    os.system(f"git remote set-url origin {auth_url}")

# Stage code, config, logs, and small result assets
os.system("git add src/ *.md requirements.txt")
os.system("git add -f config/config.yaml") if os.path.exists("config/config.yaml") else None

local_results_dir = os.path.join(REPO_PATH, "data", "results")
if os.path.exists(local_results_dir):
    for root, _, files in os.walk(local_results_dir):
        for f in files:
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, REPO_PATH)

            if f.endswith((".json", ".csv", ".png", ".md")):
                os.system(f'git add -f "{rel_path}"')
            elif f.endswith(".pt") and (os.path.getsize(file_path) / (1024 * 1024) < 95.0):
                os.system(f'git add -f "{rel_path}"')

commit_msg = f"Auto-commit ({TOKEN_TYPE.capitalize()} Run): Results updated"
os.system(f'git commit -m "{commit_msg}" || echo "No changes to commit."')
os.system(f"git push origin {GIT_BRANCH}")

# Clean token out of git remote config
os.system(f"git remote set-url origin {GIT_REPO_URL}")
print("🏁 Finished! Access token safely removed from git remote.")