# Sourced by every sbatch script. Sets up a job-scoped uv environment in /dev/shm.
# Futurama notes: /nethome/<user> is a self-referencing symlink on compute nodes, so remap to
# /nethome-instruction/<user>; /tmp is only 3.9 GB, so caches and the venv go to /dev/shm.
set -euo pipefail
fix() { [ -e "$1" ] && echo "$1" || echo "${1/#\/nethome\//\/nethome-instruction\/}"; }
export HOME=$(fix "$HOME")
REPO=$(fix "${REPO:-$SLURM_SUBMIT_DIR}")
cd "$REPO"
SCR=/dev/shm/armlab-${SLURM_JOB_ID:-$$}
mkdir -p "$SCR"
trap 'rm -rf "$SCR"' EXIT TERM INT
export UV_CACHE_DIR=$SCR/uv-cache UV_PROJECT_ENVIRONMENT=$SCR/venv UV_PYTHON_INSTALL_DIR=$SCR/python
export XDG_CACHE_HOME=$SCR/cache HF_HOME=$SCR/hf TORCHINDUCTOR_CACHE_DIR=$SCR/inductor TRITON_CACHE_DIR=$SCR/triton
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
UV=$HOME/.local/bin/uv
echo "job ${SLURM_JOB_ID:-local} on $(hostname) repo=$REPO git=$(git rev-parse --short HEAD) $(date -Is)"
"$UV" sync -q --python 3.12 ${UV_EXTRAS:-}
