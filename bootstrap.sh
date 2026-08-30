#!/usr/bin/env bash
# Runs ON the Nebius VM. Sets up MuJoCo Playground + MJX and smoke-tests the G1 env.
set -euo pipefail

echo "==> driver check (CUDA 13 wheels need >= 580)"
nvidia-smi

sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libegl1 libgl1 libglfw3 git tmux

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 ~/venv
source ~/venv/bin/activate

# NOTE: the Nebius image is CUDA 13, so this is jax[cuda13] — NOT the cuda12
# extra that every MJX tutorial tells you to install.
uv pip install -U "jax[cuda13]"
uv pip install -U mujoco mujoco-mjx playground brax mediapy wandb

echo "==> headless rendering backend"
echo 'export MUJOCO_GL=egl'   >> ~/venv/bin/activate
echo 'export PYOPENGL_PLATFORM=egl' >> ~/venv/bin/activate

echo "==> verifying JAX sees the GPU"
python -c "import jax; print('devices:', jax.devices()); assert jax.devices()[0].platform=='gpu'"

echo "==> smoke-testing the G1 env (downloads Menagerie assets on first load)"
MUJOCO_GL=egl python - <<'PY'
from mujoco_playground import locomotion
env = locomotion.load('G1JoystickFlatTerrain')
cfg = locomotion.get_default_config('G1JoystickFlatTerrain')
print('obs :', env.observation_size)
print('act :', env.action_size)
print('friction randomization is in g1/randomize.py — currently U(0.4, 1.0)')
print('registered G1 envs:', [e for e in locomotion.ALL_ENVS if e.startswith('G1')])
PY

echo
echo "READY.  source ~/venv/bin/activate"
echo "Renders are headless: MUJOCO_GL=egl, write mp4 and scp it down."
