#!/bin/bash
# In an sbatch script: source scripts/orion/games-activate.sh alfworld
GAME="${1:?Usage: source scripts/orion/games-activate.sh alfworld|webshop}"
source "$(dirname "${BASH_SOURCE[0]}")/games-common.sh" || return 1
[[ -f "$GAME_VENV/bin/activate" ]] || { echo "Missing venv: $GAME_VENV" >&2; return 1; }
source "$GAME_VENV/bin/activate"
