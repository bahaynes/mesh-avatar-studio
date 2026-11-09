#!/usr/bin/env bash
# Portrait -> drafted, built mesh avatar with drawn variants, ready for a human check in Studio.
#
#   homeserver/avatar-pipeline.sh <name> <portrait.png|webp> [x0,y0,x1,y1]
#
# Steps: crop + cut out (ComfyUI), new project, Qwen rig draft, layers, all 11 drawn variants
# (TRIES per part; Qwen picks, but its picks are unreliable: check work/judge-*.png), sprites,
# review renders. Stops before Marinara: check and fix the rig in Studio, then run
# marinara-engine/scripts/fork/mesh-avatar-to-marinara.sh <name> <character-id>.
# ponytail: homeserver-only (strata Qwen, ComfyUI on 10.10.10.114, playwright container for renders).
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?project name}; portrait=${2:?portrait image}; box=${3:-}
P=projects/$name
[[ ! -e $P ]] || { echo "$P exists; pick another name or move it aside" >&2; exit 2; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
uv run --with pillow python homeserver/prep-source.py "$portrait" "$tmp/source.png" $box
uv run tools/new-project.py "$tmp/source.png" "$name"
uv run --with pillow python homeserver/qwen-rig.py "$P"
npm run -s validate-rig -- "$P/rig.draft.json"
uv run --with numpy --with pillow --with opencv-python-headless tools/build-layers.py "$P" --rig rig.draft.json
uv run tools/variant-requests.py "$P"
uv run --with pillow --with numpy python homeserver/comfy-variants.py "$P"
uv run tools/build-sprites.py "$P"
docker run --rm --ipc=host -v "$PWD":/w -w /w mcr.microsoft.com/playwright:v1.63.0-noble npm run -s render-poses -- "$P"
echo "done: open $name in Studio, fix the rig, check $P/work/judge-*.png, then import to Marinara"
