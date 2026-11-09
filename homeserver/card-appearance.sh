#!/usr/bin/env bash
# Print a Marinara card's appearance text, for restyle.py's APPEARANCE (read-only; fine on either instance).
#   homeserver/card-appearance.sh <container> character|persona <id>
# Personas with an image-specific appearance switched on use that one. Containers serve https on 7860
# (marinara) or http (marinara-wife): BASE_URL overrides.
# ponytail: homeserver-only; talks to the container's own port via docker exec, like the import script.
set -euo pipefail
container=${1:?container}; kind=${2:?character|persona}; id=${3:?id}
base=${BASE_URL:-$([[ $container == marinara-wife ]] && echo http || echo https)://localhost:7860}
docker exec -e BASE="$base" -e KIND="$kind" -e ID="$id" -e NODE_TLS_REJECT_UNAUTHORIZED=0 -e NODE_NO_WARNINGS=1 "$container" node -e '
const { BASE, KIND, ID } = process.env;
const url = KIND === "persona" ? `${BASE}/api/characters/personas/list` : `${BASE}/api/characters`;
fetch(url).then(r => r.json()).then(list => {
  const row = list.find(x => x.id === ID);
  if (!row) { console.error(`${KIND} ${ID} not found`); process.exit(1); }
  let text;
  if (KIND === "persona") text = row.imageAppearanceEnabled && row.imageAppearance?.trim() ? row.imageAppearance : row.appearance;
  else text = JSON.parse(row.data).extensions?.appearance;
  if (!text?.trim()) { console.error("no appearance text on that card"); process.exit(1); }
  console.log(text.trim());
}).catch(e => { console.error(e.message); process.exit(1); });'
