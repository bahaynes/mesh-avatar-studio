"""Draft a mesh-avatar rig with the local Qwen (strata) reading coordinates; code derives the rest.

usage: uv run --with pillow homeserver/qwen-rig.py projects/<name>     (project made by tools/new-project.py)

Qwen only reads positions off grid images (what the vision-check measured). Every radius, band and
default is computed here from those readings, following docs/rig-fields.md and the agent guide.
Writes rig.draft.json plus work/qwen-*.json (raw answers) for review.
ponytail: spike for one character; endpoint from minecraftdocker llm/.env; no retries beyond one
re-ask on bad JSON. Upgrade path: a queue command around this once a rig proves good enough.
"""
import base64, json, math, os, re, subprocess, sys, urllib.request
from pathlib import Path

ENV = Path.home() / "git/minecraftdocker/llm/.env"
KEY = next(l.split("=", 1)[1].strip().strip("'\"") for l in ENV.read_text().splitlines() if l.startswith("LLAMA_API_KEY="))
IP = subprocess.check_output(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", "strata-backend"], text=True).split()[0]
URL = f"http://{IP}:8082/v1/chat/completions"


def ask(image: Path, question: str, log: Path):
    img = base64.b64encode(image.read_bytes()).decode()
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                         {"type": "text", "text": question + "\nAnswer with JSON only."}]}]
    for attempt in range(2):
        body = {"model": "local-model", "max_tokens": 6000, "temperature": 0, "messages": msgs}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        text = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", text, flags=re.S), re.S)
        try:
            data = json.loads(m.group(0))
            log.write_text(json.dumps({"question": question, "answer": data}, indent=1))
            return data
        except Exception:
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only."}]
    raise SystemExit(f"bad JSON from Qwen for {log.name}: {text[:400]}")


def grid(proj: Path, out: str, step: int, region=None) -> Path:
    path = proj / "work" / out
    cmd = ["uv", "run", "tools/grid.py", str(proj), "--step", str(step), "--out", str(path)]
    if region:
        cmd += ["--region", ",".join(str(int(v)) for v in region)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def crop(proj: Path, out: str, region, scale: int):
    from PIL import Image
    x0, y0, x1, y1 = (int(v) for v in region)
    img = Image.open(proj / "source.png").convert("RGBA").crop((x0, y0, x1, y1))
    bg = Image.new("RGBA", img.size, (128, 128, 128, 255)); bg.alpha_composite(img)
    bg = bg.convert("RGB").resize((img.width * scale, img.height * scale), Image.LANCZOS)
    path = proj / "work" / out; bg.save(path)
    # Qwen-VL answers on a 0-1000 grid relative to the image, whatever its pixel size.
    back = lambda p: [round(x0 + p[0] / 1000 * (x1 - x0), 1), round(y0 + p[1] / 1000 * (y1 - y0), 1)]
    return path, bg.size, back


def clampbox(x0, y0, x1, y1, w, h):
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def main():
    proj = Path(sys.argv[1])
    (proj / "work").mkdir(exist_ok=True)
    draft = json.loads((proj / "rig.draft.json").read_text())
    W, H = draft["image"]["width"], draft["image"]["height"]
    P = lambda p: [round(p[0]), round(p[1])]

    # 1. Survey: landmarks off the full 50 px grid.
    img, _, back = crop(proj, "crop-full.png", (0, 0, W, H), 1)
    raw = ask(img, f"""This is a character portrait.
Read these points as [x, y] on a 0-1000 scale relative to the image (0,0 top-left, 1000,1000 bottom-right). "Left" and "right" mean the image's left and right.
{{"hair_box": [x0, y0, x1, y1]  (bounding box of all hair and head accessories, excluding hair below the shoulders),
 "chin": [x, y], "nose_tip": [x, y], "neck": [x, y] (centre of the neck, halfway between chin and collarbone),
 "collarbone_y": y, "shoulder_left": [x, y], "shoulder_right": [x, y] (top of each shoulder),
 "eye_left": [x0, y0, x1, y1], "eye_right": [x0, y0, x1, y1] (box around each eye including lashes),
 "brow_y": y (eyebrows), "mouth_left": [x, y], "mouth_right": [x, y] (lip corners),
 "ear_left": [x, y], "ear_right": [x, y] (where each ear is, even if hidden by hair),
 "hand_near_face": true/false}}""", proj / "work/qwen-survey.json")
    sx, sy = W / 1000, H / 1000
    conv = lambda v, axis: v * (sx if axis == 0 else sy)
    s = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) == 4:
            s[k] = [conv(v[0], 0), conv(v[1], 1), conv(v[2], 0), conv(v[3], 1)]
        elif isinstance(v, list):
            s[k] = [conv(v[0], 0), conv(v[1], 1)]
        elif k.endswith("_y"):
            s[k] = conv(v, 1)
        else:
            s[k] = v
    if s.get("hand_near_face"):
        print("warning: Qwen sees a hand near the face; the hand rig is not drafted")

    eL, eR = s["eye_left"], s["eye_right"]
    ec = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    (eLx, eLy), (eRx, eRy) = ec(eL), ec(eR)
    eye_y, eye_dist = (eLy + eRy) / 2, eRx - eLx
    chin, nose, neck = s["chin"], s["nose_tip"], s["neck"]
    hx0, hy0, hx1, hy1 = s["hair_box"]
    collar = s["collarbone_y"]
    mL, mR = s["mouth_left"], s["mouth_right"]
    mcx, mcy = (mL[0] + mR[0]) / 2, (mL[1] + mR[1]) / 2

    # 2. Eyes: opening and roi polygons off a 5 px grid around each eye.
    eyes = []
    for side, b in (("left", eL), ("right", eR)):
        pad = 30
        region = clampbox(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, W, H)
        outer = "leftmost" if side == "left" else "rightmost"
        img, (cw, ch), back = crop(proj, f"crop-eye-{side}.png", region, 4)
        e = ask(img, f"""This image shows one eye close up. Give positions as [x, y] on a 0-1000 scale relative to this image (0,0 top-left, 1000,1000 bottom-right).
"opening": a polygon of 16 [x, y] points tracing the visible eye white and iris, just inside the lashes and lower lid line. Start at the outer corner of the eye (its {outer} point) and go around once.
"roi": a polygon of 12 [x, y] points a little outside the whole lash line and lower lid line, staying off hair.
{{"opening": [[x, y], ...], "roi": [[x, y], ...]}}""", proj / f"work/qwen-eye-{side}.json")
        eyes.append({"opening": [P(back(p)) for p in e["opening"]], "roi": [P(back(p)) for p in e["roi"]]})

    # 3. Mouth: corners and centre off a 5 px grid.
    half = abs(mR[0] - mL[0]) / 2
    region = clampbox(mcx - half - 40, mcy - 40, mcx + half + 40, mcy + 40, W, H)
    img, (cw, ch), back = crop(proj, "crop-mouth.png", region, 4)
    m = ask(img, f"""This image shows a mouth close up. Give positions as [x, y] on a 0-1000 scale relative to this image (0,0 top-left, 1000,1000 bottom-right).
Read the closed mouth line (where the lips meet): "left_corner" and "right_corner" [x, y], and "middle" [x, y] (where the lips meet at the centre).
{{"left_corner": [x, y], "right_corner": [x, y], "middle": [x, y]}}""", proj / "work/qwen-mouth.json")
    lc, rc, mid = back(m["left_corner"]), back(m["right_corner"]), back(m["middle"])
    mcx, mcy = (lc[0] + rc[0]) / 2, (lc[1] + rc[1]) / 2
    half = math.dist(lc, rc) / 2
    angle = math.atan2(rc[1] - lc[1], rc[0] - lc[0])

    # 4. Hair strands off a 20 px grid of the hair box.
    region = clampbox(hx0 - 10, hy0 - 10, hx1 + 10, max(hy1, chin[1]) + 10, W, H)
    img, (cw, ch), back = crop(proj, "crop-hair.png", region, 1)
    st = ask(img, f"""This image shows a character's head and hair. Give positions as [x, y] on a 0-1000 scale relative to this image (0,0 top-left, 1000,1000 bottom-right).
List the hair locks that should sway: 3 to 5 bang locks over the forehead if there are bangs, and 2 locks on each side of the face.
For each, 4 [x, y] points from root to tip, every point on hair (not skin or background), and the lock's visible width on the same 0-1000 scale.
{{"strands": [{{"name": "bang1", "nodes": [[x, y], [x, y], [x, y], [x, y]], "width": px}}, ...]}}""", proj / "work/qwen-strands.json")
    strands = [{"name": re.sub(r"[^A-Za-z0-9_-]", "_", t["name"])[:24] or f"lock{i}", "nodes": [P(back(p)) for p in t["nodes"]][:4],
                "sigma": max(6, round(t.get("width", 30) / 1000 * (region[2] - region[0]) / 2)), "k": 1, "max": 9}
               for i, t in enumerate(st["strands"]) if len(t.get("nodes", [])) >= 4]
    seen = set()
    for t in strands:  # names must be distinct
        while t["name"] in seen:
            t["name"] += "_b"
        seen.add(t["name"])

    # 5. Derive everything else (code, not the model).
    face_cx = (eLx + eRx) / 2
    # Centre on the hair + chin span, not "slightly above the eyes": with big hair the guide's centre
    # leaves the lower hair outside and the face shears under it (Ben's fix on Angharad, 2026-11-09).
    head_cy = (hy0 + max(hy1, chin[1])) / 2
    head = {"cx": round(face_cx), "cy": round(head_cy),
            "rx": round(max(face_cx - hx0, hx1 - face_cx) + 10), "ry": round(max(head_cy - hy0, chin[1] - head_cy) + 10),
            "shiftX": round(0.022 * W), "shiftY": round(0.016 * H), "pivotX": round(neck[0]), "pivotY": round(neck[1]),
            "maxRoll": 0.15, "weightBand": [round(chin[1] + 10), round(max(collar, chin[1] + 40))],
            "turnBand": [round(chin[1] + 10), round(max(collar, chin[1] + 40) + 60)]}
    shL, shR = s["shoulder_left"], s["shoulder_right"]
    span = shR[0] - shL[0]
    body = {"pivotX": round(W / 2), "pivotY": round(H * 1.37), "maxRoll": 0.035,
            "breathBand": [round(collar + 20), H], "rollBand": [round(collar), H],
            "chest": {"cx": round((shL[0] + shR[0]) / 2), "cy": round((collar + H) / 2), "rx": round(0.4 * span), "ry": round((H - collar) / 2 * 0.8)},
            "shoulders": [{"cx": round(p[0]), "cy": round(p[1]), "rx": round(0.12 * W), "ry": round(0.1 * H)} for p in (shL, shR)]}
    ell = lambda cx, cy, rx, ry: {"cx": round(cx), "cy": round(cy), "rx": max(1, round(rx)), "ry": max(1, round(ry))}
    chin_gap = chin[1] - mcy
    face = {"nose": ell(nose[0], nose[1], 0.18 * eye_dist, 0.15 * eye_dist),
            "mouth": ell(mcx, mcy, half + 8, 0.2 * eye_dist),
            "eyeA": ell(eLx, eLy, (eL[2] - eL[0]) / 2 + 8, (eL[3] - eL[1]) / 2 + 6),
            "eyeB": ell(eRx, eRy, (eR[2] - eR[0]) / 2 + 8, (eR[3] - eR[1]) / 2 + 6),
            "earL": ell(*s["ear_left"], 0.15 * eye_dist, 0.35 * eye_dist), "earR": ell(*s["ear_right"], 0.15 * eye_dist, 0.35 * eye_dist),
            "brow": {**ell(face_cx, s["brow_y"], eye_dist, 0.1 * eye_dist), "band": [round(s["brow_y"] + 0.12 * eye_dist), round(eye_y - 0.05 * eye_dist)]},
            "jaw": {**ell(face_cx, chin[1] - 0.25 * chin_gap, 0.7 * eye_dist, 0.3 * chin_gap),
                    "band": [round(mcy + 0.45 * chin_gap), round(mcy + 0.7 * chin_gap)]}}
    mouth = {"cx": round(mcx, 1), "cy": round(mcy, 1), "angle": round(angle, 3), "halfLen": round(half),
             "bow": round(mid[1] - mcy, 1), "area": {"cx": round(mcx), "cy": round(mcy), "rx": round(half + 8), "ry": round(max(20, 0.5 * chin_gap)), "angle": round(angle, 3)}}
    fine = clampbox(int(hx0 - 20), int(hy0 - 20), int(hx1 + 20), int(max(hy1, chin[1]) + 40), W, H)
    draft.update({"head": head, "body": body, "face": face, "eyes": eyes, "mouth": mouth,
                  "cheeks": [P((eLx, nose[1] - 0.05 * eye_dist)), P((eRx, nose[1] - 0.05 * eye_dist))], "strands": strands,
                  "mesh": {"baseCell": 14, "fine": dict(zip(("x0", "y0", "x1", "y1"), fine), cell=7), "handCell": 9, "tasselCell": 6,
                           "eyeBallCell": 4, "eyeCell": 3, "spriteCell": 4},
                  "view": {"padTop": 0, "padSide": 0, "gazeCenter": P((face_cx, eye_y)),
                           "irisTravel": round(0.15 * ((eL[2] - eL[0]) + (eR[2] - eR[0])) / 2, 1)}})
    (proj / "rig.draft.json").write_text(json.dumps(draft, indent=2))
    print("wrote", proj / "rig.draft.json")


if __name__ == "__main__":
    main()
