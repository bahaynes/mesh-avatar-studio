"""Draw mesh-avatar variants on a Flux.2 Klein ComfyUI: masked edit of source.png, pasted back over the source.

usage (or the /__picker page in Studio): uv run --with pillow --with numpy --with opencv-python-headless homeserver/comfy-variants.py projects/<name> [variant ...]
       ... comfy-variants.py projects/<name> --pick mouth_a=2 eyes_sad=1 ...
Reads projects/<name>/variant-requests/<v>/{mask.png,prompt.md}; writes every try to work/tries/<v>-<n>.png and a
contact sheet (original + tries) to work/sheet-<v>.png. It picks nothing: a person picks with --pick, which copies
the chosen try to variants/<v>.png.
ponytail: one-off tool for this machine's ComfyUI (COMFY env, default 10.10.10.114:8188); node names are
the Klein stack from the Marinara FluxEdit connection. Upgrade path: tools/ if it proves general.
"""
import io, json, os, sys, time, urllib.request, uuid
from pathlib import Path
from PIL import Image

COMFY = os.environ.get("COMFY", "http://10.10.10.114:8188")
DEFAULT = ["eyes_closed", "eyes_half", "eyes_smile", "eyes_sad", "eyes_surprise", "mouth_a", "mouth_a_half", "mouth_i", "mouth_o", "mouth_angry", "mouth_sad"]
# A partial set breaks mesh-avatar runtimes before 0.3.1 (Marinara bundled 0.2.1), so default to all.


# Shape-guided variants: Flux Klein copies its reference back for small mouth changes (oh, half-ah), so for
# these the reference is an edge map with the target outline drawn by code, plus the RefControl canny LoRA.
# ponytail: two hand-tuned outlines sized from the detected lips. Upgrade path: more shapes, per-character tuning.
GUIDED = [g for g in os.environ.get("GUIDE", "mouth_o,mouth_a_half").split(",") if g]
GUIDE_LORA = os.environ.get("GUIDE_LORA", "flux2_klein_9b_refcontrol_canny.safetensors")


def post(path, data, ctype):
    req = urllib.request.Request(COMFY + path, data=data, headers={"Content-Type": ctype})
    return json.load(urllib.request.urlopen(req, timeout=60))


def upload(png: Path, name: str) -> str:
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + png.read_bytes() + (
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n").encode()
    return post("/upload/image", body, f"multipart/form-data; boundary={b}")["name"]


REWRITE = {
    "Open the mouth in the Japanese vowel あ (a): a natural vertically open mouth, with a subtle tongue and upper teeth.":
        "Open the mouth naturally as when saying 'ah': opened vertically, with a little of the tongue and upper teeth showing.",
    "Draw the Japanese vowel あ (a) at half openness: a smaller, gently open mouth.":
        "Part the lips slightly, as when starting to say 'ah': a small gap between them.",
    "Draw the Japanese vowel い (i): a wide, narrow open smile with subtle upper teeth.":
        "Spread the lips wide and only slightly open, as when saying 'ee', with a little of the upper teeth showing and the mouth corners level.",
    "Draw the Japanese vowel お (o): a small rounded open mouth.":
        "Round the lips into a small open O, as when saying 'oh'.",
    ", softer and slightly moist,": ",",
    ", slightly smaller irises with clear highlights,": ", the irises unchanged in size and colour,",
    " in a gentle, soft curve, slightly wavering.": " in a gentle, soft curve.",
    " Do not draw tears on the cheeks.": " Keep the cheeks unchanged.",
    "curves gently downward, not the upward arc of a smiling closed eye.": "curves gently downward.",
}


def instruction(prompt_md: str, name: str = "") -> str:
    # The second paragraph is the drawing instruction; the rest is file-handling text Flux can't use.
    paras = [p.strip() for p in prompt_md.split("\n\n") if p.strip()]
    text = paras[1] if len(paras) > 1 else paras[0]
    # Upstream's texts (tools/variant-requests.py) say "Japanese vowel" for the mouth shapes, which Flux doesn't
    # know, and a few words push style ("moist", "wavering", "clear highlights"). Say each shape plainly. Name only
    # what to draw: Flux reads "no lipstick" or "full lips" as a request for them.
    for up, ours in REWRITE.items():
        text = text.replace(up, ours)
    if name.startswith("mouth"):
        text += " Keep the lips exactly as full or thin, same colour and shading." if os.environ.get("STYLE") else " Keep the lips exactly as full or thin, bare and natural, same colour and shading."
    if os.environ.get("STYLE"):  # generic drawing-style line, e.g. "Clean black ink outlines, flat cel colour."
        text += " " + os.environ["STYLE"]
    return text


def workflow(src, mask, text, w, h, seed):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": os.environ.get("UNET", "flux-2-klein-9b-Q4_K_M.gguf")}},
        "clip": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": os.environ.get("CLIP_GGUF", "Qwen3-8B-Q4_K_M.gguf"), "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "full_encoder_small_decoder.safetensors"}},
        "src": {"class_type": "LoadImage", "inputs": {"image": src}},
        "mask": {"class_type": "LoadImage", "inputs": {"image": mask}},  # MASK out = 1 where transparent = editable
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["src", 0], "vae": ["vae", 0]}},
        "noisemask": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["mask", 1]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["clip", 0]}},
        "ref": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["pos", 0], "latent": ["enc", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "guider": {"class_type": "CFGGuider", "inputs": {"cfg": 1, "model": ["unet", 0], "positive": ["ref", 0], "negative": ["neg", 0]}},
        "sched": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": w, "height": h}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["noisemask", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "paste": {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["src", 0], "source": ["dec", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["mask", 1]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["paste", 0], "filename_prefix": "meshvariant"}},
    }


def guided(wf, guide_name):
    """Swap the character reference for the edge-map reference, through the RefControl LoRA."""
    wf["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": GUIDE_LORA, "strength_model": float(os.environ.get("GUIDE_STRENGTH", 1.0))}}
    wf["guider"]["inputs"]["model"] = ["lora", 0]
    if os.environ.get("STYLE_LORA"):  # optional style LoRA stacked on the guide LoRA
        wf["slora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["lora", 0], "lora_name": os.environ["STYLE_LORA"], "strength_model": float(os.environ.get("STYLE_LORA_STRENGTH", 1.0))}}
        wf["guider"]["inputs"]["model"] = ["slora", 0]
    wf["guide"] = {"class_type": "LoadImage", "inputs": {"image": guide_name}}
    wf["genc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["guide", 0], "vae": ["vae", 0]}}
    wf["ref"]["inputs"]["latent"] = ["genc", 0]
    return wf


def run(wf) -> bytes:
    pid = post("/prompt", json.dumps({"prompt": wf}).encode(), "application/json")["prompt_id"]
    for _ in range(600):
        time.sleep(2)
        hist = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30))
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st)[:800])
            img = hist[pid]["outputs"]["save"]["images"][0]
            q = urllib.parse.urlencode(img)
            return urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=60).read()
    raise TimeoutError(pid)


def skin_hex(src, mask_path, ring=16):
    """Median colour of the opaque, bright (not line-art) source pixels in a ring just outside the mask."""
    import numpy as np
    from PIL import ImageFilter
    region = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0)
    r = np.asarray(region) > 0
    near = np.asarray(region.filter(ImageFilter.MaxFilter(2 * ring + 1))) > 0
    px = np.asarray(src.convert("RGBA"), float)
    sel = px[near & ~r & (px[..., 3] > 250)][:, :3]
    lum = sel.sum(1)
    sel = sel[lum >= np.median(lum)]  # drop outlines and shadow; keep the skin's lit tone
    c = np.median(sel, 0).round().astype(int)
    return "#%02x%02x%02x" % tuple(c), c


def lips_box(proj, src, skin_rgb):
    """Mouth centre and size from the rig (Qwen draft, fixed by hand in Studio); lip colour sampled there."""
    import numpy as np, cv2
    rig = json.loads((proj / "rig.json").read_text()) if (proj / "rig.json").exists() else json.loads((proj / "rig.draft.json").read_text())
    mo = rig["mouth"]
    cx, cy, half = round(mo["cx"]), round(mo["cy"]), round(mo["halfLen"])
    lh = max(12, round(mo["area"]["ry"] * .5))
    rgb = np.asarray(src.convert("RGB")).astype(int)
    m = np.zeros(rgb.shape[:2], np.uint8)
    cv2.ellipse(m, (cx, cy), (int(half * .8), int(lh * .8)), 0, 0, 360, 1, -1)
    lip = (m > 0) & (np.abs(rgb - skin_rgb).max(-1) > 45) & (rgb.sum(-1) > 90)  # not skin, not the black outline
    c = np.median(rgb[lip], 0).round().astype(int) if lip.any() else np.array(skin_rgb)
    return (cx, cy), 2 * half, lh, "#%02x%02x%02x" % tuple(c)


def guide_map(src, mask_path, v, centre, lw, lh):
    """Edge map of the face with the mouth erased and the target mouth outline drawn in."""
    import numpy as np, cv2
    rgb = np.asarray(src.convert("RGB"))
    edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)
    m = np.asarray(Image.open(mask_path).getchannel("A")) == 0
    edges[m] = 0
    cx, cy = centre  # the smile line runs past the mask; erase its ends too, or Flux joins them across the new mouth
    edges[max(0, cy - lh):cy + lh, max(0, cx - lw):cx + lw] = 0
    if v == "mouth_o":
        cv2.ellipse(edges, centre, (int(lw * .28), int(lw * .30)), 0, 0, 360, 255, 2)
        cv2.ellipse(edges, centre, (int(lw * .15), int(lw * .18)), 0, 0, 360, 255, 2)
    elif v == "mouth_a_half":
        cv2.ellipse(edges, centre, (int(lw * .45), int(lw * .20)), 0, 0, 360, 255, 2)
        cv2.ellipse(edges, centre, (int(lw * .30), int(lw * .06)), 0, 0, 360, 255, 2)
    else:
        raise SystemExit(f"{v}: no shape guide drawn for this variant")
    return Image.fromarray(edges).convert("RGB")


def match_tint(src, edit, mask_path, skin_rgb, band=6, tol=40):
    """Remove Flux's skin-colour drift inside the edit. It is not uniform (lighter toward the middle of the
    patch, e.g. +15 in green/blue there and ~0 at the edge), so measure source-minus-edit on pixels that are
    skin in both, smooth it into a field, and add that field across the patch. Lips, teeth and mouth interior
    are not skin, so they don't steer the correction."""
    import numpy as np, cv2
    from PIL import ImageFilter
    region = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0)
    inner = region.filter(ImageFilter.MinFilter(2 * band + 1))
    r = np.asarray(region) > 0
    a, b = np.asarray(src.convert("RGB"), float), np.asarray(edit.convert("RGB"), float)
    skin = lambda x: np.abs(x - skin_rgb).max(-1) < tol  # near the source's sampled skin colour
    m = (skin(a) & skin(b)).astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 12)
    cover = blur(m)
    field = np.stack([blur((a[..., c] - b[..., c]) * m) / np.maximum(cover, 1e-3) for c in range(3)], -1)
    field[cover < 0.02] = 0
    out = b.copy()
    out[r] = np.clip(b[r] + field[r], 0, 255)
    # Feather: fade the edit into the source over the band inside the mask edge, so no seam remains.
    # Outside the mask stays identical to the source.
    w = np.asarray(inner.filter(ImageFilter.GaussianBlur(band / 2)), float)[..., None] / 255 * r[..., None]
    out = a * (1 - w) + out * w
    fixed = Image.fromarray(out.astype("uint8"), "RGB")
    fixed.putalpha(edit.getchannel("A"))
    return fixed, -field[r & (m > 0)].mean(axis=0) if (r & (m > 0)).any() else np.zeros(3)


def sheet(proj, v, source, mask_path, tries):
    """Original next to each try, cropped to the mask, for a person to pick from."""
    from PIL import ImageDraw
    m = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0).getbbox()
    pad = 40
    box = (max(0, m[0] - pad), max(0, m[1] - pad), min(source.width, m[2] + pad), min(source.height, m[3] + pad))
    panels = [source] + tries
    cw, ch = box[2] - box[0], box[3] - box[1]
    scale = max(1, 400 // cw)
    out = Image.new("RGB", (cw * scale, ch * scale * len(panels) + 30 * len(panels)), "white")
    (proj / "work" / "crops").mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(panels):
        y = i * (ch * scale + 30)
        flat = Image.new("RGBA", im.size, (128, 128, 128, 255)); flat.alpha_composite(im.convert("RGBA"))
        crop = flat.convert("RGB").crop(box).resize((cw * scale, ch * scale))
        crop.save(proj / "work" / "crops" / f"{v}-{i}.png")  # 0 = original; the picker page shows these
        out.paste(crop, (0, y + 30))
        ImageDraw.Draw(out).text((6, y + 6), "ORIGINAL" if i == 0 else f"TRY {i}", fill="black")
    path = proj / "work" / f"sheet-{v}.png"; out.save(path)
    return path


def pick(proj, picks):
    import shutil
    (proj / "variants").mkdir(exist_ok=True)
    for p in picks:
        v, n = p.split("=")
        shutil.copy(proj / "work" / "tries" / f"{v}-{int(n)}.png", proj / "variants" / f"{v}.png")
        print(f"picked {v}: try {n}")


def main():
    proj = Path(sys.argv[1])
    if sys.argv[2:3] == ["--pick"]:
        return pick(proj, sys.argv[3:])
    overrides_path = proj / "work" / "prompt-overrides.json"  # written by the /__picker page
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    if sys.argv[2:3] == ["--prompts"]:  # default and current prompt per variant, for the picker page
        reqs = sorted(d for d in (proj / "variant-requests").iterdir() if d.is_dir())
        print(json.dumps({d.name: {"default": instruction((d / "prompt.md").read_text(), d.name), "override": overrides.get(d.name)} for d in reqs}))
        return
    if sys.argv[2:3] == ["--sheets"]:  # rebuild sheets and crops from saved tries, no ComfyUI
        source = Image.open(proj / "source.png").convert("RGBA")
        for v in sys.argv[3:] or DEFAULT:
            tries = sorted((proj / "work" / "tries").glob(f"{v}-*.png"), key=lambda f: int(f.stem.rsplit("-", 1)[1]))
            if tries:
                sheet(proj, v, source, proj / "variant-requests" / v / "mask.png", [Image.open(t).convert("RGBA") for t in tries])
        return
    names = sys.argv[2:] or DEFAULT
    source = Image.open(proj / "source.png").convert("RGBA")
    src_name = upload(proj / "source.png", f"{proj.name}-source.png")
    tries_n = int(os.environ.get("TRIES", 3))
    seed0 = int(os.environ.get("SEED", 1000))
    for i, v in enumerate(names):
        req = proj / "variant-requests" / v
        mask_name = upload(req / "mask.png", f"{proj.name}-{v}-mask.png")
        hexc, skin_rgb = skin_hex(source, req / "mask.png")
        # Naming the skin hex roughly halves Flux's skin drift.
        text = (overrides.get(v) or instruction((req / "prompt.md").read_text(), v)) + f" Keep the skin in color {hexc}."
        # Per-character eye look (e.g. unusual irises) for the open-eye variants only: EYES_EXTRA="...".
        if v not in overrides and v.startswith("eyes_") and v != "eyes_closed" and os.environ.get("EYES_EXTRA"):
            text += " " + os.environ["EYES_EXTRA"]
        guide_name = None
        if v in GUIDED:
            centre, lw, lh, lip_hex = lips_box(proj, source, skin_rgb)
            (proj / "work" / "guides").mkdir(parents=True, exist_ok=True)
            guide_map(source, req / "mask.png", v, centre, lw, lh).save(proj / "work" / "guides" / f"{v}.png")
            guide_name = upload(proj / "work" / "guides" / f"{v}.png", f"{proj.name}-{v}-guide.png")
            text = f"refcontrol {text} The lips are color {lip_hex}."
        tries = []
        (proj / "work" / "tries").mkdir(parents=True, exist_ok=True)
        for f in (proj / "work" / "tries").glob(f"{v}-*.png"):
            f.unlink()  # a rerun replaces this variant's tries, so no stale try survives with a lower count
        for t in range(tries_n):
            wf = workflow(src_name, mask_name, text, *source.size, seed0 + 100 * i + t)
            rgb = Image.open(io.BytesIO(run(guided(wf, guide_name) if guide_name else wf))).convert("RGB")
            rgb.putalpha(source.getchannel("A"))  # Comfy drops alpha; the source's alpha is the right one
            rgb, delta = match_tint(source, rgb, req / "mask.png", skin_rgb)
            if abs(delta).max() > 3:
                print(f"  {v} try {t + 1}: removed tint {delta.round(1).tolist()}", flush=True)
            tries.append(rgb)
            (proj / "work" / "tries").mkdir(exist_ok=True)
            rgb.save(proj / "work" / "tries" / f"{v}-{t + 1}.png")
        print(f"{v}: {tries_n} tries{', shape guide' if guide_name else ''}, skin {hexc}, sheet {sheet(proj, v, source, req / 'mask.png', tries)}", flush=True)
    print(f"pick: {sys.argv[0]} {proj} --pick " + " ".join(f"{v}=N" for v in names))

if __name__ == "__main__":
    import urllib.parse
    main()
