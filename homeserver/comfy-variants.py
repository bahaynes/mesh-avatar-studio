"""Draw mesh-avatar variants on a Flux.2 Klein ComfyUI: masked edit of source.png, pasted back over the source.

usage: uv run --with pillow --with numpy --with opencv-python-headless homeserver/comfy-variants.py projects/<name> [variant ...]
Reads projects/<name>/variant-requests/<v>/{mask.png,prompt.md}, writes projects/<name>/variants/<v>.png.
ponytail: one-off tool for this machine's ComfyUI (COMFY env, default 10.10.10.114:8188); node names are
the Klein stack from the Marinara FluxEdit connection. Upgrade path: tools/ if it proves general.
"""
import io, json, os, re, sys, time, urllib.request, uuid
from pathlib import Path
from PIL import Image

COMFY = os.environ.get("COMFY", "http://10.10.10.114:8188")
DEFAULT = ["eyes_closed", "eyes_half", "eyes_smile", "eyes_sad", "eyes_surprise", "mouth_a", "mouth_a_half", "mouth_i", "mouth_o", "mouth_angry", "mouth_sad"]
# A partial set breaks mesh-avatar runtimes before 0.3.1 (Marinara bundled 0.2.1), so default to all.


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


def instruction(prompt_md: str) -> str:
    # The second paragraph is the drawing instruction; the rest is file-handling text Flux can't use.
    paras = [p.strip() for p in prompt_md.split("\n\n") if p.strip()]
    text = paras[1] if len(paras) > 1 else paras[0]
    # Upstream's texts (tools/variant-requests.py) say "Japanese vowel" for the mouth shapes, which Flux doesn't
    # know, and a few words push style ("moist", "wavering", "clear highlights"). Say each shape plainly. Name only
    # what to draw: Flux reads "no lipstick" or "full lips" as a request for them.
    for up, ours in REWRITE.items():
        text = text.replace(up, ours)
    if "mouth" in text.lower() or "lips" in text.lower():
        text += (" Keep the mouth exactly as wide as in the reference, its corners where they are, and the lips exactly"
                 " as full or thin, bare and natural, same colour and shading; only the opening changes.")
    return text + " Match the reference image exactly in iris colour, lip colour, line weight and shading; add nothing it does not have."


def workflow(src, mask, text, w, h, seed):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux-2-klein-9b-Q5_K_M.gguf"}},
        "clip": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "Qwen3-8B-Q5_K_M.gguf", "type": "flux2"}},
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


def match_tint(src, edit, mask_path, band=6):
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
    skin = lambda x: (x[..., 0] > 200) & (x[..., 1] > 150) & (x[..., 0] - x[..., 2] > 40) & (x[..., 0] - x[..., 2] < 110)
    m = (skin(a) & skin(b)).astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 12)
    cover = blur(m)
    # ponytail: the skin test is a fixed warm-skin colour range; grey, blue or very dark skin gets no correction.
    # Upgrade path: take the skin range from the source pixels around the mask.
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


# Mouths drawn narrower than the resting lips make the mouth shrink before it opens. "o" is narrow on purpose.
WIDTH_MATCHED = {"mouth_a", "mouth_a_half", "mouth_i", "mouth_angry", "mouth_sad"}
WIDTH_TOL = float(os.environ.get("WIDTH_TOL", 0.08))  # allowed share narrower/wider than the source lips
MAX_STRETCH = float(os.environ.get("MAX_STRETCH", 1.3))  # beyond this a sideways stretch distorts teeth; flag instead


def lip_extent(img, mask_path, band=6):
    """Left/right x of the mouth: pixels inside the mask that differ from the skin colour sampled on the
    mask's inner edge (no lip-colour assumption), largest blob through the mask centre row band."""
    import numpy as np
    from PIL import ImageFilter
    region = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0)
    inner = region.filter(ImageFilter.MinFilter(2 * band + 1))
    r = np.asarray(region) > 0
    ring = r & ~(np.asarray(inner) > 0)
    a = np.asarray(img.convert("RGB"), float)
    skin = np.median(a[ring], axis=0)
    off = (np.abs(a - skin).sum(axis=2) > 90) & (np.asarray(inner) > 0)
    cols = np.nonzero(off.sum(axis=0) >= 3)[0]
    return (int(cols.min()), int(cols.max())) if len(cols) else None


def stretch_to(src_ext, img, ext, mask_path):
    """Scale the edited mouth sideways about its centre so its width matches the source lips."""
    import numpy as np
    from PIL import ImageFilter
    sx = (src_ext[1] - src_ext[0]) / max(1, ext[1] - ext[0])
    cx = (ext[0] + ext[1]) / 2
    region = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0)
    x0, y0, x1, y1 = region.getbbox()
    crop = img.crop((x0, y0, x1, y1))
    nw = round(crop.width * sx)
    lay = img.copy()
    lay.paste(crop.resize((nw, crop.height), Image.LANCZOS), (round(cx - (cx - x0) * sx), y0))
    keep = region.filter(ImageFilter.MinFilter(13)).filter(ImageFilter.GaussianBlur(3))
    return Image.composite(lay, img, keep)


def judge(qwen, proj, v, source, mask_path, tries, text):
    """Show Qwen the original next to each try (cropped to the mask) and let it pick one."""
    from PIL import ImageDraw
    m = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0).getbbox()
    pad = 40
    box = (max(0, m[0] - pad), max(0, m[1] - pad), min(source.width, m[2] + pad), min(source.height, m[3] + pad))
    panels = [source] + tries
    cw, ch = box[2] - box[0], box[3] - box[1]
    scale = max(1, 400 // cw)
    sheet = Image.new("RGB", (cw * scale, ch * scale * len(panels) + 30 * len(panels)), "white")
    for i, im in enumerate(panels):
        y = i * (ch * scale + 30)
        flat = Image.new("RGBA", im.size, (128, 128, 128, 255)); flat.alpha_composite(im.convert("RGBA"))
        sheet.paste(flat.convert("RGB").crop(box).resize((cw * scale, ch * scale)), (0, y + 30))
        ImageDraw.Draw(sheet).text((6, y + 6), "ORIGINAL" if i == 0 else f"TRY {i}", fill="black")
    path = proj / "work" / f"judge-{v}.png"; sheet.save(path)
    try:
        a = _judge_ask(qwen, path, text, proj, v)
    except SystemExit as e:  # Qwen's pick is advisory: a malformed answer must not stop the run
        m = re.search(r'"best"\s*:\s*(\d+)', str(e))
        return (int(m.group(1)) if m else 1), False, "judge answer unreadable; check by eye"
    best = int(a.get("best", 1))
    return (best if 1 <= best <= len(tries) else 1), bool(a.get("acceptable", True)), a.get("why", "")


def _judge_ask(qwen, path, text, proj, v):
    return qwen.ask(path, f"""The top panel is the ORIGINAL drawing. Each TRY panel is an edit that should do this: {text}
Pick the try that best does what was asked while keeping the original's eye colour, makeup, lip colour and line style, with both eyes matching (no wink, no odd colours, no smears).
{{"best": <try number>, "acceptable": true or false, "why": "<one sentence>"}}""", proj / "work" / f"qwen-judge-{v}.json")


def main():
    proj = Path(sys.argv[1])
    names = sys.argv[2:] or DEFAULT
    source = Image.open(proj / "source.png").convert("RGBA")
    out_dir = proj / "variants"
    out_dir.mkdir(exist_ok=True)
    src_name = upload(proj / "source.png", f"{proj.name}-source.png")
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_rig", Path(__file__).with_name("qwen-rig.py"))
    qwen = importlib.util.module_from_spec(spec); spec.loader.exec_module(qwen)
    tries_n = int(os.environ.get("TRIES", 3))
    seed0 = int(os.environ.get("SEED", 1000))
    for i, v in enumerate(names):
        req = proj / "variant-requests" / v
        mask_name = upload(req / "mask.png", f"{proj.name}-{v}-mask.png")
        text = instruction((req / "prompt.md").read_text())
        # Per-character eye look (e.g. unusual irises) for the open-eye variants only: EYES_EXTRA="...".
        if v.startswith("eyes_") and v != "eyes_closed" and os.environ.get("EYES_EXTRA"):
            text += " " + os.environ["EYES_EXTRA"]
        tries = []
        for t in range(tries_n):
            rgb = Image.open(io.BytesIO(run(workflow(src_name, mask_name, text, *source.size, seed0 + 100 * i + t)))).convert("RGB")
            rgb.putalpha(source.getchannel("A"))  # Comfy drops alpha; the source's alpha is the right one
            rgb, delta = match_tint(source, rgb, req / "mask.png")
            if abs(delta).max() > 3:
                print(f"  {v} try {t + 1}: removed tint {delta.round(1).tolist()}", flush=True)
            tries.append(rgb)
            (proj / "work" / "tries").mkdir(exist_ok=True)
            rgb.save(proj / "work" / "tries" / f"{v}-{t + 1}.png")
        pool, note = list(range(len(tries))), ""
        src_ext = lip_extent(source, req / "mask.png") if v in WIDTH_MATCHED else None
        if src_ext:
            sw = src_ext[1] - src_ext[0]
            ext = [lip_extent(t, req / "mask.png") for t in tries]
            err = [abs((e[1] - e[0]) / sw - 1) if e else 9 for e in ext]
            print(f"  {v} widths vs source {sw}px: {[e[1] - e[0] if e else None for e in ext]}", flush=True)
            pool = [k for k in pool if err[k] <= WIDTH_TOL]
            if not pool:  # none close enough: stretch the closest one
                k = min(range(len(tries)), key=err.__getitem__)
                sx = sw / max(1, ext[k][1] - ext[k][0]) if ext[k] else 9
                if sx <= MAX_STRETCH:
                    tries[k] = stretch_to(src_ext, tries[k], ext[k], req / "mask.png")
                    note = f" (stretched try {k + 1} x{sx:.2f} to source width)"
                else:
                    note = f"  WIDTH OFF x{sx:.2f}, too far to stretch: check by eye"
                pool = [k]
        sub = [tries[k] for k in pool]
        best, ok, why = judge(qwen, proj, v, source, req / "mask.png", sub, text) if len(sub) > 1 else (1, True, "")
        best = pool[best - 1] + 1
        tries[best - 1].save(out_dir / f"{v}.png")
        why += note
        print(f"wrote {v}: try {best}/{tries_n}{'' if ok else '  NOT ACCEPTABLE, check by eye'} - {why}", flush=True)


if __name__ == "__main__":
    import urllib.parse
    main()
