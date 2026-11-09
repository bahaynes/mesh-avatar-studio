"""Draw mesh-avatar variants on a Flux.2 Klein ComfyUI: masked edit of source.png, pasted back over the source.

usage: uv run --with pillow --with numpy homeserver/comfy-variants.py projects/<name> [variant ...]
Reads projects/<name>/variant-requests/<v>/{mask.png,prompt.md}, writes projects/<name>/variants/<v>.png.
ponytail: one-off tool for this machine's ComfyUI (COMFY env, default 10.10.10.114:8188); node names are
the Klein stack from the Marinara FluxEdit connection. Upgrade path: tools/ if it proves general.
"""
import io, json, os, sys, time, urllib.request, uuid
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


def instruction(prompt_md: str) -> str:
    # The second paragraph is the drawing instruction; the rest is file-handling text Flux can't use.
    paras = [p.strip() for p in prompt_md.split("\n\n") if p.strip()]
    text = paras[1] if len(paras) > 1 else paras[0]
    # Flux re-rolls iris colour and makeup from scratch inside the mask unless told otherwise.
    # Flux doesn't know the Japanese vowel names; say the shape plainly.
    text = text.replace("the Japanese vowel あ (a) at half openness: a smaller, gently open mouth", "the lips slightly parted, a small gap between them")
    if "mouth" in text.lower() and "closed" not in text.lower():
        text += " Draw full lips with the same lipstick colour, lip shape and shading as the reference; only the opening changes."
    elif "closed" in text.lower() and "mouth" in text.lower():
        text += " Draw full, closed lips with the same lipstick colour, lip shape and shading as the reference; only the lip curve changes."
    return text + " Keep the same iris colour, eyeliner, lashes, lip colour and line art as the reference image."


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
    """Remove Flux's uniform skin-colour drift: compare a band just inside the mask edge (still skin, should
    match the source) and shift the whole edited area back by the mean difference."""
    import numpy as np
    from PIL import ImageFilter
    region = Image.open(mask_path).getchannel("A").point(lambda a: 255 if a == 0 else 0)
    inner = region.filter(ImageFilter.MinFilter(2 * band + 1))
    r = np.asarray(region) > 0
    ring = r & ~(np.asarray(inner) > 0)
    a, b = np.asarray(src.convert("RGB"), float), np.asarray(edit.convert("RGB"), float)
    delta = (b[ring] - a[ring]).mean(axis=0)
    out = b.copy()
    out[r] = np.clip(b[r] - delta, 0, 255)
    # Feather: fade the edit into the source over the band inside the mask edge, so no seam (the visible
    # oval) remains. Outside the mask stays identical to the source.
    w = np.asarray(inner.filter(ImageFilter.GaussianBlur(band / 2)), float)[..., None] / 255 * r[..., None]
    out = a * (1 - w) + out * w
    fixed = Image.fromarray(out.astype("uint8"), "RGB")
    fixed.putalpha(edit.getchannel("A"))
    return fixed, delta


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
    a = qwen.ask(path, f"""The top panel is the ORIGINAL drawing. Each TRY panel is an edit that should do this: {text}
Pick the try that best does what was asked while keeping the original's eye colour, makeup, lip colour and line style, with both eyes matching (no wink, no odd colours, no smears).
{{"best": <try number>, "acceptable": true or false, "why": "<one sentence>"}}""", proj / "work" / f"qwen-judge-{v}.json")
    best = int(a.get("best", 1))
    return (best if 1 <= best <= len(tries) else 1), bool(a.get("acceptable", True)), a.get("why", "")


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
        best, ok, why = judge(qwen, proj, v, source, req / "mask.png", tries, text) if tries_n > 1 else (1, True, "")
        tries[best - 1].save(out_dir / f"{v}.png")
        print(f"wrote {v}: try {best}/{tries_n}{'' if ok else '  NOT ACCEPTABLE, check by eye'} - {why}", flush=True)


if __name__ == "__main__":
    import urllib.parse
    main()
