"""Redraw a reference portrait (photo, other style) as a front-facing head-and-shoulders illustration.

usage: uv run --with pillow homeserver/restyle.py <reference> <out-prefix> "<subject>" [tries]
  <subject>: who and what to draw, subject first (e.g. "Head-and-shoulders portrait of the woman in the
  reference image, facing the viewer, ..."). The style text and LoRA match Marinara's "Line Art" profile on
  the LocalComfy(FluxEdit) connection. Writes <out-prefix>-1.png ... for a person to pick from.
APPEARANCE env: card appearance text, e.g. APPEARANCE="$(homeserver/card-appearance.sh marinara character <id>)".
ponytail: homeserver-only; style text copied from the profile (STYLE env overrides it).
"""
import io, json, os, sys, time, urllib.parse, urllib.request, uuid
from PIL import Image

COMFY = os.environ.get("COMFY", "http://10.10.10.114:8188")
STYLE = os.environ.get("STYLE", (
    "Hand-inked fantasy illustration in the tradition of 1990s tabletop sourcebooks, finished in a flat vector "
    "comic style. Bold clean outer contours with fine pen detail inside them, light cross-hatching in the shadows, "
    "flat colour fields with hard-edged graphic shadow shapes, and a restrained palette of muted earthy tones with "
    "one accent colour."))


def post(path, data, ctype):
    req = urllib.request.Request(COMFY + path, data=data, headers={"Content-Type": ctype})
    return json.load(urllib.request.urlopen(req, timeout=60))


def upload(img: Image.Image) -> str:
    buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"restyle-{b[:8]}.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
    return post("/upload/image", body, f"multipart/form-data; boundary={b}")["name"]


def workflow(ref, text, seed):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux-2-klein-9b-Q5_K_M.gguf"}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "outlined_illustration_klein9b.safetensors", "strength_model": 1, "model": ["unet", 0]}},
        "clip": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "Qwen3-8B-Q5_K_M.gguf", "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "full_encoder_small_decoder.safetensors"}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "refscale": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["ref", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1}},
        "refenc": {"class_type": "VAEEncode", "inputs": {"pixels": ["refscale", 0], "vae": ["vae", 0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["clip", 0]}},
        "posref": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["pos", 0], "latent": ["refenc", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "guider": {"class_type": "CFGGuider", "inputs": {"cfg": 1, "model": ["lora", 0], "positive": ["posref", 0], "negative": ["neg", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "sched": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": 1024, "height": 1024}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0], "latent_image": ["latent", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "restyle"}},
    }


def run(wf) -> bytes:
    pid = post("/prompt", json.dumps({"prompt": wf}).encode(), "application/json")["prompt_id"]
    for _ in range(600):
        time.sleep(2)
        hist = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30))
        if pid in hist:
            if hist[pid].get("status", {}).get("status_str") == "error":
                raise SystemExit(json.dumps(hist[pid]["status"])[:800])
            meta = hist[pid]["outputs"]["save"]["images"][0]
            return urllib.request.urlopen(f"{COMFY}/view?{urllib.parse.urlencode(meta)}", timeout=60).read()
    raise SystemExit("timed out")


def main():
    ref, prefix, subject = sys.argv[1], sys.argv[2], sys.argv[3]
    tries = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    name = upload(Image.open(ref))
    # APPEARANCE: the card's own description (homeserver/card-appearance.sh), so age, hair and build follow
    # the card rather than the reference photo alone.
    look = os.environ.get("APPEARANCE", "").strip()
    text = f"{subject} {'Appearance: ' + look + ' ' if look else ''}{STYLE} Plain light background, no text."
    for t in range(tries):
        out = f"{prefix}-{t + 1}.png"
        open(out, "wb").write(run(workflow(name, text, 7000 + t)))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
