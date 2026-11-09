"""Make a studio source.png from a portrait: crop to head and shoulders, remove the background on ComfyUI.

usage: uv run --with pillow homeserver/prep-source.py <in-image> <out.png> [x0,y0,x1,y1]
The box is a square in input pixels; without it, Qwen finds the head and the box is derived from it.
The result is resized to 1024 x 1024.
ponytail: homeserver-only, for this machine's ComfyUI (COMFY env); BiRefNet settings copied from the
Marinara FluxEdit-Expression workflow. Upgrade path: fold into the avatar queue tool.
"""
import io, json, os, sys, time, urllib.parse, urllib.request, uuid
from PIL import Image

COMFY = os.environ.get("COMFY", "http://10.10.10.114:8188")


def post(path, data, ctype):
    req = urllib.request.Request(COMFY + path, data=data, headers={"Content-Type": ctype})
    return json.load(urllib.request.urlopen(req, timeout=60))


def auto_box(src):
    """Head-and-shoulders square: about 1.9 head heights wide, head near the top (Saskia's hand-picked box)."""
    import importlib.util, tempfile
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("qwen_rig", Path(__file__).with_name("qwen-rig.py"))
    qwen = importlib.util.module_from_spec(spec); spec.loader.exec_module(qwen)
    im = Image.open(src).convert("RGB")
    tmp = Path(tempfile.mkdtemp()) / "full.png"; im.save(tmp)
    a = qwen.ask(tmp, """This is a character portrait. On a 0-1000 scale relative to the image (0,0 top-left, 1000,1000 bottom-right), give
{"head_top": y (top of the hair), "chin": y, "face_center_x": x}""", tmp.with_suffix(".json"))
    W, H = im.size
    top, chin, cx = a["head_top"] / 1000 * H, a["chin"] / 1000 * H, a["face_center_x"] / 1000 * W
    size = min(W, H, round(1.9 * (chin - top)))
    y0 = max(0, min(H - size, round(top - 0.06 * size)))
    x0 = max(0, min(W - size, round(cx - size / 2)))
    print(f"auto box: {x0},{y0},{x0 + size},{y0 + size}")
    return (x0, y0, x0 + size, y0 + size)


def main():
    src, out = sys.argv[1], sys.argv[2]
    box = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else auto_box(src)
    if box[2] - box[0] != box[3] - box[1]:
        sys.exit("box must be square")
    img = Image.open(src).convert("RGB").crop(box).resize((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "PNG")
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"prep-{b[:8]}.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
    name = post("/upload/image", body, f"multipart/form-data; boundary={b}")["name"]
    wf = {
        "load": {"class_type": "LoadImage", "inputs": {"image": name}},
        "cut": {"class_type": "BiRefNet_Hugo", "inputs": {"image": ["load", 0], "model": "ZhengPeng7/BiRefNet",
                "load_local_model": False, "background_color_name": "transparency", "device": "auto"}},
        "m2i": {"class_type": "MaskToImage", "inputs": {"mask": ["cut", 1]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["m2i", 0], "filename_prefix": "prepmask"}},
    }
    pid = post("/prompt", json.dumps({"prompt": wf}).encode(), "application/json")["prompt_id"]
    for _ in range(300):
        time.sleep(2)
        hist = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30))
        if pid in hist:
            if hist[pid].get("status", {}).get("status_str") == "error":
                sys.exit(json.dumps(hist[pid]["status"])[:800])
            meta = hist[pid]["outputs"]["save"]["images"][0]
            mask = Image.open(io.BytesIO(urllib.request.urlopen(f"{COMFY}/view?{urllib.parse.urlencode(meta)}").read())).convert("L")
            img.putalpha(mask)
            img.save(out)
            print("wrote", out)
            return
    sys.exit("timed out")


if __name__ == "__main__":
    main()
