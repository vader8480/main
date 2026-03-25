"""
Local AI Image Editor
Uses InstructPix2Pix for instruction-based photo editing.

Run:  python image_editor.py            (web UI on http://localhost:7860)
      python image_editor.py --cli ...  (CLI mode)
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

MODEL_ID = "timbrooks/instruct-pix2pix"
_pipeline = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_id: str = MODEL_ID):
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

    device = get_device()
    print(f"[info] Loading model '{model_id}' on {device} ...")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    _pipeline = pipe
    print("[info] Model loaded.")
    return _pipeline


# ---------------------------------------------------------------------------
# Core editing function
# ---------------------------------------------------------------------------

def edit_image(
    image: Image.Image,
    instruction: str,
    num_inference_steps: int = 50,
    image_guidance_scale: float = 1.5,
    text_guidance_scale: float = 7.5,
    seed: int = -1,
) -> Image.Image:
    if not instruction.strip():
        raise ValueError("Please provide an edit instruction.")

    pipe = load_pipeline()

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=get_device()).manual_seed(seed)

    image = image.convert("RGB")
    image = _resize_to_multiple(image, multiple=8, max_size=512)

    result = pipe(
        prompt=instruction,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=text_guidance_scale,
        generator=generator,
    )
    return result.images[0]


def _resize_to_multiple(image: Image.Image, multiple: int = 8, max_size: int = 512) -> Image.Image:
    w, h = image.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_w = int(w * scale) // multiple * multiple
    new_h = int(h * scale) // multiple * multiple
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Flask web UI
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Local AI Image Editor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f2f5; color: #1a1a2e; }
  header { background: #1a1a2e; color: #fff; padding: 18px 32px; }
  header h1 { font-size: 1.4rem; }
  header p  { font-size: 0.85rem; opacity: .7; margin-top: 4px; }
  main { max-width: 1100px; margin: 32px auto; padding: 0 16px; display: flex; gap: 24px; }
  .panel { background: #fff; border-radius: 12px; padding: 24px; flex: 1; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
  h2 { font-size: 1rem; margin-bottom: 16px; color: #444; }
  label { display: block; font-size: .85rem; font-weight: 600; margin-bottom: 6px; color: #555; }
  textarea { width: 100%; border: 1px solid #ddd; border-radius: 8px; padding: 10px;
             font-size: .9rem; resize: vertical; min-height: 70px; }
  .drop-zone { border: 2px dashed #aac; border-radius: 10px; padding: 30px;
               text-align: center; color: #888; cursor: pointer; margin-bottom: 16px;
               transition: background .2s; }
  .drop-zone:hover, .drop-zone.over { background: #f0f4ff; }
  .drop-zone img { max-width: 100%; max-height: 300px; border-radius: 8px; margin-top: 12px; }
  .row { display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
  .field { flex: 1; min-width: 120px; }
  .field input[type=range] { width: 100%; }
  .field input[type=number] { width: 100%; border: 1px solid #ddd; border-radius: 6px;
                               padding: 6px 8px; font-size: .85rem; }
  .val { font-size: .8rem; color: #888; text-align: right; }
  button { width: 100%; padding: 12px; border: none; border-radius: 8px;
           background: #1a1a2e; color: #fff; font-size: 1rem; cursor: pointer; margin-top: 8px; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  #status { margin-top: 12px; font-size: .85rem; color: #555; min-height: 20px; }
  #output-img { max-width: 100%; max-height: 500px; border-radius: 8px; display: none; margin-top: 12px; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #fff;
             border-top-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  details { margin-top: 16px; }
  summary { cursor: pointer; font-size: .85rem; color: #666; }
</style>
</head>
<body>
<header>
  <h1>Local AI Image Editor</h1>
  <p>Upload a photo &amp; describe how you want it changed. Everything runs on your machine.</p>
</header>
<main>
  <div class="panel">
    <h2>Input</h2>
    <div class="drop-zone" id="drop-zone">
      <div id="dz-text">Click or drag &amp; drop an image here</div>
      <img id="preview" src="" alt="">
    </div>
    <input type="file" id="file-input" accept="image/*" style="display:none">

    <label for="instruction">Edit instruction</label>
    <textarea id="instruction" placeholder='e.g. "Make it look like a painting", "Add snow", "Turn it into night time"'></textarea>

    <details>
      <summary>Advanced settings</summary>
      <br>
      <div class="row">
        <div class="field">
          <label>Steps <span class="val" id="steps-val">50</span></label>
          <input type="range" id="steps" min="10" max="100" value="50"
                 oninput="document.getElementById('steps-val').textContent=this.value">
        </div>
        <div class="field">
          <label>Image guidance <span class="val" id="img-cfg-val">1.5</span></label>
          <input type="range" id="img-cfg" min="1.0" max="3.0" step="0.05" value="1.5"
                 oninput="document.getElementById('img-cfg-val').textContent=parseFloat(this.value).toFixed(2)">
        </div>
        <div class="field">
          <label>Text guidance <span class="val" id="txt-cfg-val">7.5</span></label>
          <input type="range" id="txt-cfg" min="1.0" max="20.0" step="0.5" value="7.5"
                 oninput="document.getElementById('txt-cfg-val').textContent=parseFloat(this.value).toFixed(1)">
        </div>
        <div class="field">
          <label>Seed (-1 = random)</label>
          <input type="number" id="seed" value="-1">
        </div>
      </div>
    </details>

    <button id="run-btn" onclick="runEdit()">Edit Image</button>
    <div id="status"></div>
  </div>

  <div class="panel">
    <h2>Result</h2>
    <p id="result-hint" style="color:#aaa;font-size:.9rem">Edited image will appear here.</p>
    <img id="output-img" src="" alt="Edited image">
    <a id="download-link" style="display:none;margin-top:10px;display:none;
       font-size:.85rem;color:#1a1a2e" download="edited.png">Download</a>
  </div>
</main>

<script>
let imageData = null;

const dz = document.getElementById('drop-zone');
const fi = document.getElementById('file-input');

dz.addEventListener('click', () => fi.click());
fi.addEventListener('change', e => loadFile(e.target.files[0]));
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    imageData = e.target.result;
    const prev = document.getElementById('preview');
    prev.src = imageData;
    prev.style.display = 'block';
    document.getElementById('dz-text').style.display = 'none';
  };
  reader.readAsDataURL(file);
}

async function runEdit() {
  if (!imageData) { alert('Please upload an image first.'); return; }
  const instruction = document.getElementById('instruction').value.trim();
  if (!instruction) { alert('Please enter an edit instruction.'); return; }

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Editing...';
  document.getElementById('status').textContent = 'Running inference (first run downloads ~3 GB model)...';

  try {
    const resp = await fetch('/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageData,
        instruction,
        steps: parseInt(document.getElementById('steps').value),
        img_cfg: parseFloat(document.getElementById('img-cfg').value),
        txt_cfg: parseFloat(document.getElementById('txt-cfg').value),
        seed: parseInt(document.getElementById('seed').value),
      })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);

    const out = document.getElementById('output-img');
    out.src = 'data:image/png;base64,' + data.image;
    out.style.display = 'block';
    document.getElementById('result-hint').style.display = 'none';

    const dl = document.getElementById('download-link');
    dl.href = out.src;
    dl.style.display = 'block';
    dl.textContent = 'Download edited image';

    document.getElementById('status').textContent = 'Done!';
  } catch (err) {
    document.getElementById('status').textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Edit Image';
  }
}
</script>
</body>
</html>
"""


def run_server(host: str = "0.0.0.0", port: int = 7860):
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/")
    def index():
        return HTML

    @app.route("/edit", methods=["POST"])
    def edit():
        try:
            data = request.get_json(force=True)
            # Decode base64 image (data URL or raw base64)
            img_b64 = data["image"]
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            raw = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")

            result = edit_image(
                image=image,
                instruction=data["instruction"],
                num_inference_steps=int(data.get("steps", 50)),
                image_guidance_scale=float(data.get("img_cfg", 1.5)),
                text_guidance_scale=float(data.get("txt_cfg", 7.5)),
                seed=int(data.get("seed", -1)),
            )

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            out_b64 = base64.b64encode(buf.getvalue()).decode()
            return jsonify({"image": out_b64})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    print(f"[info] Starting server at http://localhost:{port}")
    app.run(host=host, port=port, debug=False, threaded=False)


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def run_cli():
    parser = argparse.ArgumentParser(
        description="Local AI Image Editor (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("instruction", help='Edit instruction')
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--img-cfg", type=float, default=1.5)
    parser.add_argument("--txt-cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(sys.argv[2:])

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] File not found: {input_path}")
        sys.exit(1)

    output_path = args.output or input_path.parent / f"{input_path.stem}_edited.png"
    image = Image.open(input_path).convert("RGB")
    print(f"[info] Editing '{input_path}' — \"{args.instruction}\"")

    result = edit_image(
        image=image,
        instruction=args.instruction,
        num_inference_steps=args.steps,
        image_guidance_scale=args.img_cfg,
        text_guidance_scale=args.txt_cfg,
        seed=args.seed,
    )
    result.save(output_path)
    print(f"[info] Saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_server()
