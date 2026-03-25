"""
Local AI Image Editor
Uses InstructPix2Pix for instruction-based photo editing.
Run with:  python image_editor.py           (launches Gradio UI)
           python image_editor.py --cli      (CLI mode)
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

MODEL_ID = "timbrooks/instruct-pix2pix"
_pipeline = None  # cached pipeline


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_id: str = MODEL_ID):
    """Load (and cache) the InstructPix2Pix pipeline."""
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
        pipe.enable_xformers_memory_efficient_attention()

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
    """
    Edit *image* according to *instruction* using InstructPix2Pix.

    Parameters
    ----------
    image                : PIL Image (RGB)
    instruction          : natural-language edit instruction
    num_inference_steps  : diffusion steps (more = better quality, slower)
    image_guidance_scale : how closely to follow the original image (1–2.5)
    text_guidance_scale  : how closely to follow the text instruction (5–15)
    seed                 : random seed (-1 = random)
    """
    if not instruction.strip():
        raise ValueError("Please provide an edit instruction.")

    pipe = load_pipeline()

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=get_device()).manual_seed(seed)

    # Resize to 512×512 (model native resolution) preserving aspect ratio
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
    """Resize image so both sides are ≤ max_size and divisible by *multiple*."""
    w, h = image.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_w = int(w * scale) // multiple * multiple
    new_h = int(h * scale) // multiple * multiple
    return image.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    import gradio as gr

    with gr.Blocks(title="Local AI Image Editor") as demo:
        gr.Markdown(
            """
            # Local AI Image Editor
            Upload a photo and describe how you want it changed.
            The model runs **entirely on your machine** — no data leaves your device.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Original Image", type="pil")
                instruction = gr.Textbox(
                    label="Edit Instruction",
                    placeholder='e.g. "Make it look like a painting", "Add snow", "Turn it into night time"',
                    lines=2,
                )
                with gr.Accordion("Advanced settings", open=False):
                    steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps")
                    img_cfg = gr.Slider(1.0, 3.0, value=1.5, step=0.05,
                                        label="Image guidance (higher = closer to original)")
                    txt_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5,
                                        label="Text guidance (higher = follows instruction more)")
                    seed_box = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

                run_btn = gr.Button("Edit Image", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Edited Image", type="pil")
                status = gr.Textbox(label="Status", interactive=False)

        def _run(img, instr, n_steps, i_cfg, t_cfg, seed):
            if img is None:
                return None, "Please upload an image first."
            try:
                out = edit_image(
                    image=img,
                    instruction=instr,
                    num_inference_steps=int(n_steps),
                    image_guidance_scale=float(i_cfg),
                    text_guidance_scale=float(t_cfg),
                    seed=int(seed),
                )
                return out, "Done!"
            except Exception as exc:
                return None, f"Error: {exc}"

        run_btn.click(
            fn=_run,
            inputs=[input_image, instruction, steps, img_cfg, txt_cfg, seed_box],
            outputs=[output_image, status],
        )

        gr.Examples(
            examples=[
                [None, "Make it look like a watercolor painting"],
                [None, "Turn it into a winter scene with snow"],
                [None, "Make it look like nighttime"],
                [None, "Add a dramatic sunset sky"],
                [None, "Make it look vintage / retro"],
            ],
            inputs=[input_image, instruction],
        )

    return demo, gr


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def run_cli():
    parser = argparse.ArgumentParser(
        description="Local AI Image Editor (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("instruction", help='Edit instruction, e.g. "Make it look like a painting"')
    parser.add_argument("-o", "--output", default=None, help="Output path (default: <input>_edited.png)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--img-cfg", type=float, default=1.5, help="Image guidance scale")
    parser.add_argument("--txt-cfg", type=float, default=7.5, help="Text guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = random)")
    args = parser.parse_args(sys.argv[2:])  # skip '--cli'

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] File not found: {input_path}")
        sys.exit(1)

    output_path = args.output or input_path.parent / f"{input_path.stem}_edited.png"

    image = Image.open(input_path).convert("RGB")
    print(f"[info] Editing '{input_path}' with instruction: \"{args.instruction}\"")

    result = edit_image(
        image=image,
        instruction=args.instruction,
        num_inference_steps=args.steps,
        image_guidance_scale=args.img_cfg,
        text_guidance_scale=args.txt_cfg,
        seed=args.seed,
    )

    result.save(output_path)
    print(f"[info] Saved edited image to '{output_path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        demo, gr = build_ui()
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
