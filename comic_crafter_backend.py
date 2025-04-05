import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import textwrap
import time
import re
import os
import PyPDF2
from io import BytesIO
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF Processing Functions
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_style_references(pdf_path, num_pages=5):
    """Extract visual style references from PDF"""
    # Placeholder implementation
    return {
        "color_palette": ["#2C3E50", "#E74C3C", "#ECF0F1"],
        "panel_layout": "dynamic",
        "art_style": "manhwa"
    }

# Initialize global variables
text_gen = None
image_pipe = None
style_refs = None
panels = []

# Load models
def load_models(pdf_reference_path=None):
    global text_gen, image_pipe, style_refs
    
    # Text generation
    text_gen = pipeline(
        "text-generation",
        model="distilgpt2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Configure generation
    text_gen.model.config.do_sample = True
    text_gen.model.config.temperature = 0.7
    text_gen.model.config.top_k = 50

    # Image generation
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )

    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
        variant="fp16"
    )

    # Apply PDF style references if available
    style_refs = None
    if pdf_reference_path and os.path.exists(pdf_reference_path):
        style_refs = extract_style_references(pdf_reference_path)
        print(f"🎨 Loaded style references: {style_refs}")

    # Device optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_pipe = image_pipe.to(device)

    if torch.cuda.is_available():
        try:
            image_pipe.enable_xformers_memory_efficient_attention()
        except:
            print("⚠ Xformers not available, using default attention")
        image_pipe.enable_attention_slicing()

    return text_gen, image_pipe, style_refs

# Story generation
def generate_story(prompt, panel_count, reference_text=None):
    ref_context = ""
    if reference_text:
        vectorizer = TfidfVectorizer()
        ref_vector = vectorizer.fit_transform([reference_text[:5000]])
        prompt_vector = vectorizer.transform([prompt])
        similarity = cosine_similarity(ref_vector, prompt_vector)[0][0]

        if similarity > 0.3:
            ref_context = "\n\nMaintain the style and tone similar to the reference material."

    template = f"""Create a {panel_count}-panel comic strip about: {prompt}{ref_context}

Panel 1: [Visual description]. [Action]. Characters: [list]. Dialogue: "[text]"
Panel 2: [Visual description]. [Action]. Characters: [list]. Dialogue: "[text]"
..."""

    result = text_gen(
        template,
        max_length=1024,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=50256
    )
    return result[0]["generated_text"]

# Image generation
def generate_panel_image(description, panel_num, style_refs=None, retries=2):
    global panels
    
    style_prompt = ""
    if style_refs:
        style_prompt = (
            f", {style_refs['art_style']} style, "
            f"color scheme: {','.join(style_refs['color_palette'])}, "
            f"{style_refs['panel_layout']} composition"
        )

    prompt = (
        f"Comic panel: {description}{style_prompt}, "
        "highly detailed, vibrant, clear outlines, "
        "dynamic perspective, professional comic art"
    )

    negative_prompt = (
        "blurry, low quality, distorted, bad anatomy, "
        "text, watermark, signature, extra limbs"
    )

    for attempt in range(retries + 1):
        try:
            image = image_pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=45,
                guidance_scale=8.5,
                height=512,
                width=512
            ).images[0]

            # Add comic elements
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("Arial.ttf", 24) or ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            # Speech bubble
            bubble_h = min(120, int(image.height * 0.25))
            bubble_y = image.height - bubble_h - 20
            draw.rounded_rectangle(
                [30, bubble_y, image.width-30, image.height-20],
                radius=25,
                fill="white",
                outline="black",
                width=3
            )

            # Dialogue
            dialogue = panels[panel_num-1]["dialogue"]
            wrapped_text = textwrap.wrap(dialogue, width=28)
            for i, line in enumerate(wrapped_text[:3]):
                draw.text(
                    (50, bubble_y + 25 + i*35),
                    line,
                    fill="black",
                    font=font,
                    stroke_width=1,
                    stroke_fill="white"
                )

            # Panel number
            draw.text(
                (40, 30),
                f"Panel {panel_num}",
                fill="white",
                font=font,
                stroke_width=3,
                stroke_fill="black"
            )

            return image

        except Exception as e:
            print(f"⚠ Error generating image: {e}")
            if attempt == retries:
                return Image.new('RGB', (512, 512), color=(240, 240, 240))
            time.sleep(2)

# Main generation function
def generate_comic(prompt, panel_count=4, pdf_reference=None):
    global panels, style_refs, text_gen, image_pipe

    # Process PDF reference
    reference_text = None
    if pdf_reference:
        try:
            reference_text = extract_text_from_pdf(pdf_reference)
        except Exception as e:
            print(f"⚠ Error processing PDF: {e}")

    # Generate story
    story = generate_story(prompt, panel_count, reference_text)

    # Parse panels
    panels = []
    for i in range(1, int(panel_count)+1):
        panel_marker = f"Panel {i}:"
        if panel_marker in story:
            start = story.find(panel_marker) + len(panel_marker)
            end = story.find(f"Panel {i+1}:") if i < panel_count else len(story)
            panel_text = story[start:end].strip()

            # Extract components
            dialogue = ""
            if "Dialogue:" in panel_text:
                dialogue_part = panel_text.split("Dialogue:")[1].strip()
                if '"' in dialogue_part:
                    dialogue = dialogue_part.split('"')[1].strip()
                panel_text = panel_text.split("Dialogue:")[0].strip()

            panels.append({
                "description": panel_text,
                "dialogue": dialogue if dialogue else f"Panel {i}"
            })
        else:
            panels.append({
                "description": f"Scene {i} for: {prompt}",
                "dialogue": f"Panel {i}"
            })

    # Generate images
    comic_images = []
    for i, panel in enumerate(panels):
        img = generate_panel_image(panel["description"], i+1, style_refs)
        comic_images.append(img)

    # Create pages
    pages = []
    for i in range(0, len(comic_images), 2):
        page = Image.new('RGB', (1024, 512), (250, 250, 250))
        if i < len(comic_images):
            page.paste(comic_images[i], (10, 10))
        if i+1 < len(comic_images):
            page.paste(comic_images[i+1], (522, 10))
        pages.append(page)

    return story, pages

# Gradio interface
with gr.Blocks(title="AI Comic Generator with Reference") as demo:
    gr.Markdown("""
    # 🎨 AI Comic Generator with Reference
    Create comics inspired by your reference PDF
    """)

    with gr.Row():
        with gr.Column(scale=2):
            pdf_input = gr.File(
                label="Upload Reference PDF (Optional)",
                type="filepath",
                file_types=[".pdf"]
            )
            prompt_input = gr.Textbox(
                label="Your Comic Idea",
                placeholder="e.g., A hunter awakening in a dungeon",
                lines=2
            )
            panel_slider = gr.Slider(2, 8, value=4, step=1, label="Number of Panels")
            generate_btn = gr.Button("Generate Comic", variant="primary")

        with gr.Column(scale=3):
            story_output = gr.Textbox(label="Generated Script", lines=8)

    with gr.Tabs():
        page_outputs = []
        for i in range(4):
            with gr.Tab(f"Page {i+1}"):
                page_output = gr.Image(label=f"Page {i+1}", type="pil")
                page_outputs.append(page_output)

    def generate_with_reference(pdf_file, prompt, panel_count):
        global text_gen, image_pipe, style_refs

        # Reload models with PDF reference if provided
        pdf_path = pdf_file.name if pdf_file else None
        text_gen, image_pipe, style_refs = load_models(pdf_path)

        story, pages = generate_comic(prompt, int(panel_count), pdf_path)

        # Format output
        output_images = []
        for i in range(4):
            if i < len(pages):
                output_images.append(pages[i])
            else:
                blank = Image.new('RGB', (1024, 512), (240, 240, 240))
                output_images.append(blank)

        return [story] + output_images

    generate_btn.click(
        fn=generate_with_reference,
        inputs=[pdf_input, prompt_input, panel_slider],
        outputs=[story_output] + page_outputs
    )

# Initial load
print("🔄 Loading models...")
text_gen, image_pipe, style_refs = load_models()
print("✅ Models ready!")

# Launch with error handling
try:
    demo.launch(share=True, server_port=7860)
except OSError:
    print("🔄 Port 7860 busy, trying 7861...")
    demo.launch(share=True, server_port=7861)
