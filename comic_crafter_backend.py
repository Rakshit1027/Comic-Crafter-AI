from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont
import textwrap
import time
import os
from io import BytesIO
import base64
import tempfile
from flask_cors import CORS
import threading
from pyngrok import ngrok

# Install required packages
!pip install -q flask flask-cors torch transformers diffusers pillow PyPDF2 xformers pyngrok

# Try to import PyPDF2 with fallback
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    print("⚠ PyPDF2 not installed. PDF reference functionality will be disabled.")
    PDF_SUPPORT = False

# Initialize Flask app
app = Flask(_name_, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for models
text_gen = None
image_pipe = None
style_refs = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    if not PDF_SUPPORT:
        return None
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"⚠ Error extracting text from PDF: {e}")
        return None

def extract_style_references(pdf_path, num_pages=5):
    """Simplified style reference extraction"""
    return {
        "color_palette": ["#2C3E50", "#E74C3C", "#ECF0F1"],
        "panel_layout": "dynamic",
        "art_style": "manhwa"
    }

def load_models(pdf_reference_path=None):
    global text_gen, image_pipe, style_refs
    
    print("🔄 Loading models...")
    
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
    if pdf_reference_path and os.path.exists(pdf_reference_path) and PDF_SUPPORT:
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
    
    print("✅ Models loaded!")

def generate_story(prompt, panel_count, reference_text=None):
    ref_context = ""
    if reference_text:
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

def generate_panel_image(description, panel_num, style_refs=None, retries=2):
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
            
            # Dialogue (simplified)
            wrapped_text = textwrap.wrap(f"Panel {panel_num}", width=28)
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

def generate_comic(prompt, panel_count=4, pdf_reference=None):
    global style_refs
    
    # Process PDF reference
    reference_text = None
    if pdf_reference and PDF_SUPPORT:
        try:
            reference_text = extract_text_from_pdf(pdf_reference)
        except Exception as e:
            print(f"⚠ Error processing PDF: {e}")
    
    # Generate story
    story = generate_story(prompt, panel_count, reference_text)
    
    # Parse panels (simplified for demo)
    panels = []
    for i in range(1, int(panel_count)+1):
        panels.append({
            "description": f"Scene {i} for: {prompt}",
            "dialogue": f"Panel {i}"
        })
    
    # Generate images
    comic_images = []
    for i, panel in enumerate(panels):
        img = generate_panel_image(panel["description"], i+1, style_refs)
        
        # Convert to base64 for web
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        comic_images.append(img_str)
    
    return {
        "story": story,
        "pages": comic_images
    }

@app.route('/')
def home():
    return send_from_directory('static', 'index1.html')

@app.route('/generate-comic', methods=['POST'])
def generate_comic_endpoint():
    if 'prompt' not in request.form:
        return jsonify({"error": "Prompt is required"}), 400
    
    prompt = request.form['prompt']
    panel_count = request.form.get('panel_count', '4')
    
    pdf_reference = None
    if 'pdf_reference' in request.files:
        file = request.files['pdf_reference']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pdf_path)
            pdf_reference = pdf_path
    
    try:
        result = generate_comic(prompt, int(panel_count), pdf_reference)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=5000)

if _name_ == '_main_':
    # Load models first
    load_models()
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start ngrok tunnel
    try:
        ngrok_tunnel = ngrok.connect(5000)
        print('✨ Public URL:', ngrok_tunnel.public_url)
    except Exception as e:
        print(f"⚠ Ngrok error: {e}")
        print("Using Colab's built-in preview instead")
        from google.colab.output import eval_js
        print("Your app will be available at:", eval_js("google.colab.kernel.proxyPort(5000)"))
    
    # Keep the Colab cell running
    while True:
        time.sleep(1)