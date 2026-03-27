import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
import time
import yaml
import gradio as gr

from utils.model import FashionCombiner
from utils.vector_db import VectorDBManager

# ==========================================
# 1. INITIALIZATION: CONFIG & MODEL LOADING
# ==========================================
# Note: Models and database are loaded ONLY ONCE upon startup.

config_path = "config/config_v0.0.0.yml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

print("[*] Loading CLIP and Combiner models...")
clip_model, preprocess = clip.load(cfg['CLIP_BACKBONE'], device=cfg['DEVICE'])
clip_model.eval()

combiner = FashionCombiner(embed_dim=cfg['EMBED_DIM']).to(cfg['DEVICE'])
if os.path.exists(cfg['MODEL_WEIGHTS']):
    combiner.load_state_dict(torch.load(cfg['MODEL_WEIGHTS'], map_location=cfg['DEVICE']))
    print(f"[*] Loaded model weights: {cfg['MODEL_WEIGHTS']}")
else:
    print("[!] Warning: Trained model weights not found, running with random weights!")
combiner.eval()

# Load Vector Database (Assuming loading from cache by default)
if not os.path.exists(cfg['FEATURE_CACHE']):
    raise FileNotFoundError(f"Feature cache not found: {cfg['FEATURE_CACHE']}")
print("[*] Loading pre-extracted image features...")
features_dict = torch.load(cfg['FEATURE_CACHE'], map_location="cpu")
image_base_dir = os.path.join(cfg['ROOT_DIR'], "images") 

# Initialize database manager
db_manager = VectorDBManager(cfg, features_dict)
print("[*] System ready! Starting Gradio interface...")

# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================
def process_query(ref_image, mod_text):
    """
    Takes a PIL Image and Text from Gradio, performs the search, 
    and returns a list of images to be displayed in the gallery.
    """
    if ref_image is None:
        gr.Warning("Please upload a reference image.")
        return [], "Error: Missing image."
    if not mod_text.strip():
        gr.Warning("Please enter a modification text.")
        return [], "Error: Missing text."

    start_time = time.time()

    # Process Reference Image (Gradio directly provides a PIL object)
    img_tensor = preprocess(ref_image).unsqueeze(0).to(cfg["DEVICE"])
    
    with torch.no_grad():
        # Encode image
        ref_feat = clip_model.encode_image(img_tensor).float()
        ref_feat = F.normalize(ref_feat, dim=-1)
        
        # Encode text
        text_tokens = clip.tokenize(mod_text).to(cfg["DEVICE"])
        text_feat = clip_model.encode_text(text_tokens).float()
        text_feat = F.normalize(text_feat, dim=-1)
        
        # Pass through combiner
        query_feat = combiner(ref_feat, text_feat)
        
    # Search in Vector DB (TOP 3 only)
    top_k = 3
    results = db_manager.search(query_feat, top_k=top_k)
    
    # Prepare results for Gallery format (Image, Caption)
    output_gallery = []
    for rank, (img_id, score) in enumerate(results, 1):
        # Determine file path
        if not os.path.splitext(img_id)[1]:
            target_path = os.path.join(image_base_dir, img_id + ".jpg")
        else:
            target_path = os.path.join(image_base_dir, img_id)
            
        try:
            target_img = Image.open(target_path).convert("RGB")
            # Tuple for Gradio gallery: (Image, Caption)
            caption = f"Rank: {rank} | ID: {img_id} | Score: {score:.4f}"
            output_gallery.append((target_img, caption))
        except Exception as e:
            print(f"[!] Failed to load image: {target_path}")
            
    end_time = time.time()
    time_info = f"⏱️ Processing Time: {(end_time - start_time)*1000:.2f} ms"
    
    return output_gallery, time_info

# ==========================================
# 3. GRADIO INTERFACE DESIGN (UI)
# ==========================================
# Custom CSS for a modern look with near-white headings
custom_css = """
.title { 
    text-align: center; 
    font-weight: bold; 
    margin-bottom: 0.5rem; 
    color: #f8f9fa; /* Near-white tone */
    text-shadow: 1px 1px 4px rgba(0,0,0,0.5); /* Ensures visibility on light backgrounds */
}
.subtitle { 
    text-align: center; 
    color: #e2e8f0; /* Soft light gray/white */
    margin-bottom: 2rem; 
    font-size: 1.1rem;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
}
"""

# Using Soft theme for a clean, professional appearance
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css) as demo:
    
    gr.Markdown("<h1 class='title'>Visual & Text-Based Fashion Search</h1>")
    gr.Markdown("<p class='subtitle'>Upload a reference image and describe the desired modification (e.g., 'is shorter and black').</p>")
    
    with gr.Row():
        # LEFT COLUMN: Inputs
        with gr.Column(scale=1):
            ref_image_input = gr.Image(type="pil", label="Upload Reference Image", height=350)
            mod_text_input = gr.Textbox(
                label="Modification Text", 
                placeholder="e.g., is shorter and black", 
                lines=2
            )
            search_button = gr.Button("🔍 Search and Combine", variant="primary", size="lg")
            
        # RIGHT COLUMN: Outputs (Top 3)
        with gr.Column(scale=2):
            gr.Markdown("<h3 style='color: #f8f9fa; text-shadow: 1px 1px 3px rgba(0,0,0,0.4);'>Top 3 Results</h3>")
            # Gallery component configured to show 3 images side-by-side
            result_gallery = gr.Gallery(
                label="Matched Images",
                show_label=False,
                elem_id="gallery",
                columns=[3], # Displays 3 images horizontally
                rows=[1],
                object_fit="contain",
                height=450
            )
            status_text = gr.Markdown("System Ready. Waiting for search query...")

    # Trigger function and connections on button click
    search_button.click(
        fn=process_query,
        inputs=[ref_image_input, mod_text_input],
        outputs=[result_gallery, status_text]
    )
    
    # Also trigger search when "Enter" is pressed in the text box
    mod_text_input.submit(
        fn=process_query,
        inputs=[ref_image_input, mod_text_input],
        outputs=[result_gallery, status_text]
    )

# Launch the Application
if __name__ == "__main__":
    # Set share=True to generate a temporary public link
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)