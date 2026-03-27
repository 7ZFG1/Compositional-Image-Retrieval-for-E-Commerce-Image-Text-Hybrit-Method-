import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import clip
import time
import yaml

from utils.model import FashionCombiner
from utils.vector_db import VectorDBManager

# ==========================================
# 1. CONFIG
# ==========================================
config_path = "config/config_v0.0.0.yml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("[*] Loading CLIP and Combiner models...")
clip_model, preprocess = clip.load(cfg['CLIP_BACKBONE'], device=cfg['DEVICE'])
clip_model.eval()

combiner = FashionCombiner(embed_dim=cfg['EMBED_DIM']).to(cfg['DEVICE'])
if os.path.exists(cfg['MODEL_WEIGHTS']):
    combiner.load_state_dict(torch.load(cfg['MODEL_WEIGHTS'], map_location=cfg['DEVICE']))
    print(f"[*] Loaded model weights: {cfg['MODEL_WEIGHTS']}")
else:
    print("[!] Attention: Trained model weights not found, running with random weights!")
combiner.eval()

# ==========================================
# 3. SELECT VECTOR DATABASE
# ==========================================
print("\n" + "="*50)
print(" SELECT VECTOR DATABASE ")
print("="*50)
print("[1] Use pre-extracted features (.pt file)")
print("[2] Build New Vector Database from Custom Image Folder")

choice = input("\nSelect (1/2): ").strip()

if choice == '2':
    custom_folder = input("Enter the full path of the image folder (e.g., C:/images or ./my_catalog): ").strip()
    if not os.path.isdir(custom_folder):
        print("[!] Error: Invalid folder path. Exiting program.")
        exit(1)
    features_dict = VectorDBManager.build_custom_db(custom_folder, clip_model, preprocess, cfg['DEVICE'])
    image_base_dir = custom_folder # Gösterim için referans klasör
else:
    if not os.path.exists(cfg['FEATURE_CACHE']):
        raise FileNotFoundError(f"{cfg['FEATURE_CACHE']} bulunamadı!")
    print("[*] Önceden çıkarılmış resim özellikleri yükleniyor...")
    features_dict = torch.load(cfg['FEATURE_CACHE'], map_location="cpu")
    image_base_dir = os.path.join(cfg['ROOT_DIR'], "images") # Fashion IQ klasörü

# Start the vector database manager (FAISS or Milvus)
db_manager = VectorDBManager(cfg, features_dict)

# ==========================================
# 4. INTERACTIVE INFERENCE
# ==========================================
print("\n" + "="*50)
print(" INTERACTIVE VISUAL + TEXT SEARCH ENGINE READY ")
print("="*50)

while True:
    print("\nTo exit, type 'q'.")
    # Ask for reference image (ID or file path) and modification text
    ref_input = input("Reference Image ID or File Path: ").strip()
    if ref_input.lower() == 'q':
        break
        
    mod_text = input("Modification Text (e.g., is shorter and black): ").strip()
    start_time = time.time()
    
    # ID or file path
    if os.path.isfile(ref_input):
        try:
            ref_img_pil = Image.open(ref_input).convert("RGB")
            img_tensor = preprocess(ref_img_pil).unsqueeze(0).to(cfg["DEVICE"])
            with torch.no_grad():
                ref_feat = clip_model.encode_image(img_tensor).float()
                ref_feat = F.normalize(ref_feat, dim=-1) # Vektörü normalize et
        except Exception as e:
            print(f"[!] Error while reading image: {e}")
            continue
    elif ref_input in features_dict:
        # Standart ID
        ref_feat = features_dict[ref_input].unsqueeze(0).to(cfg["DEVICE"])
        
        if not os.path.splitext(ref_input)[1]:
            ref_img_path = os.path.join(image_base_dir, ref_input + ".jpg")
        else:
            ref_img_path = os.path.join(image_base_dir, ref_input)
            
        ref_img_pil = Image.open(ref_img_path).convert("RGB")
    else:
        print("[!] Error: The ID you entered is not in the database or the file could not be found.")
        continue
        
    # Process modification text and combine with reference feature
    with torch.no_grad():
        text_tokens = clip.tokenize(mod_text).to(cfg["DEVICE"])
        text_feat = clip_model.encode_text(text_tokens).float()
        text_feat = F.normalize(text_feat, dim=-1)
        
        # Concat and pass through the combiner
        query_feat = combiner(ref_feat, text_feat)
        
    # Search in the vector database
    results = db_manager.search(query_feat, top_k=cfg["TOP_K"])
    end_time = time.time()
    
    # ==========================================
    # SHOW RESULTS
    # ==========================================
    print(f"\n--- Processing Time: {(end_time - start_time)*1000:.2f} ms ---")
    print(f"Best Matches:")

    # Show the reference image
    ref_img_pil.show(title="Reference Image")
    
    for rank, (img_id, score) in enumerate(results, 1):
        print(f"{rank}. {img_id} (Score: {score:.4f})")
        
        # Determine the image path
        if not os.path.splitext(img_id)[1]:
            target_path = os.path.join(image_base_dir, img_id + ".jpg")
        else:
            target_path = os.path.join(image_base_dir, img_id)
            
        try:
            target_img = Image.open(target_path).convert("RGB")
            target_img.show(title=f"Result {rank}")
        except Exception as e:
            print(f"[!] Failed to display image: {target_path}")