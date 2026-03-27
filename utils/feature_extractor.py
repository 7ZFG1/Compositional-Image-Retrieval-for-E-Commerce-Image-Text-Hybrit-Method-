import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

class FeatureExtractor:
    @staticmethod
    def extract_and_save(config, clip_model, preprocess):
        if os.path.exists(config["FEATURE_CACHE"]):
            print(f"[*] Found features in: {config['FEATURE_CACHE']}")
            return
            
        print("[*] Extracting image features (This will only be done once)...")
        image_dir = os.path.join(config["ROOT_DIR"], "images")
        
        # Find all unique image IDs from the splits to ensure we only process relevant images
        all_images = set()
        for cat in config["CATEGORIES"]:
            for split in ["train", "val", "test"]:
                split_path = os.path.join(config["ROOT_DIR"], "image_splits", f"split.{cat}.{split}.json")
                if os.path.exists(split_path):
                    with open(split_path, 'r') as f:
                        all_images.update(json.load(f))
                        
        all_images = list(all_images)
        features_dict = {}
        
        clip_model.eval()
        with torch.no_grad():
            for img_id in tqdm(all_images, desc="Resimler işleniyor"):
                img_path = os.path.join(image_dir, f"{img_id}.jpg")
                if not os.path.exists(img_path):
                    continue
                
                img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(config["DEVICE"])
                feat = clip_model.encode_image(img).float()
                feat = F.normalize(feat, dim=-1).squeeze(0).cpu() 
                features_dict[img_id] = feat
                
        torch.save(features_dict, config["FEATURE_CACHE"])
        print(f"[*] Özellikler kaydedildi: {config['FEATURE_CACHE']}")