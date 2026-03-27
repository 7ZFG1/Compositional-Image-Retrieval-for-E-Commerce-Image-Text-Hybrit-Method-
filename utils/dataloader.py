import os
import json
import random
from torch.utils.data import Dataset
import clip

class FashionIQDataset(Dataset):
    def __init__(self, config, features_dict):
        self.config = config
        self.features_dict = features_dict
        self.data = []
        self.all_valid_images = []
        
        print(f"[*] {config['SPLIT'].upper()} loading database...")
        
        for cat in config['CATEGORIES']:
            # Load captions
            cap_path = os.path.join(config['ROOT_DIR'], "captions", f"cap.{cat}.{config['SPLIT']}.json")
            with open(cap_path, 'r') as f:
                self.data.extend(json.load(f))
                
            # Negative sampling
            split_path = os.path.join(config['ROOT_DIR'], "image_splits", f"split.{cat}.{config['SPLIT']}.json")
            with open(split_path, 'r') as f:
                self.all_valid_images.extend(json.load(f))
                
        self.all_valid_images = list(set(self.all_valid_images))

    def __len__(self):
        return len(self.data) * 2 # There are 2 captions per candidate image

    def __getitem__(self, idx):
        item = self.data[idx // 2]
        caption_text = item['captions'][idx % 2]
        
        cand_id = item['candidate']
        target_id = item['target']
        
        # Select a random negative image ID that is different from the target
        neg_id = random.choice(self.all_valid_images)
        while neg_id == target_id:
            neg_id = random.choice(self.all_valid_images)
            
        # Fetch features from the pre-extracted dictionary
        cand_feat = self.features_dict[cand_id]
        target_feat = self.features_dict[target_id]
        neg_feat = self.features_dict[neg_id]
        
        # Tokenize the modification text using CLIP's tokenizer
        text_tokens = clip.tokenize(caption_text, truncate=True).squeeze(0)
        
        return cand_feat, text_tokens, target_feat, neg_feat