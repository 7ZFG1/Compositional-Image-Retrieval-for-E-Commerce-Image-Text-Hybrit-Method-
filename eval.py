import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm
import yaml

from utils.model import FashionCombiner, FashionCombiner_v2

def calculate_recall(query_feats, gallery_feats, target_ids, gallery_ids, ks=[10, 50]):
    """
    query_feats: [N, 512] - Queries produced by the combiner
    gallery_feats: [M, 512] - All candidate images in the gallery
    target_ids: [N] - Ground-truth target ID for each query
    gallery_ids: [M] - ID list of gallery images
    """
    # Similarity Matrix: [N, M]
    # (N: number of queries, M: total number of gallery images)
    sim_matrix = query_feats @ gallery_feats.T
    
    # Sort by highest score (indices: [N, M])
    _, indices = torch.sort(sim_matrix, descending=True, dim=1)
    
    recalls = {}
    for k in ks:
        correct = 0
        for i in range(len(target_ids)):
            # Take IDs of top-K results
            top_k_indices = indices[i, :k]
            top_k_ids = [gallery_ids[idx] for idx in top_k_indices]
            
            # Is the ground-truth target in top-K?
            if target_ids[i] in top_k_ids:
                correct += 1
        
        recalls[f"R@{k}"] = (correct / len(target_ids)) * 100
        
    return recalls

def run_evaluation(config, combiner, clip_model, features_dict, eval_subset=True):
    final_results = {}
    combiner.eval()
    clip_model.eval()

    for cat in config['CATEGORIES']:
        print(f"\n[*] Evaluating category: {cat.upper()}...")
        
        # 1. Prepare Gallery (all valid images in that category)
        split_path = os.path.join(config['ROOT_DIR'], "image_splits", f"split.{cat}.val.json")
        with open(split_path, 'r') as f:
            gallery_ids = json.load(f)
        
        gallery_feats = torch.stack([features_dict[img_id] for img_id in gallery_ids]).to(config['DEVICE'])

        # 2. Prepare Queries
        cap_path = os.path.join(config['ROOT_DIR'], "captions", f"cap.{cat}.val.json")
        with open(cap_path, 'r') as f:
            val_data = json.load(f)

        # Select a subset for quick evaluation (optional, can be removed for full eval)
        if eval_subset:
            random.seed(42)
            sample_size = int(len(val_data) * 0.2)
            val_data = random.sample(val_data, sample_size)
            print(f"[*] For quick evaluation, using {len(val_data)} samples out of {len(gallery_ids)} total.")

        query_feats = []
        target_ids = []

        with torch.no_grad():
            for item in tqdm(val_data, desc="Creating query features"):
                target_ids.append(item['target'])
                
                # You can either average queries from both captions,
                # or use only the first one.
                # Literature typically treats both captions as separate samples.
                for caption in item['captions']:
                    ref_feat = features_dict[item['candidate']].unsqueeze(0).to(config['DEVICE'])
                    text_tokens = clip.tokenize(caption, truncate=True).to(config['DEVICE'])
                    text_feat = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)
                    
                    q_feat = combiner(ref_feat, text_feat)
                    query_feats.append(q_feat.cpu())
                    
                    # Important: add target ID for each caption
                    if len(query_feats) > len(target_ids):
                        target_ids.append(item['target'])

        # 3. Calculate Recall
        query_feats = torch.cat(query_feats).to(config['DEVICE'])
        results = calculate_recall(query_feats, gallery_feats, target_ids, gallery_ids)
        
        print(f"Results ({cat}): {results}")
        final_results[cat] = results

    return final_results

if __name__ == "__main__":
    ######## Load config ########
    config_path = "config/config_v0.0.0.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ######## Load CLIP Model ########
    print(f"[*] Loading {config['CLIP_BACKBONE']}...")
    clip_model, preprocess = clip.load(config['CLIP_BACKBONE'], device=config['DEVICE'])
    clip_model.eval()

    ######## Load features into RAM ########
    print("[*] Loading features into RAM...")
    features_dict = torch.load(config['FEATURE_CACHE'])

    ######## Initialize model and pre-trained weights ########
    # combiner = FashionCombiner(embed_dim=config['EMBED_DIM']).to(config['DEVICE'])
    combiner = FashionCombiner_v2(embed_dim=config['EMBED_DIM']).to(config['DEVICE'])
    if os.path.exists(config['MODEL_WEIGHTS']):
        checkpoint = torch.load(config['MODEL_WEIGHTS'], map_location=config['DEVICE'])
        if 'model_state_dict' in checkpoint:
            combiner.load_state_dict(checkpoint['model_state_dict'])
        else:
            combiner.load_state_dict(checkpoint) 
        combiner.eval()
        print(f"[*] Loaded model weights: {config['MODEL_WEIGHTS']}")
    else:
        print("[!] Warning: Trained model weights not found, running with random weights!")

    ######## Run evaluation ########
    eval_subset=False
    final_results = run_evaluation(config, combiner, clip_model, features_dict, eval_subset)
    print("\n[*] Final Evaluation Results:")
    for cat, res in final_results.items():
        print(f"{cat}: {res}")