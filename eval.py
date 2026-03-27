import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm
import yaml

from utils.model import FashionCombiner

def calculate_recall(query_feats, gallery_feats, target_ids, gallery_ids, ks=[10, 50]):
    """
    query_feats: [N, 512] - Combiner'dan çıkan sorgular
    gallery_feats: [M, 512] - Galerideki tüm aday resimler
    target_ids: [N] - Her sorgunun gerçek hedef ID'si
    gallery_ids: [M] - Galerideki resimlerin ID listesi
    """
    # Benzerlik Matrisi: [N, M]
    # (N: Sorgu sayısı, M: Galerideki toplam resim sayısı)
    sim_matrix = query_feats @ gallery_feats.T
    
    # En yüksek skora göre sırala (indices: [N, M])
    _, indices = torch.sort(sim_matrix, descending=True, dim=1)
    
    recalls = {}
    for k in ks:
        correct = 0
        for i in range(len(target_ids)):
            # İlk K sonuç içindeki ID'leri al
            top_k_indices = indices[i, :k]
            top_k_ids = [gallery_ids[idx] for idx in top_k_indices]
            
            # Gerçek hedef bu K resim içinde mi?
            if target_ids[i] in top_k_ids:
                correct += 1
        
        recalls[f"R@{k}"] = (correct / len(target_ids)) * 100
        
    return recalls

def run_evaluation(config, combiner, clip_model, features_dict):
    final_results = {}

    for cat in config['CATEGORIES']:
        print(f"\n[*] {cat.upper()} kategorisi değerlendiriliyor...")
        
        # 1. Galeri Hazırla (O kategorideki tüm geçerli resimler)
        split_path = os.path.join(config['ROOT_DIR'], "image_splits", f"split.{cat}.val.json")
        with open(split_path, 'r') as f:
            gallery_ids = json.load(f)
        
        gallery_feats = torch.stack([features_dict[img_id] for img_id in gallery_ids]).to(config['DEVICE'])

        # 2. Sorguları Hazırla
        cap_path = os.path.join(config['ROOT_DIR'], "captions", f"cap.{cat}.val.json")
        with open(cap_path, 'r') as f:
            val_data = json.load(f)

        query_feats = []
        target_ids = []

        with torch.no_grad():
            for item in tqdm(val_data, desc="Sorgular oluşturuluyor"):
                target_ids.append(item['target'])
                
                # Her iki caption için de sorgu üretip ortalamasını alabilir veya 
                # sadece ilkini kullanabilirsin. Literatür genelde her ikisini de ayrı örnek sayar.
                for caption in item['captions']:
                    ref_feat = features_dict[item['candidate']].unsqueeze(0).to(config['DEVICE'])
                    text_tokens = clip.tokenize(caption, truncate=True).to(config['DEVICE'])
                    text_feat = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)
                    
                    q_feat = combiner(ref_feat, text_feat)
                    query_feats.append(q_feat.cpu())
                    # Not: Hedef ID'yi her caption için eklemeyi unutma
                    if len(query_feats) > len(target_ids):
                        target_ids.append(item['target'])

        # 3. Hesapla
        query_feats = torch.cat(query_feats).to(config['DEVICE'])
        results = calculate_recall(query_feats, gallery_feats, target_ids, gallery_ids)
        
        print(f"Sonuçlar ({cat}): {results}")
        final_results[cat] = results

    return final_results

if __name__ == "__main__":
    ######## Load config ########
    config_path = "config/config_v0.0.0.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ######## Load CLIP Model ########
    print(f"[*] {config['CLIP_BACKBONE']} yükleniyor...")
    clip_model, preprocess = clip.load(config['CLIP_BACKBONE'], device=config['DEVICE'])
    clip_model.eval()

    ######## Load features into RAM ########
    print("[*] Load features into RAM...")
    features_dict = torch.load(config['FEATURE_CACHE'])

    ######## Initialize model and pre-trained weights ########
    combiner = FashionCombiner(embed_dim=config['EMBED_DIM']).to(config['DEVICE'])
    if os.path.exists(config['MODEL_WEIGHTS']):
        combiner.load_state_dict(torch.load(config['MODEL_WEIGHTS'], map_location=config['DEVICE']))
        combiner.eval()
        print(f"[*] Loaded model weights: {config['MODEL_WEIGHTS']}")
    else:
        print("[!] Warning: Trained model weights not found, running with random weights!")

    ######## Run evaluation ########
    final_results = run_evaluation(config, combiner, clip_model, features_dict)
    print("\n[*] Final Evaluation Results:")
    for cat, res in final_results.items():
        print(f"{cat}: {res}")