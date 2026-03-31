import clip
import torch
import yaml
import argparse
import os

from utils.trainer import Trainer
from utils.dataloader import FashionIQDataset
from utils.feature_extractor import FeatureExtractor
from utils.model import FashionCombiner, FashionCombiner_v2
from torch.utils.data import DataLoader

# Import evaluation function from eval.py
from eval import run_evaluation

######## Load config ########
parser = argparse.ArgumentParser(description="Train the Fashion Combiner model.")
parser.add_argument('--config', type=str, default="config/config_v0.0.0.yml", 
                    help="Path to the YAML configuration file.")
parser.add_argument('--pretrained', type=str, default=None, 
                    help="Path to pre-trained model weights (Starts from Epoch 0).")
parser.add_argument('--resume', type=str, default=None, 
                    help="Path to a checkpoint to resume training from the exact stopped point.")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

######## Load CLIP Model ########
print(f"[*] {config['CLIP_BACKBONE']} yükleniyor...")
clip_model, preprocess = clip.load(config['CLIP_BACKBONE'], device=config['DEVICE'])
clip_model.eval()

######## Extract and cache features ########
FeatureExtractor.extract_and_save(config, clip_model, preprocess)

######## Load features into RAM ########
print("[*] Load features into RAM...")
features_dict = torch.load(config['FEATURE_CACHE'])

######## Create Dataset and DataLoader ########
dataset = FashionIQDataset(config, features_dict)
dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4, drop_last=True)

######## Initialize model ########
combiner = FashionCombiner_v2(embed_dim=config['EMBED_DIM'], num_heads=8).to(config['DEVICE'])

######## Load pre-trained weights if provided (Starts from Epoch 0) ########
if args.pretrained and os.path.exists(args.pretrained):
    print(f"[*] Loading pre-trained weights from {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=config['DEVICE'])
    # Eski (sadece state_dict) ve yeni (checkpoint dict) formatları destekler
    if 'model_state_dict' in checkpoint:
        combiner.load_state_dict(checkpoint['model_state_dict'])
    else:
        combiner.load_state_dict(checkpoint)

######## Initialize trainer ########
trainer = Trainer(
    config=config, 
    model=combiner, 
    clip_model=clip_model, 
    dataloader=dataloader,
    eval_fn=run_evaluation,       
    features_dict=features_dict   
)

######## Resume training if checkpoint provided (Resumes from exact epoch, optimizer state, etc.) ########
start_epoch = 0
best_avg_r10 = 0.0
best_loss = float('inf')

if args.resume and os.path.exists(args.resume):
    print(f"[*] Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=config['DEVICE'])
    
    if 'model_state_dict' in checkpoint:
        combiner.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_avg_r10 = checkpoint.get('best_avg_r10', 0.0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"[*] Successfully resumed from Epoch {start_epoch}")
    else:
        print("[!] Warning: Provided resume file is in old format (only weights).")
        print("[!] Optimizer will start fresh from Epoch 0.")
        combiner.load_state_dict(checkpoint)

######## Start training ########
trainer.train(start_epoch=start_epoch, best_avg_r10=best_avg_r10, best_loss=best_loss)
print("[*] Training completed.")