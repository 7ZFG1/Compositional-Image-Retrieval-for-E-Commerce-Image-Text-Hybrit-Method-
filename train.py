import clip
import torch
import yaml
import argparse

from utils.trainer import Trainer
from utils.dataloader import FashionIQDataset
from utils.feature_extractor import FeatureExtractor
from utils.model import FashionCombiner
from torch.utils.data import DataLoader

# Import evaluation function from eval.py
from eval import run_evaluation

######## Load config ########

parser = argparse.ArgumentParser(description="Train the Fashion Combiner model.")
parser.add_argument('--config', type=str, default="config/config_v0.0.0.yml", 
                    help="Path to the YAML configuration file.")
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

######## Load CLIP Model ########
print(f"[*] {config['CLIP_BACKBONE']} yükleniyor...")
clip_model, preprocess = clip.load(config['CLIP_BACKBONE'], device=config['DEVICE'])
clip_model.eval()

######## Extract and cache features (if not already done) ########
FeatureExtractor.extract_and_save(config, clip_model, preprocess)

######## Load features into RAM ########
print("[*] Load features into RAM...")
features_dict = torch.load(config['FEATURE_CACHE'])

######## Create Dataset and DataLoader ########
dataset = FashionIQDataset(config, features_dict)
dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4, drop_last=True)

######## Initialize model ########
combiner = FashionCombiner(embed_dim=config['EMBED_DIM']).to(config['DEVICE'])

######## Initialize trainer ########
trainer = Trainer(
    config=config, 
    model=combiner, 
    clip_model=clip_model, 
    dataloader=dataloader,
    eval_fn=run_evaluation,       # Pass the evaluation function
    features_dict=features_dict   # Pass features for fast evaluation
)

######## Start training ########
trainer.train()
print("[*] Training completed.")