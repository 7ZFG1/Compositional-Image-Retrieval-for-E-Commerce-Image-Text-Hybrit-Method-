import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
import yaml

class Trainer:
    def __init__(self, config, model, clip_model, dataloader, eval_fn=None, features_dict=None):
        self.config = config
        self.model = model
        self.clip_model = clip_model
        self.dataloader = dataloader
        
        # Evaluation variables
        self.eval_fn = eval_fn
        self.features_dict = features_dict
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config['LR']), weight_decay=0.01)
        self.criterion = nn.TripletMarginLoss(margin=config['MARGIN'], p=2)
        self.scaler = GradScaler()

    def train(self):
        print("\n[*] Training starts...")
        best_loss = float('inf')
        best_avg_r10 = 0.0 # To track the best R@10 score
        
        # Define save path
        save_path = os.path.join(self.config.get('MODEL_SAVE_DIR', 'trained_models/v.unknown'), 
                                    self.config.get('MODEL_SAVE_NAME', 'fashion_combiner_best.pth'))
        
        # Save the config file
        config_save_dir = self.config.get('MODEL_SAVE_DIR', 'trained_models/v.unknown')
        os.makedirs(config_save_dir, exist_ok=True)
        with open(os.path.join(config_save_dir, "training_config.yml"), 'w') as f:
            yaml.dump(self.config, f)
        print(f"[*] Training configuration saved to: {os.path.join(config_save_dir, 'training_config.yml')}")
        
        for epoch in range(self.config['EPOCHS']):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['EPOCHS']}")
            for cand_feat, text_tokens, target_feat, neg_feat in pbar:
                
                cand_feat = cand_feat.to(self.config['DEVICE'])
                target_feat = target_feat.to(self.config['DEVICE'])
                neg_feat = neg_feat.to(self.config['DEVICE'])
                text_tokens = text_tokens.to(self.config['DEVICE'])
                
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    text_feat = self.clip_model.encode_text(text_tokens).float()
                    text_feat = F.normalize(text_feat, dim=-1)
                
                with autocast():
                    query_feat = self.model(cand_feat, text_feat)
                    loss = self.criterion(query_feat, target_feat, neg_feat)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")
            
            # ==========================================
            # EVALUATION & MODEL SAVING STEP
            # ==========================================
            is_eval_enabled = self.config.get('EVAL', False)
            
            if is_eval_enabled and self.eval_fn is not None:
                eval_step = self.config.get('EVAL_STEP', 1)
                
                # Check if current epoch is an evaluation step
                if (epoch + 1) % eval_step == 0:
                    print(f"\n[*] Running evaluation at epoch {epoch+1}...")
                    
                    # Run evaluation using the provided eval_fn
                    eval_results = self.eval_fn(self.config, self.model, self.clip_model, self.features_dict)
                    
                    # Calculate Average R@10 across all categories
                    total_r10 = 0.0
                    num_cats = len(eval_results)
                    
                    for cat, metrics in eval_results.items():
                        total_r10 += metrics.get('R@10', 0.0)
                        
                    avg_r10 = total_r10 / num_cats if num_cats > 0 else 0.0
                    print(f"[*] Average R@10 across categories: {avg_r10:.2f}%")
                    
                    # Save model if it achieved the best Avg R@10
                    if avg_r10 > best_avg_r10:
                        best_avg_r10 = avg_r10

                        torch.save(self.model.state_dict(), save_path)
                        print(f"[*] New best model saved with Avg R@10: {best_avg_r10:.2f}%")
                        
                    # Set model back to train mode! (Crucial after running evaluation)
                    self.model.train()
                    
            else:
                # Original save logic based on loss (If EVAL is False)
                if avg_loss < best_loss:
                    best_loss = avg_loss

                    torch.save(self.model.state_dict(), save_path)
                    print("[*] New best model saved (based on training loss).")