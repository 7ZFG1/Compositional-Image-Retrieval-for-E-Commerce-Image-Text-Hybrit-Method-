import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

from utils.loss import InfoNCELoss, MSLoss

class Trainer:
    def __init__(self, config, model, clip_model, dataloader, eval_fn=None, features_dict=None):
        self.config = config
        self.model = model
        self.clip_model = clip_model
        self.dataloader = dataloader
        
        self.eval_fn = eval_fn
        self.features_dict = features_dict
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config['LR']), weight_decay=0.01)
        self.scaler = GradScaler()
        
        # Selected loss type from config (default to 'triplet' if not specified)
        self.loss_type = self.config.get('LOSS_TYPE', 'triplet').lower()
        print(f"[*] Selected Loss Type: {self.loss_type.upper()}")
        
        if self.loss_type == 'infonce':
            self.criterion = InfoNCELoss(temperature=self.config.get('TEMPERATURE', 0.05))
        elif self.loss_type == 'ms':
            self.criterion = MSLoss(alpha=2.0, beta=50.0, base=0.5)
        else: # Default to triplet loss
            self.criterion = nn.TripletMarginLoss(margin=self.config.get('MARGIN', 0.2), p=2)

    def train(self, start_epoch=0, best_avg_r10=0.0, best_loss=float('inf')):
        print(f"\n[*] Training starts from epoch {start_epoch}...")
        
        config_save_dir = self.config.get('MODEL_SAVE_DIR', 'trained_models/v.unknown')
        os.makedirs(config_save_dir, exist_ok=True)
        
        save_path = os.path.join(config_save_dir, self.config.get('MODEL_SAVE_NAME', 'fashion_combiner_best.pth'))
        csv_path = os.path.join(config_save_dir, 'loss_log.csv')
        
        with open(os.path.join(config_save_dir, "training_config.yml"), 'w') as f:
            yaml.dump(self.config, f)
            
        # Read existing epoch losses if resuming from a checkpoint (to continue plotting later)
        epoch_losses = []
        if start_epoch > 0 and os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2 and parts[0].isdigit():
                        ep = int(parts[0])
                        if ep <= start_epoch: # Only consider epochs up to the resumed epoch
                            epoch_losses.append(float(parts[1]))
                            
        for epoch in range(start_epoch, self.config['EPOCHS']):
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
                    
                    # Calculate loss based on the selected loss type (negative sample is only used for triplet loss)
                    if self.loss_type == 'triplet':
                        loss = self.criterion(query_feat, target_feat, neg_feat)
                    else:
                        loss = self.criterion(query_feat, target_feat)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")
            
            # Save epoch loss for plotting and write to CSV
            epoch_losses.append(avg_loss)
            write_mode = 'a' if (epoch > 0 or start_epoch > 0) else 'w'
            with open(csv_path, write_mode) as f:
                if write_mode == 'w':
                    f.write("Epoch,Loss\n")
                f.write(f"{epoch+1},{avg_loss:.6f}\n")
            
            # Evaluation
            is_eval_enabled = self.config.get('EVAL', False)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_avg_r10': best_avg_r10,
                'best_loss': best_loss
            }

            if is_eval_enabled and self.eval_fn is not None:
                eval_step = self.config.get('EVAL_STEP', 1)
                if (epoch + 1) % eval_step == 0:
                    print(f"\n[*] Running evaluation at epoch {epoch+1}...")
                    
                    self.model.eval() 
                    eval_results = self.eval_fn(self.config, self.model, self.clip_model, self.features_dict)
                    
                    total_r10 = sum(metrics.get('R@10', 0.0) for metrics in eval_results.values())
                    avg_r10 = total_r10 / len(eval_results) if eval_results else 0.0
                    print(f"[*] Average R@10 across categories: {avg_r10:.2f}%")
                    
                    if avg_r10 > best_avg_r10:
                        best_avg_r10 = avg_r10
                        checkpoint['best_avg_r10'] = best_avg_r10
                        torch.save(checkpoint, save_path)
                        print(f"[*] New best model saved with Avg R@10: {best_avg_r10:.2f}%")
                        
                    self.model.train()
            else:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint['best_loss'] = best_loss
                    torch.save(checkpoint, save_path)
                    print("[*] New best model saved (based on training loss).")

        self.plot_loss(epoch_losses, config_save_dir)

    def plot_loss(self, epoch_losses, save_dir):
        if not epoch_losses:
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-', color='b')
        plt.title(f'Training Loss Curve ({self.loss_type.upper()})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plot_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"\n[*] Training completed! Loss curve saved: {plot_path}")