"""
Deepfake Trainer Engine
Author: Ramesh Prabhakaran R.
PhD Paper I: Frequency-Guided Spatial Attention Network
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..evaluation.metrics import calculate_binary_metrics


class DeepfakeTrainer:
    """
    Production-grade training loop with:
    - Automatic Mixed Precision (AMP)
    - Gradient Clipping
    - Learning rate scheduling
    - Real-time validation metrics (ROC-AUC, F1, Loss, Accuracy)
    - Early stopping & model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        experiment_name: str = "experiment",
        checkpoints_dir: str = "results/checkpoints",
        logs_dir: str = "results/logs",
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.experiment_name = experiment_name
        self.checkpoints_dir = Path(checkpoints_dir)
        self.logs_dir = Path(logs_dir)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler(enabled=self.use_amp)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
            "valid_roc_auc": [],
            "valid_f1": [],
            "lr": []
        }
        self.best_roc_auc = -1.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).unsqueeze(1)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.1f}%"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc: str = "Valid") -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_labels = []
        all_probs = []

        for batch in tqdm(data_loader, desc=desc, leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).unsqueeze(1)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            probs = torch.sigmoid(logits)
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

        val_loss = running_loss / total
        metrics = calculate_binary_metrics(np.array(all_labels), np.array(all_probs))
        return val_loss, metrics

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        print(f"\nStarting training: {self.experiment_name} on {self.device} (AMP: {self.use_amp})")
        print(f"Train samples: {len(self.train_loader.dataset):,} | Valid samples: {len(self.valid_loader.dataset):,}")
        print("=" * 80)

        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            valid_loss, val_metrics = self.evaluate(self.valid_loader, desc=f"Epoch {epoch} [Valid]")
            val_acc = val_metrics["accuracy"]
            val_roc_auc = val_metrics["roc_auc"]
            val_f1 = val_metrics["f1_score"]

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 5))
            self.history["train_acc"].append(round(train_acc, 5))
            self.history["valid_loss"].append(round(valid_loss, 5))
            self.history["valid_acc"].append(round(val_acc, 5))
            self.history["valid_roc_auc"].append(round(val_roc_auc, 5))
            self.history["valid_f1"].append(round(val_f1, 5))
            self.history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {valid_loss:.4f}, Acc: {val_acc*100:.2f}%, AUC: {val_roc_auc:.4f}, F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint best model by ROC-AUC
            if val_roc_auc > self.best_roc_auc:
                self.best_roc_auc = val_roc_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                ckpt_path = self.checkpoints_dir / f"{self.experiment_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "val_roc_auc": val_roc_auc,
                    "val_loss": valid_loss,
                    "history": self.history
                }, ckpt_path)
                print(f"  --> Saved new best checkpoint (ROC-AUC: {val_roc_auc:.4f}) to {ckpt_path.name}")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n[Early Stopping] No improvement in ROC-AUC for {self.early_stopping_patience} consecutive epochs.")
                    break

        total_time = time.time() - start_time
        print("=" * 80)
        print(f"Training completed in {total_time/60:.2f} minutes.")
        print(f"Best Validation ROC-AUC: {self.best_roc_auc:.4f} at Epoch {self.best_epoch}")

        # Save history log
        history_path = self.logs_dir / f"{self.experiment_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
