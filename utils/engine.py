import os
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE,LR,WEIGHT_DECAY,EPOCHS, CHECKPOINT_DIR

class EarlyStopping:
   
    def __init__(self, patience=7, min_delta=0.001):
        """
        patience  : how many epochs to wait after last improvement
        min_delta : minimum change to count as an improvement
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0          # epochs without improvement
        self.best_loss  = None
        self.stop       = False      # flip to True when patience runs out

    def __call__(self, val_loss, model, ckpt_path):
        if self.best_loss is None:
            # First epoch — always save
            self.best_loss = val_loss
            self._save(model, ckpt_path)

        elif val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter   = 0
            self._save(model, ckpt_path)

        else:
            # No improvement
            self.counter += 1
            print(f"  EarlyStopping: no improvement for "
                  f"{self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.stop = True

    def _save(self, model, ckpt_path):
        torch.save(model.state_dict(), ckpt_path)


def train_one_epoch(model, loader, criterion, optimizer):
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(images)

        # InceptionV3 returns (main_logits, aux_logits) during training.
        # All other models (VGG, ResNet, MobileNet, DenseNet) return a plain
        # tensor so the else branch is always taken for them — no change.
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            loss    = criterion(main_out, labels) + 0.4 * criterion(aux_out, labels)
            outputs = main_out
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (total_loss / total, correct / total, np.array(all_preds), np.array(all_labels))


def train_model(model, model_name, train_loader, val_loader, class_names, epochs=EPOCHS, patience=5):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt         = os.path.join(CHECKPOINT_DIR, f"{model_name}.pth")
    history_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_history.pt")

    model          = model.to(DEVICE)
    criterion      = nn.CrossEntropyLoss()
    optimizer      = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler      = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    print(f"\n{'='*60}")
    print(f"  Training : {model_name}  |  device: {DEVICE}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc         = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:>2}/{epochs}]  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  Model saved with Val Acc: {best_acc:.4f}")

        early_stopping(val_loss, model, ckpt)
        if early_stopping.stop:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            print(f"  Best val_loss: {early_stopping.best_loss:.4f}")
            break

    torch.save(history, history_path)
    
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, _, preds, labels = evaluate(model, val_loader, criterion)

    print(f"\n── Classification Report ({model_name}) ──")
    print(classification_report(labels, preds,
                                target_names=class_names, digits=4))
    print(f"── Confusion Matrix ({model_name}) ──")
    print(confusion_matrix(labels, preds))
 
    return model, best_acc, preds, labels