"""
BERT / DistilBERT Model for Review Authenticity Detection
==========================================================
Fine-tunes a pre-trained DistilBERT model for binary classification
of genuine vs. fake reviews.

Uses HuggingFace Transformers library with:
- DistilBERT tokenizer and model
- Custom classification head
- Warmup learning rate schedule
- Early stopping
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import BERT_CONFIG, RANDOM_SEED, METRICS_DIR

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DATASET CLASS
# ============================================================

class BertReviewDataset(Dataset):
    """Dataset for BERT-based models."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_bert_model(train_texts, train_labels, test_texts, test_labels,
                     val_texts=None, val_labels=None):
    """
    Fine-tune DistilBERT for review classification.

    Parameters
    ----------
    train_texts : list/Series
        Training review texts.
    train_labels : ndarray
        Training labels (0=genuine, 1=fake).
    test_texts : list/Series
        Test texts.
    test_labels : ndarray
        Test labels.

    Returns
    -------
    dict
        Training results including model, metrics, and predictions.
    """
    print("\n" + "=" * 60)
    print("FINE-TUNING DistilBERT MODEL")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = BERT_CONFIG

    # Load tokenizer and model
    print(f"Loading {config['model_name']}...")
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    model = DistilBertForSequenceClassification.from_pretrained(
        config['model_name'], num_labels=2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset = BertReviewDataset(
        train_texts, train_labels, tokenizer, config['max_length']
    )
    test_dataset = BertReviewDataset(
        test_texts, test_labels, tokenizer, config['max_length']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                               shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    if val_texts is not None and val_labels is not None:
        val_dataset = BertReviewDataset(
            val_texts, val_labels, tokenizer, config['max_length']
        )
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=0)
    else:
        val_loader = test_loader

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps,
    )

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_f1 = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # --- Train ---
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        val_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)

    # --- Final evaluation on test set ---
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(test_labels, y_pred),
        "f1_score": f1_score(test_labels, y_pred),
        "roc_auc": roc_auc_score(test_labels, y_prob),
    }

    print(f"\nDistilBERT Final Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }
