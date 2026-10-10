"""
XLM-RoBERTa Fine-Tuning Module
================================
Fine-tunes `xlm-roberta-base` for binary sequence classification
on Indian code-mixed hate speech detection data.

Architecture
------------
    Input Text (code-mixed)
        ↓
    XLM-RoBERTa Tokenizer
        ↓
    XLM-RoBERTa Encoder (12 layers, 768 hidden)
        ↓
    [CLS] representation
        ↓
    Dropout (0.1)
        ↓
    Linear(768 → num_labels)
        ↓
    Hate / Normal (binary)

Requirements
------------
    pip install torch transformers accelerate

IMPORTANT: Do NOT call train() until a verified research dataset
is available. The pipeline can be verified with demo_mode=True.

Usage
-----
    from src.models.xlmr_model import XLMRClassifier
    model = XLMRClassifier(config=cfg)
    model.train(train_df, val_df, demo_mode=True)  # software test only
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEMO_MARKER = "DEMO DATA — NOT PROJECT RESULTS"


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for XLM-R training.\n"
            "Install with: pip install torch transformers accelerate"
        )


def _check_transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers is required.\n"
            "Install with: pip install transformers accelerate"
        )


# ---------------------------------------------------------------------------
# Dataset class for PyTorch DataLoader
# ---------------------------------------------------------------------------

class HateSpeechDataset:
    """PyTorch Dataset for tokenized hate speech data."""

    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Main Classifier
# ---------------------------------------------------------------------------

class XLMRClassifier:
    """
    XLM-RoBERTa fine-tuned binary classifier for code-mixed hate speech.

    Parameters
    ----------
    config : dict, optional
        Full project config dict. Loaded automatically if None.
    num_labels : int
        Number of output classes. Default 2 (binary).
    """

    MODEL_NAME = "XLM-RoBERTa Base"

    def __init__(
        self,
        config: Optional[dict] = None,
        num_labels: int = 2,
    ):
        from src.utils.config import load_config, get_project_root
        self.config = config or load_config()
        self.project_root = get_project_root()
        self.xlmr_cfg = self.config.get("xlmr", {})
        self.num_labels = num_labels

        self.model_name   = self.xlmr_cfg.get("model_name", "xlm-roberta-base")
        self.max_length   = self.xlmr_cfg.get("max_length", 128)
        self.batch_size   = self.xlmr_cfg.get("batch_size", 16)
        self.lr           = self.xlmr_cfg.get("learning_rate", 2e-5)
        self.num_epochs   = self.xlmr_cfg.get("num_epochs", 5)
        self.warmup_ratio = self.xlmr_cfg.get("warmup_ratio", 0.1)
        self.weight_decay = self.xlmr_cfg.get("weight_decay", 0.01)
        self.seed         = self.xlmr_cfg.get("random_seed", 42)
        self.fp16         = self.xlmr_cfg.get("fp16", False)
        self.early_stopping_patience = self.xlmr_cfg.get("early_stopping_patience", 3)

        self.tokenizer = None
        self.model     = None
        self.is_fitted = False
        self.label_map: Dict[int, str] = {0: "NOT", 1: "HOF"}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        """Initialize tokenizer and model from HuggingFace Hub."""
        torch = _check_torch()
        tf    = _check_transformers()

        from src.utils.seed import set_seed, get_device
        set_seed(self.seed)
        self.device = get_device()

        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = tf.AutoTokenizer.from_pretrained(self.model_name)

        logger.info(f"Loading model: {self.model_name} (num_labels={self.num_labels})")
        self.model = tf.AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _tokenize(self, texts: List[str]) -> dict:
        """Tokenize a list of texts."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_col: str = "cleaned_text",
        label_col: str = "normalized_label",
        save_dir: Optional[str] = None,
        demo_mode: bool = False,
    ) -> Dict:
        """
        Fine-tune XLM-RoBERTa on the training set.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data. Must have text_col and label_col.
        val_df : pd.DataFrame
            Validation data.
        text_col : str
            Column with cleaned text. Default 'cleaned_text'.
        label_col : str
            Column with integer labels (0/1). Default 'normalized_label'.
        save_dir : str, optional
            Directory to save the best checkpoint.
            Defaults to models/xlmr/ (or models/xlmr/DEMO/).
        demo_mode : bool
            If True, uses a small subset and marks all output as DEMO.

        Returns
        -------
        dict
            Training history with per-epoch metrics.
        """
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        if demo_mode:
            print("\n" + "!" * 60)
            print(f"  {DEMO_MARKER}")
            print(f"  Running XLM-R training on demo data only.")
            print(f"  DO NOT use these results in project reports.")
            print("!" * 60 + "\n")

            debug_max = self.xlmr_cfg.get("debug_max_samples", 100)
            train_df = train_df.head(min(debug_max, len(train_df))).copy()
            val_df   = val_df.head(min(debug_max // 5, len(val_df))).copy()
            # Override epochs for speed in demo
            num_epochs = 2
        else:
            num_epochs = self.num_epochs

        self._setup()

        # Verify columns
        for df_name, df in [("train_df", train_df), ("val_df", val_df)]:
            if text_col not in df.columns:
                raise ValueError(f"Column '{text_col}' not found in {df_name}. "
                                 f"Available: {list(df.columns)}")
            if label_col not in df.columns:
                raise ValueError(f"Column '{label_col}' not found in {df_name}.")

        train_texts = train_df[text_col].astype(str).tolist()
        train_labels = train_df[label_col].astype(int).tolist()
        val_texts  = val_df[text_col].astype(str).tolist()
        val_labels = val_df[label_col].astype(int).tolist()

        logger.info(f"Tokenizing {len(train_texts)} training samples...")
        train_enc = self._tokenize(train_texts)
        val_enc   = self._tokenize(val_texts)

        train_dataset = HateSpeechDataset(train_enc, train_labels)
        val_dataset   = HateSpeechDataset(val_enc, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=False)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Determine save directory
        project_root = self.project_root
        if save_dir is None:
            subdir = "DEMO" if demo_mode else "best"
            save_dir = project_root / "models" / "xlmr" / subdir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
        best_val_f1 = -1.0
        patience_counter = 0

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        logger.info(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")

        for epoch in range(num_epochs):
            # ---- Train ----
            self.model.train()
            total_train_loss = 0.0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_train_loss += loss.item()

                if (step + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{num_epochs} | "
                                f"Step {step+1}/{len(train_loader)} | "
                                f"Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader)

            # ---- Validate ----
            val_results = self._evaluate_loader(val_loader)
            avg_val_loss = val_results["loss"]
            val_macro_f1 = val_results["macro_f1"]

            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_macro_f1"].append(round(val_macro_f1, 4))

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f}"
            )

            # ---- Save best checkpoint ----
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                patience_counter = 0
                self._save_checkpoint(save_dir)
                logger.info(f"  [OK] New best checkpoint saved (Val Macro F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        self.is_fitted = True
        history["best_val_macro_f1"] = best_val_f1
        history["demo_mode"] = demo_mode

        # Save training history
        hist_label = "DEMO_" if demo_mode else ""
        hist_path = save_dir / f"{hist_label}training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        if demo_mode:
            print(f"\n  {DEMO_MARKER}")
            print(f"  Demo training complete. Best Val Macro F1: {best_val_f1:.4f}")
            print(f"  This is a SOFTWARE TEST result, not a project result.\n")

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_loader(self, loader) -> Dict:
        """Run evaluation loop and return metrics."""
        import torch
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                logits = outputs.logits
                preds  = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        avg_loss = total_loss / len(loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "loss":      round(avg_loss, 4),
            "macro_f1":  round(macro_f1, 4),
            "all_preds": all_preds,
            "all_labels": all_labels,
        }

    def evaluate(
        self,
        test_df: pd.DataFrame,
        text_col: str = "cleaned_text",
        label_col: str = "normalized_label",
        demo_mode: bool = False,
    ) -> Dict:
        """
        Evaluate the model on a test set.

        Returns
        -------
        dict
            Dictionary with predictions, probabilities, and metrics.
        """
        import torch
        from torch.utils.data import DataLoader

        if not self.is_fitted:
            raise RuntimeError("Model not fitted or loaded. Call train() or load() first.")

        if demo_mode:
            print(f"\n  {DEMO_MARKER} — Evaluation on demo data.")

        texts  = test_df[text_col].astype(str).tolist()
        labels = test_df[label_col].astype(int).tolist()

        enc = self._tokenize(texts)
        dataset = HateSpeechDataset(enc, labels)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        results = self._evaluate_loader(loader)
        results["demo_mode"] = demo_mode
        return results

    def predict(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and probabilities for a list of texts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        tuple of (labels, probabilities)
            labels: np.ndarray of int (0 or 1)
            probabilities: np.ndarray of shape (n, num_labels)
        """
        import torch
        from torch.utils.data import DataLoader

        if not self.is_fitted:
            raise RuntimeError("Model not fitted or loaded.")

        dummy_labels = [0] * len(texts)
        enc     = self._tokenize(texts)
        dataset = HateSpeechDataset(enc, dummy_labels)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_batch = {k: v.to(self.device) for k, v in batch.items()
                               if k != "labels"}
                outputs = self.model(**input_batch)
                probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds   = np.argmax(probs, axis=-1)
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())

        return np.array(all_preds), np.array(all_probs)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, save_dir: Path) -> None:
        """Save model and tokenizer to save_dir."""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Also save config info
        meta = {
            "model_name":   self.model_name,
            "num_labels":   self.num_labels,
            "max_length":   self.max_length,
            "label_map":    self.label_map,
        }
        with open(save_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def save(self, save_dir: Optional[str] = None, demo_mode: bool = False) -> Path:
        """Save trained model and tokenizer."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Cannot save.")

        project_root = self.project_root
        if save_dir is None:
            subdir = "DEMO" if demo_mode else "final"
            save_dir = project_root / "models" / "xlmr" / subdir

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(save_dir)
        logger.info(f"Model saved to {save_dir}")
        return save_dir

    def load(self, load_dir: str) -> "XLMRClassifier":
        """Load a saved model and tokenizer from disk."""
        tf = _check_transformers()
        load_dir = Path(load_dir)

        meta_path = load_dir / "model_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.model_name  = meta.get("model_name", self.model_name)
            self.num_labels  = meta.get("num_labels", self.num_labels)
            self.max_length  = meta.get("max_length", self.max_length)
            self.label_map   = meta.get("label_map", self.label_map)

        from src.utils.seed import get_device
        self.device = get_device()

        self.tokenizer = tf.AutoTokenizer.from_pretrained(str(load_dir))
        self.model = tf.AutoModelForSequenceClassification.from_pretrained(
            str(load_dir),
            num_labels=self.num_labels,
        )
        self.model.to(self.device)
        self.is_fitted = True
        logger.info(f"Loaded XLM-R model from {load_dir}")
        return self
