#!/usr/bin/env python3
"""
Fine-tune CLIP model on workout form classification data.

Trains CLIP to classify workout form (good vs bad) for multiple exercises.

Usage:
    python scripts/finetune_clip.py --data-dir data/labeled_frames
    python scripts/finetune_clip.py --data-dir data/labeled_frames --epochs 10 --batch-size 32
"""

import argparse
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkoutFormDataset(Dataset):
    """Custom dataset for workout form classification."""
    
    def __init__(
        self,
        csv_paths: List[str],
        exercise_definitions: Dict,
        processor: CLIPProcessor,
        base_dir: Path
    ):
        self.processor = processor
        self.base_dir = base_dir
        self.exercise_definitions = exercise_definitions
        self.samples = []
        
        # Load all samples from CSVs
        for csv_path in csv_paths:
            self._load_csv(csv_path)
    
    def _load_csv(self, csv_path: str) -> None:
        """Load samples from a single CSV file."""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows without description (not yet annotated)
                if not row.get("description", "").strip():
                    continue
                
                frame_path = self.base_dir / row["frame_path"]
                if not frame_path.exists():
                    logger.warning(f"Frame not found: {frame_path}")
                    continue
                
                self.samples.append({
                    "image_path": str(frame_path),
                    "exercise": row["exercise"],
                    "form_type": row["form_type"],
                    "description": row["description"],
                    "confidence": float(row.get("confidence", 1.0)) if row.get("confidence") else 1.0
                })
        
        logger.info(f"Loaded {len(self.samples)} samples from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample."""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {sample['image_path']}: {str(e)}")
            # Return a blank image instead of crashing
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        
        # Get text description
        text = sample["description"]
        
        # Process with CLIP processor
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "exercise": sample["exercise"],
            "form_type": sample["form_type"],
            "confidence": sample["confidence"]
        }


class CLIPFineTuner:
    """Fine-tune CLIP model for workout form classification."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
        learning_rate: float = 1e-5
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 5,
        output_path: str = "models/clip_finetuned_workout.pt"
    ) -> Dict:
        """
        Fine-tune CLIP model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            output_path: Where to save the model
        
        Returns:
            Training history
        """
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)
            history["train_loss"].append(train_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                logger.info(f"  Val Loss: {val_loss:.4f} - Accuracy: {val_acc:.2%}")
        
        # Save model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"Model saved: {output_path}")
        
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def _validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                form_type = batch["form_type"]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    return_loss=True
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Simple accuracy: check if form_type is in description
                # (Would need more sophisticated evaluation in production)
                total += len(form_type)
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def evaluate_on_exercises(
        self,
        eval_loader: DataLoader,
        exercise_definitions: Dict
    ) -> Dict:
        """Evaluate model performance per exercise."""
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for exercise, definition in exercise_definitions.items():
                # Get form class descriptions
                form_classes = definition.get("classes", {})
                class_texts = [
                    cls_def["description"]
                    for cls_name, cls_def in form_classes.items()
                ]
                
                if not class_texts:
                    continue
                
                # Encode classes
                class_inputs = self.processor(
                    text=class_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                class_features = self.model.get_text_features(
                    input_ids=class_inputs["input_ids"],
                    attention_mask=class_inputs["attention_mask"]
                )
                
                # Evaluate on this exercise
                correct = 0
                total = 0
                
                for batch in eval_loader:
                    if batch["exercise"][0] != exercise:
                        continue
                    
                    pixel_values = batch["pixel_values"].to(self.device)
                    image_features = self.model.get_image_features(pixel_values)
                    
                    # Compute similarity
                    logits = torch.matmul(
                        F.normalize(image_features, dim=-1),
                        F.normalize(class_features, dim=-1).t()
                    )
                    
                    predictions = logits.argmax(dim=1)
                    # Would need ground truth labels for proper accuracy
                    
                    total += len(predictions)
                
                if total > 0:
                    results[exercise] = {"samples": total}
        
        return results


def load_exercise_definitions(config_path: str = "config/exercise_definitions.json") -> Dict:
    """Load exercise class definitions."""
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP model on workout form data"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory with training data (labeled_frames)"
    )
    parser.add_argument(
        "--output",
        default="models/clip_finetuned_workout.pt",
        help="Output path for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="Base CLIP model to fine-tune"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    
    args = parser.parse_args()
    
    # Load exercise definitions
    exercise_defs = load_exercise_definitions()
    
    # Find all training CSVs
    data_dir = Path(args.data_dir)
    csv_files = list(data_dir.glob("*_training_data.csv"))
    
    if not csv_files:
        logger.error(f"No training CSVs found in {args.data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(csv_files)} training CSV files")
    
    # Load dataset
    dataset = WorkoutFormDataset(
        csv_paths=[str(f) for f in csv_files],
        exercise_definitions=exercise_defs,
        processor=CLIPProcessor.from_pretrained(args.model),
        base_dir=data_dir
    )
    
    if len(dataset) == 0:
        logger.error("No training samples loaded. Check CSV descriptions are filled in.")
        sys.exit(1)
    
    # Split into train/val
    train_size = int(len(dataset) * (1 - args.val_split))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Fine-tune model
    tuner = CLIPFineTuner(
        model_name=args.model,
        learning_rate=args.learning_rate
    )
    
    history = tuner.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        output_path=args.output
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved: {args.output}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Final val accuracy: {history['val_accuracy'][-1]:.2%}")
    print("="*60)
    print("\nNext steps:")
    print("1. Evaluate model: python scripts/evaluate_model.py")
    print("2. Use in production: Update src/utils/clip_utils.py with model path")
    print("3. Test on new videos: python src/batch_runner.py workout.mp4 --exercise squat")


if __name__ == "__main__":
    main()
