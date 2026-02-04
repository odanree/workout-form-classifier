"""CLIP utility functions for fine-tuning and inference."""

import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPClassifier:
    """Fine-tuned CLIP model for workout form classification."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def classify(self, image, text_descriptions: list) -> dict:
        """
        Classify image against text descriptions.
        
        Args:
            image: PIL Image or tensor
            text_descriptions: List of text descriptions to compare
        
        Returns:
            Dict with scores and top classification
        """
        with torch.no_grad():
            inputs = self.processor(
                text=text_descriptions,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        scores = probs[0].cpu().numpy()
        top_idx = scores.argmax()
        
        return {
            "classification": text_descriptions[top_idx],
            "confidence": float(scores[top_idx]),
            "all_scores": {desc: float(score) for desc, score in zip(text_descriptions, scores)}
        }
