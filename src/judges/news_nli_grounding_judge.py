import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class NewsNLIGroundingJudge:
    def __init__(self, model_name_or_path: str, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        
    def score_batch(self, headlines: list[str], rationales: list[str], batch_size: int = 32):
        """
        Returns a list of dicts: [{'entailment': float, 'contradiction': float, 'neutral': float}, ...]
        Using cross-encoder/nli-deberta-v3-small.
        DeBERTa-v3-small NLI labels: 0=Contradiction, 1=Entailment, 2=Neutral
        """
        results = []
        for i in tqdm(range(0, len(headlines), batch_size), desc="NLI Grounding"):
            batch_headlines = headlines[i:i+batch_size]
            batch_rationales = rationales[i:i+batch_size]
            
            # The hypothesis is the rationale, the premise is the headline
            # We want to know if the rationale contradicts or entails the headline.
            inputs = self.tokenizer(
                batch_headlines, 
                batch_rationales, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            for p in probs:
                # DeBERTa v3 small NLI ordering:
                # 0 -> Contradiction
                # 1 -> Entailment
                # 2 -> Neutral
                results.append({
                    'contradiction': float(p[0]),
                    'entailment': float(p[1]),
                    'neutral': float(p[2])
                })
                
        return results
