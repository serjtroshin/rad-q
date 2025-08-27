
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class RoBERTaToxicityScore():
    def __init__(self):
        self.attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
        self.toxicityModel = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel")
        self.toxicityModel.eval()
        self.toxicityModel.to(self.device)
                

    def compute(self, texts) -> List[float]:  
        # return only toxicity score and zero for other attributes

        tokens = self.tokenizer(texts,
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        scores = 1.0 - torch.nn.functional.sigmoid(self.toxicityModel(**tokens).logits)
        scores = scores.to('cpu').detach().numpy()
        return scores

       

        

        