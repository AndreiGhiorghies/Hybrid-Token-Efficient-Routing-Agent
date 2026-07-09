from data.Request import Request
from data.Request import Category

import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import time

class TaskClassifier:
    CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, onnx_model_path="./models/onnx", anchors_path="./data/anchors.json"):        
        # Load tokenizer and ONNX model
        self.tokenizer = Tokenizer.from_file(f"{onnx_model_path}/tokenizer.json")
        self.tokenizer.no_padding()
        
        # Force ONNX Runtime to use only CPU
        self.session = ort.InferenceSession(
            f"{onnx_model_path}/onnx/model.onnx", 
            providers=['CPUExecutionProvider']
        )
        
        # Load the anchor database
        with open(anchors_path, "r", encoding="utf-8") as f:
            self.anchors_db = json.load(f)

        self._vectorize_anchors()
        
    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        return np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))

    def _get_embedding(self, task_data: Request) -> np.ndarray:
        encoded = self.tokenizer.encode(task_data.prompt)
        
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        outputs = self.session.run(None, onnx_inputs)
        
        embeddings = outputs[0]
        
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        return (sum_embeddings / sum_mask)[0]

    def classify(self, prompt: Request, k: int = 5) -> dict:
        prompt_vector = self._get_embedding(prompt)
        similarities = [
            (self._cosine_similarity(prompt_vector, anchor["vector"]), anchor)
            for anchor in self.anchor_vectors
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]

        category_scores = {}
        for sim, anchor in top_k:
            category_scores[anchor["category"]] = category_scores.get(anchor["category"], 0) + sim
        best_category = max(category_scores, key=category_scores.__getitem__)

        winning = [(sim, a["difficulty"]) for sim, a in top_k if a["category"] == best_category]
        avg_difficulty = sum(d for _, d in winning) / len(winning)

        return {"category" : best_category, "difficulty": round(avg_difficulty, 1)}
    
    def _vectorize_anchors(self):
        category_vectors = {}
        self.anchor_vectors = []

        for anchor in self.anchors_db:
            vector = self._get_embedding(Request("", anchor["text"]))
            anchor_entry = {
                "vector": vector,
                "category": anchor["category"],
                "difficulty": anchor["difficulty"]
            }
            self.anchor_vectors.append(anchor_entry)

            category_vectors.setdefault(anchor["category"], []).append(vector)
    
    def is_local_friendly(self, task_data: Request) -> bool:
        category, difficulty = task_data.category, task_data.difficulty
        if category == Category.SENTIMENT and difficulty < 1.0:
            return True
        if category == Category.FACTUAL_KNOWLEDGE and difficulty < 1.0:
            return True
        if category == Category.SUMMARISATION and difficulty < 1.0:
            return True
        if category == Category.NER and difficulty < 1.0:
            return True
        
        if category == Category.CODE_DEBUG and difficulty < 0.0:
            return True
        if category == Category.CODE_GENERATION and difficulty < 0.0:
            return True
        if category == Category.LOGICAL and difficulty < 0.0:
            return True
        if category == Category.MATH and difficulty < 0.0:
            return True

        return False