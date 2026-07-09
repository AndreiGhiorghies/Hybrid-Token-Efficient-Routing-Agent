import math

from data.Response import Response
from data.Request import Request
from data.Request import Category

from llama_cpp import Llama

class LocalRunner:
    def __init__(self, model_path="./models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"):
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=768,       # limit context size to avoid Out of Memory
            n_threads=2,     # 2 vCPU
            n_batch=256,
            logits_all=True,
            verbose=False
        )

    def generate(self, task_data: Request) -> Response:
        # 1600 characters =~ 400 tokens, which is a lot for a 4GB RAM model. We should send it to the API instead.
        if len(task_data.prompt) > 1600:
            return Response()

        system_content = (
            "You are a strict data extraction API. "
            "Output ONLY the exact final answer. "
            "CRITICAL: No pleasantries, no explanations, no introductions, no conversational filler."
        )

        task_data.prompt += "\n\nProvide the final answer directly, with absolutely NO additional text or reasoning."
        messages = [
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": task_data.prompt
            }
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.0,  # 100% deterministic
            top_p=0.9,
            logprobs=True,
            top_logprobs=1
        )

        final_text = response["choices"][0]["message"]["content"]

        #calculate confidence based on logprobs
        logprobs_data = response["choices"][0].get("logprobs", {})
        token_logprobs = logprobs_data.get("content", []) if logprobs_data else []

        if not token_logprobs:
            confidence = 0.0
        else:
            total_prob = 0.0
            for token_data in token_logprobs:
                total_prob += math.exp(token_data["logprob"])
            
            confidence = total_prob / len(token_logprobs)

        response = Response()
        response.success = True
        response.set_text(final_text)
        response.confidence = confidence

        return response
    
    def is_confident(self, task_data: Request, response: Response) -> bool:
        category, confidence = task_data.category, response.confidence
        if category == Category.SENTIMENT and confidence >= 0.0:
            return True
        if category == Category.FACTUAL_KNOWLEDGE and confidence >= 0.0:
            return True
        if category == Category.SUMMARISATION and confidence >= 0.0:
            return True
        if category == Category.NER and confidence >= 0.0:
            return True
        
        if category == Category.CODE_DEBUG and confidence >= 0.5:
            return True
        if category == Category.CODE_GENERATION and confidence >= 0.5:
            return True
        if category == Category.LOGICAL and confidence >= 0.5:
            return True
        if category == Category.MATH and confidence >= 0.5:
            return True

        return False