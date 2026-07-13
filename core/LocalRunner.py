import math

from data.Response import Response
from data.Request import Request
from data.Request import Category

from llama_cpp import Llama

def build_prompt(task_data):
    prompt = task_data.prompt

    if task_data.category == Category.SENTIMENT:
        system_content = "You are a precise sentiment classification assistant."
        prompt += (
            "\n\nCRITICAL RULES:"
            "\n1. If the text contains a contrastive structure (e.g. 'but', 'however', 'although'), "
            "the sentiment expressed AFTER the contrast word usually reflects the overall conclusion — weigh it more heavily."
            "\n2. Classify using ONLY the categories explicitly requested in the prompt above."
            "\n3. First think through the positive and negative elements in one internal sentence, "
            "then give your final classification and a one-sentence reason."
            "\n4. Do NOT add conversational filler beyond the required reasoning sentence."
        )

    elif task_data.category == Category.NER:
        system_content = "You are a highly precise named entity extraction API."
        prompt += (
            "\n\nCRITICAL RULES:"
            "\n1. Extract ALL named entities without omitting any."
            "\n2. Do NOT truncate dates (e.g., extract 'March 15 2023', not just '15 2023')."
            "\n3. Format exactly as requested, with NO conversational filler."
        )

    elif task_data.category in [Category.FACTUAL_KNOWLEDGE, Category.SUMMARISATION]:
        system_content = "You are a knowledgeable assistant that gives clear, complete, well-explained answers while strictly following any formatting or length constraints given."
        prompt += (
            "\n\nCRITICAL RULES:"
            "\n1. FOLLOW the EXACT length constraints (the number of sentences, the number of words, answer format, the number of bullet points if they ask for them)."
            "\n2. Provide the FULL answer without omitting any details. If the question asks for an explanation, comparison, or reasoning ('why', 'how', 'explain'), you MUST provide a complete, substantive explanation — do not give a one-word or overly terse answer."
            "\n3. Do NOT add conversational filler like 'Sure, here's the answer:'."
            "\n4. MAKE SURE TO CHECK YOUR ANSWER AND BE SURE TO BE CORRECT, DO NOT MAKE UP ANSWERS."
        )

    else:
        system_content = "You are a highly precise data extraction and processing API."
        prompt += "\n\nProvide the final answer directly, with absolutely NO additional text or reasoning."

    return system_content, prompt

class LocalRunner:
    def __init__(self, model_path="./models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"):

        self.tasks_to_local = []
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=700,       # limit context size to avoid Out of Memory
            n_threads=2,     # 2 vCPU
            n_batch=256,
            logits_all=True,
            verbose=False
        )

    def add_task(self, task_data: Request):
        self.tasks_to_local.append(task_data)

    def get_answer(self, index: int) -> Response:
        if index < 0 or index >= len(self.tasks_to_local):
            return Response()

        task_data = self.tasks_to_local[index]
        response = self.generate(task_data)
        return response

    def generate(self, task_data: Request) -> Response:
        #print(f"Generating local response for task_id={task_data.id} with question={task_data.prompt} and category={task_data.category}\n")
        # 1600 characters =~ 400 tokens, which is a lot for a 4GB RAM model. We should send it to the API instead.

        """ system_content = (
            "You are a strict data extraction API. "
            "Output ONLY the exact final answer. "
            "CRITICAL: No pleasantries, no explanations, no introductions, no conversational filler."
        )

        prompt = task_data.prompt """
        """ if task_data.category == Category.SENTIMENT:
            prompt += "\n\nProvide the sentiment of the text as one of the following: Positive, Negative, Neutral, Mixed. If a review has both good and bad elements, classify it as Mixed."
 """
        #prompt += "\n\nProvide the final answer directly, with absolutely NO additional text or reasoning."
        system_content = "You are a highly precise data extraction and processing API."
        if task_data.category in [Category.FACTUAL_KNOWLEDGE, Category.SUMMARISATION]:
            system_content = "You are a knowledgeable assistant that gives clear, complete, well-explained answers while strictly following any formatting or length constraints given."
        elif task_data.category == Category.SENTIMENT:
            system_content = "You are a precise sentiment classification assistant."
        elif task_data.category == Category.NER:
            system_content = "You are a highly precise named entity extraction API."
        else:
            system_content = "You are a highly precise data extraction and processing API."

        prompt = task_data.prompt

        # Aplicăm reguli specifice în funcție de categorie
        if task_data.category == Category.SENTIMENT:
            prompt += (
                "\n\nCRITICAL RULES:"
                "\n1. If the prompt asks for a reason, you MUST provide exactly one sentence of reasoning."
                "\n2. If the text contains BOTH positive and negative elements, your classification MUST be 'Mixed'."
                "\n3. Do NOT add conversational filler."
            )
        elif task_data.category == Category.NER:
            prompt += (
                "\n\nCRITICAL RULES:"
                "\n1. Extract ALL named entities without omitting any."
                "\n2. Do NOT truncate dates (e.g., extract 'March 15 2023', not just '15 2023')."
                "\n3. Format exactly as requested, with NO conversational filler."
            )
        elif task_data.category in [Category.FACTUAL_KNOWLEDGE, Category.SUMMARISATION]:
            prompt += (
                "\n\nCRITICAL RULES:"
                "\n1. Follow the exact length constraints (e.g., exactly two sentences, or short bullets)."
                "\n2. Provide the full answer without omitting any details. If the question asks for an explanation, a comparison, why would you do that or choose that, you MUST provide a full explanation. For sentiment classification, be careful on choosing the category, think 2 times at sentiment classification, if the text contains both positive and negative elements, you MUST classify it as 'Mixed'."
                "\n3. Do NOT add conversational filler."
            )
        else:
            # Regula ta originală, excelentă pentru Math, Logic, etc.
            prompt += "\n\nProvide the final answer directly, with absolutely NO additional text or reasoning."

        system_content, prompt = build_prompt(task_data)

        messages = [
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]

        temperature = 0.0
        if task_data.category == Category.FACTUAL_KNOWLEDGE:
            temperature = 0.3

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=temperature,
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
        #response.set_text("[Confidence: {:.2f}]".format(confidence) + "\nCategory: " + str(task_data.category) + "\nDifficulty: " + str(task_data.difficulty) + "\n" + final_text)
        response.set_text(final_text)
        response.confidence = confidence

        return response
    
    def is_confident(self, task_data: Request, response: Response) -> bool:
        return True
        category, confidence = task_data.category, response.confidence
        if category == Category.SENTIMENT and confidence >= 0.2:
            return True
        if category == Category.FACTUAL_KNOWLEDGE and confidence >= 0.2:
            return True
        if category == Category.SUMMARISATION and confidence >= 0.2:
            return True
        if category == Category.NER and confidence >= 0.2:
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