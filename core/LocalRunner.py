from data.Response import Response
from data.Request import Request
from data.Request import Category
from data.Formatter import solve_via_execution, clean_answer

import math
import time

from llama_cpp import Llama

class LocalRunner:
    def __init__(self, start_time: float, available_time: float, model_path="./models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"):

        self.tasks = []
        self.start_time = start_time
        self.available_time = available_time

        self.llm = Llama(
            model_path=model_path,
            n_ctx=700,       # limit context size to avoid Out of Memory
            n_threads=2,     # 2 vCPU
            n_batch=256,
            logits_all=True,
            verbose=False
        )

    # Add a task to the queue
    def add_task(self, task_data: Request):
        self.tasks.append(task_data)

    # Process all tasks in the queue and return their responses
    def process_tasks(self, fallback_tasks_list: list[Request]) -> list[Response]:
        answers = []

        for task in self.tasks:
            # Check if the available time has been exceeded, if so, add a fallback response to all remaining tasks to not exceed the time limit
            if self.start_time + self.available_time <= time.time():
                answers.append(Response(id = task.id, success=True, answer = "Unable to determine a confident answer within the time limit.", route = "local"))
                continue

            prompt, system_instructions = self._build_prompt(task)

            # For FACTUAL_KNOWLEDGE tasks, allow a slightly higher temperature to encourage more diverse responses
            temperature = 0.0
            if task.category == Category.FACTUAL_KNOWLEDGE:
                temperature = 0.3

            answer, confidence = self._generate_answer(prompt, system_instructions, temperature)

            # If the confidence is below the threshold for the task's category, add it to the fallback list for remote processing
            if not self.is_confident(task.category, confidence):
                fallback_tasks_list.append(task)
                continue

            answer = self._format_answer(task, answer)

            answers.append(Response(id = task.id, success=True, answer = answer, route = "local"))

        self.tasks.clear()

        return answers

    # Format the answer based on the task's category, executing code if necessary
    def _format_answer(self, task: Request, answer: str) -> str:
        if task.category in [Category.MATH, Category.LOGICAL]:
            answer_from_code = solve_via_execution(answer)

            # If the code execution fails, return the original answer cleaned up; otherwise, return the output from the code execution
            if answer_from_code in ["EXECUTION_ERROR", "UNKNOWN_ERROR"]:
                return clean_answer(answer)
            
            return clean_answer(answer_from_code)
        
        return clean_answer(answer)

    def _build_prompt(self, task: Request) -> tuple[str, str]:
        system_instruction, prompt = "", ""

        match task.category:
            case Category.NER:
                system_instruction = "You are a highly precise named entity extraction API."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. Extract ALL named entities without omitting any."
                    "\n2. Do NOT truncate dates (e.g., extract 'March 15 2023', not just '15 2023')."
                    "\n3. Format exactly as requested, with NO conversational filler."
                )
            case Category.SENTIMENT:
                system_instruction = "You are a precise sentiment classification assistant."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. If the text contains a contrastive structure (e.g. 'but', 'however', 'although'), "
                    "the sentiment expressed AFTER the contrast word usually reflects the overall conclusion — weigh it more heavily."
                    "\n2. Classify using ONLY the categories explicitly requested in the prompt above."
                    "\n3. First think through the positive and negative elements in one internal sentence, "
                    "then give your final classification and a one-sentence reason."
                    "\n4. Do NOT add conversational filler beyond the required reasoning sentence."
                )
            case Category.FACTUAL_KNOWLEDGE:
                system_instruction = "You are a knowledgeable assistant that gives clear, complete, well-explained answers while strictly following any formatting or length constraints given."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. FOLLOW the EXACT length constraints (the number of sentences, the number of words, answer format, the number of bullet points if they ask for them)."
                    "\n2. Provide the FULL answer without omitting any details. If the question asks for an explanation, comparison, or reasoning ('why', 'how', 'explain'), you MUST provide a complete, substantive explanation — do not give a one-word or overly terse answer."
                    "\n3. Do NOT add conversational filler like 'Sure, here's the answer:'."
                    "\n4. MAKE SURE TO CHECK YOUR ANSWER AND BE SURE TO BE CORRECT, DO NOT MAKE UP ANSWERS."
                )
            case Category.SUMMARISATION:
                system_instruction = "You are a knowledgeable assistant that gives clear, complete, well-explained answers while strictly following any formatting or length constraints given."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. FOLLOW the EXACT length constraints (the number of sentences, the number of words, answer format, the number of bullet points if they ask for them)."
                    "\n2. Provide the FULL answer without omitting any details."
                    "\n3. Do NOT add conversational filler like 'Sure, here's the answer:'."
                )
            case Category.CODE_GENERATION:
                system_instruction = "You are an expert code generation assistant that provides accurate and efficient code solutions."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. Ensure the code is syntactically correct and follows best practices."
                    "\n2. Provide explanations only if explicitly requested."
                    "\n3. Do NOT add conversational filler."
                )
            case Category.CODE_DEBUG:
                system_instruction = "You are an expert code debugging assistant that identifies and resolves code issues efficiently."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. Identify the root cause of the issue and provide a clear solution."
                    "\n2. Ensure the solution is syntactically correct and follows best practices."
                    "\n3. Do NOT add conversational filler."
                )
            case Category.LOGICAL | Category.MATH:
                system_instruction = "You are an expert problem-solving and code generation assistant that provides accurate and well-reasoned python solutions to logical and mathematical problems."
                prompt = task.prompt + (
                    "\n\nCRITICAL RULES:"
                    "\n1. The answer MUST be a Python script wrapped in ```python ... ``` that calculates and prints the final result"
                    "\n2. Do NOT provide any text, explanations, or greetings — only the code."
                    "\n3. Do NOT add conversational filler."
                )
            case _:
                system_instruction = "You are a highly precise data extraction and processing API."
                prompt = task.prompt + (
                    "\n\nProvide the final answer directly, with absolutely NO additional text or reasoning."
                )

        return prompt, system_instruction

    def _generate_answer(self, prompt: str, system_instructions: str, temperature: float = 0.0) -> tuple[str, float]:
        messages = [
            {
                "role": "system", 
                "content": system_instructions
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Generate the response
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=temperature,
            top_p=0.9,
            logprobs=True,
            top_logprobs=1
        )

        final_text = response["choices"][0]["message"]["content"]

        # Calculate confidence based on logprobs
        logprobs_data = response["choices"][0].get("logprobs", {})
        token_logprobs = logprobs_data.get("content", []) if logprobs_data else []

        if not token_logprobs:
            confidence = 0.0
        else:
            total_prob = 0.0
            for token_data in token_logprobs:
                total_prob += math.exp(token_data["logprob"])
            
            confidence = total_prob / len(token_logprobs)

        return final_text, confidence
    
    def is_confident(self, category: Category, confidence: float) -> bool:
        match category:
            case Category.SENTIMENT:
                return confidence >= 0.2
            case Category.FACTUAL_KNOWLEDGE:
                return confidence >= 0.2
            case Category.SUMMARISATION:
                return confidence >= 0.2
            case Category.NER:
                return confidence >= 0.2
            case Category.CODE_DEBUG:
                return confidence >= 0.9
            case Category.CODE_GENERATION:
                return confidence >= 0.9
            case Category.LOGICAL:
                return confidence >= 0.9
            case Category.MATH:
                return confidence >= 0.9
            case _:
                return False
            