import time
import requests

from data.Response import Response
from data.Request import Request, Category

import subprocess
import sys
import re

def solve_via_execution(api_response: str) -> str:
    # 1. Căutăm blocul de cod Python folosind Regex
    match = re.search(r"```(?:python)?\n(.*?)\n```", api_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        clean_code = match.group(1).strip()
    else:
        # Dacă nu a pus blocuri de cod, încercăm să rulăm tot răspunsul (riscant, dar e un fallback)
        clean_code = api_response.strip()

    # 2. Executăm codul
    try:
        result = subprocess.run(
            [sys.executable, "-c", clean_code],
            capture_output=True,
            text=True,
            timeout=3,
            check=True
        )
        return result.stdout.strip()
        
    except subprocess.CalledProcessError as e:
        #print(f"[EXEC ERROR] Scriptul a dat crash: {e.stderr}")
        return "EXECUTION_ERROR"
    except Exception as e:
        return "UNKNOWN_ERROR"

def clean_answer(answer: str) -> str:
        if answer is None:
            return ""

        cleaned = answer.strip()

        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()

        return cleaned.strip()

def build_batch_prompt(remote_tasks: list) -> str:
    tasks_text = "\n".join([f"ID:{t.id} PROMPT:{t.prompt}" for t in remote_tasks])

    system_instruction = f"""Solve all independent tasks below.
CRITICAL RULES:
1. You MUST output EXACTLY one line per task using this strict format: @@[task_id]|[your_answer]
2. For MATH and LOGICAL tasks: The [your_answer] MUST be a Python script wrapped in ```python ... ``` that prints the final result.
3. For CODE_GENERATION: Provide ONLY the raw code.
4. NO conversational text, NO XML, NO reasoning.
5. Provide the final answer directly, with absolutely NO additional text or reasoning for any task. Make the responses AS SHORT AS POSSIBLE, but still COMPLETE and ACCURATE.

Tasks:
{tasks_text}
"""
    return system_instruction

def process_batch_response(api_response: str, remote_tasks: list, final_results_list: list):
    task_categories = {t.id: t.category for t in remote_tasks}
    
    # Căutăm secțiunile care încep cu @@
    # (Folosim split('@@') și ignorăm primul element dacă e gol)
    blocks = api_response.split('@@')
    
    for block in blocks:
        if not block.strip() or '|' not in block:
            continue
            
        # Despărțim ID-ul de Răspuns
        parts = block.split('|', 1)
        task_id = parts[0].strip()
        raw_content = parts[1].strip()
        
        category = task_categories.get(task_id)
        if category in [Category.MATH, Category.LOGICAL]:
            final_answer = solve_via_execution(raw_content)
            if final_answer in ["TIMEOUT_ERROR", "EXECUTION_ERROR", "UNKNOWN_ERROR"]:
                final_answer = clean_answer(raw_content)  # fallback to raw content if execution fails
        else:
            final_answer = clean_answer(raw_content)
            
        final_results_list.append({
            "task_id": task_id,
            "answer": final_answer
        })

class RemoteClient:
    def __init__(self, api_key, base_url, allowed_models):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.allowed_models = allowed_models or []
        self.tasks_to_api = [[]]
        self.cur = 0

    def add_task(self, task_data: Request):
        #print(f"Adding task to remote queue: {task_data.id}; category: {task_data.category}\n")
        #self.tasks_to_api.append(task_data)
        self.tasks_to_api[self.cur].append(task_data)
        if len(self.tasks_to_api[self.cur]) >= 5:
            self.cur += 1
            self.tasks_to_api.append([])

    def get_answers(self, local_model, jfile, model: str, temperature: float = 0.0, max_tokens: int = 16000):
        for batch in self.tasks_to_api:
            if not batch:
                continue
            ok: bool = False
            for i in range(2):  # retry once
                try:
                    self.get_answers_help(batch, jfile, model, temperature, max_tokens)
                    ok = True
                    break  # if successful, break out of retry loop
                except Exception:
                    pass
            if not ok:
                print(f"Failed to get answers from remote model '{model}' after 2 attempts. Falling back to local model. Tasks: {[t.id for t in batch]}")
                for task in batch:
                    local_model.add_task(task)

    def get_answers_help(self, tasks_here, jfile, model: str, temperature: float = 0.0, max_tokens: int = 16000):
        prompt = build_batch_prompt(tasks_here)

        if not model.startswith("accounts/fireworks/models/"):
            model = f"accounts/fireworks/models/{model}"

        payload = {
            #"model": f"accounts/fireworks/models/{model}",
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict data processing engine. You are given multiple independent tasks."
                        "You MUST output your response for EACH task enclosed in exact XML tags like this: <answer id=\"TASK_ID\">your response here</answer>."
                        "CRITICAL RULES BASED ON CATEGORY: 1 For MATH and LOGICAL tasks: The content inside the <answer> tag MUST be a Python script wrapped in ```python ... ``` that calculates and prints the final result. No text, no greetings.\n 2. For CODE_GENERATION: Provide only the raw code. \n3. For CODE_DEBUG and FACTUAL: Provide the direct, terse answer. Maximum one sentence. NO conversational filler."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "stream": False
        }

        #print(f"Tasks for api: {prompt}")
        
        api_response = requests.post(
            self._chat_url(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=240
        )

        api_response.raise_for_status()
        data = api_response.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        #print(f"\n\nRemote model '{model}'; returned answer: {answer} for question: {prompt}\n\n")

        ans = []
        process_batch_response(answer, tasks_here, ans)
        print("USAGE: ", data.get("usage", {}))
        #print("ANSWERS!!: \n", ans)
        for item in ans:
            jfile.append(item)


    def generate(
        self,
        task_data: Request,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 160
    ) -> Response:
        response = Response()

        if not self.api_key or not self.base_url:
            response.success = False
            response.error = "Missing Fireworks API configuration"
            response.route = "remote_error"
            return response

        #prompt = self._build_prompt(task_data)
        prompt = ""
        if task_data.category == Category.MATH or task_data.category == Category.LOGICAL:
            prompt = f"""Write a Python script to solve the following problem. 
                                    You MUST wrap your code strictly inside ```python and ``` blocks.
                                    The script MUST use print() to output ONLY the final exact result (e.g., the fraction, the number, or the name). Do not print labels like "The answer is:".
                                    Do not say hello. Do not explain the code.
                                    NO conversational text, NO explanations, NO comments.

                                    Problem: {task_data.prompt}"""
        else:
            prompt = task_data.prompt + "\n\nProvide the final answer directly, with absolutely NO additional text or reasoning and NO formatting. Make the response as short as possible, but still complete and accurate."
            if task_data.category == Category.CODE_DEBUG:
                prompt += "\n\nIf the prompt asks for explanation, provide them in one sentence."

        payload = {
            "model": f"accounts/fireworks/models/{model}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise English assistant. "
                        "Output ONLY the exact final answer. "
                        "CRITICAL: No pleasantries, no explanations, no introductions, no conversational filler."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "stream": False
        }

        last_error = None

        for attempt in range(2):  # one retry
            try:
                api_response = requests.post(
                    self._chat_url(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=25
                )

                api_response.raise_for_status()
                data = api_response.json()

                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                #print(f"\n\nRemote model '{model}'; id: {task_data.id}, Category: {task_data.category} returned answer: {answer} for question: {prompt}\n\n")

                if task_data.category in [Category.MATH, Category.LOGICAL]:
                    # Execute the code and get the result
                    execution_result = solve_via_execution(answer)
                    if execution_result in ["TIMEOUT_ERROR", "EXECUTION_ERROR"]:
                        """ response.success = False
                        response.error = f"Remote model returned an error during execution: {execution_result}"
                        response.route = "remote_error"
                        response.model = model
                        response.raw_response = data
                        response.usage = data.get("usage", {})
                        return response """
                        pass
                    else:
                        answer = execution_result

                cleaned_answer = self._clean_answer(answer)

                if not cleaned_answer:
                    response.success = False
                    response.error = "Remote model returned empty answer"
                    response.route = "remote_error"
                    response.model = model
                    response.raw_response = data
                    response.usage = data.get("usage", {})
                    return response

                response.success = True
                response.set_text(cleaned_answer)
                response.raw_response = data
                response.usage = data.get("usage", {})
                response.model = model
                response.route = "remote"

                return response

            except Exception as error:
                last_error = str(error)
                time.sleep(0.5)

        response.success = False
        response.error = f"Remote call failed: {last_error}"
        response.model = model
        response.route = "remote_error"
        return response

    def _chat_url(self) -> str:
        base_url_clean = self.base_url.rstrip("/")
        if base_url_clean.endswith("/chat/completions"):
            return base_url_clean

        return f"{base_url_clean}/chat/completions"

    def _build_prompt(self, task: Request) -> str:
        category = task.category
        prompt = task.prompt

        if category == Category.SENTIMENT:
            return (
                "Classify sentiment. If the user requested a specific format, follow it. "
                "Otherwise return: Label — brief reason. "
                "Valid labels: Positive, Negative, Neutral, Mixed.\n\n"
                f"Text:\n{prompt}"
            )

        if category == Category.NER:
            return (
                "Extract named entities. If the user requested specific entity types or format, follow that. "
                "Otherwise return JSON with keys: person, organization, location, date. "
                "Use empty arrays when none.\n\n"
                f"Text:\n{prompt}"
            )

        if category == Category.SUMMARISATION:
            return (
                "Summarize exactly as requested. Respect any sentence, word, bullet, or length constraint. "
                "Be concise.\n\n"
                f"Text:\n{prompt}"
            )

        if category == Category.MATH:
            return (
                "Solve the problem carefully. Return the final answer with a brief calculation only.\n\n"
                f"Problem:\n{prompt}"
            )

        if category == Category.LOGICAL:
            return (
                "Solve all constraints. Return the final answer and a brief justification only.\n\n"
                f"Problem:\n{prompt}"
            )

        if category == Category.CODE_DEBUG:
            return (
                "Fix the code. Return corrected code only unless the prompt asks for explanation. If the prompt asks for explanation, provide them in one sentence.\n\n"
                f"Prompt:\n{prompt}"
            )

        if category == Category.CODE_GENERATION:
            return (
                "Write the requested function or code. Return clean code only unless the prompt asks for explanation.\n\n"
                f"Spec:\n{prompt}"
            )

        if category == Category.FACTUAL_KNOWLEDGE:
            return (
                "Answer accurately and concisely in English.\n\n"
                f"Question:\n{prompt}"
            )

        return prompt

    def _clean_answer(self, answer: str) -> str:
        if answer is None:
            return ""

        cleaned = answer.strip()

        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()

        return cleaned.strip()