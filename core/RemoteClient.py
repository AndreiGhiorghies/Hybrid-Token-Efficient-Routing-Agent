from data.Response import Response
from data.Request import Request, Category
from data.Formatter import solve_via_execution, clean_answer

import requests
import time

class RemoteClient:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str], start_time: float, available_time: float, batch_size: int = 7, debug: bool = False, show_usage: bool = False):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.allowed_models = allowed_models or []

        # Tasks queues for different request types
        self.text_tasks = [[]]
        self.code_tasks = [[]]
        self.batch_size = batch_size

        self.start_time = start_time
        self.available_time = available_time

        self.debug = debug
        self.show_usage = show_usage
        self.usage = {}

    # Add a task to the appropriate queue based on its category
    def add_task(self, task: Request):
        if task.category in [Category.CODE_DEBUG, Category.CODE_GENERATION, Category.LOGICAL, Category.MATH]:
            if self.code_tasks and len(self.code_tasks[-1]) >= self.batch_size:
                self.code_tasks.append([])
            
            self.code_tasks[-1].append(task)
        else:
            if self.text_tasks and len(self.text_tasks[-1]) >= self.batch_size:
                self.text_tasks.append([])

            self.text_tasks[-1].append(task)

    # Process a batch of tasks and return their responses
    def _process_batch(self, tasks: list[Request], fallback_tasks_list: list[Request]) -> list[Response]:
        if not tasks or len(tasks) == 0:
            return []

        # Build a single prompt for the batch of tasks
        prompt, system_instruction = self._build_batch_prompt(tasks)

        # Determine the appropriate model to use based on the first task's category
        model = self._get_model(tasks[0])
        answers = []

        # Check if the available time has been exceeded, if so, add a fallback response to all remaining tasks to not exceed the time limit
        if self.start_time + self.available_time <= time.time():
            for task in tasks:
                answers.append(Response(id=task.id, success=True, answer="Unable to determine a confident answer within the time limit.", route="remote", model=model))

            return answers

        batch_success = False
        
        # Attempt to call the remote API and process the response, retrying once if it fails
        for attempt in range(2):
            try:
                max_tokens = self._get_max_tokens_batch(tasks)

                response = self._call_remote_api(model, prompt, system_instruction, max_tokens=max_tokens)
                answers = self._process_batch_response(response, tasks, model)

                batch_success = True
                break

            except Exception:
                pass

        if not batch_success:
            # If the remote call fails, add the tasks to the fallback list for local processing
            fallback_tasks_list.extend(tasks)
        
        return answers

    def _process_batch_response(self, api_response: str, tasks: list[Request], model: str) -> list[Response]:
        # Dict to map task ids to their categories for easier processing
        task_categories = {task.id: task.category for task in tasks}

        response_list = []

        # Split the response into blocks based on the @@ delimiter (delimiter for each task's response)
        blocks = api_response.split('@@')
        
        for block in blocks:
            # If it does not contain a task id and answer, skip it
            if not block.strip() or '|' not in block:
                continue
                
            parts = block.split('|', 1)
            task_id = parts[0].strip()
            raw_content = parts[1].strip()

            # Get the category of the task to determine how to process the answer
            category = task_categories.get(task_id)

            if category in [Category.MATH, Category.LOGICAL]:
                # Try to execute the code to get the final answer for these categories
                final_answer = solve_via_execution(raw_content)

                # Fallback to cleaning the answer if execution fails, maybe the model the direct answer without code block
                if final_answer in ["EXECUTION_ERROR", "UNKNOWN_ERROR"]:
                    final_answer = clean_answer(raw_content)
            else:
                # The other categories should have direct answers
                final_answer = clean_answer(raw_content)
                
            response_list.append(Response(id=task_id, success=True, answer=final_answer, route="remote", model=model))

        return response_list

    # Determine the appropriate model based on the task's category
    def _get_model(self, task: Request) -> str:
        model = ""
        if task.category in [Category.CODE_DEBUG, Category.CODE_GENERATION, Category.LOGICAL, Category.MATH]:
            model = self._first_allowed_model(["kimi-k2p7-code", "minimax-m3"])
        else:
            model = self._first_allowed_model(["minimax-m3", "kimi-k2p7-code"])

        # Ensure the model name is prefixed with the required path if not already present
        if not model.startswith("accounts/fireworks/models/"):
            model = f"accounts/fireworks/models/{model}"
        
        return model

    # Get the first allowed model from a list of candidates
    def _first_allowed_model(self, candidates: list[str]) -> str:
        for model in candidates:
            for allowed_model in self.allowed_models:
                # Check if the candidate model is in the list of allowed models
                if model in allowed_model:
                    return model

        # If no allowed models are found, return the first allowed model if available
        if self.allowed_models:
            return self.allowed_models[0]

        # If no allowed models are available, raise an error
        raise RuntimeError("No allowed models available")

    def _call_remote_api(self, model: str, prompt: str, system_instruction: str, temperature: float = 0.0, max_tokens: int = 16000, reasoning_effort: str = "none") -> str:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "stream": False
        }

        api_response = requests.post(
            self._chat_url(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=180
        )

        api_response.raise_for_status()
        data = api_response.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if self.debug:
            print(f"\n\nRemote model '{model}'; returned answer: {answer} for prompt: {prompt}\n Usage: {data.get('usage', {})}\n\n")
        if self.show_usage:
            usage = data.get("usage", {})
            for key, value in usage.items():
                self.usage[key] = self.usage.get(key, 0) + value

        return answer

    # Process all batches of tasks and return their responses, adding any failed tasks to the fallback list
    def process_batches(self, fallback_tasks_list: list[Request]) -> list[Response]:
        responses = []

        for batch in self.code_tasks:
            if not batch:
                continue
            batch_responses = self._process_batch(batch, fallback_tasks_list)
            responses.extend(batch_responses)

        for batch in self.text_tasks:
            if not batch:
                continue
            batch_responses = self._process_batch(batch, fallback_tasks_list)
            responses.extend(batch_responses)

        self.code_tasks.clear()
        self.text_tasks.clear()

        return responses

    # Build a prompt that includes all tasks in a single string
    def _build_batch_prompt(self, tasks: list[Request]) -> tuple[str, str]:
        # Build a prompt that includes all tasks in a single string
        tasks_text = "\n".join([f"ID:{task.id} PROMPT:{task.prompt}" for task in tasks])

        # System instruction for the model to process the tasks
        system_instruction = (
            "You are a strict data processing engine. You are given multiple independent tasks."
            "You MUST output your response for EACH task on EXACTLY one line using this strict format: @@[task_id]|[your_answer]"
            f"""CRITICAL RULES BASED ON CATEGORY:
                1. For MATH and LOGICAL tasks: The [your_answer] MUST be a Python script wrapped in ```python ... ``` that calculates and prints the final result. No text, no greetings.
                2. For CODE_GENERATION: Provide ONLY the raw code.
                3. For CODE_DEBUG and FACTUAL: Provide the direct, terse answer. Maximum one sentence. NO conversational filler.
                4. NO conversational text, NO XML, NO reasoning.
                5. Provide the final answer directly, with absolutely NO additional text or reasoning for any task. Make the responses AS SHORT AS POSSIBLE, but still COMPLETE and ACCURATE."""
        )

        prompt = f"""
            Solve all independent tasks below.
            Tasks:
            {tasks_text}
        """
        
        return prompt, system_instruction
    
    # Determine the maximum number of tokens for a batch of tasks based on their categories and difficulties
    def _get_max_tokens_batch(self, tasks: list[Request]) -> int:
        max_tokens = 0
        for task in tasks:
            max_tokens += self._max_tokens(task.category, task.difficulty)
        
        return max_tokens

    def _max_tokens(self, category: Category, difficulty: float) -> int:
        if category == Category.SENTIMENT:
            return 100

        if category == Category.NER:
            return 120

        if category == Category.SUMMARISATION:
            return 120 if difficulty <= 6.5 else 180

        if category == Category.FACTUAL_KNOWLEDGE:
            return 140

        if category == Category.MATH:
            return 270

        if category == Category.LOGICAL:
            return 300

        if category == Category.CODE_DEBUG:
            return 450

        if category == Category.CODE_GENERATION:
            return 500

        return 160

    # Format the base URL for the chat completions endpoint
    def _chat_url(self) -> str:
        base_url_clean = self.base_url.rstrip("/")
        if base_url_clean.endswith("/chat/completions"):
            return base_url_clean

        return f"{base_url_clean}/chat/completions"

    def __del__(self):
        if self.show_usage:
            print("Total API Usage:", self.usage)