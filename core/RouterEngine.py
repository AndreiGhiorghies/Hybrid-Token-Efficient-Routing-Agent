from dataclasses import dataclass
from typing import Optional

from core.TaskClassifier import TaskClassifier
from core.LocalRunner import LocalRunner
from core.RemoteClient import RemoteClient
from data.Request import Request, Category
from data.Response import Response


@dataclass
class RouteDecision:
    route: str
    model: Optional[str]
    temperature: float
    max_tokens: int
    batchable: bool
    reason: str


class RouterEngine:
    def __init__(
        self,
        classifier: TaskClassifier,
        local_runner: LocalRunner,
        remote_client: RemoteClient
    ):
        self.classifier = classifier
        self.local_runner = local_runner
        self.remote_client = remote_client
        self.allowed_models = remote_client.allowed_models
        self.tasks_to_api = dict()

    def get_answers(self, jfile):
        decision = self.choose_route(Request(0, ""))
        try:
            self.remote_client.get_answers(self.local_runner, jfile, decision.model or "kimi-k2p7-code")
        except Exception:
            pass

    def process_request(self, raw_task: Request):
        self._ensure_classified(raw_task)

        best_local_response = None

        """ if raw_task.category != Category.CODE_DEBUG:
            return Response() """

        if self._is_local_candidate(raw_task):
            self.local_runner.add_task(raw_task)
            return
            """ best_local_response = self._safe_local_generate(raw_task)

            if (
                best_local_response
                and best_local_response.success
                and self.local_runner.is_confident(raw_task, best_local_response)
            ):
                best_local_response.route = "local"
                best_local_response.model = "local"
                return best_local_response """
            
        """ if raw_task.category in [Category.CODE_DEBUG, Category.CODE_GENERATION]:
            if "code" not in self.tasks_to_api:
                self.tasks_to_api["code"] = []
            self.tasks_to_api["code"].append(raw_task)
        if raw_task.category in [Category.LOGICAL, Category.MATH]:
            if "logical" not in self.tasks_to_api:
                self.tasks_to_api["logical"] = []
            self.tasks_to_api["python"].append(raw_task)
        if raw_task.category in [Category.SENTIMENT, Category.FACTUAL_KNOWLEDGE, Category.SUMMARISATION, Category.NER]:
            if "easy" not in self.tasks_to_api:
                self.tasks_to_api["easy"] = []
            self.tasks_to_api["easy"].append(raw_task)

        return Response() """

        self.remote_client.add_task(raw_task)

        return

        decision = self.choose_route(raw_task)

        remote_response = self.remote_client.generate(
            task_data=raw_task,
            model=decision.model,
            temperature=decision.temperature,
            max_tokens=decision.max_tokens
        )

        if remote_response.success and remote_response.get_text():
            remote_response.route = "remote"
            remote_response.model = decision.model
            return remote_response

        if best_local_response and best_local_response.get_text():
            best_local_response.route = "fallback_local"
            best_local_response.model = "local"
            return best_local_response

        """ fallback_response = self._safe_local_generate(raw_task)

        if fallback_response and fallback_response.get_text():
            fallback_response.route = "fallback_local"
            fallback_response.model = "local"
            return fallback_response """

        empty_fallback = Response()
        empty_fallback.success = False
        empty_fallback.route = "fallback"
        empty_fallback.model = "none"
        empty_fallback.set_text("Unable to determine a confident answer. here:/")
        return empty_fallback

    def _ensure_classified(self, task: Request) -> None:
        if getattr(task, "category", Category.NONE) != Category.NONE:
            return

        classified_task = self.classifier.classify(task)

        # Supports both styles:
        # 1. classifier mutates task and returns None
        # 2. classifier returns a classified Request object
        if classified_task is not None:
            task.category = getattr(classified_task, "category", task.category)
            task.difficulty = getattr(classified_task, "difficulty", task.difficulty)

    def _is_local_candidate(self, task: Request) -> bool:
        try:
            return self.classifier.is_local_friendly(task)
        except Exception:
            return False

    def _safe_local_generate(self, task: Request):
        try:
            return self.local_runner.generate(task)
        except Exception:
            return None

    def choose_route(self, task: Request) -> RouteDecision:
        category = task.category
        difficulty = float(task.difficulty)

        model = self._select_model(category, difficulty)

        return RouteDecision(
            route="remote",
            model=model,
            temperature=0.0,
            max_tokens=self._max_tokens(category, difficulty),
            batchable=self._is_batchable(category, difficulty),
            reason=f"Selected {model} for category={category.name}, difficulty={difficulty}"
        )

    def _select_model(self, category: Category, difficulty: float) -> str:
        return self._first_allowed(["kimi-k2p7-code", "minimax-m3", "gemma-4-31b-it", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it"])

        if category == Category.CODE_DEBUG:
            candidates = [
                "kimi-k2p7-code",
                "minimax-m3",
                "gemma-4-31b-it"
            ]

        elif category == Category.CODE_GENERATION:
            candidates = [
                "kimi-k2p7-code",
                "gemma-4-31b-it",
                "minimax-m3"
            ]

        elif category == Category.LOGICAL:
            if difficulty >= 7.0:
                candidates = [
                    "minimax-m3",
                    "gemma-4-31b-it",
                    "gemma-4-31b-it-nvfp4"
                ]
            else:
                candidates = [
                    "gemma-4-31b-it-nvfp4",
                    "gemma-4-31b-it",
                    "minimax-m3"
                ]

        elif category == Category.MATH:
            if difficulty >= 7.0:
                candidates = [
                    "minimax-m3",
                    "gemma-4-31b-it",
                    "gemma-4-31b-it-nvfp4"
                ]
            else:
                candidates = [
                    "gemma-4-31b-it-nvfp4",
                    "gemma-4-26b-a4b-it",
                    "minimax-m3"
                ]

        elif category in {
            Category.SENTIMENT,
            Category.NER,
            Category.SUMMARISATION
        }:
            candidates = [
                "gemma-4-26b-a4b-it",
                "gemma-4-31b-it-nvfp4",
                "gemma-4-31b-it"
            ]

        elif category == Category.FACTUAL_KNOWLEDGE:
            candidates = [
                "gemma-4-31b-it-nvfp4",
                "gemma-4-26b-a4b-it",
                "gemma-4-31b-it"
            ]

        else:
            candidates = [
                "gemma-4-31b-it-nvfp4",
                "gemma-4-26b-a4b-it",
                "minimax-m3"
            ]

        return self._first_allowed(candidates)

    def _first_allowed(self, candidates: list[str]) -> str:
        for model in candidates:
            for allowed_model in self.allowed_models:
                if model in allowed_model:
                    return model

        if self.allowed_models:
            return self.allowed_models[0]
        
        return "kimi-k2p7-code" # to be deleted

        #raise RuntimeError("No allowed models available")

    def _max_tokens(self, category: Category, difficulty: float) -> int:
        return 64000
        if category == Category.SENTIMENT:
            return 50

        if category == Category.NER:
            return 120

        if category == Category.SUMMARISATION:
            return 100 if difficulty <= 6.5 else 180

        if category == Category.FACTUAL_KNOWLEDGE:
            return 140

        if category == Category.MATH:
            return 180

        if category == Category.LOGICAL:
            return 250

        if category == Category.CODE_DEBUG:
            return 450

        if category == Category.CODE_GENERATION:
            return 500

        return 160

    def _is_batchable(self, category: Category, difficulty: float) -> bool:
        if difficulty > 6.5:
            return False

        return category in {
            Category.SENTIMENT,
            Category.NER,
            Category.SUMMARISATION,
            Category.FACTUAL_KNOWLEDGE,
            Category.MATH
        }