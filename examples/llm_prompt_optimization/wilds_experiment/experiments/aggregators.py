from collections import Counter
from typing import List

from .workers import LLMWorker, parse_rating


class MajorityVoteAggregator:
    def aggregate(self, worker_predictions: List[int]) -> int:
        return Counter(worker_predictions).most_common(1)[0][0]


class LLMAggregator:
    def __init__(self, model_name: str, prompt_template: str, api_base: str | None = None) -> None:
        self.prompt_template = prompt_template
        self.worker = LLMWorker(model_name=model_name, api_base=api_base, max_tokens=32)

    def aggregate(self, worker_outputs: List[str], review_text: str) -> int:
        outputs_str = ""
        for idx, output in enumerate(worker_outputs, 1):
            outputs_str += f"\n### Annotator {idx}:\n{output}\n"

        prompt = self.prompt_template.format(review=review_text, worker_outputs=outputs_str)
        response = self.worker._call(prompt)
        return parse_rating(response)

