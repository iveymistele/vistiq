from __future__ import annotations

from abc import ABC, abstractmethod

from vistiq.processors.types import WorkflowData


class BaseProcessor(ABC):
    name: str = "base_processor"

    @property
    def input_keys(self) -> list[str]:
        return []

    @property
    def output_keys(self) -> list[str]:
        return []

    @abstractmethod
    def run(self, data: WorkflowData) -> WorkflowData:
        raise NotImplementedError