from __future__ import annotations

from vistiq.processors.base import BaseProcessor
from vistiq.processors.types import WorkflowData



from typing import Any

import numpy as np
import pandas as pd


def summarize_state(data: WorkflowData) -> None:
    labels = data.get("labels")
    feats = data.get("region_features")
    remove_ids = data.get("labels_to_remove", [])

    if labels is not None:
        unique_labels = sorted(int(x) for x in np.unique(labels) if int(x) != 0)
        print(f"label count: {len(unique_labels)}")
        print(f"first labels: {unique_labels[:10]}")

    if feats is not None:
        if isinstance(feats, pd.DataFrame):
            print(f"feature rows: {len(feats)}")
            print(f"first feature labels: {list(feats.index[:10])}")
        elif isinstance(feats, list):
            print(f"region objects: {len(feats)}")
            first_labels = [
                getattr(r, "label", None) for r in feats[:10]
            ]
            print(f"first region labels: {first_labels}")

    if remove_ids is not None:
        if hasattr(remove_ids, "__len__"):
            print(f"labels_to_remove count: {len(remove_ids)}")
            try:
                print(f"first labels_to_remove: {list(remove_ids[:10])}")
            except Exception:
                pass

class ProcessorPipeline:
    def __init__(self, processors: list[BaseProcessor], debug: bool = False):
        self.processors = processors
        self.debug = debug

    def run(self, data: WorkflowData) -> WorkflowData:
        current = dict(data)

        if self.debug:
            print("Initial state:")
            summarize_state(current)

        for processor in self.processors:
            current = processor.run(current)

            if self.debug:
                print(f"After processor: {processor.name}")
                summarize_state(current)
       
        return current