from __future__ import annotations

from typing import Any, List, Optional, Union, Literal

import logging
import numpy as np
import pandas as pd
from prefect import task

from vistiq.core import StackProcessor, StackProcessorConfig
from vistiq.utils import ArrayIteratorConfig

from vistiq.processors.base import BaseProcessor
from vistiq.processors.types import WorkflowData

logger = logging.getLogger(__name__)


# =========================
# REAL LOGIC (moved from seg.py)
# =========================

class LabelRemoverConfig(StackProcessorConfig):
    """Configuration for label removal operations."""

    iterator_config: ArrayIteratorConfig = ArrayIteratorConfig(slice_def=())
    remap: bool = False
    output_type: Literal["stack"] = "stack"
    squeeze: bool = False


class LabelRemover(StackProcessor):
    """Remove specified labels from a label array by setting them to 0."""

    def __init__(self, config: LabelRemoverConfig):
        super().__init__(config)

    @classmethod
    def from_config(cls, config: LabelRemoverConfig) -> "LabelRemover":
        return cls(config)

    def _extract_label_ids(
        self,
        label_ids: Union[List["RegionProperties"], pd.DataFrame, List[int], np.ndarray],
    ) -> np.ndarray:
        """Extract label IDs from flexible input formats."""

        if isinstance(label_ids, pd.DataFrame):
            if "label" in label_ids.columns:
                return label_ids["label"].values.astype(np.int32)
            else:
                return label_ids.index.values.astype(np.int32)

        elif isinstance(label_ids, list) and len(label_ids) > 0:
            if hasattr(label_ids[0], "label"):
                return np.array([r.label for r in label_ids], dtype=np.int32)
            return np.array(label_ids, dtype=np.int32)

        elif isinstance(label_ids, np.ndarray):
            return label_ids.astype(np.int32)

        return np.array([], dtype=np.int32)

    def _process_slice(
        self,
        labels: np.ndarray,
        label_ids: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Remove labels from a single slice."""

        labels = np.asarray(labels)
        result = np.array(labels, dtype=np.int32, copy=True)

        if len(label_ids) > 0:
            mask = np.isin(result, label_ids)
            result[mask] = 0

        return result

    @task(name="LabelRemover.run")
    def run(
        self,
        labels: np.ndarray,
        region_properties: Union[List["RegionProperties"], pd.DataFrame, List[int], np.ndarray],
        workers: int = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Run label removal."""

        label_ids = self._extract_label_ids(region_properties)

        return super().run(
            labels,
            label_ids=label_ids,
            workers=workers,
            verbose=verbose,
            metadata=metadata,
            **kwargs,
        )


# =========================
# PROCESSOR WRAPPER
# =========================

class LabelRemoverProcessor(BaseProcessor):
    name = "label_remover"

    def __init__(self, config: LabelRemoverConfig):
        self.remover = LabelRemover.from_config(config)

    @property
    def input_keys(self) -> list[str]:
        return ["labels", "labels_to_remove"]

    @property
    def output_keys(self) -> list[str]:
        return ["labels"]

    def run(self, data: WorkflowData) -> WorkflowData:
    
        updated = dict(data)

        result = self.remover.run(
            labels=data["labels"],
            region_properties=data.get("labels_to_remove", []),
            metadata=data.get("metadata"),
        )

        # handle tuple output
        if isinstance(result, tuple):
            labels_out, metadata_out = result
            updated["labels"] = labels_out
            updated["metadata"] = metadata_out
        else:
            updated["labels"] = result

        # keep region_features consistent with updated labels
        if "region_features" in updated and updated["region_features"] is not None:
            remaining_labels = set(int(x) for x in np.unique(updated["labels"]) if int(x) != 0)
            feats = updated["region_features"]

            if isinstance(feats, pd.DataFrame):
                # if labels are the index
                if feats.index.name == "label" or "label" in str(feats.index.name).lower():
                    updated["region_features"] = feats.loc[feats.index.isin(remaining_labels)]
                # fallback if label is a column
                elif "label" in feats.columns:
                    updated["region_features"] = feats.loc[feats["label"].isin(remaining_labels)]
            elif isinstance(feats, list):
                updated["region_features"] = [
                    r for r in feats
                    if getattr(r, "label", None) in remaining_labels
                ]


        return updated