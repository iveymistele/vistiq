from typing import Any, Optional

import numpy as np
from pydantic import Field

from vistiq.analysis.coincidence import CoincidenceDetectorConfig
from vistiq.analysis.distance import DistanceCalculatorConfig
from vistiq.core import ArrayIteratorConfig, Configurable
from vistiq.workflow import Workflow, WorkflowConfig
from vistiq.segment.analysis import RegionAnalyzer, RegionAnalyzerConfig
from vistiq.utils import resolve_futures

import itertools
import logging

logger = logging.getLogger(__name__)

class AnalysisFlowConfig(WorkflowConfig):
    """Configuration for the analysis workflow.

    Args:
        config: Configuration for the analysis workflow.
    """

    region_analyzer: RegionAnalyzerConfig = Field(default_factory=lambda:RegionAnalyzerConfig(
        properties=["centroid"], 
        output_type="dataframe",
        iterator_config=ArrayIteratorConfig(slice_def=())))
    distance_calculator: Optional[DistanceCalculatorConfig] = None
    coincidence_detector: Optional[CoincidenceDetectorConfig] = None

class AnalysisFlow(Workflow):
    """Workflow that performs a series of analyses on a set of labeledimages.

    Args:
        config: Configuration for the analysis workflow.
    """

    def __init__(self, config: AnalysisFlowConfig):
        super().__init__(config)
    
    def _run(self, labels: list[np.ndarray], metadata: Optional[list[dict[str, Any]]] = None) -> np.ndarray:
        """Run the analysis workflow on a set of labeled images. The labeled images are expected to have the same shape.

        Args:
            labels: Set of labeled images.
            metadata: Metadata for the labeled images.
        """
        if len(labels) == 0:
            raise ValueError("No labels provided")
        if any(l.shape != labels[0].shape for l in labels):
            raise ValueError("All labels must have the same shape")
        if metadata is not None and len(metadata) != len(labels):
            raise ValueError("Number of metadata sets must match number of labeled image stacks")
        results = {}
        if self.config.coincidence_detector is not None:
            # do all pairwise combinations of labels
            label_index_combinations = list(itertools.combinations(range(len(labels)), 2))
            l1 = [labels[c[0]] for c in label_index_combinations]
            l2 = [labels[c[1]] for c in label_index_combinations]
            if metadata is not None:
                try:
                    sn = [(metadata[c[0]]["channel_names"][0], metadata[c[1]]["channel_names"][0]) for c in label_index_combinations]
                except:
                    sn = [(f"stack_{c[0]}", f"stack_{c[1]}") for c in label_index_combinations]
            else:
                sn = [(f"stack_{c[0]}", f"stack_{c[1]}") for c in label_index_combinations]
            logger.info(f"Setting up coincidence detector for pairs: {sn}")
            # run asynchronously
            cd = Configurable.create_from_config(self.config.coincidence_detector)
            c_results = cd.run.map(l1, l2, stack_names=sn)
            for s, c_result in zip(sn, c_results):
                results[f"coincidence: {s[0]} vs {s[1]}"] = c_result

        if self.config.region_analyzer is not None:
            # update config to add "centroid" to properties and set output type to dataframe
            basecfg = self.config.region_analyzer
            properties = list(set(basecfg.properties + ["label", "object_id", "centroid"]))
            racfg = basecfg.model_copy(update={"properties": properties, "output_type": "dataframe"})
        else:
            # create new config, region analyzer output is needed for distance calculator
            racfg = RegionAnalyzerConfig(properties=["label", "object_id", "centroid"], output_type="dataframe")
        ra = RegionAnalyzer(racfg)
        r_results = ra.run.map(labels, metadata=metadata) # can't use run.submit because we need to use the dataframe
        for meta, r_result in zip(metadata, r_results):
            results[f"region_analyzer: {meta['channel_names'][0]}"] = r_result
        """
        if self.config.distance_calculator is not None:
            # update config to add "centroid" to properties and set output type to dataframe
            if self.config.distance_calculator is not None:
                dc = Configurable.create_from_config(self.config.distance_calculator)
                centroid_cols = [c for c in region_df.columns if c.startswith("centroid")]
                centroids = region_df[centroid_cols].values
                if "object_id" in region_df.columns:
                    object_ids = region_df["object_id"].values
                else:
                    object_ids = region_df.index.values
                results["distance_calculator"] = dc.run.submit(centroids, centroids, point_annotations=tuple(object_ids, object_ids))
        """
        return resolve_futures(results)