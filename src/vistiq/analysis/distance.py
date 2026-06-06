import logging
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from prefect import task
from vistiq.utils import _torch_imports
from vistiq.core import Configuration, Configurable, generate_name

logger = logging.getLogger(__name__)

class MatrixAnalyzerConfig(Configuration):
    """Configuration for the matrix analyzer.

    Args:
        config: Configuration for the matrix analyzer.
    """

    annotate: bool = True
    output_type: Literal["array", "dataframe"] = "dataframe"

class MatrixAnalyzer(Configurable):
    """Analyzer that computes the distance between two collections of points.

    Args:
        config: Configuration for the matrix analyzer.
    """

    def __init__(self, config: MatrixAnalyzerConfig):
        super().__init__(config)

    def _calculate(self, points1: np.ndarray, points2: np.ndarray, spacing: Optional[Union[dict[str, float], tuple[float, ...]]] = None) -> np.ndarray:
        """Compute the distance between two collections of points.

        Args:
            points1: First collection of points.
            points2: Second collection of points.
            spacing: Spacing for the points.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @task(name="MatrixAnalyzer.run", task_run_name=generate_name)
    def run(self, points1: np.ndarray, points2: np.ndarray, spacing: Optional[Union[dict[str, float], tuple[float, ...]]] = None, point_annotations: Optional[tuple[tuple[str,str]]] = None) -> np.ndarray:
        """Perform matrix calculation on two collections of points.

        Args:
            points1: First collection of points.
            points2: Second collection of points.
            spacing: Spacing for the points.
            point_annotations: Annotations for the points.
        """
        raw_results = self._calculate(points1, points2, spacing=spacing, point_annotations=point_annotations)
        results = self._format(raw_results, point_annotations=point_annotations)
        return results

    def _format(self, results: np.ndarray, point_annotations: Optional[tuple[tuple[str,str]]] = None) -> np.ndarray:
        """Annotate the results with the distance between the two collections of points.

        Args:
            results: Results from the distance calculation.
        """
        if self.config.output_type == "array":
            # keep as is
            return results
        elif self.config.output_type == "dataframe":
            if self.config.annotate:
                if point_annotations is None:
                    raise ValueError("point_annotations must be provided when annotate is True")
                if len(point_annotations[0]) != results.shape[0] or len(point_annotations[1]) != results.shape[1]:
                    raise ValueError("point_annotations must have the same number of rows and columns as the results")
                col_annotations = [f"{p2}" for _, p2 in point_annotations]
                index_annotations = [f"{p1}" for p1, _ in point_annotations]
                return pd.DataFrame(results, columns=col_annotations, index=index_annotations)
            else:
                return pd.DataFrame(results)
        else:
            raise ValueError(f"Invalid output type: {self.config.output_type}")

class DistanceAnalyzerConfig(MatrixAnalyzerConfig):
    """Configuration for the distance analyzer.

    Args:
        config: Configuration for the distance analyzer.
    """

    method: Literal["euclidean", "manhattan", "chebyshev", "minkowski"] = "euclidean"

class DistanceAnalyzer(MatrixAnalyzer):
    """Analyzer that computes the distance between two labeled imagestacks.

    Args:
        config: Configuration for the distance analyzer.
    """

    def _calculate(self, points1: np.ndarray, points2: np.ndarray, spacing: Optional[Union[dict[str, float], tuple[float, ...]]] = None) -> np.ndarray:
        """Compute the distance between two collections of points. Use PyTorch for GPU acceleration.

        Args:
            points1: First collection of points.
            points2: Second collection of points.
            metadata: Metadata for the image stacks.
        """
        torch = _torch_imports()
        if spacing is not None and len(spacing) != points1.shape[1] != points2.shape[1]:
            raise ValueError(f"Spacing must have the same number of dimensions as points1, got {len(spacing)} and {points1.shape[1]}")
        logger.info(f"DistanceAnalyzer _run: points1.shape={points1.shape}, points2.shape={points2.shape}, spacing={spacing}")
        if spacing is not None:
            points1 = points1 * spacing
            points2 = points2 * spacing
        return torch.cdist(points1, points2, p=self.config.method)