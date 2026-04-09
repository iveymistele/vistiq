from __future__ import annotations

from functools import wraps
import inspect
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import field_validator, model_validator
from prefect import task
from skimage.measure import label as sk_label
from skimage.measure import regionprops, regionprops_table

from vistiq.core import StackProcessor, StackProcessorConfig
from vistiq.processors.base import BaseProcessor
from vistiq.processors.types import WorkflowData

logger = logging.getLogger(__name__)


class RegionAnalyzer(StackProcessor):
    """Analyzer that extracts region properties from labeled images."""

    default_properties: List[str] = ["label", "centroid"]

    def __init__(self, config: "RegionAnalyzerConfig"):
        super().__init__(config)

    @staticmethod
    def builtin_properties() -> List[str]:
        fake_array = np.ones((2, 2))
        labels = sk_label(fake_array)
        regions = regionprops(labels)
        return sorted([attr for attr in dir(regions[0]) if not attr.startswith("_")])

    @classmethod
    def extra_properties_funcs(cls) -> Dict[str, Callable]:
        return {
            "circularity": cls.circularity,
            "sphericity": cls.sphericity,
            "aspect_ratio": cls.aspect_ratio,
            "cross_sectional_area": cls.cross_sectional_area,
            "volume": cls.volume,
        }

    @staticmethod
    def allowed_properties() -> List[str]:
        return sorted(
            RegionAnalyzer.builtin_properties()
            + list(RegionAnalyzer.extra_properties_funcs().keys())
        )

    def used_extra_properties(self) -> List[str]:
        return sorted(
            [
                prop
                for prop in self.config.properties
                if prop in RegionAnalyzer.extra_properties_funcs().keys()
            ]
        )

    def used_extra_properties_funcs(
        self,
        spacing: Optional[Tuple[float, ...]] = None,
    ) -> List[Callable]:
        uep = self.used_extra_properties()
        base_funcs = {
            k: func
            for k, func in RegionAnalyzer.extra_properties_funcs().items()
            if k in uep
        }

        wrapped_funcs = []
        for prop_name, func in base_funcs.items():
            if spacing is not None:
                sig = inspect.signature(func)
                if "spacing" in sig.parameters:

                    def make_wrapper(f, prop_n, sp):
                        @wraps(f)
                        def wrapper(regionmask, intensity_image=None):
                            return f(regionmask, intensity_image, spacing=sp)

                        wrapper.__name__ = prop_n
                        return wrapper

                    wrapped_funcs.append(make_wrapper(func, prop_name, spacing))
                else:
                    wrapped_funcs.append(func)
            else:
                wrapped_funcs.append(func)

        return wrapped_funcs

    def used_builtin_properties(self) -> List[str]:
        return [
            prop
            for prop in self.config.properties
            if prop in RegionAnalyzer.builtin_properties()
        ]

    @classmethod
    def from_config(cls, config: "RegionAnalyzerConfig") -> "RegionAnalyzer":
        return cls(config)

    @staticmethod
    def circularity(regionmask, intensity_image=None, spacing=None):
        from skimage.measure import perimeter

        perim = perimeter(regionmask)
        area = np.sum(regionmask)
        if perim > 0:
            return float(4.0 * np.pi * area / (perim**2))
        return float("nan")

    @staticmethod
    def sphericity(regionmask, intensity_image=None, spacing=None):
        volume = np.sum(regionmask)
        if volume == 0:
            return float("nan")

        try:
            try:
                from skimage.measure import marching_cubes
            except ImportError:
                try:
                    from skimage.measure import marching_cubes_lewiner as marching_cubes
                except ImportError:
                    return float("nan")

            if spacing is not None and len(spacing) == 3:
                verts, faces, normals, values = marching_cubes(regionmask, spacing=spacing)
            else:
                verts, faces, normals, values = marching_cubes(regionmask)

            if len(faces) == 0:
                return float("nan")

            face_areas = []
            for face in faces:
                v0, v1, v2 = verts[face]
                cross = np.cross(v1 - v0, v2 - v0)
                area = 0.5 * np.linalg.norm(cross)
                face_areas.append(area)

            surface_area = sum(face_areas)

            if surface_area > 0:
                sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
                return float(sphericity)
            return float("nan")
        except (ValueError, RuntimeError, ImportError):
            return float("nan")

    @staticmethod
    def aspect_ratio(regionmask, intensity_image=None, spacing=None):
        coords = np.where(regionmask)
        if len(coords[0]) == 0:
            return float("nan")

        ndim = regionmask.ndim
        coords_array = np.array([coords[i] for i in range(ndim)], dtype=np.float64)

        if spacing is not None and len(spacing) >= ndim:
            spacing_array = np.array(spacing[:ndim], dtype=np.float64)
            coords_array = coords_array * spacing_array[:, np.newaxis]

        centroid = np.mean(coords_array, axis=1)
        coords_centered = coords_array - centroid[:, np.newaxis]

        if coords_centered.shape[1] < ndim:
            return float("nan")

        cov = np.cov(coords_centered)
        eigenvalues = np.linalg.eigvals(cov)
        if len(eigenvalues) < ndim or np.any(eigenvalues <= 0):
            return float("nan")

        eigenvalues = np.sort(eigenvalues)[::-1]
        return float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))

    @staticmethod
    def cross_sectional_area(regionmask, intensity_image=None, spacing=None):
        pixel_count = float(np.max(np.sum(regionmask, axis=(-2, -1))))

        if spacing is not None:
            pixel_area = np.abs(np.prod(spacing[-2:]))
            return pixel_count * pixel_area

        return pixel_count

    @staticmethod
    def volume(regionmask, intensity_image=None, spacing=None):
        pixel_count = float(np.sum(regionmask))

        if spacing is not None:
            voxel_volume = np.abs(np.prod(spacing))
            return pixel_count * voxel_volume

        return pixel_count

    def _process_slice(
        self,
        labels: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if metadata is None or metadata.get("scale", None) is None:
            spacing = None
        else:
            spacing = metadata.get("scale", None)

        if spacing is not None:
            spacing = spacing[-labels.ndim :]

        extra_props_funcs = self.used_extra_properties_funcs(spacing=spacing)

        if self.config.output_type == "list":
            results = regionprops(
                labels,
                extra_properties=extra_props_funcs,
                spacing=spacing,
            )
        elif self.config.output_type == "dataframe":
            results = pd.DataFrame(
                regionprops_table(
                    labels,
                    properties=self.used_builtin_properties(),
                    extra_properties=extra_props_funcs,
                    spacing=spacing,
                )
            ).set_index("label")
        else:
            raise ValueError(
                f"Invalid output type: {self.config.output_type}. "
                "Allowed output types are: list, dataframe"
            )

        return results

    def _reshape_slice_results(
        self,
        results: list[Any],
        slice_indices: list[tuple[int, ...]],
        input_shape: tuple[int, ...],
    ):
        return super()._reshape_slice_results(
            results,
            slice_indices=slice_indices,
            input_shape=input_shape,
        )

    @task(name="RegionAnalyzer.run")
    def run(
        self,
        labels: np.ndarray,
        workers: int = -1,
        verbose: int = 10,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        return super().run(
            labels,
            workers=workers,
            verbose=verbose,
            metadata=metadata,
            **kwargs,
        )


class RegionAnalyzerConfig(StackProcessorConfig):
    output_type: Literal["list", "dataframe"] = "list"
    properties: List[str] = RegionAnalyzer.default_properties

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, v: List[str]) -> List[str]:
        if v is None or len(v) == 0:
            return RegionAnalyzer.default_properties
        elif not set(v).issubset(set(RegionAnalyzer.allowed_properties())):
            raise ValueError(
                f"One or more invalid properties: {v}. "
                f"Allowed properties are: {RegionAnalyzer.allowed_properties()}"
            )
        if "label" not in v:
            v = ["label"] + v
        return v

    @model_validator(mode="after")
    def validate_properties_iterator(self) -> "RegionAnalyzerConfig":
        if self.properties is None or len(self.properties) == 0:
            self.properties = RegionAnalyzer.default_properties

        has_area = "area" in self.properties
        has_volume = "volume" in self.properties
        slice_def_len = len(self.iterator_config.slice_def)

        if has_area and slice_def_len >= 3:
            self.properties = [p for p in self.properties if p != "area"] + ["volume"]
        if has_volume and slice_def_len < 3:
            self.properties = [p for p in self.properties if p != "volume"] + ["area"]

        has_circularity = "circularity" in self.properties
        has_sphericity = "sphericity" in self.properties

        if has_circularity and slice_def_len >= 3:
            self.properties = [p for p in self.properties if p != "circularity"] + ["sphericity"]
        if has_sphericity and slice_def_len < 3:
            self.properties = [p for p in self.properties if p != "sphericity"] + ["circularity"]

        return self


class RegionAnalyzerProcessor(BaseProcessor):
    name = "region_analyzer"

    def __init__(self, config: RegionAnalyzerConfig):
        self.analyzer = RegionAnalyzer.from_config(config)

    @property
    def input_keys(self) -> list[str]:
        return ["labels"]

    @property
    def output_keys(self) -> list[str]:
        return ["region_features"]

    def run(self, data: WorkflowData) -> WorkflowData:
        updated = dict(data)
        updated["region_features"] = self.analyzer.run(
            labels=data["labels"],
            metadata=data.get("metadata"),
        )
        return updated