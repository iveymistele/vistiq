"""Tests for coincidence / overlap utilities."""

import numpy as np
import pytest

from vistiq.analysis.coincidence import labels_iou_batch_3d


def test_labels_iou_pruned_matches_dense():
    """Bbox-pruned path should match dense mask batch on small 3D volumes."""
    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    labels[4:7, 10:14, 10:14] = 2

    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1
    other[5:8, 11:15, 11:15] = 2

    dense = labels_iou_batch_3d(labels, other, prune_bboxes=False)
    pruned = labels_iou_batch_3d(
        labels, other, prune_bboxes=True, dense_pair_fraction=1.0
    )
    np.testing.assert_allclose(dense, pruned, rtol=1e-5, atol=1e-5)


def test_labels_iou_bbox_prune_skips_disjoint():
    """Disjoint regions should yield zero overlap without relying on dense masks."""
    labels = np.zeros((4, 8, 8), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1

    other = np.zeros_like(labels)
    other[2:4, 6:8, 6:8] = 1

    out = labels_iou_batch_3d(labels, other, prune_bboxes=True)
    assert out.shape == (1, 1)
    assert out[0, 0] == 0.0


def test_labels_iou_torch_matches_cpu():
    pytest.importorskip("torch")
    from vistiq.analysis.coincidence import labels_iou_batch_3d_torch

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    labels[4:7, 10:14, 10:14] = 2

    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1
    other[5:8, 11:15, 11:15] = 2

    cpu = labels_iou_batch_3d(labels, other, prune_bboxes=False)
    torch_out = labels_iou_batch_3d_torch(
        labels, other, prune_bboxes=False, device="cpu"
    )
    np.testing.assert_allclose(cpu, torch_out, rtol=1e-5, atol=1e-5)


def test_coincidence_detector_process_slice_outline_iou():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method="iou",
            mode="outline",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, ("Lobe", "Cell"))
    assert len(results) == 1
    assert results[0]["Lobe"] == 1
    assert results[0]["Cell"] == 1
    assert results[0]["score"] > 0.0
    assert results[0]["above_threshold"] is True


def test_coincidence_detector_accepts_list_stack_names():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method="iou",
            mode="outline",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, (["Scrib"], ["EdU"]))
    assert len(results) == 1
    assert results[0]["Scrib"] == 1
    assert results[0]["EdU"] == 1


def test_coincidence_detector_process_slice_bounding_box():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method="iou",
            mode="bounding_box",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, ("A", "B"))
    assert len(results) == 1
    assert results[0]["score"] > 0.0


def test_coincidence_detector_process_slice_outline_dice():
    from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
    from vistiq.utils import ArrayIteratorConfig

    labels = np.zeros((8, 16, 16), dtype=np.int32)
    labels[1:4, 2:6, 2:6] = 1
    other = np.zeros_like(labels)
    other[2:5, 3:7, 3:7] = 1

    det = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method="dice",
            mode="outline",
            iterator_config=ArrayIteratorConfig(slice_def=()),
        )
    )
    results = det._process_slice(labels, other, ("Lobe", "Cell"))
    assert len(results) == 1
    assert results[0]["score"] > 0.0
