import numpy as np

from vistiq.processors.segmentation.region_analyzer import RegionAnalyzerConfig
from vistiq.processors.segmentation.label_remover import LabelRemoverConfig
from vistiq.workflows.segmentation import build_segmentation_workflow


def test_segmentation_workflow_poc():
    labels = np.array([
        [0, 1, 1, 0],
        [2, 2, 0, 3],
        [0, 0, 3, 3],
    ], dtype=np.int32)

    workflow = build_segmentation_workflow(
        region_analyzer_config=RegionAnalyzerConfig(
            output_type="dataframe",
            properties=["label", "centroid"],
        ),
        label_remover_config=LabelRemoverConfig(),
        debug=True,   #  THIS is what I meant
    )

    result = workflow.run({
        "labels": labels,
        "labels_to_remove": [2],
        "metadata": {},
    })

    print("\nFINAL RESULT:")
    print(result.keys())
    print(result["labels"])
    print(result["region_features"])