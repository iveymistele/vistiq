from vistiq.processors.pipeline import ProcessorPipeline
from vistiq.processors.segmentation.label_remover import LabelRemoverProcessor
from vistiq.processors.segmentation.region_analyzer import RegionAnalyzerProcessor

from vistiq.analysis import RegionAnalyzerConfig


def build_segmentation_workflow(
    region_analyzer_config,
    label_remover_config,
    debug: bool = False,
):
    return ProcessorPipeline(
        processors=[
            RegionAnalyzerProcessor(region_analyzer_config),
            LabelRemoverProcessor(label_remover_config),
        ],
        debug=debug,
    )