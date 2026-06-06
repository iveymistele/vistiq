import logging

from vistiq.core import Configuration, Configurable

logger = logging.getLogger(__name__)

class GraphConfig(Configuration):
    """Configuration for the graph.

    Args:
        config: Configuration for the graph.
    """

    pass

class Graph(Configurable):
    """Graph that represents a collection of points and edges.

    Args:
        config: Configuration for the graph.
    """

    def __init__(self, config: GraphConfig):
        super().__init__(config)