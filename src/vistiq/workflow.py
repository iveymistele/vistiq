import numpy as np
from typing import List, Optional, Any
from pydantic import BaseModel
import logging

from vistiq.core import (
    Configurable,
    Configuration,
    StackProcessor,
    StackProcessorConfig,
)
from prefect import task, flow

logger = logging.getLogger(__name__)


class WorkflowConfig(Configuration):
    """Configuration for a complete workflow.

    Defines a sequence of workflow steps to be executed in order.

    Attributes:
        step_configs: List of workflow step configurations to execute.
    """

    # step_configs: List[Configuration]
    pass


class Workflow(Configurable):
    """Base class for workflow components.

    Provides common functionality for workflow steps and workflows,
    including configuration management and execution interface.

    Attributes:
        config: Configuration model instance.
    """

    def __init__(self, config: WorkflowConfig):
        """Initialize the base class.

        Args:
            config: Configuration model instance.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: StackProcessorConfig) -> "StackProcessor":
        """Create a StackProcessor instance from a configuration.

        Args:
            config: Stack processor configuration.

        Returns:
            A new StackProcessor instance.
        """
        return cls(config)

    #
    #    def name(self) -> str:
    #        """Get the name of this class.
    #
    #        Returns:
    #            The class name as a string.
    #        """
    #        return type(self).__name__

    #    def __str__(self) -> str:
    #        """String representation of the object.
    #
    #        Returns:
    #            A string describing the object and its configuration.
    #        """
    #        return f"{self.name()} with config: {self.config}"

    #    def __repr__(self) -> str:
    #        """Developer-friendly representation.
    #
    #        Returns:
    #            A string representation suitable for debugging.
    #        """
    #        return f"{self.name()}({self.config})"

    #    def get_config(self) -> BaseModel:
    #       """Get the current configuration.
    #
    #        Returns:
    #            The current configuration model instance.
    #        """
    #        return self.config

    #    def set_config(self, config: BaseModel):
    #        """Set a new configuration.

    #        Args:
    #            config: New configuration model instance.
    #        """
    #        self.config = config

    @flow(name="Workflow.run")
    def run(
        self,
        input: Any,
        *args,
        workers: int = -1,
        verbose: int = 1,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Run the workflow component on an image.

        Args:
            input: Input to be processed.
            workers: Number of parallel workers (-1 for all cores).
            verbose: Verbosity level for processing.
            metadata: Optional metadata dictionary describing the input.
            **kwargs: Additional keyword arguments.

        Returns:
            Processed image array.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method")
