from typing import Optional, Any
import logging

from vistiq.core import (
    Configurable,
    Configuration,
    StackProcessor,
    StackProcessorConfig,
    generate_flow_name,
    generate_name,
)
from vistiq.utils import resolve_futures
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

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

    @flow(name="Workflow.mapped_run", flow_run_name=generate_flow_name)
    def mapped_run(self, *args, resolve: bool = False, **kwargs) -> Any:
        """Run the workflow component on a list of images.

        Args:
            *args: list of inputs to be processed.
            resolve: If ``True``, block and return resolved mapped results.
                If ``False`` (default), return Prefect futures for chaining into
                subsequent mapped calls without introducing a barrier.
            **kwargs: list of additional keyword arguments to pass to the workflow.
        """
        futures = self._run.map(*args, **kwargs)
        if not resolve:
            return futures

        results = resolve_futures(futures)
        if not results:
            return []
        if isinstance(results[0], tuple):
            return tuple(list(items) for items in zip(*results))
        return results


    @flow(name="Workflow.run", flow_run_name=generate_flow_name)
    def run(self, *args, **kwargs) -> Any:
        """Run the workflow component on an image.

        Args:
            *args: Inputs to be processed.
            **kwargs: Additional keyword arguments to pass to the workflow.
        """
        return self._run(*args, **kwargs)

    @task(name="Workflow._run", task_run_name=generate_name)
    def _run(
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
