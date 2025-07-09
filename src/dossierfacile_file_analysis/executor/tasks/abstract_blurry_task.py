from abc import ABC, abstractmethod

from dossierfacile_file_analysis.models.blurry_execution_context import BlurryExecutionContext


class AbstractBlurryTask(ABC):
    """
    Abstract class for blurry tasks.
    This class is used to define the interface for blurry tasks.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name

    def prepare_task_data(self, context: BlurryExecutionContext):
        pass

    def has_to_apply(self, context: BlurryExecutionContext) -> bool:
        return True

    def run(self, context: BlurryExecutionContext):
        """
        Run the task with the given context.
        This method should be implemented by subclasses to define the task's execution logic.
        """
        self.prepare_task_data(context)
        print(f"Running task: {self.task_name} with context: {context.execution_id}")
        self._internal_run(context)

    @abstractmethod
    def _internal_run(self, context: BlurryExecutionContext):
        pass