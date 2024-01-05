from typing import List, Callable, Iterator, Literal, Dict, Any, NamedTuple
import ray
from collections import namedtuple
from abc import ABC, abstractmethod


class TaskSpec(NamedTuple):
    id: str
    args: List[Any]
    kwargs: Dict[str, Any]


class TaskResults(NamedTuple):
    id: str
    results: Dict[str, Any]


class SpecGenerator(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[TaskSpec]:
        pass

    def __next__(self) -> TaskSpec:
        pass


class ParallelExecutor:
    def __init__(self):
        if not ray.is_initialized():
            ray.init()
        self.gpu_fraction = 0.5
        self.mode = "debug"

    def __call__(
        self, gpu_fraction: float = 0.5, mode: Literal["debug", "release"] = "debug"
    ):
        self.gpu_fraction = gpu_fraction
        self.mode = mode
        return self

    def run(
        self,
        func: Callable,
        task_specs: List[TaskSpec] = None,
        spec_generator: SpecGenerator = None,
    ) -> List[TaskResults]:
        if task_specs is None:
            assert spec_generator, "Must provide either spec_generator or task_specs"
            results = self._run_with_generator(func, spec_generator)
        else:
            results = self._run_with_task_specs(func, task_specs)
        return results

    def _run_with_task_specs(
        self, func: Callable, task_specs: List[TaskSpec]
    ) -> List[TaskResults]:
        remote_func = ray.remote(num_gpus=self.gpu_fraction)(func)
        object_refs = {
            task_spec.id: remote_func.remote(*task_spec.args, **task_spec.kwargs)
            for task_spec in task_specs
        }
        try:
            results = [
                TaskResults(id=task_id, results=ray.get(ref))
                for task_id, ref in object_refs.items()
            ]
        except KeyboardInterrupt as e:
            print("Caught keyboard interrupt. Terminating workers.")
            ray.cancel(object_refs, force=True)
            raise e
        except Exception as e:
            print("Caught exception: {}. Terminating workers.".format(e))
            if self.mode == "debug":
                ray.cancel(object_refs, force=True)
            raise e
        print("Finished running tasks")
        return results

    def _run_with_generator(
        self, func: Callable, spec_generator: SpecGenerator
    ) -> List[TaskResults]:
        task_specs = []
        for task_spec in spec_generator:
            task_specs.append(task_spec)
        return self._run_with_task_specs(func, task_specs)
