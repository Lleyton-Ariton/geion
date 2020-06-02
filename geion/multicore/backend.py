import os
import sys
import warnings

import ray
import multiprocessing
from alive_progress.progress import alive_bar

from typing import *
from geion.genetic.individual import Individual


class RayManager:

    def __init__(self, **ray_init_params):
        self.init_params = ray_init_params

    @staticmethod
    def shutdown():
        ray.shutdown()

    @staticmethod
    def init(**ray_init_params):
        if not ray.is_initialized():
            ray.init(**ray_init_params)

    def __enter__(self):
        if not ray.is_initialized():
            ray.init(**self.init_params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ray.shutdown()


@ray.remote
def function_wrapper(func: Callable, *args) -> Callable:
    return func(*args)


@ray.remote
class IndividualWrapper:

    def __init__(self, individual: Individual):
        self._individual = individual

    def compute(self, x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> Individual:
        self._individual.fit(x_train, y_train)
        self._individual.cross_val_score(x_test, y_test)
        return self._individual


def wrap_population(population: List[Individual]) -> List[ray.ObjectID]:
    wrapped_population = [IndividualWrapper.remote(individual) for individual in population]
    return wrapped_population


def run_population(wrapped_population: List, x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> List[Individual]:
    trained_population = [individual.compute.remote(x_train, y_train, x_test, y_test) for individual in wrapped_population]
    return ray.get(trained_population)


def unpin_objects(*objects) -> None:
    for object_id in objects:
        del object_id
    return None


def kill_population_actors(population_handles: List) -> None:
    [ray.kill(handle) for handle in population_handles]
    return None


def partitioned_run(wrapped_population: List, x_train: Any, y_train: Any, x_test: Any, y_test: Any,
                    kill: bool=False, print_progress_bar: bool=True) -> List[Individual]:

    def partition(population: List, cpus: int) -> List:

        for i in range(0, len(population), cpus):
            yield population[i:i + cpus]

    total_population = []

    if multiprocessing.cpu_count() >= 2:
        num_groups = multiprocessing.cpu_count() - 1
        partitions = partition(wrapped_population, num_groups)
    else:
        num_groups = 1
        partitions = partition(wrapped_population, 1)

    with alive_bar(num_groups) as bar:
        for partition in partitions:
            total_population.extend(run_population(partition, x_train, y_train, x_test, y_test))
            if kill:
                kill_population_actors(partition)
            if print_progress_bar:
                bar()

    unpin_objects([x_train, x_test, y_train, y_test])

    return total_population


def silence_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
