import sys
import warnings

import numpy as np

from geion.genetic.individual import *
from geion.multicore import *


class FeatureOptimizer:

    def __init__(self, population_size: int, target: Union[int, float], model: Any, x_train: pd.DataFrame,
                 y_train: pd.DataFrame, x_test: Any, y_test: Any, generation_limit: int=25, mutation_rate: float=None,
                 elitist_slice: float=0.1, periodic_saves: bool=False, keep_history: bool=False,
                 print_metrics: bool=True, seed: int=False, print_warnings: bool=False, print_progress_bar: bool=True,
                 **ray_init_params):

        self._population_size = population_size
        self._target = target
        self._model = model

        if isinstance(self._model, Individual):
            self._model_kwargs = self._model.get_model_kwargs()

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.generation_limit = generation_limit

        self._mutation_rate = mutation_rate
        if self._mutation_rate is None:
            self._mutation_rate = 1/self._population_size

        self._elitist_slice = elitist_slice
        self.periodic_saves = periodic_saves

        self.keep_history = keep_history
        if self.keep_history:
            self._history = []

        self.print_metrics = print_metrics
        self.print_warnings = print_warnings
        self.print_progress_bar = print_progress_bar

        self.ray_init_params = ray_init_params

        self._seed = seed
        if self._seed:
            random.seed(self._seed)

        self._population = []

    def init_population(self) -> None:
        for _ in range(self.population_size):
            self._population.append(Individual(self.model.get_model_class(), Genome(self.x_train), **self.model.get_model_kwargs()))
        return None

    def kill_population(self) -> None:
        self._population = []
        return None

    def init_from_genomes(self, genomes: [List[List], List[Genome]]) -> None:
        if len(self.get_population()) == 0:
            for genome in genomes:
                self._population.append(Individual(self.model.get_model_class(),
                                                   genome=Genome(init_genome=genome),
                                                   **self.model.get_model_kwargs()))
            return None
        raise ValueError('The population is not empty.'
                         ' Wipe the current population to initiate a new one.')

    def set_population_size(self, new_size: int) -> None:
        if isinstance(new_size, int):
            self._population_size = new_size
            return None
        raise TypeError('new_size must be of type int')

    @property
    def population_size(self) -> int:
        return self._population_size

    def set_target(self, new_target: [int, float]) -> None:
        if isinstance(int, new_target) or isinstance(float, new_target):
            self._target = new_target
            return None
        raise TypeError('new_target must be of type int or float')

    @property
    def target(self) -> Union[int, float]:
        return self._target

    def set_model(self, model: Individual) -> None:
        if isinstance(model, Individual):
            self._model = model
            return None
        raise TypeError('model must be of type Individual')

    @property
    def model(self) -> Individual:
        return self._model

    def set_generation_limit(self, generation_limit: int) -> None:
        if isinstance(generation_limit, int):
            self.generation_limit = generation_limit
            return None
        raise TypeError('generation_limit must be of type int')

    def set_mutation_rate(self, mutation_rate: float) -> None:
        if isinstance(mutation_rate, float):
            self._mutation_rate = mutation_rate
            return None
        raise TypeError('mutation_rate must be of type float')

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    def set_elitist_slice(self, elitist_slice: float) -> None:
        if isinstance(elitist_slice, float):
            self._elitist_slice = elitist_slice
            return None
        raise TypeError('elitist_slice must be of type float')

    @property
    def elitist_slice(self) -> float:
        return self._elitist_slice

    def set_periodic_save(self, save_every: int) -> None:
        if isinstance(save_every, int):
            self.periodic_saves = save_every
            return None
        raise TypeError('save_every must be of type int')

    def get_history(self) -> List[float]:
        return self._history

    def set_seed(self, seed: int):
        self._seed = seed
        random.seed(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def __set_population(self, new_population: List[Genome]) -> None:
        if isinstance(new_population, list):
            self._population = new_population
            return None
        raise TypeError('new_population must be a list of Genomes')

    def get_population(self) -> List[Individual]:
        return self._population

    def __deprecated_parallel_training(self) -> List:
        warnings.warn('This parallel training method is not supposed to be used. '
                      'Use parallel_training method instead')

        @ray.remote
        def _parallel_fit_wrapper(individual: Individual, _x_train: Any, _y_train: Any, _x_test: Any, _y_test: Any):
            individual.fit(_x_train, _y_train)
            individual.cross_val_score(_x_test, _y_test)
            return individual

        population = []

        x_train_id, y_train_id = ray.put(self.x_train), ray.put(self.y_train)
        x_test_id, y_test_id = ray.put(self.x_test), ray.put(self.y_test)

        [population.append(_parallel_fit_wrapper.remote(individual, x_train_id, y_train_id, x_test_id, y_test_id)) for individual in self.get_population()]

        trained_population = [ray.get(individual) for individual in population]
        return trained_population

    def parallel_training(self) -> List:
        wrapped_population = wrap_population(self.get_population())

        x_train_id, y_train_id = ray.put(self.x_train), ray.put(self.y_train)
        x_test_id, y_test_id = ray.put(self.x_test), ray.put(self.y_test)

        trained_population = partitioned_run(wrapped_population, x_train_id, y_train_id,
                                             x_test_id, y_test_id, print_progress_bar=self.print_progress_bar)
        unpin_objects(x_train_id, y_train_id, x_test_id, y_test_id)

        return trained_population

    def run_optimization(self, parallel: bool=False) -> Individual:
        if not self.print_warnings:
            silence_warnings()

        generation = 0
        target_reached = False
        self.init_population()

        while not target_reached and generation <= self.generation_limit:

            if parallel:
                with RayManager(**self.ray_init_params):
                    population = self.parallel_training()

            else:
                population = self.get_population()
                for individual in population:
                    individual.fit(self.x_train, self.y_train)
                    individual.cross_val_score(self.x_test, self.y_test)

            population.sort(reverse=True)

            self.__set_population(population)
            self._history.append(population[0])

            if population[0].fitness >= self.target:
                return population[0]

            new_generation = []

            elite = int(np.ceil(self.elitist_slice * self.population_size))
            new_generation.extend(population[:elite])

            remaining = int(np.ceil((1 - self.elitist_slice) * self.population_size))
            for _ in range(remaining):
                parent1 = random.choice(population[:(len(population) // 2)])
                parent2 = random.choice(population[:(len(population) // 2)])
                child = parent1.crossover(parent2)

                if random.random() <= self.mutation_rate:
                    child = child.mutate()

                new_generation.append(child)

            self.__set_population(new_generation)

            generation += 1

            if self.print_metrics:
                sys.stdout.write(f'Generation: {generation}\t'
                                 f'Fitness: {self.get_population()[0].fitness}\t\n'
                                 f'Genome: {self.get_population()[0].chromosome}\n')

        return self.get_population()[0]
