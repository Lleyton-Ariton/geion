import random

from typing import *
from functools import total_ordering

import pandas as pd


@total_ordering
class Genome:

    def __init__(self, data: pd.DataFrame=None, init_genome: Union[List, "Genome"]=None):
        self._genome = init_genome
        self._fitness = None
        self._mapping_matrix = {}

        if self._genome is not None:
            if isinstance(self._genome, Genome):
                self._mapping_matrix = self._genome.get_mapping_matrix()
                self.chromosome = self._genome.chromosome
            elif isinstance(self._genome, list):
                self.chromosome = self._genome

        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError('data passed must be pandas DataFrame')
            self._mapping_matrix, self.chromosome = {}, []
            for index, column_name in enumerate(data.columns):
                self._mapping_matrix.update({index: column_name})
                self.chromosome.append(random.randrange(0, 2))

    def set_fitness(self, fitness: Union[int, float]) -> None:
        if isinstance(fitness, int) or isinstance(fitness, float):
            self._fitness = fitness
            return None
        raise TypeError('fitness must be of type int or float')

    def reset_fitness(self) -> None:
        self.set_fitness(0.0)
        return None

    @property
    def fitness(self) -> Union[int, float]:
        return self._fitness

    def get_mapping_matrix(self) -> Dict:
        return self._mapping_matrix

    def set_all(self, state: bool):
        self.chromosome = [int(state) for _ in range(len(self.chromosome))]

    def is_all(self) -> bool:
        return any(self.chromosome)

    def crossover(self, parent2: "Genome") -> "Genome":  # BUGGED
        if len(self.chromosome) != len(parent2.chromosome):
            raise ValueError('Genome lengths of parents mismatch')

        selected_index = random.randrange(0, len(self.chromosome))

        if random.random() >= 0.5:
            self.chromosome = self.chromosome[selected_index:] + parent2.chromosome[:selected_index]
        else:
            self.chromosome = parent2.chromosome[selected_index:] + self.chromosome[:selected_index]

        if not self.is_all():
            return self.mutate()
        return self

    def mutate(self) -> "Genome":
        random_index = random.randrange(0, len(self.chromosome))
        if self.chromosome[random_index]:
            self.chromosome[random_index] = 0
            return self
        self.chromosome[random_index] = 1
        return self

    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        if isinstance(item, str):
            return list(self.get_mapping_matrix().values()).index(item)
        if isinstance(item, int):
            return self.get_mapping_matrix()[item]
        raise TypeError('item must be of type str or int')

    def __setitem__(self, key, value) -> None:
        if isinstance(key, str):
            self.chromosome[self[key]] = value
            return None
        if isinstance(key, int):
            self.chromosome[key] = value
            return None
        raise TypeError('key must be of type str or int')

    def __lt__(self, other: "Genome"):
        return self.fitness < other.fitness

    def __len__(self) -> int:
        return len(self.chromosome)

    def __str__(self) -> str:
        return 'Genome(Chromosome({}), Fitness({}))'.format(self.chromosome, self.fitness)


def clip_to_genome(data: pd.DataFrame, genome: Genome) -> pd.DataFrame:
    for column, is_active in zip(data.columns, genome.chromosome):
        if not is_active:
            data = data.drop([column], axis=1)
    return data
