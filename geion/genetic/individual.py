import multiprocessing

from geion.genetic.genome import *
from sklearn.model_selection import cross_validate, cross_val_score


class Individual(Genome):

    def __init__(self, model: Any, genome: Union[Genome, List], **model_kwargs):
        self._model_class = model
        self._model = self._model_class(**model_kwargs)
        self.__model_kwargs = model_kwargs

        if self._model is None:
            raise TypeError('model can not be None')

        self._genome = genome
        if self._genome is None:
            raise TypeError('genome can not be None')
        self._score = None

        super().__init__(init_genome=self._genome)

    def re_init(self) -> "Individual":
        self._model = self.get_model_class()(**self.get_model_kwargs())
        return self

    def get_model_class(self) -> Any:
        return self._model_class

    def get_model_kwargs(self):
        return self.__model_kwargs

    @property
    def model(self) -> Any:
        return self._model

    def get_model(self) -> Any:
        return self.model

    @property
    def genome(self) -> Genome:
        return self._genome

    def get_genome(self) -> Genome:
        return self.genome

    @property
    def fitness(self) -> Union[int, float]:
        return self._genome.fitness

    def get_fitness(self) -> Union[int, float]:
        return self.fitness

    def fit(self, x_train: Any, y_train: Any) -> None:
        x_train = clip_to_genome(x_train, self.genome).to_numpy()
        y_train = y_train.to_numpy().reshape(-1, )

        self.model.fit(x_train, y_train)

    def cross_val_fit(self, x_train: Any, y_train: Any) -> None:
        x_train = clip_to_genome(x_train, self.genome).to_numpy()
        y_train = y_train.to_numpy().reshape(-1, )

        cross_validate(self.model, x_train, y_train, n_jobs=multiprocessing.cpu_count())

    def cross_val_score(self, x_test: Any, y_test: Any) -> float:
        # x_test = clip_to_genome(x_test, self.genome).to_numpy()
        # y_test = y_test.to_numpy().reshape(-1, )

        score = cross_val_score(self.model, x_test, y_test, n_jobs=multiprocessing.cpu_count()).mean()
        self.genome.set_fitness(score)
        self.set_fitness(score)

        return self.genome.fitness

    def average_score(self, x_test: Any, y_test: Any, iterations: int=5):

        average_score = []
        for _ in range(iterations):
            average_score.append(self.cross_val_score(x_test, y_test))

        average_score = sum(average_score) / iterations

        self.genome.set_fitness(average_score)
        self.set_fitness(average_score)

        return average_score

    def score(self, x_test: Any, y_test: Any) -> float:
        # x_test = clip_to_genome(x_test, self.genome).to_numpy()
        # y_test = y_test.to_numpy().reshape(-1, )

        score = self.model.score(x_test, y_test)
        self.genome.set_fitness(score)
        self.set_fitness(score)

        return self.genome.fitness

    def crossover(self, parent2: "Genome"):
        return Individual(self.get_model_class(), Genome(init_genome=super().crossover(parent2)), **self.get_model_kwargs())

    def mutate(self):
        return Individual(self.get_model_class(), Genome(init_genome=super().mutate()), **self.get_model_kwargs())

    def __call__(self, *args):
        self.fit(args[0], args[1])
