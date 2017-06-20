import copy
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x * np.sin(4 * x) + 1.1 * y * np.sin(2 * y)


class ContinuousGeneticAlgorithm:
    """
    This class represents simple example of continuous genetic algorithm, in which
    each chromosome an array of real numbers (floating-point numbers) as opposed to
    an array of just 0’s and 1’s.
    In this example using function of two variables, so the npar = 2.

    GA operators:
    Matting:
    Mutation:
    Selection:
    Stop condition: 100 generations
    """

    def __init__(self, func, mut_prob=0.3, pop_range=(), pop_size=12):
        self.func = func                        # Fitness function
        self.population = []                    # Population
        self.mutation_probability = mut_prob    # Mutation probability, 0.3 by default
        self.population_range = pop_range       # Function Definition Area or just range of finding :)
        self.npar = 2                           # Number of optimization variables
        self.population_size = pop_size         # Population size, 12 by default
        self.max_generations =100               # Count of generations

    def generate_start_population(self):
        """Size of population equal population_size.
        Let's create start population in range of population_range"""
        self.population = (self.population_range[1] - self.population_range[0]) * np.random.random_sample(
            (self.population_size, self.npar)) + self.population_range[0]

    def selection(self, population):
        """Selecting of population's members by rank, which will be a parents wis 'prob' probability"""
        res = np.argsort([self.func(*item) for item in population])
        new_pop = np
        new_pop = np.delete(population, res[len(res)//2:], 0)
        return new_pop

    def evolute(self):
        """Main method of evolution finding"""
        self.generate_start_population()
        keep = self.population_size//2
        prob = np.flipud(np.arange(1, keep + 1) / np.sum(np.arange(1, keep + 1)))
        odds = np.insert(np.cumsum(prob), 0, 0)
        print(print(self.population))
        for n in range(self.max_generations):
            population = copy.copy(self.population)
            new_pop = self.selection(population)
            M = (self.population_size - keep) // 2
            pick1 = np.random.rand(M)
            pick2 = np.random.rand(M)
            ma = []
            pa = []
            for ic in np.arange(M):
                for id in np.arange(1, keep + 1):
                    if pick1[ic] <= odds[id] and pick1[ic] > odds[id - 1]:
                        ma.append(id - 1)
                    if pick2[ic] <= odds[id] and pick2[ic] > odds[id - 1]:
                        pa.append(id - 1)
            print(ma, pa)



g=ContinuousGeneticAlgorithm(f, pop_range=(0,10))
res = g.evolute()