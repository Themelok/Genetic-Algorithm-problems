import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

    def __init__(self, func, mut_prob=0.2, pop_range=(), pop_size=12):
        self.func = func                        # Fitness function
        self.population = []                    # Population
        self.mutation_probability = mut_prob    # Mutation probability, 0.3 by default
        self.population_range = pop_range       # Function Definition Area or just range of finding :)
        self.npar = 2                           # Number of optimization variables
        self.population_size = pop_size         # Population size, 12 by default
        self.max_generations =100               # Count of generations
        self.keep = self.population_size//2     # Count of parents
        self.prob = np.flipud(np.arange(1, self.keep + 1) / np.sum(np.arange(1, self.keep + 1)))
        self.odds = np.insert(np.cumsum(self.prob), 0, 0)

    def generate_start_population(self):
        """Size of population equal population_size.
        Let's create start population in range of population_range"""
        self.population = (self.population_range[1] - self.population_range[0]) * np.random.random_sample(
            (self.population_size, self.npar)) + self.population_range[0]

    def selection(self, population):
        """Selecting of population's members by rank, which will be a parents wis 'prob' probability"""
        res = np.argsort([self.func(*item) for item in population])
        res = res[:self.keep]
        new_pop = np.array(self.population)[res]
        return new_pop

    def mating(self, parents):
        """Random selecting of pairs within parents.
        ma and pa contain the indices of the chromosomes that will mate
        """
        offsprings = []
        M = (self.population_size - self.keep) // 2
        pick1 = np.random.rand(M)
        pick2 = np.random.rand(M)
        ma = []
        pa = []
        for ic in np.arange(M):
            for id in np.arange(1, self.keep + 1):
                if pick1[ic] <= self.odds[id] and pick1[ic] > self.odds[id - 1]:
                    ma.append(id - 1)
                if pick2[ic] <= self.odds[id] and pick2[ic] > self.odds[id - 1]:
                    pa.append(id - 1)
        for i in range(M):
            """Here is crossover. crosover_point is number of gen in i's chromosome(parent)"""
            b=np.random.rand()
            crossover_point = np.random.randint(0,self.npar)
            mam = copy.copy(parents[ma[i]])
            pap = copy.copy(parents[pa[i]])
            mam_xy = mam[crossover_point]
            pap_xy = pap[crossover_point]
            mam[crossover_point]=(1-b)*mam_xy+b*pap_xy
            offsprings.append(mam.tolist())
            pap[crossover_point]=(1-b)*pap_xy+b*mam_xy
            offsprings.append(pap.tolist())
        return offsprings
    def mutation(self, chromosome):
        """Implements mutation of chromosomes with probability self.mutation_probability"""
        r = np.random.rand()
        if r <=self.mutation_probability:
            mutation_gen = np.random.randint(0,self.npar)
            chromosome[mutation_gen]= (self.population_range[1] - self.population_range[0])*np.random.rand()\
                                      +self.population_range[0]
        return chromosome

    def evolute(self):
        """Main method of evolution finding"""
        self.generate_start_population()
        """This is just animation block"""
        delta = 0.1
        x = np.arange(0, 10.0, delta)
        y = np.arange(0, 10, delta)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, corner_mask=True)
        ims = []
        xs=self.population[:, 0]
        ys=self.population[:, 1]
        plt.plot(xs.tolist(), ys.tolist(), 'bo', ms=15)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.colorbar()
        fig.set_size_inches(15, 10)
        """This is just animation block"""


        print(print(self.population))
        for n in range(self.max_generations):
            population = copy.copy(self.population)
            parents = self.selection(population)
            offsprings = self.mating(parents)
            new_populaion=np.concatenate((parents,offsprings))
            mutated_new_population = np.array([self.mutation(chromosome) for chromosome in new_populaion])
            print("new pop: ",mutated_new_population)
            self.population=mutated_new_population
            xs = self.population[:, 0]
            ys = self.population[:, 1]
            im = plt.plot(xs.tolist(), ys.tolist(), 'ro', ms=20)
            ims.append(im)
        ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True,
                                        repeat_delay=1000, repeat=False)

        plt.show()




g=ContinuousGeneticAlgorithm(f, pop_range=(0,10))
res = g.evolute()