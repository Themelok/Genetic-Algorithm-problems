import copy
import numpy as np
import matplotlib.pyplot as plt


def f(population, xs, ys):
    Npop, Ncity = population.shape
    back = population[:, 0]
    tour = np.insert(population, [Ncity], back.reshape(Npop, 1), axis=1)  # Замыкаем каждую хромосому на первом значении
    dcity = np.zeros((Ncity, Ncity))
    for ic in range(Ncity):
        for id in range(Ncity):
            dcity[ic, id] = np.sqrt((xs[ic] - xs[id]) ** 2 + (ys[ic] - ys[id]) ** 2)
    dist = np.zeros(Npop)
    for ic in range(Npop):
        for id in range(Ncity):
            dist[ic] = dist[ic] + dcity[tour[ic][id]][tour[ic][id + 1]]
    return dist


class TravelSalesmanGeneticAlgorithm:
    def __init__(self, func, mut_prob=0.2, pop_range=(), pop_size=20):
        self.func = func
        self.population = []
        self.mutation_probability = mut_prob
        self.population_range = pop_range
        self.npar = 20
        self.population_size = pop_size
        self.keep = self.population_size // 2
        self.max_generation = 10000
        self.city_xs = np.array([])
        self.city_ys = np.array([])
        self.odds = [0, ]

    def generate_odds(self, keep):
        odds = [0]
        for i in range(1, keep):
            odd = i * np.ones((1, i + 1), dtype=int)
            odd = odd.tolist()
            for o in odd:
                odds.extend(o)
        odds = keep - np.array(odds) - 1
        self.odds = odds

    def generate_cites(self):
        self.city_xs = np.random.random_sample((20,))
        self.city_ys = np.random.random_sample((20,))

    def generate_start_population(self):
        for i in np.arange(self.population_size):
            self.population.append(np.random.permutation(self.npar).tolist())
        self.population = np.array(self.population)

    def mating(self, parents):
        offsprings = copy.copy(parents)
        M = (self.population_size - self.keep) // 2
        pick1 = np.random.randint(0, len(self.odds), M)
        pick2 = np.random.randint(0, len(self.odds), M)
        ma = self.odds[pick1]
        pa = self.odds[pick2]
        for ic in range(M):
            mate1 = parents[ma[ic]]
            mate2 = parents[pa[ic]]
            indx = 2 * ic
            offsprings[indx] = mate1
            offsprings[indx + 1] = mate2
        return offsprings

    def selection(self, population):
        res = np.argsort(self.func(xs=self.city_xs,
                                   ys=self.city_ys,
                                   population=population))
        res = res[:self.keep]
        new_pop = np.array(self.population)[res]
        return new_pop

    def mutation(self, population):

        nmu = round(self.population_size * self.npar * self.mutation_probability)
        for ic in range(nmu):
            if np.random.rand() < self.mutation_probability:
                row1 = round(np.random.rand() * (self.population_size - 1))
                col1 = round(np.random.rand() * self.npar) - 1
                col2 = round(np.random.rand() * self.npar) - 1
                population[row1, col1], population[row1, col2] = population[row1, col2], population[row1, col1]
        return population

    def evolute(self):
        """Main method of evolution finding"""

        self.generate_cites()
        self.generate_start_population()
        self.generate_odds(self.keep)
        print(self.population)
        for n in range(self.max_generation):
            population = copy.copy(self.population)
            parents = self.selection(population)
            offsprings = self.mating(parents)
            new_populaion = np.concatenate((parents, offsprings))
            mutated_new_population = self.mutation(new_populaion)
            self.population = mutated_new_population
            min_cost = np.min(self.func(population=self.population, xs=self.city_xs, ys=self.city_ys))

            print(min_cost)

        print(self.population)
        fig = plt.figure()
        plt.plot(self.city_xs, self.city_ys, 'ro', ms=10)
        res = np.argsort(self.func(xs=self.city_xs,
                                   ys=self.city_ys,
                                   population=population))
        xss = [self.city_xs[ic] for ic in self.population[res[0]]]
        xss.append(xss[0])
        yss = [self.city_ys[ic] for ic in self.population[res[0]]]
        yss.append(yss[0])
        plt.plot(xss, yss, 'b-')
        plt.show()

if __name__ == '__main__':
    g = TravelSalesmanGeneticAlgorithm(f)
    res = g.evolute()
