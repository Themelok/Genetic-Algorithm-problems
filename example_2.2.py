import numpy as np
from fractions import Fraction
from numpy.random import rand, randint
from itertools import combinations
import copy

f = lambda x: 5 - 24 * x + 17 * x ** 2 - Fraction(11, 3) * x ** 3 + Fraction(1, 4) * x ** 4


class GeneticEvolutionNatPop:
    def __init__(self, func, mut_prob=0.3, pop_range=(), pop_size=4):
        self.func = func                        # Целевая функция
        self.population = []                    # Популяция
        self.mutation_probability = mut_prob    # Вероятность мутации
        self.population_range = pop_range       # Область определения функции
        self.population_size = pop_size         # Размер популяции
        self.chromosome_max_len = 0             # Максимальная строковая длинна бинарной хромосомы

    def generate_start_population(self):
        """Примим размер популяции равный population_size.
        Создадим начальную популяцию на заданном отрезке population_range"""
        self.population = np.random.randint(self.population_range[0],
                                            self.population_range[1] + 1,
                                            size=self.population_size).tolist()

    def crossover(self, parents):
        """Произведем процесс рекомбинации выбранных пар.
        Точка оператора рекомбинации выбирается случайно из {1,2...,pop_size-1}"""
        k = randint(1, self.chromosome_max_len)
        offsprings = [parents[0][:k] + parents[1][k:],
                      parents[1][:k] + parents[0][k:]]
        return offsprings

    def mutation(self, offspring):
        """Произведем мутации каждого из потомков с заданой вероятностью self.mutation_probability"""
        r = np.random.rand()
        if r < self.mutation_probability:
            gen_no = randint(0, self.chromosome_max_len)  # Определение номера гена для мутации
            list_offspring = list(offspring)
            list_offspring[gen_no] = '1' if list_offspring[gen_no] == '0' else '0'  # Инверсия случайного гена
            mutated_offspring = "".join(list_offspring)
        else:
            mutated_offspring = offspring
        return mutated_offspring

    def selection(self, population):
        """Произведем отбор наиболее приспособленных особей из нового поколения"""
        res = np.argsort([self.func(int(chrome, 2)) for chrome in population])
        res = res[:self.population_size]
        return np.array(population)[res].tolist()

    def evolute(self, n_step=10):
        """Основной метод генетического поиска"""
        self.generate_start_population()
        """Определение строковой длинны(число бит) максимальновозможного члена популяции"""
        self.chromosome_max_len = len(bin(self.population_range[1])[2:])

        for n in range(n_step):
            print('Шаг: ', n, " Популяция: ", self.population)
            """Запишим начальный набор хромосом в виде бинарных строк"""
            bin_population = [bin(chrom)[2:].zfill(self.chromosome_max_len) for chrom in self.population]
            new_population = copy.copy(bin_population)
            """  Составим массив возможных комбинаций родительских пар """
            probably_parents_arr = list(combinations(range(self.population_size), 2))
            for p in range(self.population_size//2):
                """ Определим пару для скрещивания """
                parents_numbers = probably_parents_arr.pop(np.random.randint(0, len(probably_parents_arr)))
                parents_binary = (bin_population[parents_numbers[0]], bin_population[parents_numbers[1]])
                """Произведем рекомбинацию отобранных потомков"""
                offsprings = self.crossover(parents_binary)
                mutated_offsprings = [self.mutation(offspring) for offspring in offsprings]
                new_population.extend(mutated_offsprings)
            bin_population = self.selection(new_population)
            self.population = [int(c, 2) for c in bin_population]
        return [float(self.func(x)) for x in self.population]


g = GeneticEvolutionNatPop(f, pop_range=(0, 7), mut_prob=0.4, pop_size=4)
res = g.evolute(n_step=20)
print('res', res)
