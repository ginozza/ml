import math
import timeit
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from bayes_opt import BayesianOptimization

class Graph:
    def __init__(self, cost_matrix: list, rank: int):
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank * rank) for _ in range(rank)] for _ in range(rank)]

class ACO:
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int, strategy: int):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = int(strategy)

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, _ in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph: Graph):
        best_cost = float('inf')
        for _ in range(self.generations):
            ants = [_Ant(self, graph) for _ in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
        return best_cost


class _Ant:
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []
        self.allowed = [i for i in range(graph.rank)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in range(graph.rank)]
        start = random.randint(0, graph.rank - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = sum(
            self.graph.pheromone[self.current][i] ** self.colony.alpha *
           self.eta[self.current][i] ** self.colony.beta
          for i in self.allowed
        )

        if denominator == 0 or math.isnan(denominator):
            selected = random.choice(self.allowed)
            self.allowed.remove(selected)
            self.tabu.append(selected)
            self.total_cost += self.graph.matrix[self.current][selected]
            self.current = selected
            return
        
        probabilities = [0] * self.graph.rank
        for i in self.allowed:
            num = (
                self.graph.pheromone[self.current][i] ** self.colony.alpha *
                self.eta[self.current][i] ** self.colony.beta
            )
            probabilities[i] = num / denominator

        rand = random.random()
        selected = None
        for i, p in enumerate(probabilities):
            rand -= p
            if rand <= 0:
                selected = i
                break

        if selected is None:
            selected = random.choice(self.allowed)

        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected


    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for _ in range(self.graph.rank)] for _ in range(self.graph.rank)]
        for k in range(1, len(self.tabu)):
            i = self.tabu[k - 1]
            j = self.tabu[k]
            if self.colony.update_strategy == 1:
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost

def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def load_graph(file_path='./input.txt'):
    cities = []
    with open(file_path) as f:
        for line in f:
            city = line.split(' ')
            cities.append(dict(index=int(city[0]), x=float(city[1]), y=float(city[2])))

    rank = len(cities)
    cost_matrix = [[distance(cities[i], cities[j]) for j in range(rank)] for i in range(rank)]
    return Graph(cost_matrix, rank)

graph = load_graph()

def objective(ant_count, generations, alpha, beta, rho, q, strategy):
    ant_count = int(ant_count)
    generations = int(generations)
    q = int(q)
    strategy = int(strategy)

    aco = ACO(ant_count, generations, alpha, beta, rho, q, strategy)

    runs = 3
    costs = []
    for _ in range(runs):
        cost = aco.solve(graph)
        costs.append(cost)
    avg_cost = np.mean(costs)

    return -avg_cost

pbounds = {
    'ant_count': (5, 50),
    'generations': (100, 800),
    'alpha': (0.5, 5),
    'beta': (0.5, 5),
    'rho': (0.1, 0.99),
    'q': (1, 10),
    'strategy': (0, 2) 
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(
    init_points=5,
    n_iter=15
)

print("\nMejores hiperparámetros encontrados:")
print(optimizer.max)