import random
random.seed(0)
import numpy as np
np.random.seed(0)
from copy import deepcopy
from deap import base
from deap import creator
from deap import tools
import time

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

import truss_env
env = truss_env.Truss(test=3)
env.reset()

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("indices", random.sample, range(env.nm), env.nm)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def func(x):
    f,c = env.func(x,functype='maxfun')
    return f,

n_individuals = 50
n_elite = 1

toolbox.register("evaluate", func)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.005)
toolbox.register("selectRoulette",tools.selRoulette,fit_attr='fitness')

def main():
    pop = toolbox.population(n=n_individuals)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    CXPB  = 0.4   # crossover rate
    MUTPB = 1.0 # mutation occurence rate

    fits = [ind.fitness.values[0] for ind in pop]

    # Begin the evolution
    for gen in range(100):
        # print(f"-- Generation {gen+1} --")
        # Select parents for the next generation individuals
        parents = list(map(toolbox.clone, toolbox.selectRoulette(pop, len(pop))))
        # Apply crossover and mutation on the offspring
        pop_add = []
        for i, (child1, child2) in enumerate(zip(parents[::2], parents[1::2])):
            if random.random() < CXPB:
                index = np.random.randint(env.nm)
                dup = [child1[index]]
                if child2[index] in dup:
                    pass
                else:
                    while True:
                        child1[index], child2[index] = child2[index], child1[index]
                        if child1[index] in dup:
                            break
                        dup.append(child1[index])
                        i1 = np.where(child1==child1[index])[0]
                        if i1[0] == index:
                            index = i1[1]
                        else:
                            index = i1[0]
                    del child1.fitness.values
                    del child2.fitness.values
                    pop_add.append(child1)
                    pop_add.append(child2)
                

        for mutant in list(map(toolbox.clone,pop)):
            if random.random() < MUTPB:
                mutant_before = deepcopy(mutant)
                toolbox.mutate(mutant)
                if not np.all(mutant==mutant_before):
                    del mutant.fitness.values
                    pop_add.append(mutant)

        # Evaluate the individuals with an invalid fitness
        fitnesses = map(toolbox.evaluate, pop_add)
        for ind, fit in zip(pop_add, fitnesses):
            ind.fitness.values = fit
        fits += [ind.fitness.values[0] for ind in pop_add]
        new_pop = pop + pop_add

        '''
        selection (1): elite preservation
        '''
        rank = np.argsort(fits)[::-1]
        pop[:n_elite] = [new_pop[i] for i in rank[:n_elite]]

        '''
        selection (2): roulette selection of remaining individuals
        '''
        p = np.array([fits[i] for i in rank[n_elite:]])
        p /= p.sum()
        pop[n_elite:] = [new_pop[i] for i in np.random.choice(rank[n_elite:],n_individuals-n_elite,replace=False,p=p)]

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        # print("Min    %s" % min(fits))
        # print("Median %s" % np.median(fits))
        # print("Max    %s" % max(fits))
        # print("Avg    %s" % np.mean(fits))
        # print("Std    %s" % np.std(fits))
    
    print(f"max. fitness func: {max(fits)}")
    # env.func_render(pop[np.argmax(fits)])

t1 = time.time()
for i in range(5):
    main()
t2 = time.time()
print("time: {:.3f} seconds".format(t2-t1))