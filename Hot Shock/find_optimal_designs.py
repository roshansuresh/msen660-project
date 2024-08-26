"""
Machine-learned material indices
Beam optimal materials optimization

Last updated: 11/21/2020

Optimization to find optimal materials that minimize mass subject to a 
maximum displacement in a square cross-section beam. Currently, the constraint
value for stiffness is hard-coded because I don't know how to pass through
additional parameters into the penalty decorator in DEAP
"""

import numpy as np
import pandas as pd

import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#Read in hot shock dataset
data = pd.read_csv('NN_dataset.csv')



def getPerfMetric(individual):
    '''
    Calculates the performance metric of the design
    Parameters
    ---------
    Inputs: individual: index of the individual that is considered
    Returns:
            perfMetric: performance metric for the considered individual. Right now hard coded to be frame fracture factor
    '''
    index = data['Sr. No.'].iloc[individual[0]]
    perfMetric = data.loc[data['Sr. No.'] == index, 'Frame Fracture Factor'].iloc[0]
    return perfMetric,


def constraintFunc(individual,targetFractureFactor=0.7):
    '''
    Constraint function to only allow stiffnesses greater than the target stiffness value
    of the form -K+targetStiffness <=0
    Parameters
    ----------
    Inputs: 
            targetFractureFactor : optimization-specified target performanceMetric
    Outputs: constraint value
    '''
    index = data['Sr. No.'].iloc[individual[0]]
    fractureFactor = data.loc[data['Sr. No.'] == index, 'Pane Fracture Factor'].iloc[0]
    if fractureFactor-targetFractureFactor > 0:
        return False
    else:
        return True


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2



def main(popSize,genSize):
    #random.seed(64)
    
    pop = toolbox.population(n=popSize)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=genSize, stats=stats,
                        halloffame=hof)
    
    #print(hof)

    return pop, stats, hof





creator.create("FitnessMin", base.Fitness, weights=(1.0,)) #Create a single-objective minimization
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

INT_MIN = 0
INT_MAX = len(data)-1

numDVs = 1

toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=numDVs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", getPerfMetric)

toolbox.decorate("evaluate", tools.DeltaPenality(constraintFunc,-1E5))
toolbox.register("mate", tools.cxUniform,indpb=0.05)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


pop,stats,hof = main(popSize=100,genSize=50)

index = hof[0]

fractureFactor = getPerfMetric(index)
paneFractureFactor = constraintFunc(index)
print('Index=',index,'Frame Fracture Factor=',fractureFactor,'Pane Fracture Factor=',paneFractureFactor)






