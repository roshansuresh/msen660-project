"""
Machine-learned material indices
Beam dataset

Last updated: 11/10/2020

Dataset creation and Ashby plotting for common engineering
materials used in beams.
"""

import numpy as np
import pandas as pd

import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

#Read in beam dataset
fileName = 'Beam Materials Dataset.xlsx'

data = pd.read_excel(fileName)

data['Index'] = pd.Series(data['Youngs Modulus (GPa)']**0.5/data['Density (Mg/m^3)'])


#After we get the optimal set of materials, perform regression to find material index
materials = ['Wood (longitudinal)','CFRP','Rigid polymer foam (LD)','Silicon']

features = ['Density (Mg/m^3)','Youngs Modulus (GPa)']

numDVs = len(features)

########################################################
###  FUNCTION INITIALIZATION
########################################################


def resizeBounds(value,bounds = [0,11]):
    '''Maps an entry of the design vector (bounds [0,1]) to the isotropic
    hardening modulus.
    Parameters
    ----------
    value : FLT
        Entry in the design vector
        
    Returns
    -------
    value : FLT
        Resized individual
    '''
    resizedValue = (bounds[1]-bounds[0])*value + bounds[0]
    
    #round down to the integer
    roundedValue = np.floor(resizedValue)
    
    possibleValues = [-3,-2,-1,-1/2,-1/3,0,1/3,1/2,1,2,3]
    print(roundedValue)
    value = possibleValues[int(roundedValue)]
    
    
    return value

def findIndex(individual,data,materials,features):
    '''
    Calculates the material index of each candidate material based on the design vector X
    Parameters
    ---------
    Inputs: individual: design vector corresponding to each power on each feature
            materials: list of candidate materials
            features: list of features to be comparing 
    Returns:
            error: The norm between all different material indices, given X
    '''
    index = np.zeros(shape=len(materials))
    errors = np.zeros(shape=len(materials))
    
    boundedIndividual = np.zeros(shape=len(individual))
    for i in range(len(individual)):
        boundedIndividual[i] = resizeBounds(individual[i])
    i = 0
    for material in materials:
        density = data.loc[data['Material'] == material, 'Density (Mg/m^3)'].iloc[0]
        modulus = data.loc[data['Material'] == material, 'Youngs Modulus (GPa)'].iloc[0]
        index[i] = density**boundedIndividual[0]*modulus**boundedIndividual[1]
        
        i +=1 
    
    meanIndex = np.mean(index)
    
    for i in range(len(materials)):
        errors[i] = abs(index[i]-meanIndex)
    error = np.mean(errors)
    
    if np.all(abs(boundedIndividual) < 0.1): #heavy-handed penalty constraint to make sure not all exponents go to zero
        error = 1E10
    
    
    return error,

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
    


########################################################
###  DEAP INITIALIZATION
########################################################
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #Create a single-objective minimization
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=numDVs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", findIndex,data=data,materials=materials, features=features)

#toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 1.0E10))
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


pop,stats,hof = main(popSize=50,genSize=50)


optimalSolution = pop[0]
boundedIndividual = np.zeros(shape=len(optimalSolution))

for i in range(len(optimalSolution)):
    boundedIndividual[i] = resizeBounds(optimalSolution[i])


print(boundedIndividual)


