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

#Read in beam dataset
fileName = 'Beam Materials Dataset.xlsx'

data = pd.read_excel(fileName)

data['Index'] = pd.Series(data['Youngs Modulus (GPa)']**0.5/data['Density (Mg/m^3)'])


def getMass(index,length=1,area=0.1):
    '''
    Calculates the mass of the beam
    Parameters
    ---------
    Inputs: density: material density (in Mg/m^3)
            length: length of the beam (in m), set to 1 as an optional argument
            area: area of the beam (in m^2), set to 0.1 m^2 as an optional argument
    Returns:
            mass: mass of the beam (in Mg)
    '''
    material = data['Material'].iloc[index[0]]
    density = data.loc[data['Material'] == material, 'Density (Mg/m^3)'].iloc[0]
    mass = density*length*area
    return mass,


def getStiffness(index,constant=3,sectionDim=0.01,length=1):
    '''
    Calculates the stiffness of the beam
    Parameters
    ----------
    Inputs: index: index of material
            constant: beam bending constant, set to 3 for a cantilever beam
                      (value taken from Ashby Appendix B.3)
            sectionDim: cross-sectional dimension of the beam, assuming a square.
                        set to 0.01 m as an optional argument
            length: length of the beam (in m), set to 1 as an optional argument
    Outputs:
        K : beam bending stiffness in GPa/m^3 
    '''
    material = data['Material'].iloc[index[0]]
    modulus = data.loc[data['Material'] == material, 'Youngs Modulus (GPa)'].iloc[0]
    I = 1.0/12.0*sectionDim**4.0 #Moment of inertia, in m^4
    K = constant*modulus*I/length**3.0 #Stiffness, in GPa/m^3
    
    return K 

def stiffnessConstraint(individual,targetStiffness=1E-10):
    '''
    Constraint function to only allow stiffnesses greater than the target stiffness value
    of the form -K+targetStiffness <=0
    Parameters
    ----------
    Inputs: K : stiffness of the individual
            target stiffness : optimization-specified target stiffness
    Outputs: constraint value
    '''
    K = getStiffness(individual)
    if -K+targetStiffness > 0:
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





creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #Create a single-objective minimization
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

INT_MIN = 0
INT_MAX = len(data)-1

numDVs = 1

toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=numDVs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", getMass)

toolbox.decorate("evaluate", tools.DeltaPenality(stiffnessConstraint,1E5))
toolbox.register("mate", tools.cxUniform,indpb=0.05)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


pop,stats,hof = main(popSize=50,genSize=50)

index = hof[0]
material = data['Material'].iloc[index[0]]
mass = getMass(index)
stiffness = getStiffness(index)
print('Material=',material,'Mass=',mass,'Stiffness=',stiffness)






