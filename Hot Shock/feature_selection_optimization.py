"""
Machine-learned material indices
Hot shock feature selection and regression

Last updated: 11/22/2020

"""

import numpy as np
import pandas as pd

import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from sklearn.preprocessing import StandardScaler

########################################################
###  FUNCTION INITIALIZATION
########################################################


def resizeBounds(value,bounds = [0,8]):
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
    
    possibleValues = [-2,-1,-1/2,-1/3,0,1/3,1/2,1,2]
    #print(roundedValue)
    value = possibleValues[int(roundedValue)]
    
    
    return value

def findIndex(individual,data,designs,features,feature_columns):
    '''
    Calculates the material index of each candidate material based on the design vector X
    Parameters
    ---------
    Inputs: individual: design vector corresponding to each power on each feature
            data: master dataframe
            designs: indexes corresponding to the optimal set
            features: list of features to be comparing 
    Returns:
            error: The norm between all different material indices, given X
    '''
    index = np.zeros(shape=len(designs))
    errors = np.zeros(shape=len(designs))
    
    featureVector = []

    #print(features)
    for feature in features:
        featureVector.append(feature_columns[feature])
    #print(featureVector)
    
    
    boundedIndividual = np.zeros(shape=len(individual))
    for i in range(len(individual)):
        boundedIndividual[i] = resizeBounds(individual[i])
    i = 0

    for design in designs:
        featureValues = []
        for feature in featureVector:
            featureValues.append(data.iloc[design][feature])
        
        
        
        index[i] = np.prod(np.power(featureValues,boundedIndividual))

        
        i +=1 
    
    meanIndex = np.mean(index)
    
    for i in range(len(designs)):
        errors[i] = abs(index[i]-meanIndex)
    error = np.mean(errors)/meanIndex  #Currently using a normalized mean absolute error
    print(designs,error,meanIndex,index)
    BREAK
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
    
# Write needed parameters to data file 
DataFile = open('features.txt','wt')
DataFile.write('Features \t Weights \t Error \n')
DataFile.close()
    


    
### Read data from dataset csv file
data = pd.read_csv('NN_dataset.csv')

#After we get the optimal set of materials, perform regression to find material index
optimalSet = [219,96,86,98,34,183,237] #indexes of the optimal set



## Separate into features and response for neural network training
feature_df = data.iloc[:,3:17]
safefac_df = data.iloc[:,17:19]

feature_columns = list(feature_df.columns)

## Preprocess features (Normalization)
feature_vals = feature_df.to_numpy()
feature_norm_vals = StandardScaler().fit_transform(feature_vals)
feature_norm_df = pd.DataFrame(feature_norm_vals,columns=feature_columns)

print('Extracted all features and outputs from csv file')

### Feature Selection using Sequential Forward Search
n_feat_total = len(feature_norm_df.columns)


#featureSet = ['Frame E [GPa]','Pane E [GPa]']

#numDVs = len(featureSet)


print('Starting Sequential Forward Search')
n_feat_sel = 9 # number of features to select
sel_feats = [0]
feat_cand = np.arange(1,n_feat_total)
best_model = None
#current_model = None
for j in range(1,n_feat_sel):
    current_best_r2 = 1.0
    for feat in feat_cand:
        feat_copy = sel_feats.copy()
        feat_copy.append(feat)
        numDVs = len(feat_copy)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #Create a single-objective minimization
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=numDVs)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", findIndex,data=feature_norm_df,designs=optimalSet,features=feat_copy,feature_columns=feature_columns)
        
        #toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 1.0E10))
        toolbox.register("mate", cxTwoPointCopy)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        
        pop,stats,hof = main(popSize=100,genSize=20)
        
        record = stats.compile(pop)
        
        
        optimalSolution = pop[0]
        boundedIndividual = np.zeros(shape=len(optimalSolution))
        
        for i in range(len(optimalSolution)):
            boundedIndividual[i] = resizeBounds(optimalSolution[i])
        
        current_adj_r2 = record['min']
        current_nn = boundedIndividual
        print(feat_copy,boundedIndividual,record['min'])
        

        #current_adj_r2, current_nn = compute_adj_R2(feature_norm_df, safefac_df, feat_copy)
        #current_model = current_nn
        if (current_adj_r2 < current_best_r2):
            current_best_r2 = current_adj_r2
            best_model = current_nn
            f_sel = feat
            print('Current features',feat,'Weights=',current_nn,'Error=',current_best_r2)

    sel_feats.append(f_sel)
    DataFile = open('features.txt','a')

    DataFile.write(str(sel_feats))
    DataFile.write('\t')
    for item in current_nn:
        DataFile.write('%.2f,' % item)
    DataFile.write('\t %.3f \n' % current_best_r2)
    DataFile.close()
    feat_cand = [f for f in feat_cand if f!=f_sel]

# Obtain corresponding feature names
sel_feat_names = []
for k in range(n_feat_sel):
    sel_feat_names.append(feature_columns[sel_feats[k]])
    
print('Best Feature Set is ' + str(sel_feat_names) + ' with adjusted R^2 metric of ' + str(current_best_r2))







