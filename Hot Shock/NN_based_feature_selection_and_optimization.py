# -*- coding: utf-8 -*-
"""
Final Project - Neural Network based Feature Selection and Optimization

@author: roshan94
"""
import pandas as pd
#from itertools import combinations
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

### Read data from dataset csv file
dataset_df = pd.read_csv('NN_dataset.csv')

## Separate into features and response for neural network training
feature_df = dataset_df.iloc[:,3:17]
safefac_df = dataset_df.iloc[:,17:19]

feature_columns = list(feature_df.columns)

## Preprocess features (Normalization)
feature_vals = feature_df.to_numpy()
feature_norm_vals = StandardScaler().fit_transform(feature_vals)
feature_norm_df = pd.DataFrame(feature_norm_vals)

print('Extracted all features and outputs from csv file')

### Feature Selection using Sequential Forward Search
n_feat_total = len(feature_norm_df.columns)

## Define function that trains the appropriate model and computes the adjusted R^2 metric 
## based on the combination of features
def compute_adj_R2(feat_data_df, Y_df, feat_tuple):
    # Extract appropriate feature values
    X_array = feat_data_df.iloc[:,list(feat_tuple)].to_numpy()
    Y_array = Y_df.to_numpy()
    
    # number of features 
    n_feat = len(list(feat_tuple))
    
    # Create Neural Network model
    nn_model = tf.keras.Sequential()
    nn_model.add(tf.keras.layers.Dense(10, input_shape=(n_feat,), activation='relu'))
    nn_model.add(tf.keras.layers.Dense(5, activation='relu'))
    nn_model.add(tf.keras.layers.Dense(2))
    nn_model.compile(optimizer='Adam', loss='mse')
    
    # Fit model to data
    nn_model.fit(X_array, Y_array, batch_size=32, epochs=20, verbose=0, shuffle=False)
    
    # Compute resubstitution RSS (which is the same as MSE)
    rss = nn_model.evaluate(X_array, Y_array, batch_size=32, verbose=0)
    
    # Compute TSS
    Y_mean = np.mean(Y_array, axis=0)
    sum_sq_diff = 0
    for i in range(Y_array.shape[0]):
        sq_diff = np.dot(np.transpose(Y_array[i]-Y_mean),Y_array[i]-Y_mean)
        sum_sq_diff += sq_diff
        
    tss = sum_sq_diff/Y_array.shape[0]
    
    #print('RSS')
    #print(rss)
    
    #print('TSS')
    #print(tss)
    
    num = rss/(n_feat_total - n_feat - 1)
    den = tss/(n_feat_total - 1)
    
    adj_R2 = 1 - (num/den)
    
    #print('adjusted R2')
    #print(adj_R2)
    
    #print('\n')
    
    return adj_R2, nn_model

## Exhaustive Search (coded but takes too much time)
#print('Starting Exhaustive Search')
#models = {}
#features = {}
#model_r2_adj = []
#feat_index_array = np.arange(n_feat_total)
#models_count = 0
#for j in range(n_feat_total):
    #for feat_comb in combinations(feat_index_array, j):
        #print('Computing adjusted R^2 metric for feature set ' + str(feat_comb))
        #current_model_r2_adj, current_model = compute_adj_R2(feature_norm_df, safefac_df, feat_comb)
        #models[models_count] = current_model
        #features[models_count] = feat_comb
        #model_r2_adj.append(current_model_r2_adj)
        #models_count += 1
    
#adj_r2_sort_indices = np.argsort(model_r2_adj)
#best_model = models[adj_r2_sort_indices[-1]]
#best_features = features[adj_r2_sort_indices[-1]]
#best_adj_r2 = model_r2_adj[adj_r2_sort_indices[-1]]

#print('Best feature set found is ' + str(best_features) + ' with adjusted R^2 value of ' + str(best_adj_r2))

## Sequential Forward Search
print('Starting Sequential Forward Search')
n_feat_sel = 8 # number of features to select
sel_feat_sets = {}
adj_r2_sets = []
sel_feats = []
feat_cand = np.arange(n_feat_total)
best_model = None
#current_best_r2 = 0 # uncomment and comment line 123 for incrementally improving feature sets
for j in range(n_feat_sel):
    current_best_r2 = 0
    for feat in feat_cand:
        feat_copy = sel_feats.copy()
        feat_copy.append(feat)
        current_adj_r2, current_nn = compute_adj_R2(feature_norm_df, safefac_df, feat_copy)
        if (current_adj_r2 > current_best_r2):
            current_best_r2 = current_adj_r2
            best_model = current_nn
            f_sel = feat
    sel_feats.append(f_sel)
    feat_cand = [f for f in feat_cand if f!=f_sel]
    sel_feat_sets[str(j+1)] = sel_feats.copy()
    adj_r2_sets.append(current_best_r2)

# Obtain corresponding feature names
sel_feat_names_sets = {}
for k in range(n_feat_sel):
    sel_feats_current = sel_feat_sets[str(k+1)]
    sel_feat_names = []
    for m in range(len(sel_feats_current)):
        sel_feat_names.append(feature_columns[sel_feats_current[m]])
    sel_feat_names_sets[str(k+1)] = sel_feat_names
    
print('\n')
print('Feature Sets with adjusted R^2 metrics:')
for s in range(n_feat_sel):
    print(str(sel_feat_names_sets[str(s+1)]) + ' : ' + str(adj_r2_sets[s]))
    
print('\n')

### GA based optimization using Neural Net 
### (currently coded to run only a single optimization for a given range of pane safety factor)
print('Starting GA based optimization')
best_feat_data = feature_norm_df.iloc[:,sel_feats]

## Define evaluation function
def get_frame_safety_fac_NN(index, best_data_df=best_feat_data, model_NN=best_model):
    X_index = best_data_df.iloc[index[0]].values
    
    Y_pred = model_NN.predict(np.array([X_index,]))
    
    return Y_pred[0][0],

## Define constraint function
def get_pane_safety_fac_constr(individual, pane_safety_fac_lb=0.05, pane_safety_fac_ub=0.075, best_data_df=best_feat_data, model_NN=best_model):
    X_index = best_data_df.iloc[individual[0]].values
    
    Y_pred = model_NN.predict(np.array([X_index,]))
    
    if pane_safety_fac_lb < Y_pred[0][1] < pane_safety_fac_ub:
        return False
    else:
        return True

## Define GA main function
def ga_main(popSize,genSize):
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

creator.create("FitnessMin", base.Fitness, weights=(1.0,)) #Create a single-objective maximization
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

INT_MIN = 0
INT_MAX = len(best_feat_data)-1

numDVs = 1

toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=numDVs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", get_frame_safety_fac_NN)

toolbox.decorate("evaluate", tools.DeltaPenalty(get_pane_safety_fac_constr,1E-5))
toolbox.register("mate", tools.cxUniform,indpb=0.05)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop,stats,hof = ga_main(popSize=100,genSize=100)

best_index = hof[0]
