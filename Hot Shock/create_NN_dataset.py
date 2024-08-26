# -*- coding: utf-8 -*-
"""
Final Project - Create dataset for Neural Network based Feature Selection and Optimization

@author: roshan94
"""
import pandas as pd
import numpy as np

### Read Material Properties.xlsx and store relevant material properties
mat_df = pd.read_excel('Material Properties2.xlsx', skipfooter=4)

# Define function to extract and store material properties
def store_mat_props(dataframe, mat_name):
    mat_k = dataframe.loc[dataframe['Material Candidates'] == mat_name,'k nominal [W/m-K]'].values
    mat_rho = dataframe.loc[dataframe['Material Candidates'] == mat_name,'rho [kg/m^3]'].values
    mat_Cp = dataframe.loc[dataframe['Material Candidates'] == mat_name,'cp [J/kg-K]'].values
    mat_diff = mat_k/(mat_rho*mat_Cp) # compute thermal diffusivity
    mat_props = [dataframe.loc[dataframe['Material Candidates'] == mat_name,'E Nominal [Gpa]'].values[0],
                dataframe.loc[dataframe['Material Candidates'] == mat_name,'K IC nominal [Mpa m^[1/2]] '].values[0], 
                dataframe.loc[dataframe['Material Candidates'] == mat_name,'alpha nominal [1E-6/K]'].values[0], mat_k[0], mat_diff[0]]
    return mat_props

## Store Pane Material Properties
# Store Alumina properties
alumina_props = store_mat_props(mat_df, 'Alumina')

# Store Slip Cast Fused Silica properties
scfs_props = store_mat_props(mat_df, 'Slip Cast Fused Silica')

# Store Reaction Bonded Silicon Nitride properties
rbsn_props = store_mat_props(mat_df, 'Reaction Bonded Silicon Nitride')

# Store Boron Nitride properties
bn_props = store_mat_props(mat_df, 'Boron Nitride (dense)')

# Store Magnesium Oxide properties
mgo_props = store_mat_props(mat_df, 'Magnesium Oxide')

# Store Beryl Oxide properties
beo_props = store_mat_props(mat_df, 'Beryl Oxide')

# Store Nextel 720: Ox-Ox properties
oxox_props = store_mat_props(mat_df, 'Nextel 720: Ox-Ox')

## Store Frame Material Properties
# Store C/C properties
cc_props = store_mat_props(mat_df, 'C/C')

# Store SiC/SiC properties
sicsic_props = store_mat_props(mat_df, 'SiC/SiC')

# Store C/SiC properties
csic_props = store_mat_props(mat_df, 'C/SiC')

print('Stored material properties')

### Read Hexagon Round 2.xlsx to obtain run data
run_df = pd.read_excel('Hexagon Round 2.xlsx')

# Remove unnecessary columns
run_array = run_df.iloc[:,[1,2,3,4,5,6,11,12,13,14]].to_numpy()

## Create data array with included material properties
nn_dataset = []
for i in range(len(run_array)):
    pane_mat_props = np.zeros(5)
    frame_mat_props = np.zeros(5)
    
    # Select appropriate frame material properties
    if (run_array[i][4] == 'C-C'):
        frame_mat_props = cc_props
    elif (run_array[i][4] == 'SiC-SiC'):
        frame_mat_props = sicsic_props
    elif (run_array[i][4] == 'C-SiC'):
        frame_mat_props = csic_props
        
    # Select appropriate pane material properties
    if (run_array[i][5] == 'Alumina'):
        pane_mat_props = alumina_props
    elif (run_array[i][5] == 'SCFS'):
        pane_mat_props = scfs_props
    elif (run_array[i][5] == 'RBSN'):
        pane_mat_props = rbsn_props
    elif (run_array[i][5] == 'BN'):
        pane_mat_props = bn_props
    elif (run_array[i][5] == 'MgO'):
        pane_mat_props = mgo_props
    elif (run_array[i][5] == 'BeO'):
        pane_mat_props = beo_props
    elif (run_array[i][5] == 'Ox-Ox'):
        pane_mat_props = oxox_props
        
    # Append material properties along with available data
    nn_dataset.append([i+1, run_array[i][4], run_array[i][5]] + list(frame_mat_props) 
                      + list(pane_mat_props) + list(run_array[i][0:4]) + list(run_array[i][-4:]))

print('Created appended dataset array')

### Save to a csv file
nn_data_df = pd.DataFrame(np.array(nn_dataset), columns=['Sr. No.', 'Frame Material', 'Pane Material', 
                                                         'Frame E [GPa]', 'Frame K IC [MPa m^[1/2]]', 'Frame alpha [1E-6/K]', 'Frame k [W/m-K]', 'Frame kappa [W m^2/J]', 
                                                         'Pane E [GPa]', 'Pane K IC [MPa m^[1/2]]', 'Pane alpha [1E-6/K]', 'Pane k [W/m-K]', 'Pane kappa [W m^2/J]', 
                                                         'Pane Area', 'Fillet Scale', 'Pane Depth', 'Frame Scale', 
                                                         'Frame Safety Factor', 'Pane Safety Factor', 'Frame Fracture Factor', 'Pane Fracture Factor'])

nn_data_df.to_csv('NN_dataset.csv', index=False)
print('Saved dataset to csv file')