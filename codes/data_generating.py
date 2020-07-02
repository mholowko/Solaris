# 21/03/2020

# This code is used to generate the ready to use data 
# based on the first round result
import numpy as np
import pandas as pd
import os

sheet_name = '4h'
Log_flag = False # indicates whether take log label
Norm_method = 'mean' # indicates how to normalize label (one of 'mean', 'minmax', None)
Folder_path = os.getcwd() # folder path might need to change for different devices

First_round_results_path = '/data/First_round_results/Results - First and Second Plate 3 reps.xlsx'

def normalize(df, col_name):
    # take log FC -- possibly provide Gaussian distribution?
    if Log_flag:
        df[col_name] = np.log(df[col_name])
    if Norm_method == 'mean':
        # mean normalization
        df[col_name] = (df[col_name] - df[col_name].mean())/df[col_name].std()
    elif Norm_method == 'minmax':
        # min-max normalization 
        df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())
    else:
        assert Norm_method == None
        
    return df

#-------------------------------------------------------------------------------------------
# Process first round results file
# Columns: 
# RBS: 20-base RBS sequences
# Name: Group and index 
# Rep1, Rep2, Rep3: GFPOD for the 4h (using derivatives) for three biological replicates.
# AVERAGE: average value of replicates
# STD: standard divation of replicates
# PERC: STD/AVERAGE	
# Group: one of 'bps', 'uni random', 'prob random', 'bandit' 

Path_new = Folder_path + First_round_results_path
df_new = pd.read_excel(Path_new, sheet_name= sheet_name)
df_new['RBS'] = df_new['RBS'].str.upper() # convert to upper case
df_new['RBS6'] = df_new['RBS'].str[7:13] # extract core part

# exclude missing data (with AVERAGE > 100)
df_new = df_new[df_new['AVERAGE'] < 100]

# First step: normalise each Rep resectively (zero mean and unit variance)

for col_name in ['Rep1', 'Rep2', 'Rep3']:
    df_new_norm = normalize(df_new, col_name)

# Second step: drop outliers, where Rep3 may have some too big labels

df_new_norm['AVERAGE'] = df_new_norm.loc[: , "Rep1":"Rep3"].mean(axis=1)
df_new_norm['STD'] = df_new_norm.loc[: , "Rep1":"Rep3"].std(axis=1)

outliers = df_new_norm['STD'] > 1.5

print('Outliers:')
print(df_new_norm[outliers])

df_new_norm.at[outliers,'Rep3'] = np.nan

# Third step: calculate AVERAGE and STD

df_new_norm.at[outliers,'AVERAGE'] = df_new_norm.loc[outliers , "Rep1":"Rep2"].mean(axis=1)
df_new_norm.at[outliers,'STD'] = df_new_norm.loc[outliers , "Rep1":"Rep2"].std(axis=1)

df_new_norm.to_csv(Folder_path + '/data/firstRound_' + sheet_name + '.csv')

# Fourth step: put each replicate as one row

df_new_norm_melt = pd.melt(df_new_norm, id_vars=['RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep1', 'Rep2', 'Rep3'])
df_new_norm_melt = df_new_norm_melt.rename(columns = {'value': 'label'})

df_new_norm_melt = df_new_norm_melt.dropna()
#df_new_norm_melt['Group'] = 'First round result'

# -----------------------------------------------------------------------------------------
# Data downloaded from https://github.com/synbiochem/opt-mva
# Paper https://pubs.acs.org/doi/abs/10.1021/acssynbio.8b00398

# A: whole RBS sequence (len: 29)
# B: extracted 20-base RBS seqeunce (A[7:27], len: 20), training features
# C: TIR labels
# D: the design part RBS (B[7:13], len: 6) 

Path = Folder_path + '/data/Baseline_data/RBS_seqs.csv'

df = pd.read_csv(Path)
df.columns = ['Long_RBS', 'RBS', 'label']
df = normalize(df, 'label')

df['RBS6'] = df['RBS'].str[7:13]
df = df.drop_duplicates(subset = ['RBS', 'label'])
df['variable'] = 'no'
df['Group'] = 'Baseline data'
df['AVERAGE'] = df['label'].groupby(df['RBS']).transform('mean')
df['STD'] = df['label'].groupby(df['RBS']).transform('std')

df_baseline_norm = df[['RBS', 'RBS6', 'AVERAGE', 'STD', 'variable', 'label', 'Group']]

all_df = df_new_norm_melt.append(df_baseline_norm)
print(all_df)

all_df.to_csv(Folder_path + '/data/firstRound_' + sheet_name + '+Baseline.csv')
