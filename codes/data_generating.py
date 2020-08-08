# 21/03/2020

# This code is used to generate the ready to use data 
# based on the first round result
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Generate valid data.')
parser.add_argument('normalize_flag', default= 'True', help = 'indicates whether normalize labels. True for normalized label.')
parser.add_argument('Format', default='Seq', help = 'Seq for rows as sequences; Sample for rows as samples')

args = parser.parse_args()
normalize_flag = str(args.normalize_flag) # str True or False
data_format = str(args.Format)

sheet_name = 'Microplate'
Log_flag = True # indicates whether take log label
Norm_method = 'mean' # indicates how to normalize label (one of 'mean', 'minmax', None)
Use_partial_rep = False
Folder_path = os.getcwd() # folder path might need to change for different devices

First_round_results_path = '/data/First_round_results/Results - First and Second Plate 3 reps.xlsx'
Generated_File_Path = '/data/firstRound_' + sheet_name + '_norm' + str(normalize_flag) + '_format' + data_format + '_log' + str(Log_flag) + '.csv'
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
# Rep1 - Rep5: GFPOD for the 4h (using derivatives) for three biological replicates.
# AVERAGE: average value of replicates
# STD: standard divation of replicates
# PERC: STD/AVERAGE	
# Group: one of 'bps', 'uni random', 'prob random', 'bandit'
# Usable: Yes if valid; otherwise No
# Replicates: the idx of replicates are valid to use

Path_new = Folder_path + First_round_results_path
df_new = pd.read_excel(Path_new, sheet_name= sheet_name)

# select only usable seq
df_new_usable = df_new[df_new['Usable'] == 'Yes']

if Use_partial_rep:
    # select only valid replicates
    df_new_valid = pd.DataFrame()
    df_new_valid[['Name', 'Group']] = df_new_usable[['Name', 'Group']]
    df_new_valid['RBS'] = df_new_usable['RBS'].str.upper() # convert to upper case
    df_new_valid['RBS6'] = df_new_usable['RBS'].str[7:13] # extract core part
    for idx, row in df_new_usable.iterrows():
        vad_reps = []
        for rep_idx in row['Replicates'].split(','):
            rep_name = 'Rep' + str(rep_idx)
            df_new_valid.loc[idx, rep_name] = row[rep_name]
            vad_reps.append(row[rep_name])
        df_new_valid.loc[idx, 'AVERAGE'] = np.mean(vad_reps)
        df_new_valid.loc[idx, 'STD'] = np.std(vad_reps)
else:
    df_new_valid = df_new_usable.copy()
    df_new_valid['RBS'] = df_new_usable['RBS'].str.upper() # convert to upper case
    df_new_valid['RBS6'] = df_new_usable['RBS'].str[7:13] # extract core part

# reorder columns
df_new_valid = df_new_valid[['Name', 'Group', 'RBS', 'RBS6', 'Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'AVERAGE', 'STD']]

# normalise each Rep respectively (zero mean and unit variance)

if normalize_flag == 'True':
    for col_name in ['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5']:
        df_new_norm = normalize(df_new_valid, col_name)

    df_new_norm['AVERAGE'] = df_new_norm.loc[: , "Rep1":"Rep5"].mean(axis=1)
    df_new_norm['STD'] = df_new_norm.loc[: , "Rep1":"Rep5"].std(axis=1)
    if data_format == 'Seq':
        df_new_norm.to_csv(Folder_path + Generated_File_Path)
    else: # sample
        df_new_norm_melt = pd.melt(df_new_norm, id_vars=['Name', 'RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep1'])
        df_new_norm_melt = df_new_norm_melt.rename(columns = {'value': 'label'})
        df_new_norm_melt = df_new_norm_melt.dropna()
        df_new_norm_melt.to_csv(Folder_path + Generated_File_Path)
elif data_format == 'Seq':
    print('seq, no normalises')
    df_new_valid.to_csv(Folder_path + Generated_File_Path)
else: # no normalise + samples
    df_new_valid_melt = pd.melt(df_new_valid, id_vars=['Name', 'RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep1'])
    df_new_valid_melt = df_new_valid_melt.rename(columns = {'value': 'label'})
    df_new_valid_melt = df_new_valid_melt.dropna()
    df_new_valid_melt.to_csv(Folder_path + Generated_File_Path)

"""

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
"""