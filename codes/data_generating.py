# 23/10/2020 Update: normalisation for each plate each replicate, 
# substracting mean of reference sequence in each plate

# 04/11/2020 Update: add Rep7,8,9

# This code is used to generate the ready to use data 
# based on the first round result
import numpy as np
import pandas as pd
import os
import argparse


# input parameters
parser = argparse.ArgumentParser(description='Generate valid data.')
parser.add_argument('partial_rep_flag', default= 'True', help = 'indicates whether only using partial replicates (indicating in Replicate col). True for using partial reps.')
parser.add_argument('normalize_flag', default= 'True', help = 'indicates whether normalize labels. True for normalized label.')
parser.add_argument('Format', default='Seq', help = 'Seq for rows as sequences; Sample for rows as samples')

args = parser.parse_args()
normalize_flag = str(args.normalize_flag) # str True or False
Use_partial_rep = str(args.partial_rep_flag) # str True or False
data_format = str(args.Format) # str Seq or Sample

# settings
how_to_normalize = 'roundRep' # choices:  'plateRep', 'roundRep'
COMPLETE_REP_SET = {'1','2','3','4','5','6','7','8','9'}
sheet_name = 'Microplate' # for masterfile
Log_flag = True # indicates whether take log label
Norm_method = 'mean' # indicates how to normalize label (one of 'mean', 'minmax', None)

# path 
Folder_path = os.getcwd() # folder path might need to change for different devices
Results_path = '/data/Results_Masterfile.xlsx'
Predictions_path = '/data/Designs/design_pred.xlsx'
if normalize_flag == 'True':
    Generated_File_Path = '/data/Results_' + sheet_name + '_partial' + str(Use_partial_rep) + '_norm' + str(normalize_flag) + '_' + how_to_normalize + '_format' + data_format + '_log' + str(Log_flag) + '.csv'
else:
    Generated_File_Path = '/data/Results_' + sheet_name + '_partial' + str(Use_partial_rep) + '_norm' + str(normalize_flag) + '_format' + data_format + '_log' + str(Log_flag) + '.csv'

def normalize(df, col_name):
    if how_to_normalize == 'plateRep': # normalise for each plate each replicate
        by_column = 'Plate'
    elif how_to_normalize  == 'roundRep':
        by_column  = 'Round'
    else:
        print('Unknown normalized type. Use plateRep instead')
        by_column = 'Plate'

    normalised_df = pd.DataFrame()
    for name, group in df.groupby(by_column):
        # step1: substract the mean of the reference sequence in each group
        # the reason to do that is the values of each group (plate) turns out to be different
        # and the only same sequence is the reference sequence.
        ref_seq_mean = group.loc[group['Group'] == 'reference', col_name].stack().mean()
        print(name)
        print(ref_seq_mean)
        print(group.loc[group['Group'] == 'reference', col_name])
        # +100 to avoid invalid value in log
        group[col_name] = group[col_name] - ref_seq_mean + 100
        # step2: take log
        if Log_flag:
            group[col_name] = np.log(group[col_name])
        # check the mean of reference sequences to be the same
        # print(group.loc[group['Group'] == 'reference', col_name].stack().mean())
        # step3: normalisation
        if Norm_method == 'mean':
            # Z normalization
            group[col_name] = (group[col_name] - group[col_name].mean())/group[col_name].std()
        elif Norm_method == 'minmax':
            # min-max normalization 
            group[col_name] = (group[col_name] - group[col_name].min())/(group[col_name].max() - group[col_name].min())
        else:
            assert Norm_method == None
        
        normalised_df = normalised_df.append(group)
        # print(name)
        # print(normalised_df)
    return normalised_df


    # # Todo: if still want to use ref/all, you need to substract mean of concensus sequence somehow
    # elif how_to_normalize == 'rep':
    #     if Log_flag:
    #         df[col_name] = np.log(df[col_name])
    #     if Norm_method == 'mean':
    #         # mean normalization
    #         df[col_name] = (df[col_name] - df[col_name].mean())/df[col_name].std()
    #     elif Norm_method == 'minmax':
    #         # min-max normalization 
    #         df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())
    #     else:
    #         assert Norm_method == None
    #     return df
    # elif how_to_normalize == 'all':
    #     if Log_flag:
    #         df[col_name] = np.log(df[col_name])
    #     if Norm_method == 'mean':
    #         # mean normalization
    #         # print(df[col_name].stack().std().type)
    #         df[col_name] = (df[col_name] - df[col_name].stack().mean())/df[col_name].stack().std()
    #     elif Norm_method == 'minmax':
    #         # min-max normalization 
    #         df[col_name] = (df[col_name] - df[col_name].stack().min(axis=1))/(df[col_name].stack().max() - df[col_name].stack().min())
    #     else:
    #         assert Norm_method == None
    #     return df
    # else: 
    #     print('Unknown Normalization, output unnormalised data.')
    #     return df


#-------------------------------------------------------------------------------------------
# Process first round results file
# Columns: 
# RBS: 20-base RBS sequences
# Name: Group and index 
# Rep1 - Rep6: GFPOD for the 4h (using derivatives) for three biological replicates.
# AVERAGE: average value of replicates
# STD: standard divation of replicates
# PERC: STD/AVERAGE	
# Group: one of 'bps', 'uni random', 'prob random', 'bandit'
# Usable: Yes if valid; otherwise No
# Replicates: the idx of replicates are valid to use

df_new = pd.read_excel(Folder_path + Results_path, sheet_name= sheet_name)
df_pred = pd.read_excel(Folder_path + Predictions_path, sheet_name= 'gpbucb_alpha2_beta2')

# select only usable seq
df_new_usable = df_new[df_new['Usable'] == 'Yes']

df_new_valid = df_new_usable.copy()
df_new_valid['RBS'] = df_new_usable['RBS'].str.upper() # convert to upper case
df_new_valid['RBS6'] = df_new_usable['RBS'].str[7:13] # extract core part

if Use_partial_rep == 'True':
    # remove unused columns
    for idx, row in df_new_usable.iterrows():
        remove_columns = COMPLETE_REP_SET - set(row['Replicates'].split(','))
        # print(remove_columns)
        for rep_idx in remove_columns:
            rep_name = 'Rep' + str(rep_idx)
            df_new_valid.loc[idx, rep_name] = np.nan   
    df_new_valid = df_new_valid.merge(df_pred, how = 'left', on = 'RBS').drop(columns = ['RBS6_y', 'Group_y'])  
    df_new_valid = df_new_valid.rename(columns = {'Group_x': 'Group', 'RBS6_x': 'RBS6'})
else:
    # only add pred for normalised data
    if normalize_flag == 'True': 
        df_new_valid = df_new_valid.merge(df_pred, how = 'left', on = 'RBS').drop(columns = ['RBS6_y', 'Group_y'])  
        df_new_valid = df_new_valid.rename(columns = {'Group_x': 'Group', 'RBS6_x': 'RBS6'})
        df_new_valid = df_new_valid[['Name', 'Group', 'Plate', 'Round', 'RBS', 'RBS6', 'Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep7', 'Rep8', 'Rep9',
                                    'AVERAGE', 'STD', 'Pred Mean', 'Pred Std', 'Pred UCB']]
    else:
        # reorder columns
        df_new_valid = df_new_valid[['Name', 'Group', 'Plate', 'Round', 'RBS', 'RBS6', 'Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep7', 'Rep8', 'Rep9',
                                    'AVERAGE', 'STD']]

# normalise each Rep respectively (zero mean and unit variance)
# the normalisation should be done in terms of each plate
if normalize_flag == 'True':
    df_new_norm = normalize(df_new_valid, ['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep7', 'Rep8', 'Rep9'])
    # else:
    #     for col_name in ['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6']:
    #         df_new_norm = normalize(df_new_valid, col_name)
    df_new_norm['AVERAGE'] = df_new_norm.loc[: , "Rep1":"Rep6"].mean(axis=1)
    df_new_norm['STD'] = df_new_norm.loc[: , "Rep1":"Rep6"].std(axis=1)
    if data_format == 'Seq':
        df_new_norm.to_csv(Folder_path + Generated_File_Path)
    else: # sample
        df_new_norm_melt = pd.melt(df_new_norm, id_vars=['Name', 'RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], 
                                    value_vars=['Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep7', 'Rep8', 'Rep9', 'Rep1'])
        df_new_norm_melt = df_new_norm_melt.rename(columns = {'value': 'label'})
        df_new_norm_melt = df_new_norm_melt.dropna()
        df_new_norm_melt.to_csv(Folder_path + Generated_File_Path)
elif data_format == 'Seq':
    print('seq, no normalises')
    df_new_valid.to_csv(Folder_path + Generated_File_Path)
else: # no normalise + samples
    df_new_valid_melt = pd.melt(df_new_valid, id_vars=['Name', 'RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep7', 'Rep8', 'Rep9', 'Rep1'])
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