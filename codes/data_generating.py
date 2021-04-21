# 25/11/2020 Update: add unique index corresponding to 'idx_seq_dict.npz' 

# 23/10/2020 Update: normalisation for each plate each replicate, 
# substracting mean of reference sequence in each plate

# 04/11/2020 Update: add Rep7,8,9

# This code is used to generate the ready to use data 
# based on the first round result
import numpy as np
import pandas as pd
import os
import argparse
# from generate_idx_seq import TwoWayDict
import pickle


# input parameters
parser = argparse.ArgumentParser(description='Generate valid data.')
parser.add_argument('normalize_flag', default= 'True', help = 'indicates whether normalize labels. True for normalized label.')
parser.add_argument('to_design_round', default= '3', help = 'specify the round to design.')

args = parser.parse_args()
normalize_flag = str(args.normalize_flag) # str True or False
to_design_round = str(args.to_design_round) # 0-4

# other settings
Use_partial_rep = True  # Update: for all data points, we uses replicates 1-6
# parser.add_argument('partial_rep_flag', default= 'True', help = 'indicates whether only using partial replicates (indicating in Replicate col). True for using partial reps.')
# str(args.partial_rep_flag) 
data_format = 'Seq' # Update: we only use seq format in downstream tasks, so constrained to 'Seq'
# str(args.Format) # str Seq or Sample
# parser.add_argument('Format', default='Seq', help = 'Seq for rows as sequences; Sample for rows as samples')
how_to_normalize = 'roundRep' # choices:  'plateRep', 'roundRep'
# WARNING: if one want to use plateRep, consensus sequence needs to be in each plate
COMPLETE_REP_SET = {'1','2','3','4','5','6'}
sheet_name = 'Microplate' # for masterfile
Log_flag = True # indicates whether take log label
Norm_method = 'mean' # indicates how to normalize label (one of 'mean', 'minmax', None)
round_normalisation = False

#-------------------------------------------------------------------------------------------------------------
# path 
Folder_path = os.getcwd() # folder path might need to change for different devices
Results_path = '/data/Results_Masterfile.xlsx'
Predictions_path = '/data/Designs/design_pred.xlsx'
idx_seq_path = '/data/idx_seq.pickle'

if normalize_flag == 'True':
    Generated_File_Path = '/data/pipeline_data/Results_' + sheet_name \
        + '_partial' + str(Use_partial_rep) + '_norm' + str(normalize_flag) \
        + '_' + str(Norm_method) + '_' + how_to_normalize + '_format' \
        + data_format + '_log' + str(Log_flag) +'_Round' + str(to_design_round) \
        + '_RN' + str(round_normalisation)+ '.csv'
else:
    Generated_File_Path = '/data/pipeline_data/Results_' + sheet_name + '_partial' + str(Use_partial_rep) + '_norm' + str(normalize_flag) + '_format' + data_format + '.csv'
#------------------------------------------------------------------------------------------------------------

def normalize(df, col_name):
    if how_to_normalize == 'plateRep': # normalise for each plate each replicate
        by_column = 'Plate'
    elif how_to_normalize  == 'roundRep':
        by_column  = 'Round'
    else:
        print('Unknown normalized type. Use roundRep instead')
        by_column = 'Round'

    normalised_df = pd.DataFrame()
    for name, group in df.groupby(by_column):
        # step1:
        # When we were about to design round 2, 
        # we observed that the TIR of the same RBS in each round turns out to be very different.
        # So we subtract the mean of the reference sequence (the only one RBS tested repeated in different rounds) in each round.
        if to_design_round in {'2', '3', '4'}: 
            ref_seq_mean = group.loc[group['Group'] == 'reference', col_name].stack().mean()
            print('col name: ', col_name)
            print('round: ', name)
            print('before substracting mean: ', group[col_name].mean())
            print('reference mean: ', ref_seq_mean)
            # print(group.loc[group['Group'] == 'reference', col_name])
            # +100 to avoid invalid value in log
            group[col_name] = group[col_name] - ref_seq_mean + 100
            print('after substracting mean: ', group[col_name].mean())
        #--------------------------------------------------------------------------------------
        # step2: take log
        if Log_flag:
            group[col_name] = np.log(group[col_name])
        # check the mean of reference sequences to be the same
        # print(group.loc[group['Group'] == 'reference', col_name].stack().mean())
        #--------------------------------------------------------------------------------------- 
        # step3: normalisation
        if round_normalisation:
            if Norm_method == 'mean':
                # Z normalization
                group[col_name] = (group[col_name] - group[col_name].mean())/group[col_name].std()
            elif Norm_method == 'minmax':
                # min-max normalization 
                group[col_name] = (group[col_name] - group[col_name].min())/(group[col_name].max() - group[col_name].min())
            else:
                assert Norm_method == None
            
        normalised_df = normalised_df.append(group)

    normalised_df[col_name] = (normalised_df[col_name] - normalised_df[col_name].mean())/normalised_df[col_name].std()
        # print(name)
        # print(normalised_df)
    return normalised_df

# rename group names
def rename_group_names(df):
    df['Group'] = df['Group'].replace({
                        'consensus': 'Consensus',
                        'reference': 'Reference', 
                        'bps_core':'BPS-C', 'bps_noncore': 'BPS-NC', 
                        'uni random': 'UNI', 'prob random': 'PPM', 
                        'bandit': 'Bandit-0', 'bandit2': 'Bandit-1',
                        'bandit3': 'Bandit-2', 'bandit4': 'Bandit-3'})
    return df

# add index
def assign_idx(df):
    with open(Folder_path + idx_seq_path, 'rb') as handle:
        idx_seq_dict = pickle.load(handle)['idx_seq_dict']
    for df_index, row in df.iterrows():
        print(idx_seq_dict[str(row['RBS']).upper()])
        df.loc[df_index,'idx'] = idx_seq_dict[str(row['RBS']).upper()]
    
    # put index at first col
    first_col = df.pop('idx')
    df.insert(0, 'idx', first_col)
    return df

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

# only output the round before to_design_round
df_new = df_new[df_new['Round'] < int(to_design_round)]

df_new = assign_idx(df_new)
# df_pred = assign_idx(df_pred)

# select only usable seq
df_new_usable = df_new[df_new['Usable'] == 'Yes']

df_new_valid = df_new_usable.copy()
df_new_valid['RBS'] = df_new_usable['RBS'].str.upper() # convert to upper case
df_new_valid['RBS6'] = df_new_usable['RBS'].str[7:13] # extract core part

if Use_partial_rep == 'True':
    # remove unused columns
    for idx, row in df_new_usable.iterrows():
        remove_columns = COMPLETE_REP_SET - set(str(row['Replicates']).split(','))
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
        df_new_valid = df_new_valid[['Name', 'Group', 'Plate', 'Round', 'RBS', 'RBS6', 'Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 
                                    'AVERAGE', 'STD', 'Pred Mean', 'Pred Std', 'Pred UCB']]
    else:
        # reorder columns
        df_new_valid = df_new_valid[['Name', 'Group', 'Plate', 'Round', 'RBS', 'RBS6', 'Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 
                                    'AVERAGE', 'STD']]

# normalise each Rep respectively (zero mean and unit variance)
# the normalisation should be done in terms of each plate
if normalize_flag == 'True':
    df_new_norm = normalize(df_new_valid, ['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6'])
    # else:
    #     for col_name in ['Rep1', 'Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6']:
    #         df_new_norm = normalize(df_new_valid, col_name)
    df_new_norm['AVERAGE'] = df_new_norm.loc[: , "Rep1":"Rep6"].mean(axis=1)
    df_new_norm['STD'] = df_new_norm.loc[: , "Rep1":"Rep6"].std(axis=1)
    if data_format == 'Seq':
        rename_group_names(df_new_norm).to_csv(Folder_path + Generated_File_Path, index = False)
    else: # sample
        df_new_norm_melt = pd.melt(df_new_norm, id_vars=['Name', 'RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], 
                                    value_vars=['Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep1'])
        df_new_norm_melt = df_new_norm_melt.rename(columns = {'value': 'label'})
        df_new_norm_melt = df_new_norm_melt.dropna()
        rename_group_names(df_new_norm_melt).to_csv(Folder_path + Generated_File_Path, index = False)
elif data_format == 'Seq':
    print('seq, no normalises')
    rename_group_names(df_new_valid).to_csv(Folder_path + Generated_File_Path, index = False)
else: # no normalise + samples
    df_new_valid_melt = pd.melt(df_new_valid, id_vars=['Name', 'RBS', 'RBS6', 'AVERAGE', 'STD', 'Group'], value_vars=['Rep2', 'Rep3', 'Rep4', 'Rep5', 'Rep6', 'Rep7', 'Rep8', 'Rep9', 'Rep1'])
    df_new_valid_melt = df_new_valid_melt.rename(columns = {'value': 'label'})
    df_new_valid_melt = df_new_valid_melt.dropna()
    rename_group_names(df_new_valid_melt).to_csv(Folder_path + Generated_File_Path, index = False)

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