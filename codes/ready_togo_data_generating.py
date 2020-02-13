# This code is used to generate the ready to use data 
# based on the first round result

import numpy as np
import pandas as pd

Log_flag = False # indicates whether take log label
Norm_method = 'mean' # indicates how to normalize label (one of 'mean', 'minmax', None)

def normalize(df):
    # take log FC -- possiblely provide Gaussain distribution?
    if Log_flag:
        df['label'] = np.log(df['label'])
    if Norm_method == 'mean':
        # mean normalization
        df['label'] = (df['label'] - df['label'].mean())/df['label'].std()
    elif Norm_method == 'minmax':
        # min-max normalization 
        df['label'] = (df['label'] - df['label'].min())/(df['label'].max() - df['label'].min())
    else:
        assert Norm_method == None
        
    return df

# Data downloaded from https://github.com/synbiochem/opt-mva
# Paper https://pubs.acs.org/doi/abs/10.1021/acssynbio.8b00398

# A: whole RBS sequence (len: 29)
# B: extracted 20-base RBS seqeunce (A[7:27], len: 20), training features
# C: TIR labels
# D: the design part RBS (B[7:13], len: 6) 

Path_new = '../data/First_round_results/Results - First Plate 3 reps.csv'

df_new = pd.read_csv(Path_new)
df_new['RBS6'] = df_new['RBS'].str[7:13]

# TODO: drop outliers


# unpivot data

df_new_melt = pd.melt(df_new, id_vars=['RBS', 'RBS6'], value_vars=['Rep1', 'Rep2', 'Rep3'])
df_new_melt = df_new_melt.rename(columns = {'value': 'label'})

df_new_melt_norm = normalize(df_new_melt).dropna()
df_new_melt_norm['Group'] = 'First round result'

# -----------------------------------------------------------------------------------------
# Data downloaded from https://github.com/synbiochem/opt-mva
# Paper https://pubs.acs.org/doi/abs/10.1021/acssynbio.8b00398

# A: whole RBS sequence (len: 29)
# B: extracted 20-base RBS seqeunce (A[7:27], len: 20), training features
# C: TIR labels
# D: the design part RBS (B[7:13], len: 6) 

Path = '../data/Baseline_data/RBS_seqs.csv'

df = pd.read_csv(Path)
df.columns = ['Long_RBS', 'RBS', 'label']
df['RBS6'] = df['RBS'].str[7:13]
df = df.drop_duplicates(subset = ['RBS', 'label'])
df['variable'] = 'no'
df['Group'] = 'Baseline data'
df_baseline_norm = normalize(df[['RBS', 'RBS6', 'variable', 'label', 'Group']])

all_df = df_new_melt_norm.append(df_baseline_norm)
print(all_df)

all_df.to_csv('../data/firstRound+Baseline.csv')

