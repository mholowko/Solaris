# direct to proper path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import itertools
from collections import defaultdict
import math
import json

sim_rec_path = 'all_recs_topnUCB.npy'
result_path = '../data/Results_Salis.csv'
whole_size = 4096
# data_path = '../data/Comparison_Data/Model_Comparisons.csv'
# data_df = pd.read_csv(data_path, header= 0)


print('Load simulated recommendation file from {}'.format(sim_rec_path))
sim_recs = np.array(np.load(sim_rec_path, allow_pickle = True))
n_trial, n_round, n_batch = sim_recs.shape
# sim_recs = np.concatenate(sim_recs, axis = 0)
print('sim_recs shape: ', np.array(sim_recs).shape)

print('Load result file from {}'.format(result_path))
df = pd.read_csv(result_path, header = 0)
print(df.sort_values(by = ['AVERAGE'])['AVERAGE'])

all_rec_averages = []
for i in range(n_trial):
    rec_averages = []
    for j in range(n_round):
        rec_average = df.loc[df['RBS'].isin(sim_recs[i,j,:]), 'AVERAGE']
        rec_averages.append(rec_average)
        per_trial_max = np.sort(np.concatenate(rec_averages, axis = 0))[::-1][:3]
        print('Trial {} round {} max {} '.format(i, j ,per_trial_max))
    print()
    all_rec_averages.append(rec_averages)
print('all rec average shape: ', np.array(all_rec_averages).shape)

# all_rec_averages = np.array(all_rec_averages)
# all_rec_averages = all_rec_averages.
    



