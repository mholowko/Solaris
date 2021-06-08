# sanity check for reproducibility 
# direct to proper path
# import os
# import sys
# module_path = os.path.abspath(os.path.join('../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# export PYTHONPATH=$PYTHONPATH:~/ownCloud/git/SynbioML

import numpy as np
import pandas as pd
from codes.batch_ucb import *
import codes.config

DRIVE_PATH = '/localdata/u6015325/SynbioML_drive/'


design_round = 1

rec_size = 90
l = 6
s = 1
alpha = 2
sigma_0 = 1
kernel = 'WD_Kernel_Shift'
embedding = 'label'
kernel_norm_flag = True
centering_flag = True
unit_norm_flag = True

check_dataset_flag = False
check_rec_flag = True


# ----------------------------------------------------------------------------------------------

def check_dataset(data1, data2, design_round):
    data1 = data1.where(data1['Round'] < design_round)
    data2 = data2.where(data1['Round'] < design_round)
    for index1, row1 in data1.iterrows():
        for index2, row2 in data2.iterrows():
            if row1['RBS'] == row2['RBS'] and row1['RBS'] != 'TTTAAGAAGGAGATATACAT':
                print(row1['RBS'])
                # print('ave diff: ', np.abs(row1['AVERAGE'] - row2['AVERAGE']))
                # print('std diff: ', np.abs(row1['STD'] - row2['STD']))
                
                # Instead of using average, we use a single replicate to check
                # Since AVERAGE calculation in round 1 and 2 were wrong 
                assert np.abs(row1['Rep6'] - row2['Rep6']) < 1e-3
                # assert np.abs(row1['STD'] - row2['STD'])< 1e-3
                break
    print('Dataset check for round '+ str(design_round) + ' passed!')

# if design_round == 0:
#     source1 = "data/pipeline_data/Results_d1.csv"
#     source2 = "SynBio_repo_designs/SynbioML-round0/data/firstRound_Microplate_normTrue_formatSeq_logTrue.csv"
if design_round == 1:  
    source1 = "data/pipeline_data/Results_bc1.csv"
    source2 = DRIVE_PATH + "SynBio_repo_designs/SynbioML-round1/data/firstRound_Microplate_normTrue_formatSeq_logTrue.csv"
elif design_round == 2:
    source1 = "data/pipeline_data/Results_abc1.csv"
    source2 = DRIVE_PATH + "SynBio_repo_designs/SynbioML-round2/data/Results_Microplate_partialTrue_normTrue_roundRep_formatSeq_logTrue.csv"
elif design_round == 3:
    source1 = "data/pipeline_data/Results_abc1.csv"
    source2 = DRIVE_PATH + "SynBio_repo_designs/SynbioML-round3/data/Results_Microplate_partialTrue_normTrue_mean_roundRep_formatSeq_logTrue.csv"

df_source1 = pd.read_csv(source1)
df_source2 = pd.read_csv(source2)

if check_dataset_flag:
    check_dataset(df_source2, df_source2, design_round)

# Log: round 1-3 passed the test (tested rep6 only)
# REVIEW: need to be careful about round 2, whether used the "correct" (original rep 1-3, 7-9) set of replicates to train?

# --------------------------------------------------------------------------------------------------------------------------


if design_round == 3:
    beta = 0
else:
    beta = 2

if design_round == 1: 
    kernel_over_all_flag = False
else:
    kernel_over_all_flag = True

if check_rec_flag:
    gpbucb = GP_BUCB(df_source1[df_source1['Round'] < design_round], kernel_name=kernel, l=l, s=s, sigma_0=sigma_0,
                    embedding=embedding, alpha=alpha, rec_size=rec_size, beta=beta, 
                    kernel_norm_flag=kernel_norm_flag, centering_flag = centering_flag,              
                    unit_norm_flag=unit_norm_flag, kernel_over_all_flag = kernel_over_all_flag)

    gpbucb_rec_df = gpbucb.run_experiment()
    new_rec = set(np.asarray(gpbucb_rec_df['RBS6']))
    lib_rec = set(np.asarray(df_source1[df_source1['Round'] == design_round]['RBS6']))
    # lib_rec = set(np.asarray(df_source1['RBS6']))
    num_overlap = len(new_rec.intersection(lib_rec))
    print('The overlap for round ' + str(design_round) + ' is ' + str(num_overlap))

# Log: round 3 rec passed the test (overlap 90, gpbucb with beta = 0/top-n)
# round 2 overlap 74 -> 90 (kernel normalisation over all design space, gpbucb with beta = 2)
# round 1 overlap 74 -> 88 (kernel normalisation over known features)