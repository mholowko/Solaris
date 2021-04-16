# sanity check for reproducibility 
# direct to proper path
import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd

def check_dataset(data1, data2):
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
    
source1 = "data/pipeline_data/Results_Microplate_partialTrue_normTrue_mean_roundRep_formatSeq_logTrue_Round2.csv"
# source2 = "/home/admin-u6015325/ownCloud/SynBio_repo_designs/SynbioML-round1/data/firstRound_Microplate_normTrue_formatSeq_logTrue.csv"
source2 = "/home/admin-u6015325/ownCloud/SynBio_repo_designs/SynbioML-round2/data/Results_Microplate_partialTrue_normTrue_roundRep_formatSeq_logTrue.csv"
check_dataset(pd.read_csv(source1), pd.read_csv(source2))

# Log: round 1-3 passed the test (tested rep6 only)
# REVIEW: need to be careful about round 2, whether used the "correct" set of replicates to train?
