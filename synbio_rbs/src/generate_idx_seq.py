# Nov.2020 Mengyan Zhang 
# This file generates a two-way dict (index, sequences) for all design RBS sequences,
# including 4^6 (core-part) + 3*14 (bps non-core group) = 4138 sequences. 

# usage:
# with open(Path, 'rb') as handle:
#    idx_seq = pickle.load(handle)

# idx_seq has three keys
# dict_keys(['idx_seq_dict', 'idx_list', 'seq_list'])

# idx_list and seq_list gives you the idx/seq in the same order
# idx_seq_dict is a two way dict, 
# you can do idx_seq_dict[20-long-RBS] = idx or idx_seq_dict[idx] = 20-long-RBS

import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import numpy as np
import itertools
import pickle

# class TwoWayDict(dict):
#     """bijective map between index and sequences.
#     code from Sasha Chedygov's answer https://stackoverflow.com/questions/1456373/two-way-reverse-map
#     """
#     def __setitem__(self, key, value):
#         # Remove any previous connections with these values
#         if key in self:
#             del self[key]
#         if value in self:
#             del self[value]
#         dict.__setitem__(self, key, value)
#         dict.__setitem__(self, value, key)

#     def __delitem__(self, key):
#         dict.__delitem__(self, self[key])
#         dict.__delitem__(self, key)

#     def __len__(self):
#         """Returns the number of connections"""
#         return dict.__len__(self) // 2 


# Setting
char_sets = ['A','C','G','T']
design_len = 6
pre_design = 'TTTAAGA'
pos_design = 'TATACAT'
index = 0
index_list = []
seq_list = []
idx_seq_dict = {}

for combo in itertools.product(char_sets, repeat= design_len):
    combo = pre_design + ''.join(combo) + pos_design
    index_list.append(index)
    seq_list.append(combo)
    idx_seq_dict[index] = combo
    idx_seq_dict[combo] = index
    index += 1
assert len(idx_seq_dict)/2 == len(char_sets) ** design_len

df = pd.read_excel(os.getcwd()+'/data/Results_Masterfile.xlsx', sheet_name= 'Microplate')
bps_noncore = df[df['Group'] == 'bps_noncore']['RBS']
for seq in bps_noncore:
    index_list.append(index)
    seq_list.append(seq.upper())
    idx_seq_dict[index] = seq.upper()
    idx_seq_dict[seq.upper()] = index
    index += 1
assert len(idx_seq_dict)/2 == len(char_sets) ** design_len + (len(char_sets) - 1) * (20-design_len)
print(len(idx_seq_dict)/2)

# np.savez(os.getcwd()+'/data/idx_seq.npz',idxList = index_list, seqList = seq_list, idxSeqDict = idx_seq_dict)
# print('two-way dict saved to ../data/idx_seq.npz')
# outfile = np.load(os.getcwd()+'/data/idx_seq.npz')
# for i in outfile['seqList']:
#     print(i)

save_dict = {}
save_dict['idx_seq_dict'] = idx_seq_dict
save_dict['idx_list'] = index_list
save_dict['seq_list'] = seq_list

with open(os.getcwd()+'/data/idx_seq.pickle', 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open(os.getcwd()+'/data/idx_seq.pickle', 'rb') as handle:
#     b = pickle.load(handle)
# print(b == idx_seq_dict)

# sanity check
# print('consensus index 648: ', idx_seq_dict[648])
# print('consensus seq TTTAAGAAGGAGATATACAT: ', idx_seq_dict['TTTAAGAAGGAGATATACAT'])
# print('index 3699: ', idx_seq_dict[3699])
# print('bandit design TTTAAGATGCTATTATACAT: ', idx_seq_dict['TTTAAGATGCTATTATACAT'])
# print('noncore index 4096: ', idx_seq_dict[4096])
# print('noncore seq CTTAAGAAGGAGATATACAT: ', idx_seq_dict['CTTAAGAAGGAGATATACAT'])