# provide configuration, e.g. global variable, path

BASES = ['A','C','G','T'] # embedding

SAVED_IDX_SEQ_PATH = '../data/idx_seq.pickle'
# SAVED_IDX_SEQ_PATH = '../data/idx_seq_lit.pickle'

# Usage:
# with open(SAVED_IDX_SEQ_PATH, 'rb') as handle:
#     saved_idx_seq = pickle.load(handle)
# seq_list = saved_idx_seq['seq_list']
# idx_list = saved_idx_seq['idx_list']
# idx_seq_dict = saved_idx_seq['idx_seq_dict']

SAVED_KERNEL_PATH = '../data/saved_kernel/'
# SAVED_KERNEL_PATH = '/data4/u6015325/Solaris/saved_kernel_lit/'

# ---------------------------------------------------------------------

# DRIVE_PATH = '/localdata/u6015325/SynbioML_drive/'
# SAVED_KERNEL_PATH = DRIVE_PATH+ 'saved_normalised_kernel.pickle'

# with open('../data/saved_normalised_kernel.pickle', 'rb') as handle:
#     kernel = pickle.load(handle)
    
# new_kernel = {}

# for key in kernel.keys():
    
#     with open('../data/' + key + '.pickle', 'wb') as handle:
#         pickle.dump(generate_triu(kernel[key]), handle, protocol=pickle.HIGHEST_PROTOCOL)

