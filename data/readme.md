# Data Folder Description

- known_design.csv: range(0,150) first round result; range(150, ..) design space

- firstRound_4h_partial_data: contains pre-precessed data for partial first round results (150 sequences before lockdown), with 2,4,5 replicates for first plate and 1,2,3 for the second plate. 

- firstRound_Microplate_normFalse_formatSample_logTrue.csv
  firstRound_Microplate_normFalse_formatSeq_logTrue.csv
  firstRound_Microplate_normTrue_formatSample_logTrue.csv
  firstRound_Microplate_normTrue_formatSeq_logTrue.csv  
  Newest results (5 replicates), with log transformed  


- saved_normalised_kernel.npz: 
  - 'kernel': 4138 * 4138 normalised kernel matrix: wd kernel (l = 6, length_scale = 1, sigma_0 = 1) with shift (s = 1)

- idx_seq.npz: 
  - 'idxList': list of index, from 0 to 4137
  - 'seqList': list of RBS sequences
  - 'idxSeqDict': two-way dict for index and seq
