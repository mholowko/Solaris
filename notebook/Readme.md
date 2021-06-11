This folder shows experiments for SynBio with machine learning algorithms.

## Regression Design

- Evaluate based on the RMSE scoreThis folder shows experiments for SynBio with machine learning algorithms.

## Regression Design

- Evaluate based on the RMSE score (the smaller the better). 
- Labels are min-max normalized (between 0 and 1).
- Regression models: Gaussian process regression.
- Kernels: 
  - Kernels from sklearn library: DotProduct; RBF
  - [String kernels](https://dx.plos.org/10.1371/journal.pcbi.1000173): spectrum, mixed spectrum, weighted degree, weighted degree with shifting

## Recommendation design: Multi-armed Bandits

- Model: Upper confidence bound
- Comparison: Random selection
- Evaluation metric: expected cumulative regrets

## Design notebooks

### First round

Three approaches:
1. Baseline sequence (20') + change one position once (61 sequence totally).
2. Random (6') (4^6 combs): [random_seq_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/First_round_random_seq_design.ipynb) 
3. Bandit recommendations based on the current experiment:  [bandit_sequence_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/First_round_bandit_design.ipynb) 

The bandit experiments show our approaches better than random (not the arm space is now only limited to the available choices for sake of evaluation). 
We decide to design 61 (baseline) + 60 (random) + 60 (bandit). 

### Second Round

Based on the first-round result, we design the 90 second round sequences with [similar approaches](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/Second_round_design.ipynb).

### Analysis

- First round result: [violinplot](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/Label_violinplot.ipynb) which compares the performance of different groups; [regression on first round result](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/first_round_result_regression-Eva_on_ave.ipynb); [kernel matrix analysis](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/kernel_matrix_analysis.ipynb)
- Using clustering to show how spectrum kernel covers the design space and how well our design is: [kmeans](https://github.com/mholowko/SynbioML/tree/master/notebooks/rec_design/Clustering)
- [Sanity check](https://github.com/mholowko/SynbioML/tree/master/notebooks/rec_design/Sanity_check)
 (the smaller the better). 
- Labels are min-max normalized (between 0 and 1).
- Regression models: Gaussian process regression.
- Kernels: 
  - Kernels from sklearn library: DotProduct; RBF
  - [String kernels](https://dx.plos.org/10.1371/journal.pcbi.1000173): spectrum, mixed spectrum, weighted degree, weighted degree with shifting

## Recommendation desgin: Multi-armed Bandits

- Model: Upper confidence bound
- Comparison: Random selection
- Evaluation metric: expected cumulative regrets

## Design notebooks

### First round

Three approaches:
1. Baseline sequence (20') + change one position once (61 sequence totally).
2. Random (6') (4^6 combs): [random_seq_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/First_round_random_seq_design.ipynb) 
3. Bandit recommendations based on the current experiment:  [bandit_sequence_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/First_round_bandit_design.ipynb) 

The bandit experiments shows our approaches better than random (not the arm space is now only limited to the available choices for sake of evaluation). 
We decide to design 61 (baseline) + 60 (random) + 60 (bandit). 

### Second Round

Based on the first round result, we design the 90 second round sequences with [similar approaches](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/Second_round_design.ipynb).

### Analysis

- First round result: [violinplot](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/Label_violinplot.ipynb) which compares the performance of different groups; [regression on first round result](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/first_round_result_regression-Eva_on_ave.ipynb); [kernel matrix analysis](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/kernel_matrix_analysis.ipynb)
- Using clustering to show how spectrum kernel covers the design space and how well our design is: [kmeans](https://github.com/mholowko/SynbioML/tree/master/notebooks/rec_design/Clustering)
- [Sanity check](https://github.com/mholowko/SynbioML/tree/master/notebooks/rec_design/Sanity_check)
